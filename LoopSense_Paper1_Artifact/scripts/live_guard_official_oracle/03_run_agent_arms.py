from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from datasets import load_dataset

from official_common import (
    ARMS,
    DATASET_NAME,
    DATASET_SPLIT,
    MINI_SWE_PACKAGE,
    PREV_LIVE_SCRIPT_SUBDIR,
    append_csv,
    apply_deepseek_openai_env,
    ensure_output,
    read_csv_dicts,
    redactor,
    repo_root,
    stable_hash,
    utc_now,
    write_json,
)


RUN_FIELDS = [
    "released_task_id",
    "benchmark_instance_id",
    "arm",
    "run_id",
    "status",
    "model",
    "runner",
    "task_source",
    "start_time",
    "end_time",
    "wall_clock",
    "steps",
    "tokens_input",
    "tokens_output",
    "estimated_cost",
    "guard_triggered",
    "trigger_step",
    "trigger_count",
    "stop_reason",
    "patch_generated",
    "patch_path_local",
    "patch_bytes",
    "changed_files_count",
    "repeated_file_count",
    "repeated_tool_count",
    "repeated_file_rate",
    "repeated_tool_rate",
    "post_trigger_repetition_rate",
    "raw_log_path_local",
    "error_summary",
]

PATCH_FIELDS = [
    "released_task_id",
    "benchmark_instance_id",
    "arm",
    "run_id",
    "model_name_or_path",
    "patch_generated",
    "patch_path_local",
    "patch_bytes",
    "changed_files_count",
    "status",
    "notes",
]


def add_paths(root: Path) -> None:
    apply_deepseek_openai_env()
    for p in [root / MINI_SWE_PACKAGE, root / PREV_LIVE_SCRIPT_SUBDIR]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load_config(root: Path, max_steps: int, timeout_minutes: int) -> dict[str, Any]:
    add_paths(root)
    from minisweagent.config import get_config_from_spec
    from minisweagent.utils.serialize import recursive_merge

    model = os.environ.get("DEEPSEEK_MODEL", "")
    config_path = ensure_output(root) / "runner_config" / "deepseek_litellm_swebench.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "model:",
                f'  model_name: "openai/{model}"',
                '  model_class: "litellm"',
                "  model_kwargs:",
                "    drop_params: true",
                "    temperature: 0.0",
                "    timeout: 60",
                '  cost_tracking: "ignore_errors"',
                "agent:",
                f"  step_limit: {max_steps}",
                "  cost_limit: 0.0",
                f"  wall_time_limit_seconds: {timeout_minutes * 60}",
                "environment:",
                "  timeout: 60",
                "  container_timeout: 3h",
                "  pull_timeout: 900",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return recursive_merge(
        get_config_from_spec("swebench.yaml"),
        get_config_from_spec(str(config_path)),
        {"agent": {"step_limit": max_steps, "wall_time_limit_seconds": timeout_minutes * 60}},
    )


def guarded_agent_class(root: Path):
    add_paths(root)
    from live_guard_policies import GuardState, count_repetitions
    from minisweagent.agents.default import DefaultAgent

    class GuardedAgent(DefaultAgent):
        def __init__(self, *args, arm: str, **kwargs):
            super().__init__(*args, **kwargs)
            self.arm = arm
            self.guard_state = GuardState(arm=arm)
            self.actions_by_step: list[list[dict[str, Any]]] = []

        def execute_actions(self, message: dict) -> list[dict]:
            actions = message.get("extra", {}).get("actions", []) or []
            outputs = [self.env.execute(action) for action in actions]
            observations = self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))
            self.actions_by_step.append(actions)
            event = self.guard_state.observe_actions(self.n_calls, actions)
            if event and event.decision == "warn_replan":
                self.add_messages(self.guard_state.warning_message())
            elif event and event.decision == "hard_stop":
                self.add_messages(
                    {
                        "role": "exit",
                        "content": "HardStopGuardTriggered",
                        "extra": {
                            "exit_status": "HardStopGuardTriggered",
                            "submission": "",
                            "guard_event": event.__dict__,
                        },
                    }
                )
            return observations

        def serialize(self, *extra_dicts):
            payload = {
                "official_oracle_live_guard": {
                    "arm": self.arm,
                    "guard_events": [event.__dict__ for event in self.guard_state.events],
                    "actions_by_step": self.actions_by_step,
                }
            }
            return super().serialize(payload, *extra_dicts)

    return GuardedAgent, count_repetitions


def load_instances(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    ids = {row["benchmark_instance_id"] for row in rows}
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    return {str(row["instance_id"]): dict(row) for row in ds if str(row["instance_id"]) in ids}


def selected_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    rows = sorted(rows, key=lambda r: int(r.get("selected_order") or 999999))
    if args.task_range:
        start, end = parse_task_range(args.task_range)
        return [r for r in rows if start <= int(r.get("selected_order") or 999999) <= end]
    if args.dry_run is not None:
        return rows[: args.dry_run]
    if args.pilot is not None:
        return rows[: args.pilot]
    if args.full:
        return rows[: args.full_limit]
    raise SystemExit("Specify --dry-run N, --pilot N, or --full.")


def parse_task_range(value: str) -> tuple[int, int]:
    if ":" not in value:
        n = int(value)
        return n, n
    start, end = value.split(":", 1)
    return int(start), int(end)


def existing_pairs(path: Path, retry_infra: bool) -> set[tuple[str, str]]:
    pairs = set()
    for row in read_csv_dicts(path):
        if retry_infra and row.get("status") == "infrastructure_failed":
            continue
        if row.get("status"):
            pairs.add((row.get("released_task_id", ""), row.get("arm", "")))
    return pairs


def usage(messages: list[dict[str, Any]]) -> tuple[int, int]:
    prompt = 0
    completion = 0
    for msg in messages:
        u = (((msg.get("extra") or {}).get("response") or {}).get("usage") or {})
        prompt += int(u.get("prompt_tokens") or 0)
        completion += int(u.get("completion_tokens") or 0)
    return prompt, completion


def docker_rm(container_id: str | None) -> None:
    if not container_id:
        return
    subprocess.run(["docker", "rm", "-f", container_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def capture_patch(env: Any) -> tuple[str, int]:
    try:
        out = env.execute({"command": "git diff --binary"}, timeout=180)
        patch = out.get("output", "") if isinstance(out, dict) else ""
    except Exception:
        patch = ""
    try:
        files_out = env.execute({"command": "git diff --name-only"}, timeout=60)
        changed = [x for x in str(files_out.get("output", "")).splitlines() if x.strip()]
    except Exception:
        changed = []
    return patch, len(changed)


def run_one(root: Path, out: Path, sample_row: dict[str, str], instance: dict[str, Any], arm: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    add_paths(root)
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    config = load_config(root, args.max_steps, args.timeout_minutes)
    run_id = stable_hash(sample_row["released_task_id"], arm, utc_now(), n=12)
    run_dir = out / "agent_raw_runs" / sample_row["released_task_id"] / arm / run_id
    patch_dir = out / "patches" / arm
    run_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)
    traj_path = run_dir / f"{run_id}.traj.json"
    patch_path = patch_dir / f"{sample_row['released_task_id']}__{arm}__{run_id}.patch"
    config["agent"]["output_path"] = traj_path
    start = time.perf_counter()
    start_iso = utc_now()
    model = get_model(config=config.get("model", {}))
    AgentClass, count_repetitions = guarded_agent_class(root)
    env = None
    agent = None
    info: dict[str, Any] = {}
    status = "completed"
    error = ""
    patch = ""
    changed_files_count = 0
    try:
        env = get_sb_environment(copy.deepcopy(config), instance)
        agent = AgentClass(model, env, arm=arm, **config.get("agent", {}))
        info = agent.run(instance.get("problem_statement", ""))
        patch, changed_files_count = capture_patch(env)
    except Exception as exc:
        status = "infrastructure_failed"
        error = redactor()(f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}")[:1600]
        if env is not None:
            patch, changed_files_count = capture_patch(env)
    finally:
        if agent is not None:
            try:
                agent.save(traj_path)
            except Exception:
                pass
        if env is not None:
            docker_rm(getattr(env, "container_id", None))
    if not patch and isinstance(info, dict):
        patch = str(info.get("submission") or "")
    patch_generated = bool(patch.strip())
    if patch_generated:
        patch_path.write_text(patch, encoding="utf-8", errors="replace")
    else:
        patch_path.write_text("", encoding="utf-8")
    messages = agent.messages if agent is not None else []
    pt, ct = usage(messages)
    reps = count_repetitions(agent.actions_by_step if agent is not None else [])
    events = agent.guard_state.events if agent is not None else []
    trigger_step = events[0].step if events else ""
    post_rate = reps["repeated_file_rate"]
    exit_status = info.get("exit_status", "") if isinstance(info, dict) else ""
    stop_reason = "exception" if error else (exit_status or "completed")
    if exit_status == "HardStopGuardTriggered":
        stop_reason = "hard_stop_guard"
    row = {
        "released_task_id": sample_row["released_task_id"],
        "benchmark_instance_id": sample_row["benchmark_instance_id"],
        "arm": arm,
        "run_id": run_id,
        "status": status,
        "model": os.environ.get("DEEPSEEK_MODEL", ""),
        "runner": "mini-swe-agent",
        "task_source": sample_row.get("task_source", ""),
        "start_time": start_iso,
        "end_time": utc_now(),
        "wall_clock": f"{time.perf_counter() - start:.3f}",
        "steps": agent.n_calls if agent is not None else 0,
        "tokens_input": pt,
        "tokens_output": ct,
        "estimated_cost": f"{agent.cost:.8f}" if agent is not None else "0",
        "guard_triggered": bool(events),
        "trigger_step": trigger_step,
        "trigger_count": len(events),
        "stop_reason": stop_reason,
        "patch_generated": patch_generated,
        "patch_path_local": str(patch_path),
        "patch_bytes": len(patch.encode("utf-8", errors="replace")),
        "changed_files_count": changed_files_count,
        "repeated_file_count": reps["repeated_file_count"],
        "repeated_tool_count": reps["repeated_tool_count"],
        "repeated_file_rate": f"{reps['repeated_file_rate']:.6f}",
        "repeated_tool_rate": f"{reps['repeated_tool_rate']:.6f}",
        "post_trigger_repetition_rate": f"{post_rate:.6f}",
        "raw_log_path_local": str(traj_path),
        "error_summary": error,
    }
    patch_row = {
        "released_task_id": sample_row["released_task_id"],
        "benchmark_instance_id": sample_row["benchmark_instance_id"],
        "arm": arm,
        "run_id": run_id,
        "model_name_or_path": os.environ.get("DEEPSEEK_MODEL", ""),
        "patch_generated": patch_generated,
        "patch_path_local": str(patch_path),
        "patch_bytes": len(patch.encode("utf-8", errors="replace")),
        "changed_files_count": changed_files_count,
        "status": status,
        "notes": stop_reason,
    }
    return row, patch_row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mini-swe-agent arms for official-oracle live guard experiment.")
    parser.add_argument("--dry-run", type=int, default=None)
    parser.add_argument("--pilot", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--full-limit", type=int, default=30)
    parser.add_argument("--task-range", default=None, help="Inclusive selected_order range, e.g. 31:60.")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--timeout-minutes", type=int, default=45)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-infrastructure-failed", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    invalid = [arm for arm in args.arms if arm not in ARMS]
    if invalid:
        raise SystemExit(f"Unknown arms: {invalid}")
    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)
    sample = out / "official_oracle_task_sample.csv"
    if not sample.exists():
        raise SystemExit(f"Missing sample: {sample}")
    rows = selected_rows(read_csv_dicts(sample), args)
    instances = load_instances(rows)
    run_log = out / "agent_run_log.csv"
    patch_manifest = out / "patch_manifest.csv"
    seen = set() if args.force or not args.resume else existing_pairs(run_log, args.retry_infrastructure_failed)
    plan_name = "official_oracle_run_plan_60.json" if args.task_range or args.full_limit >= 60 else "official_oracle_run_plan.json"
    write_json(
        out / plan_name,
        {
            "generated_at": utc_now(),
            "mode": "task_range" if args.task_range else "dry_run" if args.dry_run is not None else "pilot" if args.pilot is not None else "full",
            "task_range": args.task_range or "",
            "tasks": len(rows),
            "arms": args.arms,
            "max_steps": args.max_steps,
            "timeout_minutes": args.timeout_minutes,
            "resume": args.resume,
        },
    )
    n = 0
    for row in rows:
        instance = instances.get(row["benchmark_instance_id"])
        if not instance:
            continue
        for arm in args.arms:
            if (row["released_task_id"], arm) in seen:
                continue
            print(f"[{utc_now()}] running {row['selected_order']} {row['benchmark_instance_id']} {arm}", flush=True)
            run_row, patch_row = run_one(root, out, row, instance, arm, args)
            append_csv(run_log, run_row, RUN_FIELDS)
            append_csv(patch_manifest, patch_row, PATCH_FIELDS)
            n += 1
            if args.sleep:
                time.sleep(args.sleep)
    print(f"New task-arm runs: {n}")
    print(f"Run log: {run_log}")
    print(f"Patch manifest: {patch_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
