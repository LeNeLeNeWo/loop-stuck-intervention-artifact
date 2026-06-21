from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from datasets import load_dataset

from live_guard_common import (
    ARMS,
    MINI_SWE_PACKAGE,
    append_csv,
    apply_deepseek_openai_env,
    ensure_output,
    read_csv_dicts,
    redacted_env_for_litellm,
    redactor,
    repo_root,
    run_cmd,
    silence_mini_swe_startup,
    stable_hash,
    utc_now,
    write_json,
)
from live_guard_policies import GuardState, count_repetitions


RUN_LOG_FIELDS = [
    "released_task_id",
    "arm",
    "run_id",
    "status",
    "model",
    "runner",
    "task_source",
    "traceability_level",
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
    "oracle_available",
    "resolved",
    "tests_pass",
    "patch_generated",
    "repeated_file_count",
    "repeated_tool_count",
    "repeated_file_rate",
    "repeated_tool_rate",
    "unique_files_touched",
    "error_summary",
    "raw_log_path_local",
]


def add_mini_swe_to_path(root: Path) -> None:
    silence_mini_swe_startup()
    apply_deepseek_openai_env()
    pkg = root / MINI_SWE_PACKAGE
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))


def load_instances(sample_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    by_source: dict[tuple[str, str], set[str]] = {}
    for row in sample_rows:
        source = row["task_source"]
        dataset_path, split = source.rsplit(":", 1)
        by_source.setdefault((dataset_path, split), set()).add(row["benchmark_instance_id"])
    instances: dict[str, dict[str, Any]] = {}
    for (dataset_path, split), ids in by_source.items():
        ds = load_dataset(dataset_path, split=split)
        for item in ds:
            iid = str(item.get("instance_id", ""))
            if iid in ids:
                instances[iid] = dict(item)
    return instances


def load_runner_config(root: Path, max_steps: int, timeout_minutes: int) -> dict[str, Any]:
    add_mini_swe_to_path(root)
    from minisweagent.config import get_config_from_spec
    from minisweagent.utils.serialize import recursive_merge

    config_path = root / "outputs/paper1_live_guard_experiment/runner_config/deepseek_litellm_swebench.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"runner config missing: {config_path}")
    configs = [
        get_config_from_spec("swebench.yaml"),
        get_config_from_spec(str(config_path)),
        {"agent": {"step_limit": max_steps, "wall_time_limit_seconds": timeout_minutes * 60}},
    ]
    return recursive_merge(*configs)


class GuardedAgent:  # wrapper subclass is created dynamically after mini-swe-agent import
    pass


def make_guarded_agent_class(root: Path):
    add_mini_swe_to_path(root)
    from minisweagent.agents.default import DefaultAgent

    class _GuardedAgent(DefaultAgent):
        def __init__(self, *args, arm: str, run_metadata: dict[str, Any] | None = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.guard_state = GuardState(arm=arm)
            self.arm = arm
            self.run_metadata = run_metadata or {}
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
            guard_payload = {
                "live_guard_experiment": {
                    "arm": self.arm,
                    "run_metadata": self.run_metadata,
                    "guard_events": [event.__dict__ for event in self.guard_state.events],
                    "actions_by_step": self.actions_by_step,
                }
            }
            return super().serialize(guard_payload, *extra_dicts)

    return _GuardedAgent


def token_usage_from_messages(messages: list[dict[str, Any]]) -> tuple[int, int]:
    prompt = 0
    completion = 0
    for msg in messages:
        usage = (((msg.get("extra") or {}).get("response") or {}).get("usage") or {})
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
    return prompt, completion


def existing_pairs(log_path: Path, retry_infrastructure_failed: bool = False) -> set[tuple[str, str]]:
    rows = read_csv_dicts(log_path)
    pairs = set()
    for row in rows:
        status = row.get("status", "")
        if retry_infrastructure_failed and status == "infrastructure_failed":
            continue
        if status:
            pairs.add((row.get("released_task_id", ""), row.get("arm", "")))
    return pairs


def select_rows(sample_rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected = [row for row in sample_rows if str(row.get("selected_for_run", "")).lower() in {"yes", "true", "1"}]
    if args.dry_run is not None:
        return selected[: args.dry_run]
    if args.pilot is not None:
        return selected[: args.pilot]
    if args.full:
        return selected
    raise SystemExit("Specify one of --dry-run N, --pilot N, or --full.")


def run_one(root: Path, out: Path, row: dict[str, str], instance: dict[str, Any], arm: str, args: argparse.Namespace) -> dict[str, Any]:
    add_mini_swe_to_path(root)
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    config = load_runner_config(root, args.max_steps, args.timeout_minutes)
    run_id = stable_hash(row["released_task_id"], arm, utc_now(), n=12)
    raw_dir = out / "live_guard_raw_runs" / row["released_task_id"] / arm
    raw_dir.mkdir(parents=True, exist_ok=True)
    traj_path = raw_dir / f"{run_id}.traj.json"
    config["agent"]["output_path"] = traj_path
    model = get_model(config=config.get("model", {}))
    env = None
    agent = None
    start = time.perf_counter()
    start_iso = utc_now()
    status = "completed"
    error_summary = ""
    info: dict[str, Any] = {}
    try:
        env = get_sb_environment(copy.deepcopy(config), instance)
        AgentClass = make_guarded_agent_class(root)
        agent = AgentClass(
            model,
            env,
            arm=arm,
            run_metadata={
                "released_task_id": row["released_task_id"],
                "task_source": row["task_source"],
                "traceability_level": row["traceability_level"],
            },
            **config.get("agent", {}),
        )
        info = agent.run(instance.get("problem_statement", ""))
    except Exception as exc:
        status = "infrastructure_failed"
        error_summary = redactor()(f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")[:1000]
    finally:
        if agent is not None:
            try:
                agent.save(traj_path)
            except Exception:
                pass
        if env is not None and hasattr(env, "cleanup"):
            try:
                env.cleanup()
            except Exception:
                pass
    wall = time.perf_counter() - start
    messages = agent.messages if agent is not None else []
    prompt_tokens, completion_tokens = token_usage_from_messages(messages)
    events = agent.guard_state.events if agent is not None else []
    reps = count_repetitions(agent.actions_by_step if agent is not None else [])
    exit_status = info.get("exit_status", "") if isinstance(info, dict) else ""
    if exit_status == "LimitsExceeded":
        stop_reason = "step_or_cost_limit"
    elif exit_status == "TimeExceeded":
        stop_reason = "wall_time_limit"
    elif exit_status == "HardStopGuardTriggered":
        stop_reason = "hard_stop_guard"
    elif exit_status == "Submitted":
        stop_reason = "submitted"
    elif error_summary:
        stop_reason = "exception"
    else:
        stop_reason = exit_status or "completed"
    patch = info.get("submission", "") if isinstance(info, dict) else ""
    row_out = {
        "released_task_id": row["released_task_id"],
        "arm": arm,
        "run_id": run_id,
        "status": status,
        "model": os.environ.get("DEEPSEEK_MODEL", ""),
        "runner": "mini-swe-agent",
        "task_source": row.get("task_source", ""),
        "traceability_level": row.get("traceability_level", ""),
        "start_time": start_iso,
        "end_time": utc_now(),
        "wall_clock": f"{wall:.3f}",
        "steps": agent.n_calls if agent is not None else 0,
        "tokens_input": prompt_tokens,
        "tokens_output": completion_tokens,
        "estimated_cost": f"{agent.cost:.8f}" if agent is not None else "0",
        "guard_triggered": bool(events),
        "trigger_step": events[0].step if events else "",
        "trigger_count": len(events),
        "stop_reason": stop_reason,
        "oracle_available": False,
        "resolved": "",
        "tests_pass": "",
        "patch_generated": bool(str(patch).strip()),
        "repeated_file_count": reps["repeated_file_count"],
        "repeated_tool_count": reps["repeated_tool_count"],
        "repeated_file_rate": f"{reps['repeated_file_rate']:.6f}",
        "repeated_tool_rate": f"{reps['repeated_tool_rate']:.6f}",
        "unique_files_touched": reps["unique_targets"],
        "error_summary": error_summary,
        "raw_log_path_local": str(traj_path),
    }
    return row_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live guard-policy experiment using mini-swe-agent and DeepSeek.")
    parser.add_argument("--dry-run", type=int, default=None)
    parser.add_argument("--pilot", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-infrastructure-failed", action="store_true")
    parser.add_argument("--arms", nargs="+", default=["NO_GUARD", "WARN_REPLAN_GUARD", "HARD_STOP_GUARD"])
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--timeout-minutes", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    invalid = [arm for arm in args.arms if arm not in ARMS]
    if invalid:
        raise SystemExit(f"Unknown arms: {invalid}")
    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)
    sample_path = out / "live_task_sample.csv"
    if not sample_path.exists():
        raise SystemExit(f"Missing sample file: {sample_path}")
    sample_rows = select_rows(read_csv_dicts(sample_path), args)
    instances = load_instances(sample_rows)
    log_path = out / "live_guard_run_log.csv"
    seen = existing_pairs(log_path, retry_infrastructure_failed=args.retry_infrastructure_failed) if args.resume else set()
    write_json(
        out / "live_guard_run_plan.json",
        {
            "generated_at": utc_now(),
            "mode": "dry_run" if args.dry_run is not None else "pilot" if args.pilot is not None else "full",
            "task_count": len(sample_rows),
            "arms": args.arms,
            "max_steps": args.max_steps,
            "timeout_minutes": args.timeout_minutes,
            "resume": args.resume,
        },
    )
    completed = 0
    for row in sample_rows:
        instance = instances.get(row["benchmark_instance_id"])
        if not instance:
            for arm in args.arms:
                append_csv(
                    log_path,
                    {
                        **{field: "" for field in RUN_LOG_FIELDS},
                        "released_task_id": row["released_task_id"],
                        "arm": arm,
                        "status": "skipped",
                        "model": os.environ.get("DEEPSEEK_MODEL", ""),
                        "runner": "mini-swe-agent",
                        "task_source": row.get("task_source", ""),
                        "traceability_level": row.get("traceability_level", ""),
                        "error_summary": "benchmark instance not found in loaded dataset",
                    },
                    RUN_LOG_FIELDS,
                )
            continue
        for arm in args.arms:
            if (row["released_task_id"], arm) in seen:
                continue
            print(f"[{utc_now()}] running {row['released_task_id']} {arm}", flush=True)
            result = run_one(root, out, row, instance, arm, args)
            append_csv(log_path, result, RUN_LOG_FIELDS)
            completed += 1
            if args.sleep:
                time.sleep(args.sleep)
    print(f"Live guard run complete. New task-arm rows: {completed}")
    print(f"Run log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
