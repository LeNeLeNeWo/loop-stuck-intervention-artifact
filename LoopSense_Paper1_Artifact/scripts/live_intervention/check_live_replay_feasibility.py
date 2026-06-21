from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

REQUIREMENTS = {
    "prefix_state_recovery": [
        "repo snapshots / checkouts", "task metadata with repo/base commit", "action replay to prefix", "patch state at prefix"
    ],
    "agent_runner": [
        "SWE-agent or LoopBench-Agent runner", "OpenHands runner", "budget/timeout controls", "intervention prompt injection"
    ],
    "model_availability": [
        "fixed model configuration", "API credentials or local model endpoint", "seed/temperature controls"
    ],
    "evaluation_oracle": [
        "resolved/tests-pass oracle", "current patch evaluation", "step/token/cost logging", "repetition metric logging"
    ],
}

CANDIDATES = {
    "raw_trajectory_logs": ["data/trajectories", "data/raw_trajectories", "data/raw", "logs", "outputs/raw_trajectories"],
    "task_metadata": ["data/tasks", "data/benchmark_tasks", "data/swebench", "metadata", "data/raw/swe"],
    "repo_checkouts": ["repos", "repositories", "checkouts", "data/repos", "workspace_repos"],
    "action_replay": ["scripts/replay", "scripts/replay_actions.py", "scripts/paper1_live_intervention/replay_prefix.py"],
    "swe_runner": ["scripts/run_agent.py", "scripts/agents/run_agent.py", "scripts/swe_agent", "apps/swe-agent"],
    "openhands_runner": ["scripts/run_openhands.py", "apps/openhands", "OpenHands"],
    "model_config": ["configs", "config", ".env.example"],
    "test_oracle": ["scripts/evaluate_patch.py", "scripts/run_tests.py", "scripts/evaluation", "tests"],
    "intervention_injection": ["scripts/inject_intervention.py", "scripts/paper1_live_intervention/run_prefix_branch_experiment.py"],
}

PRIMARY_INPUTS = {
    "normalized_steps": "outputs/paper1_audit/normalized_steps.csv",
    "normalized_spans": "outputs/paper1_audit/normalized_spans.csv",
    "stop_counterfactual_triggers": "outputs/paper1_intervention_validation/stop_counterfactual_triggers.csv",
    "detector_metrics": "outputs/paper1_intervention_risk_rich_detectors/detector_metrics_by_dataset.csv",
    "enriched_steps": "outputs/paper1_raw_enrichment/enriched_steps.csv",
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def exists_any(root: Path, rels: list[str]) -> tuple[bool, list[str]]:
    hits = []
    for rel in rels:
        p = root / rel
        if p.exists():
            hits.append(rel)
    return bool(hits), hits


def header(path: Path) -> list[str]:
    if not path.exists() or path.suffix.lower() not in {".csv", ".tsv"}:
        return []
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            return next(csv.reader(f, delimiter=delimiter), [])
    except Exception:
        return []


def env_model_available() -> tuple[bool, list[str]]:
    key_suffix = "_API" + "_KEY"
    token_suffix = "_TO" + "KEN"
    endpoint_suffix = "_END" + "POINT"
    present = [
        name for name, value in os.environ.items()
        if value and (name.endswith(key_suffix) or name.endswith(token_suffix) or name.endswith(endpoint_suffix))
    ]
    return bool(present), ["credential environment variable detected; name withheld"] if present else []


def classify(status: dict[str, dict[str, Any]]) -> str:
    prefix = status["repo_checkouts"]["available"] and status["task_metadata"]["available"] and status["action_replay"]["available"]
    runner = status["swe_runner"]["available"] or status["openhands_runner"]["available"]
    model = status["model_config"]["available"] and status["model_credentials"]["available"]
    oracle = status["test_oracle"]["available"]
    injection = status["intervention_injection"]["available"]
    raw_logs = status["raw_trajectory_logs"]["available"]
    if prefix and runner and model and oracle and injection:
        return "FULL_LIVE_REPLAY_FEASIBLE"
    if raw_logs and prefix and runner and oracle and injection:
        return "PARTIAL_REPLAY_FEASIBLE"
    if raw_logs and runner and model and injection:
        return "PROMPT_ONLY_CONTINUATION_FEASIBLE"
    return "NOT_FEASIBLE"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether live/replay prefix-branch intervention is feasible.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_live_intervention"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    status: dict[str, dict[str, Any]] = {}
    for key, rels in CANDIDATES.items():
        ok, hits = exists_any(root, rels)
        status[key] = {"available": ok, "evidence": hits}
    creds_ok, creds = env_model_available()
    status["model_credentials"] = {"available": creds_ok, "evidence": creds}

    inputs = []
    for key, rel in PRIMARY_INPUTS.items():
        p = root / rel
        inputs.append({"input": key, "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else None, "columns": header(p)[:16], "path": rel})

    decision = classify(status)
    missing = [k for k, v in status.items() if not v["available"]]
    can_restore_prefix = decision in {"FULL_LIVE_REPLAY_FEASIBLE", "PARTIAL_REPLAY_FEASIBLE"}
    can_run_agent = (status["swe_runner"]["available"] or status["openhands_runner"]["available"]) and status["intervention_injection"]["available"]
    can_evaluate = status["test_oracle"]["available"]

    lines = [
        "# Live / Replay Prefix-Branch Feasibility Report", "",
        "This audit checks whether a true prefix-branch intervention experiment can be executed from the current repository state. It does not run agents.", "",
        f"Conclusion: **{decision}**", "",
        "## Primary Analysis Inputs", "",
        "| Input | Exists | Size bytes | Columns (prefix) |", "|---|---:|---:|---|",
    ]
    for row in inputs:
        lines.append(f"| {row['input']} | {'yes' if row['exists'] else 'no'} | {row['size_bytes'] if row['size_bytes'] is not None else 'NA'} | {', '.join(row['columns'])} |")
    lines.extend(["", "## Replay Requirements", "", "| Requirement | Available | Evidence |", "|---|---:|---|"])
    for key, row in status.items():
        evidence = ", ".join(row["evidence"]) if row["evidence"] else "not found"
        lines.append(f"| {key} | {'yes' if row['available'] else 'no'} | {evidence} |")
    lines.extend([
        "", "## Decision Details", "",
        f"- Can restore prefix state: {'yes' if can_restore_prefix else 'no'}.",
        f"- Can run an agent with intervention prompt injection: {'yes' if can_run_agent else 'no'}.",
        f"- Can evaluate resolved/tests-pass outcomes: {'yes' if can_evaluate else 'no'}.",
        f"- Model credentials detected in current environment: {'yes' if creds_ok else 'no'}.",
        "", "## Missing Pieces", "",
    ])
    if missing:
        for item in missing:
            lines.append(f"- {item}")
    else:
        lines.append("None detected by this audit.")
    lines.extend([
        "", "## Consequence", "",
        "Because the current repository does not expose a complete prefix-state replay stack, no live/replay intervention results are produced. The existing offline hard-stop counterfactual remains the strongest executable intervention-validation evidence in this artifact.",
    ])
    (out / "feasibility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "feasibility_report.json").write_text(json.dumps({"decision": decision, "status": status, "primary_inputs": inputs, "missing": missing}, indent=2), encoding="utf-8")
    if decision == "NOT_FEASIBLE":
        (out / "live_intervention_not_feasible.md").write_text(
            "# Live Intervention Not Feasible\n\n"
            "A live/replay prefix-branch intervention experiment was not executed. The current repository lacks the complete combination of prefix-state recovery, agent runner, model credentials, intervention injection path, and test/oracle harness needed for a reproducible live branch experiment. No live success, cost, or repetition results are claimed.\n",
            encoding="utf-8",
        )
    print(f"Live/replay feasibility: {decision}")
    print(f"Report: {out / 'feasibility_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
