from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RESULT_COLUMNS = [
    "pool", "trajectory_id", "prefix_step", "arm", "success", "patch_valid", "steps_after_prefix", "tokens", "cost",
    "wall_clock_seconds", "repeated_action_count", "repeated_tool_count", "repeated_file_count", "repeated_error_count",
    "new_file_touched_count", "changed_error_count", "test_pass_event_count", "final_status", "failure_mode",
]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prefix-branch intervention arms when a replay stack is available.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_live_intervention"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    decision = "NOT_FEASIBLE"
    feas = out / "feasibility_report.json"
    if feas.exists():
        decision = json.loads(feas.read_text(encoding="utf-8")).get("decision", "NOT_FEASIBLE")
    if decision not in {"FULL_LIVE_REPLAY_FEASIBLE", "PARTIAL_REPLAY_FEASIBLE", "PROMPT_ONLY_CONTINUATION_FEASIBLE"}:
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(out / "prefix_branch_results.csv", index=False, encoding="utf-8-sig")
        (out / "run_report.md").write_text(
            "# Prefix-Branch Run Report\n\nNo live/replay arms were executed because feasibility is `NOT_FEASIBLE`. No success, cost, or repetition results are claimed.\n",
            encoding="utf-8",
        )
        print("Run skipped: live/replay not feasible.")
        return 0
    raise NotImplementedError("A replay-capable agent runner was detected, but this artifact does not yet define a safe runner adapter. Configure a runner adapter before executing live branches.")


if __name__ == "__main__":
    raise SystemExit(main())
