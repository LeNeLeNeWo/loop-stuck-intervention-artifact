from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze prefix-branch intervention results when available.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_live_intervention"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    results = out / "prefix_branch_results.csv"
    df = pd.read_csv(results) if results.exists() else pd.DataFrame()
    if df.empty:
        pd.DataFrame(columns=["group", "arm", "N", "success_rate", "mean_steps", "repetition_rate", "delta_success_vs_continue", "delta_steps_vs_continue", "main_interpretation"]).to_csv(out / "prefix_branch_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["group", "arm", "N", "success", "mean_steps", "repetition_rate", "delta_success_vs_continue", "delta_steps_vs_continue", "main_interpretation"]).to_csv(out / "table_live_intervention_summary.csv", index=False, encoding="utf-8-sig")
        (out / "live_intervention_report.md").write_text(
            "# Live Intervention Report\n\nNo live/replay results are available. The current artifact records feasibility only; no success, cost, or repetition effects are claimed.\n",
            encoding="utf-8",
        )
        print("No live results to analyze.")
        return 0
    raise NotImplementedError("Analysis for non-empty live results requires runner-specific schema validation.")


if __name__ == "__main__":
    raise SystemExit(main())
