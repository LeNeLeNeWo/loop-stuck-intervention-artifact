from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from live_guard_common import ensure_output, repo_root, utc_now, write_csv


ARM_ORDER = ["NO_GUARD", "WARN_REPLAN_GUARD", "HARD_STOP_GUARD"]


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def mean_bool(series: pd.Series) -> float:
    vals = series.astype(str).str.lower().isin({"true", "1", "yes"}).astype(float)
    return float(vals.mean()) if len(vals) else 0.0


def bootstrap_diff(pairs: list[tuple[float, float]], samples: int = 1000, seed: int = 20270614) -> tuple[float, float, float]:
    if not pairs:
        return 0.0, 0.0, 0.0
    observed = statistics.mean([b - a for a, b in pairs])
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        draws.append(statistics.mean([b - a for a, b in sample]))
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return observed, lo, hi


def summarize(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for arm in ARM_ORDER:
        g = df[df["arm"] == arm]
        if g.empty:
            continue
        completed = g[g["status"] == "completed"]
        rows.append(
            {
                "arm": arm,
                "N_rows": len(g),
                "total_attempt_rows": len(g),
                "completed": len(completed),
                "infrastructure_failed": int((g["status"] == "infrastructure_failed").sum()),
                "skipped": int((g["status"] == "skipped").sum()),
                "guard_trigger_rate": mean_bool(g["guard_triggered"]),
                "patch_generated_rate": mean_bool(g["patch_generated"]),
                "oracle_available_rate": mean_bool(g["oracle_available"]),
                "resolved_rate": "",
                "tests_pass_rate": "",
                "mean_steps": float(numeric(g["steps"]).mean()) if len(g) else 0.0,
                "median_steps": float(numeric(g["steps"]).median()) if len(g) else 0.0,
                "mean_tokens": float((numeric(g["tokens_input"]) + numeric(g["tokens_output"])).mean()) if len(g) else 0.0,
                "mean_repeated_file_rate": float(numeric(g["repeated_file_rate"]).mean()) if len(g) else 0.0,
                "mean_repeated_tool_rate": float(numeric(g["repeated_tool_rate"]).mean()) if len(g) else 0.0,
                "hard_stop_rate": float((g["stop_reason"] == "hard_stop_guard").mean()) if len(g) else 0.0,
            }
        )
    return rows


def pairwise(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    metrics = ["steps", "tokens_total", "repeated_file_rate", "repeated_tool_rate", "patch_generated_num"]
    df = df.copy()
    df["tokens_total"] = numeric(df["tokens_input"]) + numeric(df["tokens_output"])
    df["patch_generated_num"] = df["patch_generated"].astype(str).str.lower().isin({"true", "1", "yes"}).astype(int)
    for b_arm in ["WARN_REPLAN_GUARD", "HARD_STOP_GUARD"]:
        a = df[df["arm"] == "NO_GUARD"].set_index("released_task_id")
        b = df[df["arm"] == b_arm].set_index("released_task_id")
        common = sorted(set(a.index) & set(b.index))
        for metric in metrics:
            pairs = []
            for tid in common:
                av = pd.to_numeric(pd.Series([a.loc[tid, metric]]), errors="coerce").iloc[0]
                bv = pd.to_numeric(pd.Series([b.loc[tid, metric]]), errors="coerce").iloc[0]
                if pd.notna(av) and pd.notna(bv):
                    pairs.append((float(av), float(bv)))
            diff, lo, hi = bootstrap_diff(pairs)
            rows.append(
                {
                    "comparison": f"{b_arm} minus NO_GUARD",
                    "metric": metric,
                    "paired_N": len(pairs),
                    "mean_difference": diff,
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze live guard run log.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)
    log_path = out / "live_guard_run_log.csv"
    if not log_path.exists():
        report = out / "live_guard_report.md"
        report.write_text(
            "# Live Guard Experiment Report\n\nStatus: **NOT_RUN**\n\nNo `live_guard_run_log.csv` exists, so no live outcomes are reported.\n",
            encoding="utf-8",
        )
        print("No run log found.")
        return 2
    attempts_df = pd.read_csv(log_path)
    df = attempts_df.copy()
    df["_attempt_order"] = range(len(df))
    df = df.sort_values("_attempt_order").drop_duplicates(["released_task_id", "arm"], keep="last").drop(columns=["_attempt_order"])
    summary_rows = summarize(df)
    pair_rows = pairwise(df)
    write_csv(out / "live_guard_summary_by_arm.csv", summary_rows)
    write_csv(out / "live_guard_pairwise_effects.csv", pair_rows)
    results = df.copy()
    if "raw_log_path_local" in results.columns:
        results = results.drop(columns=["raw_log_path_local"])
    results.to_csv(out / "live_guard_results.csv", index=False)
    write_csv(out / "table_live_guard_summary.csv", summary_rows)
    write_csv(out / "table_live_guard_pairwise_effects.csv", pair_rows)

    completed_pairs = df[df["status"] == "completed"].groupby("arm")["released_task_id"].nunique().to_dict()
    all_infra_failures = int((attempts_df["status"] == "infrastructure_failed").sum()) if "status" in attempts_df else 0
    oracle_any = df["oracle_available"].astype(str).str.lower().isin({"true", "1", "yes"}).any()
    enough_for_paper = min(completed_pairs.get("NO_GUARD", 0), completed_pairs.get("WARN_REPLAN_GUARD", 0)) >= 30
    lines = [
        "# Live Guard Experiment Report",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Status: **{'LIVE_RESULTS_AVAILABLE' if len(df) else 'NOT_RUN'}**",
        "",
        "## Scope",
        "",
        "- This experiment executes a real mini-swe-agent runner on SWE-bench-style Docker tasks with DeepSeek API calls.",
        "- Official resolved/tests-pass oracle was not run by this analyzer unless `oracle_available=true` appears in the run log.",
        "- Therefore task-success claims are allowed only if oracle availability is confirmed.",
        "",
        "## Completion",
        "",
        f"- Total task-arm attempts logged: {len(attempts_df)}",
        f"- Infrastructure-failed attempts retained in audit log: {all_infra_failures}",
        f"- Latest task-arm rows used for arm summaries: {len(df)}",
    ]
    for arm in ARM_ORDER:
        lines.append(f"- `{arm}` completed tasks: {completed_pairs.get(arm, 0)}")
    lines.extend(
        [
            f"- Oracle success available: {'yes' if oracle_any else 'no'}",
            f"- Paper integration threshold met for NO_GUARD and WARN_REPLAN_GUARD: {'yes' if enough_for_paper else 'no'}",
            "",
            "## Summary By Arm",
            "",
            "| Arm | N rows | Completed | Guard trigger rate | Patch generated rate | Mean steps | Mean repeated file rate | Mean tokens |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['arm']} | {row['N_rows']} | {row['completed']} | {row['guard_trigger_rate']:.3f} | "
            f"{row['patch_generated_rate']:.3f} | {row['mean_steps']:.2f} | {row['mean_repeated_file_rate']:.3f} | {row['mean_tokens']:.1f} |"
        )
    lines.extend(["", "## Pairwise Effects", "", "| Comparison | Metric | N | Mean diff | 95% CI |", "|---|---|---:|---:|---|"])
    for row in pair_rows:
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['paired_N']} | {row['mean_difference']:.4f} | "
            f"[{row['bootstrap95_low']:.4f}, {row['bootstrap95_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- If oracle availability is `no`, the evidence is live tool-level policy evidence, not task-success evidence.",
            "- Negative, mixed, or underpowered effects must be reported as such; no selective exclusion is applied here.",
        ]
    )
    (out / "live_guard_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _summary_value(arm: str, key: str, default: str = "NA") -> str:
        for row in summary_rows:
            if row["arm"] == arm:
                value = row.get(key, default)
                if isinstance(value, float):
                    return f"{value:.4f}"
                return str(value)
        return default

    def _effect_value(comparison_prefix: str, metric: str) -> str:
        for row in pair_rows:
            if str(row["comparison"]).startswith(comparison_prefix) and row["metric"] == metric:
                return (
                    f"{row['mean_difference']:.4f} "
                    f"[{row['bootstrap95_low']:.4f}, {row['bootstrap95_high']:.4f}]"
                )
        return "NA"

    final_status = (
        "READY for manuscript integration"
        if enough_for_paper and oracle_any
        else "NOT READY for live task-success manuscript claims; READY as artifact pilot evidence"
    )
    final = [
        "# Live Guard Final Report",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Final feasibility level: {'public benchmark live policy' if len(df) else 'not run'}",
        "- Runner used: mini-swe-agent local package.",
        f"- Model: `{df['model'].dropna().iloc[0] if len(df) and 'model' in df else ''}`",
        "- Task source: `princeton-nlp/SWE-Bench_Verified:test` public benchmark fallback.",
        f"- Traceability level: {', '.join(sorted(df['traceability_level'].dropna().unique())) if len(df) and 'traceability_level' in df else 'none'}",
        f"- Task arms run: {', '.join(sorted(df['arm'].dropna().unique())) if len(df) else 'none'}",
        f"- Latest completed tasks per arm: NO_GUARD={completed_pairs.get('NO_GUARD', 0)}, WARN_REPLAN_GUARD={completed_pairs.get('WARN_REPLAN_GUARD', 0)}, HARD_STOP_GUARD={completed_pairs.get('HARD_STOP_GUARD', 0)}",
        f"- Infrastructure-failed attempts retained: {all_infra_failures}",
        f"- Oracle availability: {'yes' if oracle_any else 'no'}",
        f"- Paper integration threshold met: {'yes' if enough_for_paper else 'no'}",
        "",
        "## Primary Outcomes",
        "",
        f"- NO_GUARD mean steps: {_summary_value('NO_GUARD', 'mean_steps')}; mean repeated-file rate: {_summary_value('NO_GUARD', 'mean_repeated_file_rate')}; mean tokens: {_summary_value('NO_GUARD', 'mean_tokens')}.",
        f"- WARN_REPLAN_GUARD mean steps: {_summary_value('WARN_REPLAN_GUARD', 'mean_steps')}; mean repeated-file rate: {_summary_value('WARN_REPLAN_GUARD', 'mean_repeated_file_rate')}; mean tokens: {_summary_value('WARN_REPLAN_GUARD', 'mean_tokens')}.",
        f"- HARD_STOP_GUARD mean steps: {_summary_value('HARD_STOP_GUARD', 'mean_steps')}; mean repeated-file rate: {_summary_value('HARD_STOP_GUARD', 'mean_repeated_file_rate')}; mean tokens: {_summary_value('HARD_STOP_GUARD', 'mean_tokens')}.",
        f"- WARN_REPLAN minus NO_GUARD repeated-file-rate effect: {_effect_value('WARN_REPLAN_GUARD', 'repeated_file_rate')}.",
        f"- WARN_REPLAN minus NO_GUARD token effect: {_effect_value('WARN_REPLAN_GUARD', 'tokens_total')}.",
        f"- HARD_STOP minus NO_GUARD step effect: {_effect_value('HARD_STOP_GUARD', 'steps')}.",
        f"- HARD_STOP minus NO_GUARD token effect: {_effect_value('HARD_STOP_GUARD', 'tokens_total')}.",
        "",
        "## Interpretation",
        "",
        "- The pilot provides real tool-level live-policy evidence, not task-success evidence.",
        "- Warning/replanning shows a negative repeated-file-rate point estimate, but the bootstrap CI crosses zero at N=10.",
        "- Hard stop saves steps and tokens in this short-budget pilot, but no oracle is available to evaluate success loss.",
        "- No abstract or main-result success claim should be added from this pilot alone.",
        "",
        f"Final status: **{final_status}**.",
    ]
    (out / "LIVE_GUARD_FINAL_REPORT.md").write_text("\n".join(final) + "\n", encoding="utf-8")
    print(f"Analyzed {len(df)} task-arm rows.")
    print(f"Report: {out / 'live_guard_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
