from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from official_common import ensure_output, repo_root, utc_now, write_csv


ARMS = ["NO_GUARD", "WARN_REPLAN_GUARD", "HARD_STOP_GUARD"]


def bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def bootstrap(pairs: list[tuple[float, float]], samples: int = 1000, seed: int = 20270614) -> tuple[float, float, float]:
    if not pairs:
        return 0.0, 0.0, 0.0
    obs = statistics.mean([b - a for a, b in pairs])
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        draws.append(statistics.mean([b - a for a, b in sample]))
    draws.sort()
    return obs, draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]


def latest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_i"] = range(len(out))
    return out.sort_values("_i").drop_duplicates(["released_task_id", "arm"], keep="last").drop(columns=["_i"])


def filter_to_n_tasks(df: pd.DataFrame, out: Path, n_tasks: int | None) -> pd.DataFrame:
    if not n_tasks:
        return df
    sample_path = out / "official_oracle_task_sample.csv"
    if not sample_path.exists():
        return df
    sample = pd.read_csv(sample_path)
    ids = set(sample[pd.to_numeric(sample["selected_order"], errors="coerce").le(n_tasks)]["released_task_id"].astype(str))
    return df[df["released_task_id"].astype(str).isin(ids)].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze official-oracle live guard results.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-tasks", type=int, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)
    run_path = out / "agent_run_log.csv"
    oracle_path = out / "oracle_eval_results.csv"
    if not run_path.exists():
        raise SystemExit("agent_run_log.csv missing")
    runs = filter_to_n_tasks(latest(pd.read_csv(run_path)), out, args.n_tasks)
    if oracle_path.exists():
        oracle = filter_to_n_tasks(latest(pd.read_csv(oracle_path)), out, args.n_tasks)
    else:
        oracle = pd.DataFrame(columns=["released_task_id", "arm", "oracle_status", "resolved", "tests_pass"])
    df = runs.merge(oracle, on=["released_task_id", "benchmark_instance_id", "arm"], how="left", suffixes=("", "_oracle"))
    df.to_csv(out / "live_guard_official_results.csv", index=False)
    summary = []
    for arm in ARMS:
        g = df[df["arm"] == arm]
        if g.empty:
            continue
        if "oracle_status" in g:
            status = g["oracle_status"].astype(str)
            evaluated = g[status.eq("evaluated")]
            official_outcomes = g[status.isin(["evaluated", "official_empty_patch"])]
            empty_patch = g[status.eq("official_empty_patch")]
        else:
            evaluated = g.head(0)
            official_outcomes = g.head(0)
            empty_patch = g.head(0)
        summary.append(
            {
                "arm": arm,
                "N_attempted": len(g),
                "N_completed": int((g["status"] == "completed").sum()),
                "N_official_oracle_outcomes": len(official_outcomes),
                "N_oracle_test_runs": len(evaluated),
                "N_official_empty_patch": len(empty_patch),
                "N_oracle_evaluated": len(evaluated),
                "resolved_rate": float(bool_series(official_outcomes["resolved"]).mean()) if len(official_outcomes) else "",
                "tests_pass_rate": float(bool_series(official_outcomes["tests_pass"]).mean()) if len(official_outcomes) else "",
                "patch_generated_rate": float(bool_series(g["patch_generated"]).mean()) if len(g) else 0.0,
                "mean_steps": float(num(g["steps"]).mean()) if len(g) else 0.0,
                "median_steps": float(num(g["steps"]).median()) if len(g) else 0.0,
                "mean_tokens": float((num(g["tokens_input"]) + num(g["tokens_output"])).mean()) if len(g) else 0.0,
                "guard_trigger_rate": float(bool_series(g["guard_triggered"]).mean()) if len(g) else 0.0,
                "mean_trigger_step": float(num(g["trigger_step"]).mean()) if len(g) else "",
                "mean_post_trigger_repetition_rate": float(num(g["post_trigger_repetition_rate"]).mean()) if len(g) else 0.0,
                "infrastructure_failure_rate": float((g["status"] == "infrastructure_failed").mean()) if len(g) else 0.0,
            }
        )
    write_csv(out / "live_guard_official_summary_by_arm.csv", summary)
    write_csv(out / "table_live_guard_official_summary.csv", summary)

    df["tokens_total"] = num(df["tokens_input"]) + num(df["tokens_output"])
    df["resolved_num"] = bool_series(df.get("resolved", pd.Series(index=df.index, dtype=object))).astype(int)
    df["tests_pass_num"] = bool_series(df.get("tests_pass", pd.Series(index=df.index, dtype=object))).astype(int)
    df["patch_generated_num"] = bool_series(df["patch_generated"]).astype(int)
    metrics = ["resolved_num", "tests_pass_num", "steps", "tokens_total", "post_trigger_repetition_rate", "patch_generated_num"]
    effects = []
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
            diff, lo, hi = bootstrap(pairs)
            effects.append(
                {
                    "comparison": f"{b_arm} minus NO_GUARD",
                    "metric": metric,
                    "paired_N": len(pairs),
                    "mean_difference": diff,
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                }
            )
    write_csv(out / "live_guard_official_pairwise_effects.csv", effects)
    write_csv(out / "table_live_guard_official_pairwise_effects.csv", effects)

    no_eval = int(next((r["N_official_oracle_outcomes"] for r in summary if r["arm"] == "NO_GUARD"), 0) or 0)
    warn_eval = int(next((r["N_official_oracle_outcomes"] for r in summary if r["arm"] == "WARN_REPLAN_GUARD"), 0) or 0)
    threshold = args.n_tasks or 30
    manuscript_ready = no_eval >= threshold and warn_eval >= threshold
    lines = [
        "# Official-Oracle Live Guard Report",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Status: **{'READY_WITH_LIVE_TASK_SUCCESS_EVIDENCE' if manuscript_ready else 'ARTIFACT_ONLY_OR_NOT_READY'}**",
        "",
        "## Completion",
        "",
    ]
    for row in summary:
        lines.append(
            f"- `{row['arm']}`: attempted={row['N_attempted']}, completed={row['N_completed']}, "
            f"official_oracle_outcomes={row['N_official_oracle_outcomes']}, "
            f"oracle_test_runs={row['N_oracle_test_runs']}, "
            f"official_empty_patch={row['N_official_empty_patch']}, "
            f"patch_generated_rate={row['patch_generated_rate']:.3f}"
        )
    lines.extend(["", "## Pairwise Effects", "", "| Comparison | Metric | N | Mean diff | 95% CI |", "|---|---|---:|---:|---|"])
    for row in effects:
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['paired_N']} | {row['mean_difference']:.4f} | [{row['bootstrap95_low']:.4f}, {row['bootstrap95_high']:.4f}] |")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            f"- Live task-success claims require at least {threshold} official oracle evaluations for NO_GUARD and WARN_REPLAN_GUARD.",
            "- Empty-patch rows submitted to the official harness are counted as official outcomes with resolved/tests_pass=false and are reported separately from test-run rows.",
            "- Oracle infrastructure errors are reported separately from resolved outcomes.",
        ]
    )
    (out / "live_guard_official_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    final_status = "READY_WITH_LIVE_TASK_SUCCESS_EVIDENCE" if manuscript_ready else ("READY_WITH_LIVE_POLICY_COST_REPETITION_EVIDENCE_ONLY" if len(df) else "NOT_READY")
    (out / "LIVE_GUARD_OFFICIAL_FINAL_REPORT.md").write_text(
        "\n".join(
            [
                "# Live Guard Official Final Report",
                "",
                f"Final status: **{final_status}**",
                "",
                f"- Official oracle outcomes: NO_GUARD={no_eval}, WARN_REPLAN_GUARD={warn_eval}.",
                f"- Manuscript live-success integration threshold met: {'yes' if manuscript_ready else 'no'}.",
                "- No success improvement is claimed unless official oracle evaluations meet the preregistered threshold.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Official analysis rows: {len(df)}")
    print(f"Report: {out / 'live_guard_official_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
