from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_SUBDIR = Path("outputs/paper1_api_replan_intervention")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def b01(value: Any) -> int:
    return 1 if str(value).strip().lower() in {"true", "1", "yes", "y"} else 0


def info_gain01(value: Any) -> int:
    return 1 if str(value).strip().lower() in {"medium", "high"} else 0


def normalize_target(value: Any) -> str:
    out = str(value or "").strip().lower()
    out = " ".join(out.split())
    if out in {"", "none", "n/a", "na", "unknown"}:
        return ""
    return out


def deterministic_repetition(row: pd.Series) -> int:
    if b01(row.get("repeats_recent_pattern", "")):
        return 1
    target = normalize_target(row.get("next_action_target", ""))
    action_type = normalize_target(row.get("next_action_type", ""))
    pattern = normalize_target(row.get("repeated_pattern_summary", ""))
    recent_actions = normalize_target(row.get("recent_actions_summary", ""))
    if target and len(target) >= 4 and (target in pattern or target in recent_actions):
        return 1
    if action_type and f"step_type={action_type}" in recent_actions and not b01(row.get("uses_new_evidence", "")):
        return 1
    return 0


def paired_bootstrap(diff_values: np.ndarray, seed: int = 20270614, n_boot: int = 5000) -> tuple[float, float, float]:
    if len(diff_values) == 0:
        return math.nan, math.nan, math.nan
    point = float(np.mean(diff_values))
    rng = np.random.default_rng(seed)
    vals = np.asarray(diff_values, dtype=float)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(vals), len(vals))
        boots[i] = float(np.mean(vals[idx]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def mcnemar_counts(neutral: np.ndarray, warn: np.ndarray) -> tuple[int, int, float, float]:
    b = int(((neutral == 1) & (warn == 0)).sum())
    c = int(((neutral == 0) & (warn == 1)).sum())
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_approx = math.erfc(math.sqrt(chi2 / 2.0))
    return b, c, float(chi2), float(p_approx)


def prepare(parsed: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    keep = parsed.copy()
    keep = keep[(keep["status"].astype(str) == "ok") & (keep["json_parse_success"].astype(str).str.lower().isin(["true", "1"]))]
    cols = [
        "released_prefix_id",
        "stratum",
        "dataset_display",
        "trigger_location_type",
        "detector_family",
        "detector_config",
        "recent_actions_summary",
        "repeated_pattern_summary",
    ]
    keep = keep.merge(sample[cols], on=["released_prefix_id", "stratum", "dataset_display"], how="left")
    keep["repeat_pattern"] = keep.apply(deterministic_repetition, axis=1)
    keep["uses_new_evidence_bin"] = keep["uses_new_evidence"].map(b01)
    keep["info_gain_bin"] = keep["expected_information_gain"].map(info_gain01)
    keep["new_evidence_action"] = ((keep["uses_new_evidence_bin"] == 1) | (keep["info_gain_bin"] == 1)).astype(int)
    keep["should_continue_bin"] = keep["should_continue"].map(b01)
    keep["unsafe_overstop"] = ((keep["stratum"].eq("productive_risk")) & (keep["should_continue_bin"] == 0)).astype(int)
    keep["progress_seeking_score"] = keep["uses_new_evidence_bin"] + keep["info_gain_bin"] + (1 - keep["repeat_pattern"])
    keep["next_action_type_norm"] = keep["next_action_type"].map(normalize_target)
    keep["next_action_target_norm"] = keep["next_action_target"].map(normalize_target)
    return keep


def action_shift_flags(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, group in scored.groupby("released_prefix_id"):
        arms = {row["arm"]: row for _, row in group.iterrows()}
        if "NEUTRAL_CONTINUE" not in arms or "WARN_REPLAN" not in arms:
            continue
        n = arms["NEUTRAL_CONTINUE"]
        w = arms["WARN_REPLAN"]
        shifted = int(
            str(n["next_action_type_norm"]) != str(w["next_action_type_norm"])
            or str(n["next_action_target_norm"]) != str(w["next_action_target_norm"])
        )
        rows.append(
            {
                "released_prefix_id": pid,
                "stratum": w["stratum"],
                "dataset_display": w["dataset_display"],
                "action_shift": shifted,
            }
        )
    return pd.DataFrame(rows)


def summary_by_group(scored: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    shift_by_stratum = shifts.groupby("stratum")["action_shift"].mean().to_dict() if not shifts.empty else {}
    for (stratum, arm), group in scored.groupby(["stratum", "arm"]):
        rows.append(
            {
                "stratum": stratum,
                "arm": arm,
                "N": int(group["released_prefix_id"].nunique()),
                "repeat_pattern_rate": float(group["repeat_pattern"].mean()),
                "new_evidence_action_rate": float(group["new_evidence_action"].mean()),
                "action_shift_rate": float(shift_by_stratum.get(stratum, math.nan)) if arm == "WARN_REPLAN" else math.nan,
                "progress_seeking_score_mean": float(group["progress_seeking_score"].mean()),
                "continue_rate": float(group["should_continue_bin"].mean()),
                "unsafe_overstop_rate": float(group["unsafe_overstop"].mean()) if stratum == "productive_risk" else math.nan,
                "json_parse_success_rate": 1.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["stratum", "arm"])


def pairwise(scored: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("repeat_pattern_rate", "repeat_pattern"),
        ("new_evidence_action_rate", "new_evidence_action"),
        ("progress_seeking_score_mean", "progress_seeking_score"),
        ("continue_rate", "should_continue_bin"),
        ("unsafe_overstop_rate", "unsafe_overstop"),
    ]
    rows = []
    strata = sorted(scored["stratum"].dropna().unique().tolist())
    strata.append("all")
    for stratum in strata:
        sub = scored if stratum == "all" else scored[scored["stratum"].eq(stratum)]
        pivot_base = sub.pivot_table(index="released_prefix_id", columns="arm", values="stratum", aggfunc="first")
        valid_prefixes = [idx for idx in pivot_base.index if "NEUTRAL_CONTINUE" in pivot_base.columns and "WARN_REPLAN" in pivot_base.columns and pd.notna(pivot_base.loc[idx].get("NEUTRAL_CONTINUE")) and pd.notna(pivot_base.loc[idx].get("WARN_REPLAN"))]
        for label, col in metrics:
            if stratum != "productive_risk" and col == "unsafe_overstop":
                continue
            piv = sub.pivot_table(index="released_prefix_id", columns="arm", values=col, aggfunc="first")
            if "NEUTRAL_CONTINUE" not in piv.columns or "WARN_REPLAN" not in piv.columns:
                continue
            piv = piv.dropna(subset=["NEUTRAL_CONTINUE", "WARN_REPLAN"])
            diff = piv["WARN_REPLAN"].to_numpy(dtype=float) - piv["NEUTRAL_CONTINUE"].to_numpy(dtype=float)
            point, lo, hi = paired_bootstrap(diff)
            row = {
                "stratum": stratum,
                "metric": label,
                "N_pairs": int(len(piv)),
                "neutral_mean": float(piv["NEUTRAL_CONTINUE"].mean()) if len(piv) else math.nan,
                "warn_replan_mean": float(piv["WARN_REPLAN"].mean()) if len(piv) else math.nan,
                "warn_minus_neutral": point,
                "ci_low": lo,
                "ci_high": hi,
                "mcnemar_b_neutral_yes_warn_no": "",
                "mcnemar_c_neutral_no_warn_yes": "",
                "mcnemar_chi2": "",
                "mcnemar_p_approx": "",
            }
            if col == "repeat_pattern" and len(piv):
                b, c, chi2, p = mcnemar_counts(piv["NEUTRAL_CONTINUE"].to_numpy(dtype=int), piv["WARN_REPLAN"].to_numpy(dtype=int))
                row.update(
                    {
                        "mcnemar_b_neutral_yes_warn_no": b,
                        "mcnemar_c_neutral_no_warn_yes": c,
                        "mcnemar_chi2": chi2,
                        "mcnemar_p_approx": p,
                    }
                )
            rows.append(row)
        shift_sub = shifts if stratum == "all" else shifts[shifts["stratum"].eq(stratum)]
        if not shift_sub.empty:
            vals = shift_sub["action_shift"].to_numpy(dtype=float)
            point, lo, hi = paired_bootstrap(vals)
            rows.append(
                {
                    "stratum": stratum,
                    "metric": "action_shift_rate",
                    "N_pairs": int(len(vals)),
                    "neutral_mean": 0.0,
                    "warn_replan_mean": point,
                    "warn_minus_neutral": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "mcnemar_b_neutral_yes_warn_no": "",
                    "mcnemar_c_neutral_no_warn_yes": "",
                    "mcnemar_chi2": "",
                    "mcnemar_p_approx": "",
                }
            )
    return pd.DataFrame(rows)


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100*x:.1f}%"


def fmt_num(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.2f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    work = work.fillna("").astype(str)
    cols = list(work.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in work.iterrows():
        vals = [str(row[c]).replace("|", "\\|") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze API-assisted prefix replan intervention results.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = repo_root()
    out = args.output_dir or root / OUTPUT_SUBDIR
    parsed_path = out / "api_replan_parsed_responses.csv"
    sample_path = out / "prefix_replan_sample.csv"
    if not parsed_path.exists():
        raise FileNotFoundError(parsed_path)
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    parsed = pd.read_csv(parsed_path)
    sample = pd.read_csv(sample_path)
    scored = prepare(parsed, sample)
    shifts = action_shift_flags(scored)
    summary = summary_by_group(scored, shifts)
    effects = pairwise(scored, shifts)

    scored.to_csv(out / "api_replan_scored_responses.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "api_replan_summary_by_group.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(out / "api_replan_pairwise_effects.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "table_api_replan_summary.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(out / "table_api_replan_pairwise_effects.csv", index=False, encoding="utf-8-sig")

    prompt_rows = len(parsed)
    ok_rows = int(((parsed["status"].astype(str) == "ok") & parsed["json_parse_success"].astype(str).str.lower().isin(["true", "1"])).sum())
    prefixes_with_pairs = int(effects[effects["metric"].eq("repeat_pattern_rate") & effects["stratum"].eq("all")]["N_pairs"].iloc[0]) if not effects.empty and ((effects["metric"].eq("repeat_pattern_rate")) & effects["stratum"].eq("all")).any() else 0

    manifest = {
        "experiment": "paper1_api_replan_intervention",
        "prompt_rows": int(prompt_rows),
        "ok_parsed_rows": ok_rows,
        "json_parse_success_rate": ok_rows / prompt_rows if prompt_rows else 0,
        "prefixes_with_complete_pairs": prefixes_with_pairs,
        "outputs": [
            "api_replan_parsed_responses.csv",
            "api_replan_scored_responses.csv",
            "api_replan_summary_by_group.csv",
            "api_replan_pairwise_effects.csv",
            "table_api_replan_summary.csv",
            "table_api_replan_pairwise_effects.csv",
        ],
    }
    (out / "api_replan_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# API-Assisted Prefix Replan Intervention Report",
        "",
        "This experiment calls a real DeepSeek OpenAI-compatible chat-completions endpoint on sanitized debugging prefixes. It is a prompt-level prefix replan experiment: it does not restore repository state, execute tools, run tests, or estimate final task success.",
        "",
        "## Run Summary",
        "",
        f"- Prompt rows parsed: {prompt_rows}",
        f"- JSON parsed OK rows: {ok_rows}",
        f"- JSON parse success rate: {fmt_pct(manifest['json_parse_success_rate'])}",
        f"- Prefixes with complete paired arms: {prefixes_with_pairs}",
        "",
        "## Group-by-Arm Summary",
        "",
        markdown_table(summary),
        "",
        "## Paired Effects",
        "",
        markdown_table(effects),
        "",
        "## Interpretation Boundary",
        "",
        "- The experiment measures whether a generic warning/replan prompt changes proposed next-step plans on real debugging prefixes.",
        "- It does not claim task success improvement, tests-passed improvement, or full-agent deployment effects.",
        "- Productive-risk over-stop is tracked as `unsafe_overstop_rate` so that a warning is not rewarded for simply stopping productive prefixes.",
    ]
    (out / "api_replan_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Parsed OK rows: {ok_rows}/{prompt_rows}")
    print(f"Complete paired prefixes: {prefixes_with_pairs}")
    print(f"Summary: {out / 'api_replan_summary_by_group.csv'}")
    return 0 if prefixes_with_pairs > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
