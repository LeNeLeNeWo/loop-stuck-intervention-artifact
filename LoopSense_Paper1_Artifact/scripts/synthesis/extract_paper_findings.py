from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.paper1_audit.load_paper1_datasets import repo_root_from_script


DATASET_ORDER = ["Natural500", "Final500", "StressFresh100", "OpenHandsExternal330"]
DETECTOR_FAMILIES = [
    "max_step_guard",
    "exact_action_repeat_k",
    "tool_repeat_k",
    "tool_sequence_repeat",
    "action_observation_repeat",
    "action_observation_similarity",
    "error_signature_repeat_k",
    "file_revisit_guard",
    "repeat_without_progress_guard",
]
PROGRESS_AWARE = {"repeat_without_progress_guard"}


def pct(value: Any, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{100 * float(value):.{digits}f}%"


def num(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def count(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(int(round(float(value))))


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    show = df if max_rows is None else df.head(max_rows)
    if show.empty:
        return "_No rows._"
    return show.to_markdown(index=False)


def read_csv(root: Path, rel: str) -> pd.DataFrame:
    return pd.read_csv(root / rel)


def row_by(df: pd.DataFrame, **conds: Any) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in conds.items():
        mask &= df[key].astype(str).eq(str(value))
    rows = df[mask]
    if rows.empty:
        raise KeyError(f"No row for {conds}")
    return rows.iloc[0]


def metric_lookup(metrics: pd.DataFrame, dataset: str, detector: str, config: str) -> pd.Series:
    return row_by(metrics, dataset=dataset, detector=detector, **{"threshold/config": config})


def make_headline_findings(
    size: pd.DataFrame,
    prevalence: pd.DataFrame,
    length_summary: pd.DataFrame,
    overlap: pd.DataFrame,
    metrics: pd.DataFrame,
    raw_avail: pd.DataFrame,
) -> pd.DataFrame:
    findings: list[dict[str, Any]] = []

    def add(theme: str, dataset: str, metric_1: str, value_1: str, metric_2: str, value_2: str, comparison: str, claim: str, source: str) -> None:
        findings.append(
            {
                "finding_id": f"F{len(findings) + 1:02d}",
                "theme": theme,
                "dataset": dataset,
                "metric_1": metric_1,
                "value_1": value_1,
                "metric_2": metric_2,
                "value_2": value_2,
                "comparison": comparison,
                "paper_claim_supported": claim,
                "table_or_figure_source": source,
            }
        )

    oh_len = row_by(length_summary, comparison="OpenHands_external")
    swe_len = row_by(length_summary, comparison="SWE_agent_datasets")
    add(
        "Length is not UC",
        "OpenHandsExternal330 vs SWE-agent datasets",
        "OpenHands mean length",
        num(oh_len["mean_length"]),
        "OpenHands UC prevalence",
        pct(oh_len["contains_uc_pct"]),
        f"OpenHandsExternal330 is much longer on average than SWE-agent datasets ({num(oh_len['mean_length'])} vs {num(swe_len['mean_length'])} steps), but has lower UC prevalence ({pct(oh_len['contains_uc_pct'])} vs {pct(swe_len['contains_uc_pct'])}).",
        "Long-horizon trajectories are not sufficient evidence of unproductive cycling.",
        "outputs/paper1_construct_mismatch/tables/length_comparison_summary.csv",
    )

    top = row_by(length_summary, comparison="global_top_10pct_longest")
    rest = row_by(length_summary, comparison="global_remaining_90pct")
    add(
        "Length is not UC",
        "All four evidence pools, reported as a diagnostic contrast",
        "UC prevalence in global top 10% longest",
        pct(top["contains_uc_pct"]),
        "UC prevalence in remaining 90%",
        pct(rest["contains_uc_pct"]),
        f"The global top decile of longest trajectories has lower UC prevalence than the remaining trajectories ({pct(top['contains_uc_pct'])} vs {pct(rest['contains_uc_pct'])}).",
        "Length alone is an ambiguous signal for loop/stuck behavior.",
        "outputs/paper1_construct_mismatch/tables/length_comparison_summary.csv",
    )

    for dataset in ["Natural500", "Final500", "StressFresh100", "OpenHandsExternal330"]:
        prev = row_by(prevalence, dataset=dataset)
        if dataset == "OpenHandsExternal330":
            add(
                "Construct mismatch",
                dataset,
                "PI prevalence",
                pct(prev["contains_pi_pct"]),
                "UC prevalence",
                pct(prev["contains_uc_pct"]),
                f"{dataset} has near-universal PI ({count(prev['contains_pi_count'])}/{count(prev['trajectory_count'])}) but very rare UC ({count(prev['contains_uc_count'])}/{count(prev['trajectory_count'])}).",
                "Productive iteration and unproductive cycling are related but non-equivalent constructs.",
                "outputs/paper1_construct_mismatch/construct_prevalence_by_dataset.csv",
            )
        if dataset == "StressFresh100":
            add(
                "Evidence-pool contrast",
                dataset,
                "UC prevalence",
                pct(prev["contains_uc_pct"]),
                "HN prevalence",
                pct(prev["contains_hn_pct"]),
                f"{dataset} is a clean stress challenge with high UC prevalence ({pct(prev['contains_uc_pct'])}) while still containing HN cases ({pct(prev['contains_hn_pct'])}).",
                "Stress evidence pools expose intervention-risk cases that are less visible in natural-distribution controls.",
                "outputs/paper1_construct_mismatch/construct_prevalence_by_dataset.csv",
            )

    final_hn = row_by(overlap, dataset="Final500", category="HN_only")
    final_uc = row_by(overlap, dataset="Final500", category="UC_only")
    final_mixed = row_by(overlap, dataset="Final500", category="mixed_HN_and_UC")
    add(
        "HN/UC non-equivalence",
        "Final500",
        "HN-only trajectories",
        f"{count(final_hn['trajectory_count'])} ({pct(final_hn['trajectory_pct'])})",
        "UC-only trajectories",
        f"{count(final_uc['trajectory_count'])} ({pct(final_uc['trajectory_pct'])})",
        f"Final500 contains HN-only, UC-only, and mixed HN+UC trajectories; mixed HN+UC is {count(final_mixed['trajectory_count'])} trajectories ({pct(final_mixed['trajectory_pct'])}).",
        "HN and UC should not be collapsed into a single binary stuck construct.",
        "outputs/paper1_construct_mismatch/tables/hn_uc_overlap_trajectory_categories.csv",
    )

    final_step50 = metric_lookup(metrics, "Final500", "max_step_guard", "threshold=50")
    final_file_w5r3 = metric_lookup(metrics, "Final500", "file_revisit_guard", "window=5;repeats=3")
    add(
        "F1 hides trigger-site risk",
        "Final500",
        "max_step_guard threshold=50 F1 / PIIR",
        f"{num(final_step50['F1'])} / {num(final_step50['PIIR'])}",
        "file_revisit_guard window=5;repeats=3 F1 / PIIR",
        f"{num(final_file_w5r3['F1'])} / {num(final_file_w5r3['PIIR'])}",
        f"These two detectors differ by only {num(abs(final_step50['F1'] - final_file_w5r3['F1']))} F1, but PIIR differs by {num(abs(final_step50['PIIR'] - final_file_w5r3['PIIR']))} and HN-FPR by {num(abs(final_step50['HN_FPR'] - final_file_w5r3['HN_FPR']))}.",
        "Similar aggregate F1 can mask substantially different productive-interruption risk.",
        "outputs/paper1_intervention_risk_rich_detectors/near_f1_risk_pairs.csv",
    )

    final_file_w10r3 = metric_lookup(metrics, "Final500", "file_revisit_guard", "window=10;repeats=3")
    add(
        "Progress-insensitive repetition risk",
        "Final500",
        "file_revisit_guard window=10;repeats=3 UC recall",
        pct(final_file_w10r3["UC_recall"]),
        "PIIR / HN-FPR",
        f"{num(final_file_w10r3['PIIR'])} / {num(final_file_w10r3['HN_FPR'])}",
        f"The file revisit guard recalls {pct(final_file_w10r3['UC_recall'])} of UC spans, but also interrupts {pct(final_file_w10r3['PIIR'])} of PI spans and triggers on {pct(final_file_w10r3['HN_FPR'])} of HN-eligible steps.",
        "Repetition-based guards can be useful coarse diagnostics but require trigger-site risk evaluation.",
        "outputs/paper1_intervention_risk_rich_detectors/detector_metrics_by_dataset.csv",
    )

    stress_file_w10r3 = metric_lookup(metrics, "StressFresh100", "file_revisit_guard", "window=10;repeats=3")
    add(
        "Stress challenge exposes risk",
        "StressFresh100",
        "file_revisit_guard window=10;repeats=3 UC recall",
        pct(stress_file_w10r3["UC_recall"]),
        "PIIR / HN-FPR",
        f"{num(stress_file_w10r3['PIIR'])} / {num(stress_file_w10r3['HN_FPR'])}",
        f"In StressFresh100, the same family reaches very high UC recall ({pct(stress_file_w10r3['UC_recall'])}) but also very high PIIR and HN-FPR.",
        "StressFresh100 surfaces intervention risks that a natural-control-only evaluation could understate.",
        "outputs/paper1_intervention_risk_rich_detectors/detector_metrics_by_dataset.csv",
    )

    oh_step25 = metric_lookup(metrics, "OpenHandsExternal330", "max_step_guard", "threshold=25")
    oh_step150 = metric_lookup(metrics, "OpenHandsExternal330", "max_step_guard", "threshold=150")
    add(
        "Step caps are budget guards",
        "OpenHandsExternal330",
        "max_step_guard threshold=25 burden / PIIR",
        f"{num(oh_step25['burden'])} / {num(oh_step25['PIIR'])}",
        "max_step_guard threshold=150 F1 / PIIR",
        f"{num(oh_step150['F1'])} / {num(oh_step150['PIIR'])}",
        f"On long OpenHands trajectories with rare UC, low step caps trigger broadly (threshold=25 burden {pct(oh_step25['burden'])}, PIIR {pct(oh_step25['PIIR'])}); even threshold=150 has F1 {num(oh_step150['F1'])} with PIIR {pct(oh_step150['PIIR'])}.",
        "Step caps can control budget but should not be interpreted as progress-aware loop detectors.",
        "outputs/paper1_intervention_risk_rich_detectors/detector_metrics_by_dataset.csv",
    )

    final_repeat = metric_lookup(metrics, "Final500", "exact_action_repeat_k", "k=3")
    final_progress = metric_lookup(metrics, "Final500", "repeat_without_progress_guard", "base=exact_action;k=3;progress=runtime")
    add(
        "Progress-aware suppression",
        "Final500",
        "exact_action_repeat_k k=3 F1 / PIIR / HN-FPR",
        f"{num(final_repeat['F1'])} / {num(final_repeat['PIIR'])} / {num(final_repeat['HN_FPR'])}",
        "repeat_without_progress_guard runtime F1 / PIIR / HN-FPR",
        f"{num(final_progress['F1'])} / {num(final_progress['PIIR'])} / {num(final_progress['HN_FPR'])}",
        f"The progress-aware guard lowers PIIR from {num(final_repeat['PIIR'])} to {num(final_progress['PIIR'])} and HN-FPR from {num(final_repeat['HN_FPR'])} to {num(final_progress['HN_FPR'])}, with F1 changing from {num(final_repeat['F1'])} to {num(final_progress['F1'])}.",
        "Simple progress-aware suppression is consistent with lower interruption risk, although utility tradeoffs remain.",
        "outputs/paper1_intervention_risk_rich_detectors/detector_metrics_by_dataset.csv",
    )

    stress_tool = metric_lookup(metrics, "StressFresh100", "tool_repeat_k", "k=3")
    stress_progress_tool = metric_lookup(metrics, "StressFresh100", "repeat_without_progress_guard", "base=tool;k=3;progress=runtime")
    add(
        "Progress-aware suppression",
        "StressFresh100",
        "tool_repeat_k k=3 F1 / PIIR / HN-FPR",
        f"{num(stress_tool['F1'])} / {num(stress_tool['PIIR'])} / {num(stress_tool['HN_FPR'])}",
        "repeat_without_progress_guard tool-runtime F1 / PIIR / HN-FPR",
        f"{num(stress_progress_tool['F1'])} / {num(stress_progress_tool['PIIR'])} / {num(stress_progress_tool['HN_FPR'])}",
        f"On StressFresh100, progress-aware suppression reduces PIIR from {num(stress_tool['PIIR'])} to {num(stress_progress_tool['PIIR'])} and HN-FPR from {num(stress_tool['HN_FPR'])} to {num(stress_progress_tool['HN_FPR'])}, while F1 decreases from {num(stress_tool['F1'])} to {num(stress_progress_tool['F1'])}.",
        "Progress awareness can reduce trigger-site risk, but the paper should report the corresponding utility tradeoff.",
        "outputs/paper1_intervention_risk_rich_detectors/detector_metrics_by_dataset.csv",
    )

    raw_final = row_by(raw_avail, dataset="Final500", field="raw_action_text")
    raw_stress = row_by(raw_avail, dataset="StressFresh100", field="raw_action_text")
    raw_open = row_by(raw_avail, dataset="OpenHandsExternal330", field="raw_action_text")
    add(
        "Raw-field availability caveat",
        "Final500, StressFresh100, OpenHandsExternal330",
        "raw action coverage Final500 / StressFresh100",
        f"{pct(raw_final['availability_rate'])} / {pct(raw_stress['availability_rate'])}",
        "raw action coverage OpenHandsExternal330",
        pct(raw_open["availability_rate"]),
        "Richer repetition detectors are enabled mainly where raw runtime fields are recoverable; OpenHandsExternal330 active labels expose no raw action text in the current artifact.",
        "Detector-family comparisons should be interpreted with dataset-specific raw-field availability in mind.",
        "outputs/paper1_raw_enrichment/raw_field_availability_by_dataset.csv",
    )

    return pd.DataFrame(findings)


def write_headline_findings(output_dir: Path, findings: pd.DataFrame) -> None:
    lines = [
        "# Headline Empirical Findings",
        "",
        "These findings are extracted from the completed Paper1 audit, construct-mismatch, raw-enrichment, and intervention-risk outputs. They use exact values from the generated CSV files and avoid pooling datasets unless the source analysis explicitly reports a diagnostic cross-pool comparison.",
        "",
    ]
    for row in findings.itertuples(index=False):
        lines.extend(
            [
                f"## {row.finding_id}. {row.theme}",
                "",
                f"- **Dataset / contrast**: {row.dataset}",
                f"- **Metric 1**: {row.metric_1} = `{row.value_1}`",
                f"- **Metric 2**: {row.metric_2} = `{row.value_2}`",
                f"- **Comparison**: {row.comparison}",
                f"- **Paper claim supported**: {row.paper_claim_supported}",
                f"- **Source**: `{row.table_or_figure_source}`",
                "",
                "Cautious interpretation: this finding suggests a construct or intervention-risk pattern in the corresponding evidence pool; it should not be read as a claim that human labels are wrong or that UC is the only valid construct.",
                "",
            ]
        )
    (output_dir / "headline_findings.md").write_text("\n".join(lines), encoding="utf-8")
    findings.to_csv(output_dir / "headline_findings.csv", index=False, encoding="utf-8-sig")


def write_near_f1_pairs(output_dir: Path, pairs: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lookup = {
        (row.dataset, row.detector, row.config): row
        for row in metrics.rename(columns={"threshold/config": "config"}).itertuples(index=False)
    }
    enriched_rows = []
    for _, row in pairs.sort_values(["abs_PIIR_diff", "abs_HN_FPR_diff"], ascending=False).head(10).iterrows():
        a = lookup.get((row["dataset"], row["detector_a"], row["config_a"]))
        b = lookup.get((row["dataset"], row["detector_b"], row["config_b"]))
        burden_a = getattr(a, "burden", math.nan) if a is not None else math.nan
        burden_b = getattr(b, "burden", math.nan) if b is not None else math.nan
        enriched_rows.append({**row.to_dict(), "burden_a": burden_a, "burden_b": burden_b, "abs_burden_diff": abs(burden_a - burden_b) if pd.notna(burden_a) and pd.notna(burden_b) else math.nan})
    top = pd.DataFrame(enriched_rows)
    top.to_csv(output_dir / "strongest_near_f1_risk_pairs.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Strongest Near-F1 Risk Pairs",
        "",
        "The table below ranks detector pairs with absolute F1 difference <= 0.02 by largest PIIR gap, then HN-FPR gap. This is the clearest evidence that aggregate F1 can hide trigger-site risk.",
        "",
    ]
    for i, row in enumerate(top.itertuples(index=False), start=1):
        lines.extend(
            [
                f"## Pair {i}: {row.dataset}",
                "",
                f"- **Detector A**: `{row.detector_a}` / `{row.config_a}`",
                f"- **Detector B**: `{row.detector_b}` / `{row.config_b}`",
                f"- **F1**: `{num(row.F1_a)}` vs `{num(row.F1_b)}` (delta `{num(row.abs_F1_diff)}`)",
                f"- **PIIR**: `{num(row.PIIR_a)}` vs `{num(row.PIIR_b)}` (delta `{num(row.abs_PIIR_diff)}`)",
                f"- **HN-FPR**: `{num(row.HN_FPR_a)}` vs `{num(row.HN_FPR_b)}` (delta `{num(row.abs_HN_FPR_diff)}`)",
                f"- **Burden**: `{num(row.burden_a)}` vs `{num(row.burden_b)}` (delta `{num(row.abs_burden_diff)}`)",
                "- **Interpretation**: These configurations have nearly indistinguishable aggregate F1 but materially different productive-interruption risk. This supports reporting PIIR/HN-FPR alongside F1 for intervention mechanisms.",
                "",
            ]
        )
    (output_dir / "strongest_near_f1_risk_pairs.md").write_text("\n".join(lines), encoding="utf-8")


def write_detector_family_summary(output_dir: Path, metrics: pd.DataFrame, availability: pd.DataFrame) -> None:
    rows = []
    lines = [
        "# Detector Family Summary",
        "",
        "Best risk-utility setting is selected by a descriptive score `F1 - 0.5*PIIR - 0.5*HN_FPR`. This score is used only to summarize configurations; it is not proposed as a paper metric.",
        "",
    ]
    for family in DETECTOR_FAMILIES:
        fam_avail = availability[availability["detector_family"] == family]
        available_datasets = fam_avail[fam_avail["available"]]["dataset"].tolist()
        fam_metrics = metrics[metrics["detector_family"] == family].copy()
        progress_type = "progress-aware" if family in PROGRESS_AWARE else "progress-insensitive"
        if fam_metrics.empty:
            best_desc = "not evaluated because required raw fields were unavailable"
            worst_desc = "not evaluated"
            best_row = None
            worst_piir = None
            worst_hn = None
        else:
            fam_metrics["risk_utility_score"] = fam_metrics["F1"] - 0.5 * fam_metrics["PIIR"] - 0.5 * fam_metrics["HN_FPR"]
            best = fam_metrics.sort_values("risk_utility_score", ascending=False).iloc[0]
            worst_piir = fam_metrics.sort_values("PIIR", ascending=False).iloc[0]
            worst_hn = fam_metrics.sort_values("HN_FPR", ascending=False).iloc[0]
            best_row = best
            best_desc = f"{best['dataset']} `{best['threshold/config']}` (F1 {num(best['F1'])}, PIIR {num(best['PIIR'])}, HN-FPR {num(best['HN_FPR'])})"
            worst_desc = f"worst PIIR: {worst_piir['dataset']} `{worst_piir['threshold/config']}` = {num(worst_piir['PIIR'])}; worst HN-FPR: {worst_hn['dataset']} `{worst_hn['threshold/config']}` = {num(worst_hn['HN_FPR'])}"
        rows.append(
            {
                "detector_family": family,
                "available_datasets": ", ".join(available_datasets) if available_datasets else "none",
                "progress_type": progress_type,
                "best_risk_utility_setting": best_desc,
                "worst_piir_hn_fpr_setting": worst_desc,
            }
        )
        lines.extend(
            [
                f"## {family}",
                "",
                f"- **Type**: {progress_type}",
                f"- **Available in**: {', '.join(available_datasets) if available_datasets else 'none in the current enriched artifact'}",
                f"- **Best risk-utility setting**: {best_desc}",
                f"- **Worst PIIR/HN-FPR setting**: {worst_desc}",
                "- **Interpretation**: Use this family as a detector-site risk profile, not as evidence that the underlying labels are wrong.",
                "",
            ]
        )
    pd.DataFrame(rows).to_csv(output_dir / "detector_family_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "detector_family_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_dataset_role_summary(output_dir: Path, size: pd.DataFrame, prevalence: pd.DataFrame, raw_avail: pd.DataFrame) -> None:
    role_caveats = {
        "Natural500": "Natural-distribution control; current active artifact exposes evidence/signal sidecars but no raw action/tool/observation fields for rich repetition detectors.",
        "Final500": "Enriched difficult loop-heavy set; raw runtime text is partially recoverable, especially for manually reviewed disagreement steps.",
        "StressFresh100": "Clean stress challenge; raw visible step fields are broadly available and support the richest repetition-detector analysis.",
        "OpenHandsExternal330": "External long-horizon generalization set; active label files do not expose raw runtime fields, so rich detectors are limited to step caps.",
    }
    rows = []
    for dataset in DATASET_ORDER:
        s = row_by(size, dataset=dataset)
        p = row_by(prevalence, dataset=dataset)
        raw_action = row_by(raw_avail, dataset=dataset, field="raw_action_text")
        tool = row_by(raw_avail, dataset=dataset, field="tool_name")
        obs = row_by(raw_avail, dataset=dataset, field="observation_text")
        rows.append(
            {
                "dataset": dataset,
                "trajectory_count": int(s["trajectory_count"]),
                "step_count": int(s["step_count"]),
                "avg_length": float(s["avg_trajectory_length"]),
                "pi_prevalence": float(p["contains_pi_pct"]),
                "hn_prevalence": float(p["contains_hn_pct"]),
                "uc_prevalence": float(p["contains_uc_pct"]),
                "raw_action_coverage": float(raw_action["availability_rate"]),
                "tool_coverage": float(tool["availability_rate"]),
                "observation_coverage": float(obs["availability_rate"]),
                "caveat": role_caveats[dataset],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "dataset_role_summary.csv", index=False, encoding="utf-8-sig")
    display = df.copy()
    for col in ["pi_prevalence", "hn_prevalence", "uc_prevalence", "raw_action_coverage", "tool_coverage", "observation_coverage"]:
        display[col] = display[col].map(pct)
    display["avg_length"] = display["avg_length"].map(lambda v: num(v))
    lines = [
        "# Dataset Role Summary",
        "",
        "The four evidence pools have distinct sampling roles and should not be pooled as one natural distribution.",
        "",
        md_table(display),
        "",
    ]
    (output_dir / "dataset_role_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_figure_table_recommendations(output_dir: Path) -> None:
    rows = [
        {
            "title": "Dataset roles and construct prevalence",
            "source": "outputs/paper1_audit/table1_dataset_role_size_summary.csv; outputs/paper1_construct_mismatch/construct_prevalence_by_dataset.csv",
            "key_message": "The four evidence pools differ sharply in size, length, and PI/HN/UC prevalence; they should be reported separately.",
            "placement": "main paper",
        },
        {
            "title": "Length is not unproductive cycling",
            "source": "outputs/paper1_construct_mismatch/tables/length_comparison_summary.csv; outputs/paper1_construct_mismatch/figures/length_vs_construct_density.png",
            "key_message": "OpenHands and the global longest decile show that long trajectories do not necessarily have high UC prevalence.",
            "placement": "main paper",
        },
        {
            "title": "HN/UC overlap categories",
            "source": "outputs/paper1_construct_mismatch/tables/hn_uc_overlap_trajectory_categories.csv; outputs/paper1_construct_mismatch/figures/hn_uc_overlap_by_dataset.png",
            "key_message": "HN-only, UC-only, and mixed trajectories coexist, supporting construct non-equivalence.",
            "placement": "main paper",
        },
        {
            "title": "F1 vs PIIR by detector and dataset",
            "source": "outputs/paper1_intervention_risk_rich_detectors/figures/figureA_f1_vs_piir_by_dataset.png; detector_metrics_by_dataset.csv",
            "key_message": "Aggregate F1 and productive-interruption risk can move differently across detector families.",
            "placement": "main paper",
        },
        {
            "title": "F1 vs HN-FPR by detector and dataset",
            "source": "outputs/paper1_intervention_risk_rich_detectors/figures/figureB_f1_vs_hn_fpr_by_dataset.png",
            "key_message": "False positives on hard-negative eligible steps are not captured by F1 alone.",
            "placement": "main paper",
        },
        {
            "title": "Near-F1 risk pairs",
            "source": "outputs/paper1_paper_findings/strongest_near_f1_risk_pairs.md; outputs/paper1_intervention_risk_rich_detectors/near_f1_risk_pairs.csv",
            "key_message": "Near-identical F1 can mask large PIIR/HN-FPR gaps.",
            "placement": "main paper",
        },
        {
            "title": "UC recall vs PIIR frontier",
            "source": "outputs/paper1_intervention_risk_rich_detectors/figures/figureC_uc_recall_vs_piir_frontier.png",
            "key_message": "Detector settings trade off UC recall against productive-interruption risk.",
            "placement": "main paper or appendix depending on space",
        },
        {
            "title": "Trigger location distribution",
            "source": "outputs/paper1_intervention_risk_rich_detectors/figures/figureE_trigger_location_distribution.png; trigger_location_distribution.csv",
            "key_message": "Detector triggers land in UC, PI/HN, and other regions; site of intervention matters.",
            "placement": "main paper",
        },
        {
            "title": "Raw-field availability",
            "source": "outputs/paper1_raw_enrichment/raw_field_availability_by_dataset.csv; raw_enrichment_report.md",
            "key_message": "Rich detector families are evaluated only where runtime fields are recoverable.",
            "placement": "appendix",
        },
        {
            "title": "Bootstrap confidence intervals",
            "source": "outputs/paper1_intervention_risk_rich_detectors/bootstrap_ci.csv",
            "key_message": "Trajectory-level uncertainty estimates support the main detector-risk comparisons.",
            "placement": "appendix",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "figure_table_recommendations.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Figure and Table Recommendations",
        "",
        md_table(df),
        "",
    ]
    (output_dir / "figure_table_recommendations.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract paper-ready empirical findings from Paper1 outputs.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_paper_findings"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    size = read_csv(root, "outputs/paper1_audit/table1_dataset_role_size_summary.csv")
    prevalence = read_csv(root, "outputs/paper1_construct_mismatch/construct_prevalence_by_dataset.csv")
    length_summary = read_csv(root, "outputs/paper1_construct_mismatch/tables/length_comparison_summary.csv")
    overlap = read_csv(root, "outputs/paper1_construct_mismatch/tables/hn_uc_overlap_trajectory_categories.csv")
    metrics = read_csv(root, "outputs/paper1_intervention_risk_rich_detectors/detector_metrics_by_dataset.csv")
    pairs = read_csv(root, "outputs/paper1_intervention_risk_rich_detectors/near_f1_risk_pairs.csv")
    availability = read_csv(root, "outputs/paper1_intervention_risk_rich_detectors/detector_availability.csv")
    raw_avail = read_csv(root, "outputs/paper1_raw_enrichment/raw_field_availability_by_dataset.csv")

    findings = make_headline_findings(size, prevalence, length_summary, overlap, metrics, raw_avail)
    write_headline_findings(output_dir, findings)
    write_near_f1_pairs(output_dir, pairs, metrics)
    write_detector_family_summary(output_dir, metrics, availability)
    write_dataset_role_summary(output_dir, size, prevalence, raw_avail)
    write_figure_table_recommendations(output_dir)

    print(f"Paper findings extracted: {output_dir}")
    print(f"Headline findings: {len(findings)}")


if __name__ == "__main__":
    main()
