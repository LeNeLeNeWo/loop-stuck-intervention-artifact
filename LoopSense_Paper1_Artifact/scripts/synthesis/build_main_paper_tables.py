from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_ORDER = ["Natural500", "Final500", "StressFresh100", "OpenHandsExternal330"]

SOURCE_FRAMEWORK = {
    "Natural500": "SWE-agent / LoopBench-Agent",
    "Final500": "SWE-agent / LoopBench-Agent",
    "StressFresh100": "SWE-agent / LoopBench-Agent",
    "OpenHandsExternal330": "OpenHands external",
}

SAMPLING_ROLE = {
    "Natural500": "Natural-distribution control",
    "Final500": "Enriched difficult loop-heavy set",
    "StressFresh100": "Clean stress challenge",
    "OpenHandsExternal330": "External long-horizon generalization set",
}

MAIN_USE = {
    "Natural500": "Estimate construct behavior in a less enriched SWE-agent pool.",
    "Final500": "Study difficult loop-heavy cases and HN/UC boundary behavior.",
    "StressFresh100": "Stress-test intervention risk and UC/HN boundary cases.",
    "OpenHandsExternal330": "Probe external long-horizon generalization and length effects.",
}

TABLE2_INTERPRETATION = {
    "Natural500": "Natural control still contains PI/HN/UC mixtures, but with lower UC concentration.",
    "Final500": "HN-only, UC-only, and mixed cases coexist, so HN and UC should not be collapsed.",
    "StressFresh100": "High UC prevalence with remaining HN cases makes this a focused stress pool.",
    "OpenHandsExternal330": "Long-horizon PI is common while UC is rare in this external pool.",
}

TABLE3_INTERPRETATION = {
    ("Final500", "max_step_guard", "threshold=50"): "A simple step cap gives strong F1 with lower PIIR than file-revisit guards in this pool.",
    ("Final500", "file_revisit_guard", "window=5;repeats=3"): "Similar F1 to the step cap but substantially higher productive-interruption risk.",
    ("Final500", "file_revisit_guard", "window=10;repeats=3"): "Higher UC recall, but risk rises further on PI spans and HN-eligible steps.",
    ("StressFresh100", "file_revisit_guard", "window=10;repeats=3"): "Very high UC recall, paired with high PIIR/HN-FPR in the stress pool.",
    ("StressFresh100", "tool_repeat_k", "k=3"): "A coarse repetition guard with moderate F1 and non-trivial productive-interruption risk.",
    (
        "StressFresh100",
        "repeat_without_progress_guard",
        "base=tool;k=3;progress=runtime",
    ): "A progress-aware guard reduces PIIR/HN-FPR, with a utility tradeoff in F1/UC recall.",
    ("OpenHandsExternal330", "max_step_guard", "threshold=25"): "A short step cap triggers broadly on long productive trajectories; useful as a budget guard, not a progress-aware detector.",
    ("OpenHandsExternal330", "max_step_guard", "threshold=150"): "A later cap reduces burden but still interrupts many PI/HN regions while UC is rare.",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def out_dir(root: Path) -> Path:
    return root / "outputs" / "paper1_main_tables"


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def ordered(df: pd.DataFrame, column: str = "Dataset") -> pd.DataFrame:
    if column not in df.columns:
        return df
    order = {name: i for i, name in enumerate(DATASET_ORDER)}
    return (
        df.assign(_order=df[column].map(order).fillna(999))
        .sort_values(["_order", column])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def round_float(value: Any, digits: int = 3) -> float | str:
    if value is None or pd.isna(value):
        return "NA"
    return round(float(value), digits)


def round_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].round(3)
    return out


def count_rate(count: Any, rate: Any) -> str:
    if pd.isna(count) or pd.isna(rate):
        return "NA"
    return f"{int(count)} ({float(rate):.3f})"


def sanitize_latex_label(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def write_table(df: pd.DataFrame, stem: str, output_dir: Path, caption: str, label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    display = round_numeric_columns(df)
    display.to_csv(output_dir / f"{stem}.csv", index=False)
    display.to_markdown(output_dir / f"{stem}.md", index=False)
    latex = display.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=f"tab:{sanitize_latex_label(label)}",
        longtable=False,
    )
    (output_dir / f"{stem}.tex").write_text(latex, encoding="utf-8")


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "size": root / "outputs" / "paper1_audit" / "table1_dataset_role_size_summary.csv",
        "prevalence": root / "outputs" / "paper1_construct_mismatch" / "construct_prevalence_by_dataset.csv",
        "overlap": root
        / "outputs"
        / "paper1_construct_mismatch"
        / "tables"
        / "hn_uc_overlap_trajectory_categories.csv",
        "metrics": root
        / "outputs"
        / "paper1_intervention_risk_rich_detectors"
        / "detector_metrics_by_dataset.csv",
        "near_f1": root
        / "outputs"
        / "paper1_intervention_risk_rich_detectors"
        / "near_f1_risk_pairs.csv",
    }
    loaded: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        loaded[name] = pd.read_csv(path)
    return loaded


def build_table1(size: pd.DataFrame, prevalence: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(
        size,
        ["dataset", "trajectory_count", "step_count", "avg_trajectory_length"],
        "dataset size table",
    )
    ensure_columns(
        prevalence,
        ["dataset", "contains_pi_pct", "contains_hn_pct", "contains_uc_pct"],
        "construct prevalence table",
    )
    merged = size.merge(
        prevalence[
            [
                "dataset",
                "contains_pi_pct",
                "contains_hn_pct",
                "contains_uc_pct",
            ]
        ],
        on="dataset",
        how="left",
        validate="one_to_one",
    )
    rows = []
    for _, row in merged.iterrows():
        dataset = row["dataset"]
        rows.append(
            {
                "Evidence pool": dataset,
                "Source framework": SOURCE_FRAMEWORK.get(dataset, "not documented"),
                "Trajectories": int(row["trajectory_count"]),
                "Steps": int(row["step_count"]),
                "Avg. steps": round_float(row["avg_trajectory_length"]),
                "PI prevalence": round_float(row["contains_pi_pct"]),
                "HN prevalence": round_float(row["contains_hn_pct"]),
                "UC prevalence": round_float(row["contains_uc_pct"]),
                "Sampling role": SAMPLING_ROLE.get(dataset, str(row.get("dataset_role", ""))),
                "Main use": MAIN_USE.get(dataset, "Construct analysis."),
            }
        )
    return ordered(pd.DataFrame(rows), "Evidence pool")


def pivot_overlap(overlap: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(overlap, ["dataset", "category", "trajectory_count", "trajectory_pct"], "overlap table")
    pivot = overlap.pivot_table(
        index="dataset",
        columns="category",
        values=["trajectory_count", "trajectory_pct"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}__{category}" for metric, category in pivot.columns]
    return pivot.reset_index()


def build_table2(prevalence: pd.DataFrame, overlap: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(
        prevalence,
        ["dataset", "contains_pi_pct", "contains_hn_pct", "contains_uc_pct"],
        "construct prevalence table",
    )
    overlap_wide = pivot_overlap(overlap)
    merged = prevalence.merge(overlap_wide, on="dataset", how="left", validate="one_to_one")
    rows = []
    for _, row in merged.iterrows():
        dataset = row["dataset"]
        rows.append(
            {
                "Dataset": dataset,
                "PI prevalence": round_float(row["contains_pi_pct"]),
                "HN prevalence": round_float(row["contains_hn_pct"]),
                "UC prevalence": round_float(row["contains_uc_pct"]),
                "HN-only": count_rate(
                    row.get("trajectory_count__HN_only"),
                    row.get("trajectory_pct__HN_only"),
                ),
                "UC-only": count_rate(
                    row.get("trajectory_count__UC_only"),
                    row.get("trajectory_pct__UC_only"),
                ),
                "HN+UC mixed": count_rate(
                    row.get("trajectory_count__mixed_HN_and_UC"),
                    row.get("trajectory_pct__mixed_HN_and_UC"),
                ),
                "Main interpretation": TABLE2_INTERPRETATION.get(dataset, "Constructs are related but distinct."),
            }
        )
    return ordered(pd.DataFrame(rows))


def metric_row(metrics: pd.DataFrame, dataset: str, detector: str, config: str) -> pd.Series:
    match = metrics[
        (metrics["dataset"] == dataset)
        & (metrics["detector"] == detector)
        & (metrics["threshold/config"] == config)
    ]
    if len(match) != 1:
        raise ValueError(f"Expected exactly one detector metric row for {dataset}, {detector}, {config}; got {len(match)}")
    return match.iloc[0]


def build_table3(metrics: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(
        metrics,
        ["dataset", "detector", "threshold/config", "F1", "UC_recall", "PIIR", "HN_FPR", "burden"],
        "detector metrics table",
    )
    selected = [
        ("Final500", "max_step_guard", "threshold=50"),
        ("Final500", "file_revisit_guard", "window=5;repeats=3"),
        ("Final500", "file_revisit_guard", "window=10;repeats=3"),
        ("StressFresh100", "file_revisit_guard", "window=10;repeats=3"),
        ("StressFresh100", "tool_repeat_k", "k=3"),
        ("StressFresh100", "repeat_without_progress_guard", "base=tool;k=3;progress=runtime"),
        ("OpenHandsExternal330", "max_step_guard", "threshold=25"),
        ("OpenHandsExternal330", "max_step_guard", "threshold=150"),
    ]
    rows = []
    for dataset, detector, config in selected:
        row = metric_row(metrics, dataset, detector, config)
        rows.append(
            {
                "Dataset": dataset,
                "Detector": detector,
                "Config": config,
                "F1": round_float(row["F1"]),
                "UC Recall": round_float(row["UC_recall"]),
                "PIIR": round_float(row["PIIR"]),
                "HN-FPR": round_float(row["HN_FPR"]),
                "Burden": round_float(row["burden"]),
                "Interpretation": TABLE3_INTERPRETATION[(dataset, detector, config)],
            }
        )
    return pd.DataFrame(rows)


def add_burden_to_pairs(pairs: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, pair in pairs.iterrows():
        row = pair.to_dict()
        for side in ["a", "b"]:
            detector = row[f"detector_{side}"]
            config = row[f"config_{side}"]
            dataset = row["dataset"]
            metric = metric_row(metrics, dataset, detector, config)
            row[f"burden_{side}"] = float(metric["burden"])
        row["abs_burden_diff"] = abs(row["burden_a"] - row["burden_b"])
        rows.append(row)
    return pd.DataFrame(rows)


def pair_interpretation(row: pd.Series) -> str:
    dataset = row["dataset"]
    if row["abs_PIIR_diff"] >= 0.20:
        return (
            f"In {dataset}, near-identical F1 is paired with a large PIIR gap, "
            "suggesting that aggregate F1 alone is incomplete for intervention mechanisms."
        )
    if row["abs_HN_FPR_diff"] >= 0.10:
        return (
            f"In {dataset}, similar F1 hides a meaningful HN-FPR difference, "
            "which matters for productive hard-negative regions."
        )
    return (
        f"In {dataset}, the pair has similar F1 but different trigger-site risk, "
        "supporting risk-aware reporting."
    )


def build_table4(metrics: pd.DataFrame, near_f1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_columns(
        near_f1,
        [
            "dataset",
            "detector_a",
            "config_a",
            "F1_a",
            "PIIR_a",
            "HN_FPR_a",
            "detector_b",
            "config_b",
            "F1_b",
            "PIIR_b",
            "HN_FPR_b",
            "abs_F1_diff",
            "abs_PIIR_diff",
            "abs_HN_FPR_diff",
        ],
        "near-F1 pair table",
    )
    pairs = add_burden_to_pairs(near_f1, metrics)
    pairs["max_f1"] = pairs[["F1_a", "F1_b"]].max(axis=1)

    candidate_main = pairs[
        pairs["dataset"].isin(["Final500", "StressFresh100"])
        & (pairs["max_f1"] >= 0.10)
        & (pairs["abs_F1_diff"] <= 0.02)
    ].copy()
    main_parts = []
    for dataset in ["Final500", "StressFresh100"]:
        dataset_pairs = candidate_main[candidate_main["dataset"] == dataset].copy()
        dataset_pairs = dataset_pairs.sort_values(
            ["abs_PIIR_diff", "abs_HN_FPR_diff"],
            ascending=False,
        ).head(4)
        main_parts.append(dataset_pairs)
    main = pd.concat(main_parts, ignore_index=True) if main_parts else pd.DataFrame()

    appendix = pairs[
        (pairs["dataset"] == "OpenHandsExternal330")
        & (pairs["max_f1"] < 0.05)
        & (pairs["abs_F1_diff"] <= 0.02)
    ].copy()
    appendix = appendix.sort_values(["abs_PIIR_diff", "abs_HN_FPR_diff"], ascending=False).head(10)

    def format_pairs(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            rows.append(
                {
                    "Dataset": row["dataset"],
                    "Detector A": f"{row['detector_a']} ({row['config_a']})",
                    "Detector B": f"{row['detector_b']} ({row['config_b']})",
                    "F1 A": round_float(row["F1_a"]),
                    "F1 B": round_float(row["F1_b"]),
                    "Delta F1": round_float(row["abs_F1_diff"]),
                    "PIIR A": round_float(row["PIIR_a"]),
                    "PIIR B": round_float(row["PIIR_b"]),
                    "Delta PIIR": round_float(row["abs_PIIR_diff"]),
                    "HN-FPR A": round_float(row["HN_FPR_a"]),
                    "HN-FPR B": round_float(row["HN_FPR_b"]),
                    "Delta HN-FPR": round_float(row["abs_HN_FPR_diff"]),
                    "Interpretation": pair_interpretation(row),
                }
            )
        return pd.DataFrame(rows)

    return format_pairs(main), format_pairs(appendix)


def write_notes(output_dir: Path) -> None:
    notes = """# Main Table Notes

These tables are built from the completed Paper1 audit, construct-mismatch, and rich detector intervention-risk outputs. Datasets are reported separately because they represent different evidence pools rather than one pooled natural distribution.

## Table 1: Evidence Pools

This table belongs in the main paper because it anchors the empirical setting: source framework, dataset size, average trajectory length, construct prevalence, sampling role, and main analytical use. It helps readers see that Natural500, Final500, StressFresh100, and OpenHandsExternal330 serve different roles.

## Table 2: Construct Non-Equivalence

This table belongs in the main paper because it directly supports the construct claim. HN-only, UC-only, and mixed HN+UC trajectories all appear, suggesting that productive iteration, hard negatives, and unproductive cycles are related but should not be collapsed into one binary stuck label.

## Table 3: Focused Detector Risk Comparison

This table belongs in the main paper because it shows intervention-risk metrics next to conventional F1 and UC recall. The selected rows emphasize non-degenerate, interpretable detector settings: step caps, file revisits, tool repetition, and a simple progress-aware suppression baseline.

## Table 4: Near-F1 Risk Divergence

This table belongs in the main paper because it shows that near-identical F1 can correspond to substantially different PIIR and HN-FPR. OpenHands near-zero-F1 pairs are excluded from the main table because they are diagnostic of long-horizon step-cap behavior but less useful as a focused main-paper near-F1 comparison; they are exported separately as an appendix table.
"""
    (output_dir / "main_table_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    root = repo_root()
    output_dir = out_dir(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(root)

    table1 = build_table1(inputs["size"], inputs["prevalence"])
    table2 = build_table2(inputs["prevalence"], inputs["overlap"])
    table3 = build_table3(inputs["metrics"])
    table4, appendix_table = build_table4(inputs["metrics"], inputs["near_f1"])

    write_table(
        table1,
        "table1_evidence_pools",
        output_dir,
        "Evidence pools used in the Paper1 analysis.",
        "table1_evidence_pools",
    )
    write_table(
        table2,
        "table2_construct_non_equivalence",
        output_dir,
        "Trajectory-level construct non-equivalence across evidence pools.",
        "table2_construct_non_equivalence",
    )
    write_table(
        table3,
        "table3_focused_detector_risk_comparison",
        output_dir,
        "Focused detector risk comparison for selected non-degenerate configurations.",
        "table3_focused_detector_risk_comparison",
    )
    write_table(
        table4,
        "table4_near_f1_risk_divergence",
        output_dir,
        "Near-F1 detector pairs with divergent productive-interruption risk.",
        "table4_near_f1_risk_divergence",
    )
    if not appendix_table.empty:
        write_table(
            appendix_table,
            "appendix_openhands_degenerate_near_f1_pairs",
            output_dir,
            "OpenHands near-zero-F1 diagnostic near-F1 pairs.",
            "appendix_openhands_degenerate_near_f1_pairs",
        )
    write_notes(output_dir)

    required = [
        output_dir / "table1_evidence_pools.csv",
        output_dir / "table1_evidence_pools.md",
        output_dir / "table1_evidence_pools.tex",
        output_dir / "table2_construct_non_equivalence.csv",
        output_dir / "table2_construct_non_equivalence.md",
        output_dir / "table2_construct_non_equivalence.tex",
        output_dir / "table3_focused_detector_risk_comparison.csv",
        output_dir / "table3_focused_detector_risk_comparison.md",
        output_dir / "table3_focused_detector_risk_comparison.tex",
        output_dir / "table4_near_f1_risk_divergence.csv",
        output_dir / "table4_near_f1_risk_divergence.md",
        output_dir / "table4_near_f1_risk_divergence.tex",
        output_dir / "main_table_notes.md",
    ]
    empty = [path for path in required if not path.exists() or path.stat().st_size == 0]
    if empty:
        raise RuntimeError(f"Missing or empty required outputs: {empty}")

    print(f"Main paper tables written to {output_dir}")


if __name__ == "__main__":
    main()
