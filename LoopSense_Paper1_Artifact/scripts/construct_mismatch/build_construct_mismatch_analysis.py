from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.paper1_audit.audit_dataset_integrity import ensure_normalized
from scripts.paper1_audit.construct_distribution_analysis import compute_trajectory_flags
from scripts.paper1_audit.load_paper1_datasets import repo_root_from_script


DATASET_ORDER = ["Natural500", "Final500", "StressFresh100", "OpenHandsExternal330"]
SWE_AGENT_DATASETS = {"Natural500", "Final500", "StressFresh100"}


def pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100 * value:.1f}%"


def save_fig(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{name}.png", dpi=240)
    fig.savefig(figures_dir / f"{name}.pdf")
    plt.close(fig)


def ordered(df: pd.DataFrame, column: str = "dataset") -> pd.DataFrame:
    if column not in df.columns:
        return df
    order = {name: i for i, name in enumerate(DATASET_ORDER)}
    return df.assign(_order=df[column].map(order).fillna(999)).sort_values(["_order", column]).drop(columns="_order")


def bool_col(series: pd.Series) -> pd.Series:
    def parse(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        text = str(value).strip().lower()
        return text in {"true", "1", "yes", "y", "t"}

    return series.map(parse)


def construct_prevalence(flags: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, role), group in flags.groupby(["dataset", "dataset_role"], dropna=False):
        n = len(group)
        measures = {
            "contains_pi": group["has_pi"],
            "contains_hn": group["has_hn"],
            "contains_uc": group["has_uc"],
            "contains_both_hn_and_uc": group["has_hn"] & group["has_uc"],
            "contains_pi_but_no_uc": group["has_pi"] & ~group["has_uc"],
            "contains_hn_but_no_uc": group["has_hn"] & ~group["has_uc"],
        }
        row: dict[str, Any] = {
            "dataset": dataset,
            "dataset_role": role,
            "trajectory_count": n,
        }
        for name, mask in measures.items():
            count = int(mask.sum())
            row[f"{name}_count"] = count
            row[f"{name}_pct"] = count / n if n else math.nan
        rows.append(row)
    return ordered(pd.DataFrame(rows))


def length_vs_constructs(flags: pd.DataFrame) -> pd.DataFrame:
    out = flags.copy()
    out["trajectory_length"] = out["step_count"].astype(int)
    out["source_family"] = out["dataset"].map(
        lambda d: "OpenHands external" if d == "OpenHandsExternal330" else "SWE-agent datasets"
    )
    global_threshold = out["trajectory_length"].quantile(0.9)
    out["top_10pct_longest_global"] = out["trajectory_length"] >= global_threshold
    thresholds = out.groupby("dataset")["trajectory_length"].quantile(0.9).to_dict()
    out["dataset_top_10pct_length_threshold"] = out["dataset"].map(thresholds)
    out["top_10pct_longest_within_dataset"] = out["trajectory_length"] >= out["dataset_top_10pct_length_threshold"]
    keep = [
        "dataset",
        "dataset_role",
        "trajectory_id",
        "source_family",
        "trajectory_length",
        "top_10pct_longest_global",
        "top_10pct_longest_within_dataset",
        "pi_density",
        "hn_density",
        "uc_density",
        "has_pi",
        "has_hn",
        "has_uc",
        "pi_step_count",
        "hn_step_count",
        "uc_step_count",
        "pi_projected_step_count",
        "hn_projected_step_count",
        "uc_projected_step_count",
    ]
    return ordered(out[keep])


def add_projected_construct_densities(flags: pd.DataFrame, steps: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    coverage: dict[tuple[str, str], dict[str, set[int]]] = {}

    def bucket(dataset: Any, trajectory_id: Any) -> dict[str, set[int]]:
        key = (str(dataset), str(trajectory_id))
        if key not in coverage:
            coverage[key] = {"pi": set(), "hn": set(), "uc": set()}
        return coverage[key]

    step_cols = [
        "dataset",
        "trajectory_id",
        "step_index",
        "span_label_norm",
        "hard_negative",
        "productive_iteration_flag",
        "unproductive_cycle_flag",
    ]
    for row in steps[step_cols].itertuples(index=False):
        try:
            step_index = int(row.step_index)
        except (TypeError, ValueError):
            continue
        b = bucket(row.dataset, row.trajectory_id)
        span_label = "" if pd.isna(row.span_label_norm) else str(row.span_label_norm)
        if span_label == "productive_iteration" or bool_col(pd.Series([row.productive_iteration_flag])).iloc[0]:
            b["pi"].add(step_index)
        if bool_col(pd.Series([row.hard_negative])).iloc[0]:
            b["hn"].add(step_index)
        if span_label == "unproductive_cycle" or bool_col(pd.Series([row.unproductive_cycle_flag])).iloc[0]:
            b["uc"].add(step_index)

    if not spans.empty:
        span_cols = [
            "dataset",
            "trajectory_id",
            "start_step_index",
            "end_step_index",
            "span_label_norm",
            "hard_negative",
            "productive_iteration_flag",
            "unproductive_cycle_flag",
        ]
        available_cols = [c for c in span_cols if c in spans.columns]
        span_rows = spans[available_cols].copy()
        for missing in set(span_cols) - set(available_cols):
            span_rows[missing] = None
        for row in span_rows[span_cols].itertuples(index=False):
            try:
                start = int(float(row.start_step_index))
                end = int(float(row.end_step_index))
            except (TypeError, ValueError):
                continue
            if end < start:
                start, end = end, start
            # Clamp pathological spans to a conservative range. Current datasets are
            # short enough that materializing intervals is simpler and transparent.
            if end - start > 10000:
                continue
            b = bucket(row.dataset, row.trajectory_id)
            span_label = "" if pd.isna(row.span_label_norm) else str(row.span_label_norm)
            indices = range(start, end + 1)
            if span_label == "productive_iteration" or bool_col(pd.Series([row.productive_iteration_flag])).iloc[0]:
                b["pi"].update(indices)
            if bool_col(pd.Series([row.hard_negative])).iloc[0]:
                b["hn"].update(indices)
            if span_label == "unproductive_cycle" or bool_col(pd.Series([row.unproductive_cycle_flag])).iloc[0]:
                b["uc"].update(indices)

    rows: list[dict[str, Any]] = []
    for (dataset, trajectory_id), values in coverage.items():
        rows.append(
            {
                "dataset": dataset,
                "trajectory_id": trajectory_id,
                "pi_projected_step_count": len(values["pi"]),
                "hn_projected_step_count": len(values["hn"]),
                "uc_projected_step_count": len(values["uc"]),
            }
        )
    projected = pd.DataFrame(rows)
    out = flags.merge(projected, on=["dataset", "trajectory_id"], how="left")
    for col in ["pi_projected_step_count", "hn_projected_step_count", "uc_projected_step_count"]:
        out[col] = out[col].fillna(0).astype(int)
    out["pi_density"] = (out["pi_projected_step_count"] / out["step_count"]).fillna(0)
    out["hn_density"] = (out["hn_projected_step_count"] / out["step_count"]).fillna(0)
    out["uc_density"] = (out["uc_projected_step_count"] / out["step_count"]).fillna(0)
    return out


def length_comparison_summary(length_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_group(name: str, mask: pd.Series) -> None:
        group = length_df[mask]
        if group.empty:
            return
        rows.append(
            {
                "comparison": name,
                "trajectory_count": len(group),
                "mean_length": group["trajectory_length"].mean(),
                "median_length": group["trajectory_length"].median(),
                "p90_length": group["trajectory_length"].quantile(0.9),
                "contains_uc_count": int(group["has_uc"].sum()),
                "contains_uc_pct": group["has_uc"].mean(),
                "contains_hn_count": int(group["has_hn"].sum()),
                "contains_hn_pct": group["has_hn"].mean(),
                "mean_pi_density": group["pi_density"].mean(),
                "mean_hn_density": group["hn_density"].mean(),
                "mean_uc_density": group["uc_density"].mean(),
            }
        )

    add_group("global_top_10pct_longest", length_df["top_10pct_longest_global"])
    add_group("global_remaining_90pct", ~length_df["top_10pct_longest_global"])
    add_group("OpenHands_external", length_df["source_family"] == "OpenHands external")
    add_group("SWE_agent_datasets", length_df["source_family"] == "SWE-agent datasets")
    for dataset in DATASET_ORDER:
        add_group(f"{dataset}_within_dataset_top_10pct_longest", (length_df["dataset"] == dataset) & length_df["top_10pct_longest_within_dataset"])
        add_group(f"{dataset}_within_dataset_remaining", (length_df["dataset"] == dataset) & ~length_df["top_10pct_longest_within_dataset"])
    return pd.DataFrame(rows)


def interval_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    if right < left:
        return 0.0
    inter = right - left + 1
    union = (a_end - a_start + 1) + (b_end - b_start + 1) - inter
    return inter / union if union else 0.0


def hn_uc_overlap(flags: pd.DataFrame, spans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    category_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for (dataset, role), group in flags.groupby(["dataset", "dataset_role"], dropna=False):
        categories = {
            "HN_only": group["has_hn"] & ~group["has_uc"],
            "UC_only": group["has_uc"] & ~group["has_hn"],
            "mixed_HN_and_UC": group["has_hn"] & group["has_uc"],
            "neither_HN_nor_UC": ~group["has_hn"] & ~group["has_uc"],
        }
        n = len(group)
        for name, mask in categories.items():
            ids = group.loc[mask, "trajectory_id"].astype(str).tolist()
            category_rows.append(
                {
                    "dataset": dataset,
                    "dataset_role": role,
                    "category": name,
                    "trajectory_count": len(ids),
                    "trajectory_pct": len(ids) / n if n else math.nan,
                    "example_trajectory_ids": ";".join(ids[:12]),
                }
            )
            for tid in ids[:25]:
                example_rows.append({"dataset": dataset, "category": name, "trajectory_id": tid})

    span_rows: list[dict[str, Any]] = []
    if not spans.empty:
        use = spans.copy()
        use["start_step_index"] = pd.to_numeric(use["start_step_index"], errors="coerce")
        use["end_step_index"] = pd.to_numeric(use["end_step_index"], errors="coerce")
        use = use.dropna(subset=["start_step_index", "end_step_index"]).copy()
        use["start_step_index"] = use["start_step_index"].astype(int)
        use["end_step_index"] = use["end_step_index"].astype(int)
        use["is_hn_span"] = bool_col(use.get("hard_negative", pd.Series(False, index=use.index)))
        use["is_uc_span"] = use["span_label_norm"].fillna("").eq("unproductive_cycle") | bool_col(
            use.get("unproductive_cycle_flag", pd.Series(False, index=use.index))
        )
        for (dataset, role), g in use.groupby(["dataset", "dataset_role"], dropna=False):
            hn_spans = g[g["is_hn_span"]]
            uc_spans = g[g["is_uc_span"]]
            overlapping_pairs = 0
            hn_overlap_ids: set[Any] = set()
            uc_overlap_ids: set[Any] = set()
            ious: list[float] = []
            for tid, hn_t in hn_spans.groupby("trajectory_id"):
                uc_t = uc_spans[uc_spans["trajectory_id"] == tid]
                for hn_idx, hn_row in hn_t.iterrows():
                    for uc_idx, uc_row in uc_t.iterrows():
                        iou = interval_iou(
                            int(hn_row["start_step_index"]),
                            int(hn_row["end_step_index"]),
                            int(uc_row["start_step_index"]),
                            int(uc_row["end_step_index"]),
                        )
                        if iou > 0:
                            overlapping_pairs += 1
                            hn_overlap_ids.add(hn_idx)
                            uc_overlap_ids.add(uc_idx)
                            ious.append(iou)
            span_rows.append(
                {
                    "dataset": dataset,
                    "dataset_role": role,
                    "hn_span_count": len(hn_spans),
                    "uc_span_count": len(uc_spans),
                    "overlapping_hn_uc_span_pairs": overlapping_pairs,
                    "hn_spans_overlapping_uc_count": len(hn_overlap_ids),
                    "uc_spans_overlapping_hn_count": len(uc_overlap_ids),
                    "mean_iou_for_overlapping_pairs": sum(ious) / len(ious) if ious else 0.0,
                }
            )
    return ordered(pd.DataFrame(category_rows)), ordered(pd.DataFrame(span_rows)), ordered(pd.DataFrame(example_rows))


def parse_extra_json(value: Any) -> dict[str, Any]:
    if pd.isna(value) or value == "":
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_signal_score(extra: dict[str, Any], name: str) -> float | None:
    signals = extra.get("signals")
    if not isinstance(signals, dict):
        return None
    sig = signals.get(name)
    if not isinstance(sig, dict):
        return None
    value = sig.get("score")
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def add_local_signal_columns(steps: pd.DataFrame) -> pd.DataFrame:
    out = steps[[
        "dataset",
        "dataset_role",
        "trajectory_id",
        "step_index",
        "label_norm",
        "span_label_norm",
        "hard_negative",
        "productive_iteration_flag",
        "unproductive_cycle_flag",
        "uncertain_flag",
        "raw_extra_json",
    ]].copy()
    extras = out["raw_extra_json"].map(parse_extra_json)
    for name in ["progress", "recurrence", "novelty", "error_persistence", "budget_pressure"]:
        out[f"{name}_score"] = extras.map(lambda obj, key=name: extract_signal_score(obj, key))
    out["has_legacy_signal_scores"] = out[["progress_score", "recurrence_score", "novelty_score", "error_persistence_score"]].notna().any(axis=1)
    out["hard_negative"] = bool_col(out["hard_negative"])
    out["productive_iteration_flag"] = bool_col(out["productive_iteration_flag"])
    out["unproductive_cycle_flag"] = bool_col(out["unproductive_cycle_flag"])
    out["uncertain_flag"] = bool_col(out["uncertain_flag"])
    out["region_type"] = "other"
    out.loc[out["productive_iteration_flag"], "region_type"] = "PI_non_HN"
    out.loc[out["unproductive_cycle_flag"], "region_type"] = "UC"
    out.loc[out["hard_negative"], "region_type"] = "HN"
    out.loc[out["uncertain_flag"], "region_type"] = "uncertain_or_unresolved"
    return out


def local_signal_summary(steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = add_local_signal_columns(steps)
    score_cols = ["progress_score", "recurrence_score", "novelty_score", "error_persistence_score", "budget_pressure_score"]
    rows: list[dict[str, Any]] = []
    for (dataset, region), group in enriched.groupby(["dataset", "region_type"], dropna=False):
        row: dict[str, Any] = {
            "dataset": dataset,
            "region_type": region,
            "step_count": len(group),
            "rows_with_legacy_signal_scores": int(group["has_legacy_signal_scores"].sum()),
            "legacy_signal_coverage_pct": group["has_legacy_signal_scores"].mean() if len(group) else math.nan,
        }
        for col in score_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_median"] = group[col].median()
        rows.append(row)
    feature_summary = ordered(pd.DataFrame(rows))

    availability_rows = [
        {
            "feature": "repeated same action",
            "status": "unavailable",
            "reason": "The normalized audit tables do not preserve raw action text or canonical action identifiers.",
        },
        {
            "feature": "repeated tool",
            "status": "unavailable",
            "reason": "The normalized audit tables do not preserve tool_name/tool_call fields.",
        },
        {
            "feature": "repeated file",
            "status": "unavailable",
            "reason": "The normalized audit tables do not preserve file paths or edited/read file fields.",
        },
        {
            "feature": "repeated error signature",
            "status": "partially_available_for_Final500_and_Natural500",
            "reason": "Historical release-derived signal scores include error_persistence and recurrence, but normalized tables do not preserve raw stack traces.",
        },
        {
            "feature": "new file/function touched",
            "status": "unavailable",
            "reason": "The normalized audit tables do not preserve file/function identifiers.",
        },
        {
            "feature": "action change after feedback",
            "status": "unavailable",
            "reason": "The normalized audit tables do not preserve consecutive action semantics.",
        },
        {
            "feature": "new observation or evidence",
            "status": "partially_available_for_Final500_and_Natural500",
            "reason": "Historical release-derived signal scores include novelty and progress proxies, but not the raw observations themselves.",
        },
        {
            "feature": "same-error persistence",
            "status": "partially_available_for_Final500_and_Natural500",
            "reason": "Historical release-derived signal scores include error_persistence proxies.",
        },
    ]
    availability = pd.DataFrame(availability_rows)
    return feature_summary, availability


def make_prevalence_figure(prev: pd.DataFrame, figures_dir: Path) -> None:
    plot = prev.set_index("dataset")[
        [
            "contains_pi_pct",
            "contains_hn_pct",
            "contains_uc_pct",
            "contains_both_hn_and_uc_pct",
            "contains_pi_but_no_uc_pct",
            "contains_hn_but_no_uc_pct",
        ]
    ]
    plot = plot.loc[[d for d in DATASET_ORDER if d in plot.index]]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of trajectories")
    ax.set_title("Construct prevalence by dataset")
    ax.legend(["PI", "HN", "UC", "HN+UC", "PI no UC", "HN no UC"], ncol=3, fontsize=8)
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, figures_dir, "construct_prevalence_by_dataset")


def make_length_figure(length_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, field, title in zip(axes, ["pi_density", "hn_density", "uc_density"], ["PI density", "HN density", "UC density"]):
        for dataset, group in ordered(length_df).groupby("dataset"):
            ax.scatter(group["trajectory_length"], group[field], s=13, alpha=0.55, label=dataset)
        ax.set_xlabel("Trajectory length")
        ax.set_title(title)
    axes[0].set_ylabel("Step-level density")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.subplots_adjust(bottom=0.25)
    save_fig(fig, figures_dir, "length_vs_construct_density")


def make_top_decile_figure(summary: pd.DataFrame, figures_dir: Path) -> None:
    keep = summary[summary["comparison"].isin(["global_top_10pct_longest", "global_remaining_90pct", "OpenHands_external", "SWE_agent_datasets"])].copy()
    keep = keep.set_index("comparison")[["contains_uc_pct", "contains_hn_pct", "mean_pi_density", "mean_uc_density"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    keep.plot(kind="bar", ax=ax)
    ax.set_ylim(0, max(1.05, float(keep.max().max()) * 1.15))
    ax.set_ylabel("Rate or mean density")
    ax.set_title("Length and source-family comparison")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, figures_dir, "top_decile_and_source_family_comparison")


def make_overlap_figure(overlap_categories: pd.DataFrame, figures_dir: Path) -> None:
    keep = overlap_categories[overlap_categories["category"].isin(["HN_only", "UC_only", "mixed_HN_and_UC", "neither_HN_nor_UC"])]
    pivot = keep.pivot(index="dataset", columns="category", values="trajectory_pct").fillna(0)
    order = [c for c in ["HN_only", "UC_only", "mixed_HN_and_UC", "neither_HN_nor_UC"] if c in pivot.columns]
    pivot = pivot.loc[[d for d in DATASET_ORDER if d in pivot.index]]
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot[order].plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of trajectories")
    ax.set_title("HN and UC trajectory-level overlap")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, figures_dir, "hn_uc_overlap_by_dataset")


def make_funnel_figures(prev: pd.DataFrame, figures_dir: Path) -> None:
    metrics = [
        ("All trajectories", "trajectory_count"),
        ("Productive iteration", "contains_pi_count"),
        ("Hard negatives", "contains_hn_count"),
        ("Unproductive cycles", "contains_uc_count"),
        ("Mixed HN+UC", "contains_both_hn_and_uc_count"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=False)
    axes_flat = axes.ravel()
    for ax, (_, row) in zip(axes_flat, ordered(prev).iterrows()):
        names = [name for name, _ in metrics]
        values = [int(row[col]) for _, col in metrics]
        ax.barh(names, values, color=["#4c78a8", "#59a14f", "#f28e2b", "#e15759", "#b07aa1"])
        ax.invert_yaxis()
        ax.set_title(str(row["dataset"]))
        ax.set_xlabel("Trajectories")
        for i, value in enumerate(values):
            ax.text(value, i, f" {value}", va="center", fontsize=8)
    for ax in axes_flat[len(prev) :]:
        ax.axis("off")
    save_fig(fig, figures_dir, "construct_funnel_by_dataset")

    for _, row in ordered(prev).iterrows():
        names = [name for name, _ in metrics]
        values = [int(row[col]) for _, col in metrics]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(names, values, color=["#4c78a8", "#59a14f", "#f28e2b", "#e15759", "#b07aa1"])
        ax.invert_yaxis()
        ax.set_title(f"Construct funnel: {row['dataset']}")
        ax.set_xlabel("Trajectories")
        for i, value in enumerate(values):
            ax.text(value, i, f" {value}", va="center", fontsize=8)
        safe = str(row["dataset"]).lower()
        save_fig(fig, figures_dir, f"construct_funnel_{safe}")


def make_local_signal_figure(local_summary: pd.DataFrame, figures_dir: Path) -> None:
    signal_rows = local_summary[local_summary["rows_with_legacy_signal_scores"] > 0].copy()
    if signal_rows.empty:
        return
    keep_regions = ["HN", "PI_non_HN", "UC", "other"]
    signal_rows = signal_rows[signal_rows["region_type"].isin(keep_regions)]
    metrics = ["progress_score_mean", "novelty_score_mean", "recurrence_score_mean", "error_persistence_score_mean"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, metric in zip(axes.ravel(), metrics):
        pivot = signal_rows.pivot(index="dataset", columns="region_type", values=metric)
        pivot = pivot.loc[[d for d in DATASET_ORDER if d in pivot.index]]
        cols = [c for c in keep_regions if c in pivot.columns]
        pivot[cols].plot(kind="bar", ax=ax)
        ax.set_title(metric.replace("_score_mean", "").replace("_", " ").title())
        ax.set_ylabel("Mean signal score")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
    save_fig(fig, figures_dir, "local_process_signal_proxy_comparison")


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    show = df if max_rows is None else df.head(max_rows)
    if show.empty:
        return "_No rows._"
    return show.to_markdown(index=False)


def report_lines(
    prev: pd.DataFrame,
    length_summary: pd.DataFrame,
    overlap_categories: pd.DataFrame,
    span_overlap: pd.DataFrame,
    local_summary: pd.DataFrame,
    feature_availability: pd.DataFrame,
) -> list[str]:
    prev_display = prev.copy()
    for col in [c for c in prev_display.columns if c.endswith("_pct")]:
        prev_display[col] = prev_display[col].map(pct)
    length_display = length_summary.copy()
    for col in [c for c in length_display.columns if c.endswith("_pct") or c.startswith("mean_")]:
        if col.endswith("_pct"):
            length_display[col] = length_display[col].map(pct)
        else:
            length_display[col] = length_display[col].map(lambda v: f"{v:.3f}" if not pd.isna(v) else "NA")
    overlap_display = overlap_categories.drop(columns=["example_trajectory_ids"], errors="ignore").copy()
    overlap_display["trajectory_pct"] = overlap_display["trajectory_pct"].map(pct)
    span_display = span_overlap.copy()
    if not span_display.empty:
        for col in ["mean_iou_for_overlapping_pairs"]:
            span_display[col] = span_display[col].map(lambda v: f"{v:.3f}" if not pd.isna(v) else "NA")

    top = length_summary.set_index("comparison")
    openhands_uc = top.loc["OpenHands_external", "contains_uc_pct"] if "OpenHands_external" in top.index else math.nan
    swe_uc = top.loc["SWE_agent_datasets", "contains_uc_pct"] if "SWE_agent_datasets" in top.index else math.nan
    top_uc = top.loc["global_top_10pct_longest", "contains_uc_pct"] if "global_top_10pct_longest" in top.index else math.nan
    rest_uc = top.loc["global_remaining_90pct", "contains_uc_pct"] if "global_remaining_90pct" in top.index else math.nan

    lines = [
        "# Construct Mismatch Analysis",
        "",
        "This analysis uses the normalized step and span tables from `outputs/paper1_audit/` and writes derived construct tables and figures to `outputs/paper1_construct_mismatch/`.",
        "",
        "## Core Claim",
        "",
        "In software-engineering agent debugging, repetition and long trajectories are ambiguous evidence. Productive iteration, hard negatives, and unproductive cycles are related but distinct constructs. These results motivate evaluation designs that measure intervention risk rather than treating loop or stuck detection as a simple binary classification task.",
        "",
        "## RQ1: How different are the four evidence pools?",
        "",
        "The four datasets differ in sampling role and construct prevalence. Natural500 is the natural-distribution control, Final500 is enriched for difficult loop-heavy behavior, StressFresh100 concentrates clean challenge cases, and OpenHandsExternal330 is an external long-horizon generalization set. They should therefore be reported separately rather than pooled as a single natural distribution.",
        "",
        md_table(prev_display),
        "",
        "## RQ2: Does long or repetitive behavior imply unproductive cycling?",
        "",
        f"The global top decile of longest trajectories has UC prevalence `{pct(top_uc)}`, compared with `{pct(rest_uc)}` for the remaining trajectories. OpenHands external trajectories are much longer on average, but their UC prevalence is `{pct(openhands_uc)}`, compared with `{pct(swe_uc)}` for the SWE-agent datasets. This supports the paper's caution that length is not a direct substitute for unproductive cycling.",
        "",
        md_table(length_display[length_display["comparison"].isin(["global_top_10pct_longest", "global_remaining_90pct", "OpenHands_external", "SWE_agent_datasets"])]),
        "",
        "## RQ3: How often do hard negatives and unproductive cycles overlap?",
        "",
        "HN and UC overlap at the trajectory level in some datasets, but neither construct contains the other. HN-only trajectories are intervention-risk cases because they contain behavior that may look repetitive while remaining productive. UC-only trajectories capture unproductive cycling without the hard-negative caveat. Mixed HN+UC trajectories show that local labels can coexist within the same long debugging run.",
        "",
        md_table(overlap_display),
        "",
        "### Span-Level HN/UC Overlap",
        "",
        md_table(span_display),
        "",
        "## RQ4: Do productive-looking and unproductive regions differ in local process signals?",
        "",
        "The normalized tables do not retain enough raw action/tool/file/error text to compute every requested local feature directly. For Final500 and Natural500, historical release-derived signal scores provide partial proxies for progress, novelty, recurrence, and error persistence. For StressFresh100 and OpenHandsExternal330, these raw local process signals are not available in the normalized audit tables, so the analysis documents the limitation instead of inventing features.",
        "",
        md_table(feature_availability),
        "",
        "### Local Signal Proxy Summary",
        "",
        md_table(local_summary[local_summary["rows_with_legacy_signal_scores"] > 0].round(3)),
        "",
        "## Paper-Ready Findings",
        "",
        "- The four evidence pools occupy different construct regimes and should not be pooled as one natural distribution.",
        "- Productive iteration is common across datasets, but hard negatives are a narrower construct tied to intervention risk.",
        "- Unproductive cycles are dataset-dependent: they are prominent in Final500 and StressFresh100 but rare in OpenHandsExternal330 despite its much longer trajectories.",
        "- Long trajectories do not necessarily imply unproductive cycling; length and repetition need local semantic interpretation.",
        "- HN-only and UC-only trajectories both exist, showing that hard negatives and unproductive cycles are related but non-equivalent.",
        "- Mixed HN+UC trajectories show that a single debugging run can contain productive repetition and unproductive cycling in different local regions.",
        "- The available local signal proxies support construct separation, but full action/tool/file/error feature analysis requires retaining raw process fields in the normalized tables.",
        "- These findings motivate intervention-risk evaluation: a detector should distinguish when to interrupt unproductive cycling from when to allow productive iteration to continue.",
        "",
        "## Reproducibility Notes",
        "",
        "- Inputs: `outputs/paper1_audit/normalized_steps.csv` and `outputs/paper1_audit/normalized_spans.csv`.",
        "- No raw annotation files were modified.",
        "- The analysis treats UC, HN, and productive iteration as annotated constructs, not as claims that one label family is the only ground truth.",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Paper1 construct-mismatch analysis from normalized audit tables.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/paper1_audit"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_construct_mismatch"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    audit_dir = (repo_root / args.audit_dir).resolve() if not args.audit_dir.is_absolute() else args.audit_dir
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    steps, spans = ensure_normalized(repo_root, audit_dir)
    flags = compute_trajectory_flags(steps, spans)
    flags = add_projected_construct_densities(flags, steps, spans)
    flags = ordered(flags)

    prev = construct_prevalence(flags)
    length_df = length_vs_constructs(flags)
    length_summary = length_comparison_summary(length_df)
    overlap_categories, span_overlap, overlap_examples = hn_uc_overlap(flags, spans)
    local_summary, feature_availability = local_signal_summary(steps)

    # Required root-level files plus full tables directory copies.
    prev.to_csv(output_dir / "construct_prevalence_by_dataset.csv", index=False, encoding="utf-8-sig")
    length_df.to_csv(output_dir / "length_vs_constructs.csv", index=False, encoding="utf-8-sig")

    outputs = {
        "construct_prevalence_by_dataset": prev,
        "length_vs_constructs": length_df,
        "length_comparison_summary": length_summary,
        "hn_uc_overlap_trajectory_categories": overlap_categories,
        "hn_uc_overlap_span_level": span_overlap,
        "hn_uc_overlap_example_trajectory_ids": overlap_examples,
        "local_process_signal_proxy_summary": local_summary,
        "local_process_feature_availability": feature_availability,
        "construct_trajectory_flags": flags,
    }
    for name, df in outputs.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    make_prevalence_figure(prev, figures_dir)
    make_length_figure(length_df, figures_dir)
    make_top_decile_figure(length_summary, figures_dir)
    make_overlap_figure(overlap_categories, figures_dir)
    make_funnel_figures(prev, figures_dir)
    make_local_signal_figure(local_summary, figures_dir)

    lines = report_lines(prev, length_summary, overlap_categories, span_overlap, local_summary, feature_availability)
    (output_dir / "construct_mismatch_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Construct mismatch analysis complete: {output_dir}")
    print(f"Tables: {tables_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
