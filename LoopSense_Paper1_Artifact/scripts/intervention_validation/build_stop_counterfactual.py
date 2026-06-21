from __future__ import annotations

import argparse
import bisect
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.paper1_intervention_risk.build_rich_detector_evaluation import (  # noqa: E402
    DATASET_ORDER,
    availability,
    compute_trigger,
    detector_specs,
    prepare_steps,
)

LOCATION_ORDER = ["UC", "HN", "PI", "other", "uncertain"]
DISPLAY_DATASET = {"Natural500": "Natural500", "Final500": "Enriched500", "StressFresh100": "StressFresh100", "OpenHandsExternal330": "OpenHandsExternal330"}

def display_dataset(value: str) -> str:
    return DISPLAY_DATASET.get(str(value), str(value))

def with_display_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dataset" in out.columns:
        out["dataset"] = out["dataset"].map(display_dataset)
    return out

DATASET_COLORS = {
    "Natural500": "#0072B2",
    "Final500": "#009E73",
    "StressFresh100": "#E69F00",
    "OpenHandsExternal330": "#CC79A7",
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def ensure_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def add_suffix_step_counts(steps: pd.DataFrame) -> pd.DataFrame:
    steps = steps.sort_values(["dataset", "trajectory_id", "step_index"]).copy()
    steps["in_productive"] = steps["in_pi"].astype(bool) | steps["in_hn"].astype(bool)
    specs = {
        "remaining_PI_steps_after_trigger": "in_pi",
        "remaining_HN_steps_after_trigger": "in_hn",
        "remaining_UC_steps_after_trigger": "in_uc",
        "remaining_productive_steps_after_trigger": "in_productive",
    }
    group_keys = ["dataset", "trajectory_id"]
    steps["remaining_steps_after_trigger"] = steps.groupby(group_keys)["step_index"].transform("size") - steps.groupby(group_keys).cumcount() - 1
    for out_col, flag_col in specs.items():
        flag = steps[flag_col].astype(int)
        steps[out_col] = flag.groupby([steps["dataset"], steps["trajectory_id"]]).transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1] - s)
    return steps


def make_span_end_maps(spans: pd.DataFrame) -> tuple[dict[tuple[str, str], list[int]], dict[tuple[str, str], list[int]]]:
    prod: dict[tuple[str, str], list[int]] = {}
    uc: dict[tuple[str, str], list[int]] = {}
    if spans.empty:
        return prod, uc
    use = spans.copy()
    for col in ["dataset", "trajectory_id", "span_label_norm", "productive_iteration_flag", "unproductive_cycle_flag", "hard_negative", "end_step_index"]:
        if col not in use.columns:
            use[col] = ""
    use["end_step_index"] = pd.to_numeric(use["end_step_index"], errors="coerce")
    use = use.dropna(subset=["end_step_index"])
    for row in use.itertuples(index=False):
        key = (str(getattr(row, "dataset")), str(getattr(row, "trajectory_id")))
        end = int(getattr(row, "end_step_index"))
        label = str(getattr(row, "span_label_norm", ""))
        is_prod = label == "productive_iteration" or parse_bool(getattr(row, "productive_iteration_flag", False)) or parse_bool(getattr(row, "hard_negative", False))
        is_uc = label == "unproductive_cycle" or parse_bool(getattr(row, "unproductive_cycle_flag", False))
        if is_prod:
            prod.setdefault(key, []).append(end)
        if is_uc:
            uc.setdefault(key, []).append(end)
    for mapping in [prod, uc]:
        for key in list(mapping):
            mapping[key] = sorted(mapping[key])
    return prod, uc


def count_spans_after(mapping: dict[tuple[str, str], list[int]], dataset: str, trajectory_id: str, step_index: int) -> int:
    ends = mapping.get((dataset, trajectory_id), [])
    if not ends:
        return 0
    return len(ends) - bisect.bisect_right(ends, int(step_index))


def location_for_rows(df: pd.DataFrame) -> pd.Series:
    conditions = [df["in_uc"].astype(bool), df["in_hn"].astype(bool), df["in_pi"].astype(bool), df["in_uncertain"].astype(bool)]
    choices = ["UC", "HN", "PI", "uncertain"]
    return pd.Series(np.select(conditions, choices, default="other"), index=df.index)


def selected_specs_for_report(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    focus = metrics[
        ((metrics["dataset"] == "Final500") & (metrics["detector"].isin(["max_step_guard", "file_revisit_guard"])) & (metrics["threshold/config"].isin(["threshold=50", "window=5;repeats=3"])))
        | ((metrics["dataset"] == "StressFresh100") & (metrics["detector"] == "file_revisit_guard") & (metrics["threshold/config"] == "window=10;repeats=3"))
        | (metrics["detector"] == "repeat_without_progress_guard")
    ].copy()
    return focus


def build_trigger_table(steps: pd.DataFrame, spans: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    steps = add_suffix_step_counts(steps)
    prod_span_ends, uc_span_ends = make_span_end_maps(spans)
    detector_avail = availability(steps)
    specs = detector_specs(steps, detector_avail)
    available_map = {
        (str(row.dataset), str(row.detector_family)): parse_bool(row.available)
        for row in detector_avail.itertuples(index=False)
    }
    records: list[pd.DataFrame] = []
    for spec in specs:
        trigger = compute_trigger(steps, spec)
        if not trigger.any():
            continue
        cols = [
            "dataset",
            "dataset_role",
            "trajectory_id",
            "step_index",
            "step_position",
            "in_pi",
            "in_hn",
            "in_uc",
            "in_uncertain",
            "remaining_steps_after_trigger",
            "remaining_PI_steps_after_trigger",
            "remaining_HN_steps_after_trigger",
            "remaining_UC_steps_after_trigger",
            "remaining_productive_steps_after_trigger",
        ]
        trig = steps.loc[trigger, cols].copy()
        trig = trig[trig.apply(lambda r: available_map.get((str(r["dataset"]), str(spec["family"])), False), axis=1)]
        if trig.empty:
            continue
        trig["detector"] = spec["detector"]
        trig["detector_family"] = spec["family"]
        trig["detector_config"] = spec["config"]
        trig["uses_annotation_signal_proxy"] = bool(spec.get("uses_annotation_signal_proxy", False))
        records.append(trig)
    if not records:
        return pd.DataFrame()
    out = pd.concat(records, ignore_index=True)
    out["trigger_location_type"] = location_for_rows(out)
    out["remaining_productive_spans_after_trigger"] = [
        count_spans_after(prod_span_ends, str(d), str(tid), int(step)) for d, tid, step in zip(out["dataset"], out["trajectory_id"], out["step_index"])
    ]
    out["remaining_UC_spans_after_trigger"] = [
        count_spans_after(uc_span_ends, str(d), str(tid), int(step)) for d, tid, step in zip(out["dataset"], out["trajectory_id"], out["step_index"])
    ]
    out = out.sort_values(["dataset", "detector", "detector_config", "trajectory_id", "step_index"]).reset_index(drop=True)
    out["is_first_trigger_for_config_trajectory"] = out.groupby(["dataset", "detector", "detector_config", "trajectory_id"]).cumcount().eq(0)
    out["would_cut_productive_suffix"] = out["remaining_productive_steps_after_trigger"].gt(0) | out["remaining_productive_spans_after_trigger"].gt(0)
    out["would_save_UC_suffix"] = out["remaining_UC_steps_after_trigger"].gt(0) | out["remaining_UC_spans_after_trigger"].gt(0)
    out["trajectory_final_success"] = pd.NA
    out["current_prefix_success"] = pd.NA
    out["post_trigger_success"] = pd.NA
    out["post_trigger_validation_event"] = pd.NA
    out["would_stop_before_observed_success"] = pd.NA
    out["evidence_available_flag"] = True
    out["success_evidence_available_flag"] = False
    out.insert(0, "trigger_id", [f"trig_{i+1:08d}" for i in range(len(out))])
    metric_cols = ["dataset", "detector", "threshold/config", "F1", "UC_recall", "PIIR", "HN_FPR", "burden"]
    if not metrics.empty and set(metric_cols).issubset(metrics.columns):
        m = metrics[metric_cols].rename(columns={"threshold/config": "detector_config"})
        out = out.merge(m, on=["dataset", "detector", "detector_config"], how="left")
    return out


def summarize_by_detector(triggers: pd.DataFrame) -> pd.DataFrame:
    use = triggers[triggers["is_first_trigger_for_config_trajectory"]].copy()
    rows: list[dict[str, Any]] = []
    for keys, g in use.groupby(["dataset", "detector", "detector_family", "detector_config"], dropna=False):
        dataset, detector, family, config = keys
        row: dict[str, Any] = {
            "dataset": dataset,
            "detector": detector,
            "detector_family": family,
            "detector_config": config,
            "first_trigger_count": len(g),
            "triggered_trajectory_count": g["trajectory_id"].nunique(),
            "productive_suffix_cut_mean": g["remaining_productive_steps_after_trigger"].mean(),
            "productive_suffix_cut_median": g["remaining_productive_steps_after_trigger"].median(),
            "PI_suffix_cut_mean": g["remaining_PI_steps_after_trigger"].mean(),
            "HN_suffix_cut_mean": g["remaining_HN_steps_after_trigger"].mean(),
            "UC_suffix_saved_mean": g["remaining_UC_steps_after_trigger"].mean(),
            "UC_suffix_saved_median": g["remaining_UC_steps_after_trigger"].median(),
            "productive_spans_cut_mean": g["remaining_productive_spans_after_trigger"].mean(),
            "UC_spans_saved_mean": g["remaining_UC_spans_after_trigger"].mean(),
            "would_cut_productive_suffix_rate": g["would_cut_productive_suffix"].mean(),
            "would_save_UC_suffix_rate": g["would_save_UC_suffix"].mean(),
            "stop_harm_rate_available": False,
            "stop_harm_rate": pd.NA,
            "success_destroying_trigger_count": pd.NA,
        }
        for col in ["F1", "UC_recall", "PIIR", "HN_FPR", "burden"]:
            row[col] = g[col].dropna().iloc[0] if col in g.columns and g[col].notna().any() else pd.NA
        for loc in LOCATION_ORDER:
            row[f"first_triggers_in_{loc}"] = int((g["trigger_location_type"] == loc).sum())
            row[f"first_trigger_pct_{loc}"] = float((g["trigger_location_type"] == loc).mean()) if len(g) else math.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {d: i for i, d in enumerate(DATASET_ORDER)}
    out = out.assign(_order=out["dataset"].map(order).fillna(999)).sort_values(["_order", "dataset", "detector", "detector_config"]).drop(columns="_order")
    return out


def summarize_by_location(triggers: pd.DataFrame) -> pd.DataFrame:
    use = triggers[triggers["is_first_trigger_for_config_trajectory"]].copy()
    rows: list[dict[str, Any]] = []
    for keys, g in use.groupby(["dataset", "trigger_location_type"], dropna=False):
        dataset, loc = keys
        rows.append(
            {
                "dataset": dataset,
                "trigger_location_type": loc,
                "first_trigger_count": len(g),
                "productive_suffix_cut_mean": g["remaining_productive_steps_after_trigger"].mean(),
                "productive_suffix_cut_median": g["remaining_productive_steps_after_trigger"].median(),
                "PI_suffix_cut_mean": g["remaining_PI_steps_after_trigger"].mean(),
                "HN_suffix_cut_mean": g["remaining_HN_steps_after_trigger"].mean(),
                "UC_suffix_saved_mean": g["remaining_UC_steps_after_trigger"].mean(),
                "UC_suffix_saved_median": g["remaining_UC_steps_after_trigger"].median(),
                "would_cut_productive_suffix_rate": g["would_cut_productive_suffix"].mean(),
                "would_save_UC_suffix_rate": g["would_save_UC_suffix"].mean(),
                "stop_harm_rate_available": False,
                "stop_harm_rate": pd.NA,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    dataset_order = {d: i for i, d in enumerate(DATASET_ORDER)}
    loc_order = {d: i for i, d in enumerate(LOCATION_ORDER)}
    out = out.assign(_d=out["dataset"].map(dataset_order).fillna(999), _l=out["trigger_location_type"].map(loc_order).fillna(999)).sort_values(["_d", "_l"]).drop(columns=["_d", "_l"])
    return out


def near_f1_stop_summary(near_pairs: pd.DataFrame, detector_summary: pd.DataFrame) -> pd.DataFrame:
    if near_pairs.empty or detector_summary.empty:
        return pd.DataFrame()
    key_cols = ["dataset", "detector", "detector_config"]
    lookup = detector_summary.set_index(key_cols)
    rows: list[dict[str, Any]] = []
    for row in near_pairs.itertuples(index=False):
        dataset = str(getattr(row, "dataset"))
        a_key = (dataset, str(getattr(row, "detector_a")), str(getattr(row, "config_a")))
        b_key = (dataset, str(getattr(row, "detector_b")), str(getattr(row, "config_b")))
        if a_key not in lookup.index or b_key not in lookup.index:
            continue
        a = lookup.loc[a_key]
        b = lookup.loc[b_key]
        rows.append(
            {
                "dataset": dataset,
                "detector_a": a_key[1],
                "config_a": a_key[2],
                "detector_b": b_key[1],
                "config_b": b_key[2],
                "F1_a": getattr(row, "F1_a"),
                "F1_b": getattr(row, "F1_b"),
                "PIIR_a": getattr(row, "PIIR_a"),
                "PIIR_b": getattr(row, "PIIR_b"),
                "HN_FPR_a": getattr(row, "HN_FPR_a"),
                "HN_FPR_b": getattr(row, "HN_FPR_b"),
                "abs_F1_diff": getattr(row, "abs_F1_diff"),
                "abs_PIIR_diff": getattr(row, "abs_PIIR_diff"),
                "abs_HN_FPR_diff": getattr(row, "abs_HN_FPR_diff"),
                "productive_suffix_cut_mean_a": a["productive_suffix_cut_mean"],
                "productive_suffix_cut_mean_b": b["productive_suffix_cut_mean"],
                "UC_suffix_saved_mean_a": a["UC_suffix_saved_mean"],
                "UC_suffix_saved_mean_b": b["UC_suffix_saved_mean"],
                "abs_productive_suffix_cut_mean_diff": abs(float(a["productive_suffix_cut_mean"]) - float(b["productive_suffix_cut_mean"])),
                "abs_UC_suffix_saved_mean_diff": abs(float(a["UC_suffix_saved_mean"]) - float(b["UC_suffix_saved_mean"])),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["dataset", "abs_productive_suffix_cut_mean_diff"], ascending=[True, False])


def save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(output_dir / f"{name}.{ext}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(summary: pd.DataFrame, near_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.4))
    for dataset, group in summary.groupby("dataset"):
        ax.scatter(
            group["UC_suffix_saved_mean"],
            group["productive_suffix_cut_mean"],
            s=34,
            alpha=0.45,
            color=DATASET_COLORS.get(dataset, "#666666"),
            label=display_dataset(dataset),
            edgecolor="none",
        )
    final_pair = near_summary[(near_summary["dataset"] == "Final500") & (near_summary["detector_a"] == "max_step_guard") & (near_summary["config_a"] == "threshold=50")]
    if not final_pair.empty:
        r = final_pair.iloc[0]
        points = [
            ("A", r["UC_suffix_saved_mean_a"], r["productive_suffix_cut_mean_a"]),
            ("B", r["UC_suffix_saved_mean_b"], r["productive_suffix_cut_mean_b"]),
        ]
        ax.plot([p[1] for p in points], [p[2] for p in points], color="#1f4e99", lw=1.5, zorder=4)
        for label, x, y in points:
            ax.scatter([x], [y], s=135, color=DATASET_COLORS["Final500"], edgecolor="#1f4e99", linewidth=1.8, zorder=5)
            dx, dy = (1.7, -3.2) if label == "A" else (1.3, 3.2)
            ax.annotate(
                label,
                xy=(x, y),
                xycoords="data",
                xytext=(x + dx, y + dy),
                textcoords="data",
                ha="center",
                va="center",
                fontsize=8.8,
                fontweight="bold",
                color="#1f4e99",
                arrowprops=dict(arrowstyle="-", color="#1f4e99", lw=0.7, shrinkA=1, shrinkB=5),
                zorder=6,
            )
    ax.margins(x=0.05, y=0.07)
    ax.set_xlabel("Mean remaining UC steps saved by hard stop")
    ax.set_ylabel("Mean remaining productive steps cut")
    ax.set_title("Offline hard-stop tradeoff by detector configuration", fontsize=11, weight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    save_fig(fig, output_dir, "fig_stop_counterfactual_tradeoff")


def plot_harm_by_location(location_summary: pd.DataFrame, output_dir: Path) -> None:
    use = location_summary.groupby("trigger_location_type", as_index=False).agg(
        productive_suffix_cut_mean=("productive_suffix_cut_mean", "mean"),
        UC_suffix_saved_mean=("UC_suffix_saved_mean", "mean"),
        first_trigger_count=("first_trigger_count", "sum"),
    )
    use["_order"] = use["trigger_location_type"].map({v: i for i, v in enumerate(LOCATION_ORDER)}).fillna(999)
    use = use.sort_values("_order")
    x = np.arange(len(use))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.1, 3.9))
    ax.bar(x - width / 2, use["productive_suffix_cut_mean"], width, label="productive suffix cut", color="#0072B2", alpha=0.82)
    ax.bar(x + width / 2, use["UC_suffix_saved_mean"], width, label="UC suffix saved", color="#E69F00", alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(use["trigger_location_type"])
    ax.set_ylabel("Mean remaining steps after first trigger")
    ax.set_title("Hard-stop counterfactual by trigger location", fontsize=11, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, output_dir, "fig_stop_harm_by_location")


def plot_near_f1(near_summary: pd.DataFrame, output_dir: Path) -> None:
    pair = near_summary[(near_summary["dataset"] == "Final500") & (near_summary["detector_a"] == "max_step_guard") & (near_summary["config_a"] == "threshold=50")]
    if pair.empty:
        pair = near_summary.head(1)
    if pair.empty:
        return
    r = pair.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4))
    labels = ["A\nmax-step\nT=50", "B\nfile-revisit\nw=5,r=3"]
    metrics = ["F1", "PIIR", "HN-FPR"]
    vals_a = [r["F1_a"], r["PIIR_a"], r["HN_FPR_a"]]
    vals_b = [r["F1_b"], r["PIIR_b"], r["HN_FPR_b"]]
    xx = np.arange(len(metrics))
    axes[0].bar(xx - 0.18, vals_a, 0.36, label="A", color="#9ecae1", edgecolor="#1f4e99")
    axes[0].bar(xx + 0.18, vals_b, 0.36, label="B", color="#74c69d", edgecolor="#1f4e99")
    axes[0].set_xticks(xx)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylim(0, max(1.0, max(vals_a + vals_b) * 1.12))
    axes[0].set_title("Aggregate and proxy risk")
    axes[0].legend(frameon=False, fontsize=8)
    suffix_metrics = ["productive\ncut", "UC\nsaved"]
    suffix_a = [r["productive_suffix_cut_mean_a"], r["UC_suffix_saved_mean_a"]]
    suffix_b = [r["productive_suffix_cut_mean_b"], r["UC_suffix_saved_mean_b"]]
    xx2 = np.arange(len(suffix_metrics))
    axes[1].bar(xx2 - 0.18, suffix_a, 0.36, label=labels[0], color="#9ecae1", edgecolor="#1f4e99")
    axes[1].bar(xx2 + 0.18, suffix_b, 0.36, label=labels[1], color="#74c69d", edgecolor="#1f4e99")
    axes[1].set_xticks(xx2)
    axes[1].set_xticklabels(suffix_metrics)
    axes[1].set_ylabel("Mean remaining steps")
    axes[1].set_title("Hard-stop suffix effect")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Enriched500 near-F1 pair: offline stop-counterfactual", fontsize=11, weight="bold")
    save_fig(fig, output_dir, "fig_near_f1_stop_counterfactual")


def fmt(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    use = df[cols].head(max_rows).copy()
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in use.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(fmt(v) if isinstance(v, (float, int, np.floating, np.integer)) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, summary: pd.DataFrame, loc_summary: pd.DataFrame, near_summary: pd.DataFrame, feasibility_path: Path) -> None:
    focus = summary[
        ((summary["dataset"] == "Final500") & (summary["detector"].isin(["max_step_guard", "file_revisit_guard"])) & (summary["detector_config"].isin(["threshold=50", "window=5;repeats=3"])))
        | ((summary["dataset"] == "StressFresh100") & (summary["detector"] == "file_revisit_guard") & (summary["detector_config"] == "window=10;repeats=3"))
        | (summary["detector"] == "repeat_without_progress_guard")
    ]
    lines = [
        "# Offline Stop-Counterfactual Intervention Validation",
        "",
        "This analysis asks: if a detector/guard trigger were converted into a hard stop at step `t`, what observed suffix would that stop cut off? It is an offline hard-stop counterfactual over observed traces, not a full causal rerun and not evidence about warn/replan/handoff policies.",
        "",
        "## Feasibility",
        "",
        f"Feasibility audit: `{feasibility_path}`.",
        "Final task success and structured post-trigger successful validation were not available in the primary normalized/rich inputs. Therefore Stop-Harm Rate and success-destroying trigger counts are marked unavailable rather than fabricated.",
        "",
        "## Metric Definitions",
        "",
        "- Productive Suffix Loss (PSL): remaining productive steps after a first trigger, where productive means PI or HN.",
        "- UC Waste Saved (UWS): remaining UC steps after a first trigger.",
        "- Stop-Harm Rate (SHR): unavailable in this artifact because observed final success/post-trigger validation fields are not structured in the primary inputs.",
        "- Detector summaries use the first trigger per dataset/trajectory/detector/config, matching a hard-stop policy.",
        "",
        "## Focused Detector Configurations",
        "",
        markdown_table(with_display_dataset(focus), ["dataset", "detector", "detector_config", "F1", "PIIR", "HN_FPR", "productive_suffix_cut_mean", "UC_suffix_saved_mean", "would_cut_productive_suffix_rate", "would_save_UC_suffix_rate"], max_rows=40),
        "",
        "## By Trigger Location",
        "",
        markdown_table(with_display_dataset(loc_summary), ["dataset", "trigger_location_type", "first_trigger_count", "productive_suffix_cut_mean", "UC_suffix_saved_mean", "would_cut_productive_suffix_rate", "would_save_UC_suffix_rate"], max_rows=40),
        "",
        "## Near-F1 Stop-Counterfactual Pairs",
        "",
        markdown_table(with_display_dataset(near_summary), ["dataset", "detector_a", "config_a", "detector_b", "config_b", "abs_F1_diff", "abs_PIIR_diff", "abs_HN_FPR_diff", "productive_suffix_cut_mean_a", "productive_suffix_cut_mean_b", "UC_suffix_saved_mean_a", "UC_suffix_saved_mean_b"], max_rows=20),
        "",
        "## Interpretation",
        "",
        "PIIR and HN-FPR remain trigger-site risk proxies. This offline analysis strengthens the intervention interpretation by showing whether high-risk triggers also cut off observed productive suffixes under a hard-stop policy. It still does not estimate stochastic live rerun outcomes or softer interventions such as warn, summarize, replan, escalate, or handoff.",
    ]
    (out_dir / "stop_counterfactual_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_master_report(out_dir: Path, summary: pd.DataFrame, loc_summary: pd.DataFrame, near_summary: pd.DataFrame) -> None:
    live_note = out_dir / "prefix_branch_feasibility_only.md"
    lines = [
        "# Intervention Validation Master Report",
        "",
        "## 1. Feasibility Audit Result",
        "",
        "The audit recommends offline stop-counterfactual validation only. Live prefix-branch reruns were not executed because the repository does not expose the complete replay stack needed to restore prefix state, rerun an agent, and evaluate task success under controlled arms.",
        "",
        "## 2. Offline Stop-Counterfactual Metrics",
        "",
        f"Detector/config rows summarized: {len(summary)}.",
        f"Trigger-location rows summarized: {len(loc_summary)}.",
        "Stop-Harm Rate is unavailable because final success and post-trigger successful validation are not structured in the primary inputs.",
        "",
        "## 3. Live Prefix-Branch Rerun Feasibility",
        "",
        f"Feasibility-only note: `{live_note.name}`.",
        "",
        "## 4. Figures Generated",
        "",
        "- `fig_stop_counterfactual_tradeoff.pdf/png/svg`",
        "- `fig_stop_harm_by_location.pdf/png/svg`",
        "- `fig_near_f1_stop_counterfactual.pdf/png/svg`",
        "",
        "## 5. Tables Generated",
        "",
        "- `stop_counterfactual_triggers.csv`",
        "- `stop_counterfactual_summary_by_detector.csv`",
        "- `stop_counterfactual_summary_by_location.csv`",
        "- `stop_counterfactual_near_f1_pairs.csv`",
        "",
        "## 6. Remaining Limitations",
        "",
        "The offline analysis evaluates hard-stop consequences on observed suffixes. It cannot estimate stochastic rerun behavior, warn/replan/handoff effects, or final task success deltas without a replayable live experiment.",
    ]
    (out_dir / "intervention_validation_master_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline stop-counterfactual intervention validation tables and figures.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_intervention_validation"))
    parser.add_argument("--enriched-steps", type=Path, default=Path("outputs/paper1_raw_enrichment/enriched_steps.csv"))
    parser.add_argument("--spans", type=Path, default=Path("outputs/paper1_audit/normalized_spans.csv"))
    parser.add_argument("--detector-metrics", type=Path, default=Path("outputs/paper1_intervention_risk_rich_detectors/detector_metrics_by_dataset.csv"))
    parser.add_argument("--near-pairs", type=Path, default=Path("outputs/paper1_intervention_risk_rich_detectors/near_f1_risk_pairs.csv"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched_path = args.enriched_steps if args.enriched_steps.is_absolute() else repo_root / args.enriched_steps
    spans_path = args.spans if args.spans.is_absolute() else repo_root / args.spans
    metrics_path = args.detector_metrics if args.detector_metrics.is_absolute() else repo_root / args.detector_metrics
    near_path = args.near_pairs if args.near_pairs.is_absolute() else repo_root / args.near_pairs

    enriched = pd.read_csv(enriched_path, dtype=str, keep_default_na=False)
    spans = pd.read_csv(spans_path)
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    near_pairs = pd.read_csv(near_path) if near_path.exists() else pd.DataFrame()

    steps = prepare_steps(enriched, spans)
    triggers = build_trigger_table(steps, spans, metrics)
    if triggers.empty:
        raise RuntimeError("No detector triggers could be reconstructed from existing detector definitions.")

    trigger_cols = [
        "trigger_id", "dataset", "trajectory_id", "step_index", "detector", "detector_family", "detector_config", "trigger_location_type",
        "trajectory_final_success", "current_prefix_success", "post_trigger_success", "post_trigger_validation_event",
        "remaining_steps_after_trigger", "remaining_productive_steps_after_trigger", "remaining_PI_steps_after_trigger", "remaining_HN_steps_after_trigger", "remaining_UC_steps_after_trigger",
        "remaining_productive_spans_after_trigger", "remaining_UC_spans_after_trigger", "would_stop_before_observed_success", "would_cut_productive_suffix", "would_save_UC_suffix",
        "is_first_trigger_for_config_trajectory", "evidence_available_flag", "success_evidence_available_flag", "F1", "UC_recall", "PIIR", "HN_FPR", "burden",
    ]
    triggers[trigger_cols].to_csv(out_dir / "stop_counterfactual_triggers.csv", index=False, encoding="utf-8-sig")

    summary = summarize_by_detector(triggers)
    loc_summary = summarize_by_location(triggers)
    near_summary = near_f1_stop_summary(near_pairs, summary)
    summary.to_csv(out_dir / "stop_counterfactual_summary_by_detector.csv", index=False, encoding="utf-8-sig")
    loc_summary.to_csv(out_dir / "stop_counterfactual_summary_by_location.csv", index=False, encoding="utf-8-sig")
    near_summary.to_csv(out_dir / "stop_counterfactual_near_f1_pairs.csv", index=False, encoding="utf-8-sig")

    plot_tradeoff(summary, near_summary, out_dir)
    plot_harm_by_location(loc_summary, out_dir)
    plot_near_f1(near_summary, out_dir)

    feasibility_path = out_dir / "feasibility_report.md"
    if not (out_dir / "prefix_branch_feasibility_only.md").exists():
        (out_dir / "prefix_branch_feasibility_only.md").write_text(
            "# Prefix-Branch Feasibility Only\n\nLive prefix-branch reruns were not executed. See `feasibility_report.md`.\n",
            encoding="utf-8",
        )
    write_report(out_dir, summary, loc_summary, near_summary, feasibility_path)
    write_master_report(out_dir, summary, loc_summary, near_summary)
    print(f"Wrote stop-counterfactual outputs to {out_dir}")
    print(f"Trigger rows: {len(triggers)}")
    print(f"First-trigger detector summaries: {len(summary)}")


if __name__ == "__main__":
    main()
