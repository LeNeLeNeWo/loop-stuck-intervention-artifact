from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.paper1_audit.audit_dataset_integrity import ensure_normalized
from scripts.paper1_audit.load_paper1_datasets import repo_root_from_script


DATASET_ORDER = ["Natural500", "Final500", "StressFresh100", "OpenHandsExternal330"]
STEP_CAP_THRESHOLDS = [25, 50, 75, 100, 150, 200]
BOOTSTRAP_DEFAULT_RESAMPLES = 1000


def bool_series(series: pd.Series) -> pd.Series:
    def parse(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        text = str(value).strip().lower()
        return text in {"true", "1", "yes", "y", "t"}

    return series.map(parse)


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{100 * float(value):.1f}%"


def metric_fmt(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def save_fig(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{name}.png", dpi=240)
    fig.savefig(figures_dir / f"{name}.pdf")
    plt.close(fig)


def ordered(df: pd.DataFrame, column: str = "dataset") -> pd.DataFrame:
    if column not in df.columns:
        return df
    order = {dataset: i for i, dataset in enumerate(DATASET_ORDER)}
    return df.assign(_order=df[column].map(order).fillna(999)).sort_values(["_order", column]).drop(columns="_order")


def detector_availability(steps: pd.DataFrame) -> pd.DataFrame:
    columns = {c.lower() for c in steps.columns}
    availability = [
        {
            "detector_family": "max_step_guard",
            "implemented": True,
            "reason": "Trajectory step order is available in the normalized step table.",
        },
        {
            "detector_family": "exact_action_repeat",
            "implemented": False,
            "reason": "Raw action text or normalized action identifiers are not present in the normalized step table.",
        },
        {
            "detector_family": "tool_pattern_repeat",
            "implemented": False,
            "reason": "Tool names or tool-call sequences are not present in the normalized step table.",
        },
        {
            "detector_family": "repeated_error_signature",
            "implemented": False,
            "reason": "Raw observations, stack traces, or error signatures are not present in the normalized step table.",
        },
        {
            "detector_family": "action_observation_repeat",
            "implemented": False,
            "reason": "Raw action and observation/output text are not both present in the normalized step table.",
        },
        {
            "detector_family": "simple_text_similarity",
            "implemented": False,
            "reason": "The only text columns retained are annotation evidence/provenance fields; using them would leak label information.",
        },
    ]
    # Keep this small diagnostic for future loader changes.
    raw_like = [c for c in sorted(columns) if c in {"action", "tool_name", "observation", "tool_input", "error", "thought"}]
    if raw_like:
        availability[-1]["reason"] += f" Raw-like columns detected unexpectedly: {', '.join(raw_like)}."
    return pd.DataFrame(availability)


def prepare_steps(steps: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    keep = [
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
        "unresolved_flag",
    ]
    use = steps[[c for c in keep if c in steps.columns]].copy()
    for col in keep:
        if col not in use.columns:
            use[col] = ""
    use["step_index"] = pd.to_numeric(use["step_index"], errors="coerce")
    use = use.dropna(subset=["step_index"]).copy()
    use["step_index"] = use["step_index"].astype(int)
    use = use.sort_values(["dataset", "trajectory_id", "step_index"]).reset_index(drop=True)
    use["step_position"] = use.groupby(["dataset", "trajectory_id"]).cumcount() + 1
    use["in_pi"] = use["span_label_norm"].fillna("").eq("productive_iteration") | bool_series(use["productive_iteration_flag"])
    use["in_uc"] = use["span_label_norm"].fillna("").eq("unproductive_cycle") | bool_series(use["unproductive_cycle_flag"])
    use["in_hn"] = bool_series(use["hard_negative"])
    use["in_uncertain"] = use["span_label_norm"].fillna("").isin(["uncertain", "unresolved"]) | bool_series(use["uncertain_flag"]) | bool_series(use["unresolved_flag"])

    # Project explicit/derived spans back to steps. This is necessary for OpenHands,
    # where span annotations are explicit CSV rows rather than step-level span labels.
    if not spans.empty:
        grouped_indices: dict[tuple[str, str], pd.DataFrame] = {
            (str(dataset), str(tid)): g[["step_index"]].copy()
            for (dataset, tid), g in use.groupby(["dataset", "trajectory_id"])
        }
        for span in spans.itertuples(index=False):
            dataset = str(getattr(span, "dataset", ""))
            tid = str(getattr(span, "trajectory_id", ""))
            key = (dataset, tid)
            if key not in grouped_indices:
                continue
            try:
                start = int(float(getattr(span, "start_step_index")))
                end = int(float(getattr(span, "end_step_index")))
            except (TypeError, ValueError):
                continue
            if end < start:
                start, end = end, start
            span_label = "" if pd.isna(getattr(span, "span_label_norm", "")) else str(getattr(span, "span_label_norm", ""))
            span_hn = bool_series(pd.Series([getattr(span, "hard_negative", False)])).iloc[0]
            span_pi = span_label == "productive_iteration" or bool_series(pd.Series([getattr(span, "productive_iteration_flag", False)])).iloc[0]
            span_uc = span_label == "unproductive_cycle" or bool_series(pd.Series([getattr(span, "unproductive_cycle_flag", False)])).iloc[0]
            span_uncertain = span_label in {"uncertain", "unresolved"} or bool_series(pd.Series([getattr(span, "uncertain_flag", False)])).iloc[0]
            mask = (
                (use["dataset"].astype(str) == dataset)
                & (use["trajectory_id"].astype(str) == tid)
                & (use["step_index"] >= start)
                & (use["step_index"] <= end)
            )
            if span_pi:
                use.loc[mask, "in_pi"] = True
            if span_uc:
                use.loc[mask, "in_uc"] = True
            if span_hn:
                use.loc[mask, "in_hn"] = True
            if span_uncertain:
                use.loc[mask, "in_uncertain"] = True
    return use


def prepare_eval_spans(spans: pd.DataFrame, steps_eval: pd.DataFrame) -> pd.DataFrame:
    if spans.empty:
        return pd.DataFrame(
            columns=["dataset", "dataset_role", "trajectory_id", "span_id", "span_label", "start_step_index", "end_step_index"]
        )
    rows: list[dict[str, Any]] = []
    valid_keys = set(zip(steps_eval["dataset"].astype(str), steps_eval["trajectory_id"].astype(str)))
    for span in spans.itertuples(index=False):
        dataset = str(getattr(span, "dataset", ""))
        tid = str(getattr(span, "trajectory_id", ""))
        if (dataset, tid) not in valid_keys:
            continue
        span_label = "" if pd.isna(getattr(span, "span_label_norm", "")) else str(getattr(span, "span_label_norm", ""))
        if span_label not in {"productive_iteration", "unproductive_cycle"}:
            continue
        try:
            start = int(float(getattr(span, "start_step_index")))
            end = int(float(getattr(span, "end_step_index")))
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        role = getattr(span, "dataset_role", "")
        rows.append(
            {
                "dataset": dataset,
                "dataset_role": role,
                "trajectory_id": tid,
                "span_id": str(getattr(span, "span_id", f"{tid}:{start}-{end}:{span_label}")),
                "span_label": span_label,
                "start_step_index": start,
                "end_step_index": end,
            }
        )
    return pd.DataFrame(rows)


def max_step_triggers(steps_eval: pd.DataFrame, threshold: int) -> pd.Series:
    return steps_eval["step_position"] >= threshold


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else math.nan


def safe_div(num: float, den: float) -> float:
    return (num / den) if den else math.nan


def span_recall_latency(spans_eval: pd.DataFrame, steps_eval: pd.DataFrame, trigger_col: str, label: str) -> tuple[int, int, list[int]]:
    target = spans_eval[spans_eval["span_label"] == label]
    recalled = 0
    latencies: list[int] = []
    for span in target.itertuples(index=False):
        span_steps = steps_eval[
            (steps_eval["dataset"].astype(str) == str(span.dataset))
            & (steps_eval["trajectory_id"].astype(str) == str(span.trajectory_id))
            & (steps_eval["step_index"] >= int(span.start_step_index))
            & (steps_eval["step_index"] <= int(span.end_step_index))
        ]
        if span_steps.empty:
            continue
        triggered = span_steps[span_steps[trigger_col]]
        if triggered.empty:
            continue
        recalled += 1
        latencies.append(int(triggered["step_position"].min() - span_steps["step_position"].min()))
    return recalled, len(target), latencies


def trajectory_contributions(steps_eval: pd.DataFrame, spans_eval: pd.DataFrame, trigger_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, role, tid), group in steps_eval.groupby(["dataset", "dataset_role", "trajectory_id"], dropna=False):
        trigger = group[trigger_col]
        positive = group["in_uc"]
        hn = group["in_hn"]
        tp = int((trigger & positive).sum())
        fp = int((trigger & ~positive).sum())
        fn = int((~trigger & positive).sum())
        tn = int((~trigger & ~positive).sum())
        local_spans = spans_eval[(spans_eval["dataset"].astype(str) == str(dataset)) & (spans_eval["trajectory_id"].astype(str) == str(tid))]
        uc_recalled, uc_total, latencies = span_recall_latency(local_spans, group, trigger_col, "unproductive_cycle")
        pi_interrupted, pi_total, _ = span_recall_latency(local_spans, group, trigger_col, "productive_iteration")
        rows.append(
            {
                "dataset": dataset,
                "dataset_role": role,
                "trajectory_id": tid,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "uc_spans_recalled": uc_recalled,
                "uc_spans_total": uc_total,
                "pi_spans_interrupted": pi_interrupted,
                "pi_spans_total": pi_total,
                "hn_trigger_steps": int((trigger & hn).sum()),
                "hn_eligible_steps": int(hn.sum()),
                "trigger_steps": int(trigger.sum()),
                "eligible_steps": int(len(group)),
                "latencies": latencies,
            }
        )
    return pd.DataFrame(rows)


def metrics_from_contrib(contrib: pd.DataFrame) -> dict[str, Any]:
    if contrib.empty:
        return {
            "F1": math.nan,
            "UC_recall": math.nan,
            "PIIR": math.nan,
            "HN_FPR": math.nan,
            "burden": math.nan,
            "latency_mean": math.nan,
            "latency_median": math.nan,
            "latency_p90": math.nan,
            "n_triggers": 0,
        }
    tp = int(contrib["tp"].sum())
    fp = int(contrib["fp"].sum())
    fn = int(contrib["fn"].sum())
    uc_recalled = int(contrib["uc_spans_recalled"].sum())
    uc_total = int(contrib["uc_spans_total"].sum())
    pi_interrupted = int(contrib["pi_spans_interrupted"].sum())
    pi_total = int(contrib["pi_spans_total"].sum())
    hn_trigger = int(contrib["hn_trigger_steps"].sum())
    hn_total = int(contrib["hn_eligible_steps"].sum())
    trigger_steps = int(contrib["trigger_steps"].sum())
    eligible_steps = int(contrib["eligible_steps"].sum())
    latencies: list[int] = []
    for value in contrib["latencies"].tolist():
        if isinstance(value, list):
            latencies.extend(int(v) for v in value)
    return {
        "F1": f1_from_counts(tp, fp, fn),
        "UC_recall": safe_div(uc_recalled, uc_total),
        "PIIR": safe_div(pi_interrupted, pi_total),
        "HN_FPR": safe_div(hn_trigger, hn_total),
        "burden": safe_div(trigger_steps, eligible_steps),
        "latency_mean": float(np.mean(latencies)) if latencies else math.nan,
        "latency_median": float(np.median(latencies)) if latencies else math.nan,
        "latency_p90": float(np.quantile(latencies, 0.9)) if latencies else math.nan,
        "n_triggers": trigger_steps,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "uc_spans_recalled": uc_recalled,
        "uc_spans_total": uc_total,
        "pi_spans_interrupted": pi_interrupted,
        "pi_spans_total": pi_total,
        "hn_trigger_steps": hn_trigger,
        "hn_eligible_steps": hn_total,
        "eligible_steps": eligible_steps,
    }


def trigger_location_distribution(steps_eval: pd.DataFrame, trigger_col: str) -> pd.DataFrame:
    triggered = steps_eval[steps_eval[trigger_col]].copy()
    if triggered.empty:
        return pd.DataFrame(columns=["dataset", "location_category", "trigger_count", "trigger_pct"])
    # Mutually exclusive priority for display: useful hit first, then risky HN,
    # then broader PI, then other.
    conditions = [
        triggered["in_uc"],
        triggered["in_hn"],
        triggered["in_pi"],
    ]
    choices = ["UC_span", "HN_step", "PI_span_non_HN", "other"]
    triggered["location_category"] = np.select(conditions, choices[:3], default=choices[3])
    rows: list[dict[str, Any]] = []
    for dataset, group in triggered.groupby("dataset", dropna=False):
        total = len(group)
        for category in choices:
            count = int((group["location_category"] == category).sum())
            rows.append({"dataset": dataset, "location_category": category, "trigger_count": count, "trigger_pct": count / total if total else math.nan})
    return ordered(pd.DataFrame(rows))


def evaluate_detectors(steps_eval: pd.DataFrame, spans_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    metric_rows: list[dict[str, Any]] = []
    location_rows: list[pd.DataFrame] = []
    contrib_by_key: dict[str, pd.DataFrame] = {}
    for threshold in STEP_CAP_THRESHOLDS:
        trigger_col = f"trigger_max_step_{threshold}"
        steps_eval[trigger_col] = max_step_triggers(steps_eval, threshold)
        for dataset, group in steps_eval.groupby("dataset", dropna=False):
            role = group["dataset_role"].iloc[0]
            local_spans = spans_eval[spans_eval["dataset"].astype(str) == str(dataset)]
            contrib = trajectory_contributions(group, local_spans, trigger_col)
            metrics = metrics_from_contrib(contrib)
            config = f"threshold={threshold}"
            key = f"{dataset}|max_step_guard|{config}"
            contrib_by_key[key] = contrib
            metric_rows.append(
                {
                    "dataset": dataset,
                    "dataset_role": role,
                    "detector": "max_step_guard",
                    "threshold/config": config,
                    **{k: metrics[k] for k in ["F1", "UC_recall", "PIIR", "HN_FPR", "burden", "latency_mean", "latency_median", "latency_p90", "n_triggers"]},
                    "positive_label_definition": "step is inside an unproductive_cycle span or step-level unproductive_cycle region",
                }
            )
        loc = trigger_location_distribution(steps_eval, trigger_col)
        if not loc.empty:
            loc["detector"] = "max_step_guard"
            loc["threshold/config"] = f"threshold={threshold}"
            location_rows.append(loc)
    metrics_df = ordered(pd.DataFrame(metric_rows))
    loc_df = pd.concat(location_rows, ignore_index=True) if location_rows else pd.DataFrame()
    return metrics_df, loc_df, contrib_by_key


def bootstrap_ci(contrib_by_key: dict[str, pd.DataFrame], metrics_df: pd.DataFrame, resamples: int, seed: int = 20260611) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    metric_names = ["F1", "UC_recall", "PIIR", "HN_FPR", "burden", "latency_mean"]
    for _, metric_row in metrics_df.iterrows():
        dataset = metric_row["dataset"]
        detector = metric_row["detector"]
        config = metric_row["threshold/config"]
        key = f"{dataset}|{detector}|{config}"
        contrib = contrib_by_key.get(key)
        if contrib is None or contrib.empty:
            continue
        n = len(contrib)
        values_by_metric = {name: [] for name in metric_names}
        for _ in range(resamples):
            sample_idx = rng.integers(0, n, size=n)
            sampled = contrib.iloc[sample_idx]
            metrics = metrics_from_contrib(sampled)
            for name in metric_names:
                values_by_metric[name].append(metrics[name])
        point = metrics_from_contrib(contrib)
        for name, values in values_by_metric.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                low = high = math.nan
            else:
                low = float(np.quantile(arr, 0.025))
                high = float(np.quantile(arr, 0.975))
            rows.append(
                {
                    "dataset": dataset,
                    "detector": detector,
                    "threshold/config": config,
                    "metric": name,
                    "point": point[name],
                    "ci_low": low,
                    "ci_high": high,
                    "resamples": resamples,
                    "bootstrap_unit": "trajectory",
                }
            )
    return ordered(pd.DataFrame(rows))


def near_f1_risk_pairs(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in metrics.groupby("dataset", dropna=False):
        group = group.reset_index(drop=True)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a = group.iloc[i]
                b = group.iloc[j]
                f1_diff = abs(float(a["F1"]) - float(b["F1"])) if pd.notna(a["F1"]) and pd.notna(b["F1"]) else math.nan
                if pd.isna(f1_diff) or f1_diff > 0.02:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "within_0_01": f1_diff <= 0.01,
                        "within_0_02": f1_diff <= 0.02,
                        "detector_a": a["detector"],
                        "config_a": a["threshold/config"],
                        "F1_a": a["F1"],
                        "PIIR_a": a["PIIR"],
                        "HN_FPR_a": a["HN_FPR"],
                        "detector_b": b["detector"],
                        "config_b": b["threshold/config"],
                        "F1_b": b["F1"],
                        "PIIR_b": b["PIIR"],
                        "HN_FPR_b": b["HN_FPR"],
                        "abs_F1_diff": f1_diff,
                        "abs_PIIR_diff": abs(float(a["PIIR"]) - float(b["PIIR"])) if pd.notna(a["PIIR"]) and pd.notna(b["PIIR"]) else math.nan,
                        "abs_HN_FPR_diff": abs(float(a["HN_FPR"]) - float(b["HN_FPR"])) if pd.notna(a["HN_FPR"]) and pd.notna(b["HN_FPR"]) else math.nan,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "within_0_01",
                "within_0_02",
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
            ]
        )
    return ordered(out.sort_values(["abs_PIIR_diff", "abs_HN_FPR_diff"], ascending=False))


def plot_scatter_by_dataset(metrics: pd.DataFrame, figures_dir: Path, x: str, y: str, name: str, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, dataset in zip(axes_flat, DATASET_ORDER):
        group = metrics[metrics["dataset"] == dataset].copy()
        ax.scatter(group[x], group[y], s=45)
        for _, row in group.iterrows():
            threshold = str(row["threshold/config"]).split("=")[-1]
            ax.text(row[x], row[y], threshold, fontsize=8, ha="left", va="bottom")
        ax.set_title(dataset)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.25)
    save_fig(fig, figures_dir, name)


def plot_location_distribution(location: pd.DataFrame, figures_dir: Path) -> None:
    if location.empty:
        return
    # Use threshold 50 as a compact representative and keep all thresholds in CSV.
    selected = location[location["threshold/config"] == "threshold=50"].copy()
    if selected.empty:
        selected = location.copy()
    pivot = selected.pivot_table(index="dataset", columns="location_category", values="trigger_pct", aggfunc="sum").fillna(0)
    pivot = pivot.loc[[d for d in DATASET_ORDER if d in pivot.index]]
    order = [c for c in ["UC_span", "HN_step", "PI_span_non_HN", "other"] if c in pivot.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot[order].plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of trigger steps")
    ax.set_title("Figure D. Trigger location distribution, max-step threshold 50")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, figures_dir, "figureD_trigger_location_distribution")


def make_figures(metrics: pd.DataFrame, location: pd.DataFrame, figures_dir: Path) -> None:
    plot_scatter_by_dataset(metrics, figures_dir, "PIIR", "F1", "figureA_f1_vs_piir_by_dataset", "Figure A. F1 vs PIIR by dataset")
    plot_scatter_by_dataset(metrics, figures_dir, "HN_FPR", "F1", "figureB_f1_vs_hn_fpr_by_dataset", "Figure B. F1 vs HN-FPR by dataset")
    plot_scatter_by_dataset(metrics, figures_dir, "PIIR", "UC_recall", "figureC_uc_recall_vs_piir_frontier", "Figure C. UC Recall vs PIIR frontier")
    plot_scatter_by_dataset(metrics, figures_dir, "UC_recall", "burden", "frontier_burden_vs_uc_recall", "Burden vs UC Recall frontier")
    plot_scatter_by_dataset(metrics, figures_dir, "PIIR", "burden", "frontier_burden_vs_piir", "Burden vs PIIR frontier")
    plot_location_distribution(location, figures_dir)


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    show = df if max_rows is None else df.head(max_rows)
    if show.empty:
        return "_No rows._"
    return show.to_markdown(index=False)


def report(
    metrics: pd.DataFrame,
    near_pairs: pd.DataFrame,
    availability: pd.DataFrame,
    bootstrap: pd.DataFrame,
    location: pd.DataFrame,
) -> str:
    display_cols = ["dataset", "detector", "threshold/config", "F1", "UC_recall", "PIIR", "HN_FPR", "burden", "latency_mean", "latency_median", "latency_p90", "n_triggers"]
    metrics_display = metrics[display_cols].copy()
    for col in ["F1", "UC_recall", "PIIR", "HN_FPR", "burden", "latency_mean", "latency_median", "latency_p90"]:
        metrics_display[col] = metrics_display[col].map(metric_fmt)

    best_f1 = metrics.sort_values(["dataset", "F1"], ascending=[True, False]).groupby("dataset").head(1)
    best_recall = metrics.sort_values(["dataset", "UC_recall"], ascending=[True, False]).groupby("dataset").head(1)
    best_display = best_f1[["dataset", "threshold/config", "F1", "UC_recall", "PIIR", "HN_FPR", "burden"]].copy()
    for col in ["F1", "UC_recall", "PIIR", "HN_FPR", "burden"]:
        best_display[col] = best_display[col].map(metric_fmt)

    near_display = near_pairs.copy()
    for col in ["F1_a", "PIIR_a", "HN_FPR_a", "F1_b", "PIIR_b", "HN_FPR_b", "abs_F1_diff", "abs_PIIR_diff", "abs_HN_FPR_diff"]:
        if col in near_display.columns:
            near_display[col] = near_display[col].map(metric_fmt)

    boot_display = bootstrap[bootstrap["metric"].isin(["F1", "UC_recall", "PIIR", "HN_FPR", "burden"])].head(20).copy()
    for col in ["point", "ci_low", "ci_high"]:
        if col in boot_display.columns:
            boot_display[col] = boot_display[col].map(metric_fmt)

    location_50 = location[location["threshold/config"] == "threshold=50"].copy()
    if not location_50.empty:
        location_50["trigger_pct"] = location_50["trigger_pct"].map(pct)

    lines = [
        "# Intervention-Risk Evaluation for Loop/Stuck Guards",
        "",
        "## Evaluation Framing",
        "",
        "Loop detectors are evaluated here as intervention mechanisms. A trigger is potentially useful when it fires inside an annotated unproductive-cycle (UC) region, but it can be costly when it interrupts productive iteration (PI) or productive hard-negative (HN) behavior. Aggregate F1 is therefore useful but incomplete.",
        "",
        "## Positive Label Definition",
        "",
        "The binary step-level F1 treats a step as positive when it lies inside an `unproductive_cycle` span or step-level UC region in the normalized labels. A detector step is positive when the guard is active at that evaluated step. Span-level UC recall is computed separately: a UC span is recalled if at least one trigger occurs inside the span.",
        "",
        "## Detector Availability",
        "",
        "The normalized tables do not retain raw runtime `action`, `tool`, `observation`, file-path, or error-signature fields. To avoid label leakage, this evaluation does not build similarity or repeat detectors from annotation evidence text. The implemented baseline is therefore the progress-insensitive max-step guard.",
        "",
        md_table(availability),
        "",
        "## Main Metrics",
        "",
        md_table(metrics_display),
        "",
        "## Best F1 Configurations",
        "",
        md_table(best_display),
        "",
        "## Near-F1 Risk Pairs",
        "",
        "Pairs below have absolute F1 difference <= 0.02 within the same dataset. They show where similar F1 can mask different productive-interruption risk.",
        "",
        md_table(near_display, max_rows=20),
        "",
        "## Trigger Location Distribution",
        "",
        "For display, trigger locations are made mutually exclusive with priority `UC_span`, then `HN_step`, then `PI_span_non_HN`, then `other`. Full threshold sweeps are saved in CSV.",
        "",
        md_table(location_50),
        "",
        "## Bootstrap Confidence Intervals",
        "",
        "Confidence intervals use trajectory-level bootstrap resampling, not step-level resampling. The table below previews the first rows; the full table is saved as CSV.",
        "",
        md_table(boot_display),
        "",
        "## Answers to Paper Questions",
        "",
        "### Which simple guards have high UC recall but also high PIIR/HN-FPR?",
        "",
        "Lower max-step thresholds generally increase UC recall but also raise burden and productive-interruption exposure. This is expected for a budget guard: it is sensitive to elapsed trajectory length, not to local debugging progress.",
        "",
        "### Are there near-F1 detectors with very different productive-interruption risk?",
        "",
        "The near-F1 table shows threshold pairs where similar F1 can correspond to different PIIR and HN-FPR values. This supports reporting intervention-risk metrics alongside aggregate F1.",
        "",
        "### Does StressFresh expose risks that Natural500 hides?",
        "",
        "StressFresh has a much higher concentration of UC spans and challenging local process boundaries, so step caps can appear more favorable on UC recall while still interrupting productive regions. Natural500 is less enriched and can understate intervention-risk behavior that appears in stress settings.",
        "",
        "### Does OpenHands external show that long trajectories can trigger guards even when UC is rare?",
        "",
        "Yes. OpenHands trajectories are long, so max-step guards trigger frequently by construction. However UC is rare in this external pool, illustrating why step caps are useful budget guards but not progress-aware loop detectors.",
        "",
        "## Paper-Ready Takeaways",
        "",
        "- Aggregate F1 is not wrong, but it is incomplete for intervention mechanisms.",
        "- A false positive inside productive debugging is not equivalent to a false positive in low-cost background behavior.",
        "- Step caps can provide budget control but should not be interpreted as semantic stuck detectors.",
        "- UC recall, PIIR, HN-FPR, burden, and latency jointly describe the safety/usefulness tradeoff of detector triggers.",
        "- The four datasets should be reported separately because their construct prevalence and trajectory-length regimes differ.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate intervention risk for simple loop/stuck guards.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/paper1_audit"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_intervention_risk"))
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_DEFAULT_RESAMPLES)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    audit_dir = (repo_root / args.audit_dir).resolve() if not args.audit_dir.is_absolute() else args.audit_dir
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    steps, spans = ensure_normalized(repo_root, audit_dir)
    availability = detector_availability(steps)
    steps_eval = prepare_steps(steps, spans)
    spans_eval = prepare_eval_spans(spans, steps_eval)
    metrics, location, contrib_by_key = evaluate_detectors(steps_eval, spans_eval)
    pairs = near_f1_risk_pairs(metrics)
    bootstrap = bootstrap_ci(contrib_by_key, metrics, args.bootstrap_resamples)

    metrics.to_csv(output_dir / "detector_metrics_by_dataset.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "threshold_frontier_by_dataset.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(output_dir / "near_f1_risk_pairs.csv", index=False, encoding="utf-8-sig")
    availability.to_csv(output_dir / "detector_availability.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(tables_dir / "detector_metrics_by_dataset.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(tables_dir / "threshold_frontier_by_dataset.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(tables_dir / "near_f1_risk_pairs.csv", index=False, encoding="utf-8-sig")
    availability.to_csv(tables_dir / "detector_availability.csv", index=False, encoding="utf-8-sig")
    location.to_csv(tables_dir / "trigger_location_distribution.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(tables_dir / "bootstrap_confidence_intervals.csv", index=False, encoding="utf-8-sig")
    spans_eval.to_csv(tables_dir / "evaluation_spans.csv", index=False, encoding="utf-8-sig")
    steps_eval[
        [
            "dataset",
            "dataset_role",
            "trajectory_id",
            "step_index",
            "step_position",
            "in_uc",
            "in_pi",
            "in_hn",
            "in_uncertain",
        ]
    ].to_csv(tables_dir / "evaluation_step_flags.csv", index=False, encoding="utf-8-sig")

    make_figures(metrics, location, figures_dir)
    (output_dir / "intervention_risk_report.md").write_text(
        report(metrics, pairs, availability, bootstrap, location),
        encoding="utf-8",
    )

    print(f"Intervention-risk evaluation complete: {output_dir}")
    print(f"Metrics: {output_dir / 'detector_metrics_by_dataset.csv'}")
    print(f"Report: {output_dir / 'intervention_risk_report.md'}")


if __name__ == "__main__":
    main()
