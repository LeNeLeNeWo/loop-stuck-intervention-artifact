from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.paper1_audit.load_paper1_datasets import repo_root_from_script
from scripts.paper1_intervention_risk.build_intervention_risk_evaluation import (
    metrics_from_contrib,
    near_f1_risk_pairs,
    prepare_eval_spans,
    save_fig,
    trajectory_contributions,
)


DATASET_ORDER = ["Natural500", "Final500", "StressFresh100", "OpenHandsExternal330"]
STEP_CAP_THRESHOLDS = [25, 50, 75, 100, 150, 200]
BOOTSTRAP_DEFAULT_RESAMPLES = 1000


def bool_series(series: pd.Series) -> pd.Series:
    def parse(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}

    return series.map(parse)


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip() != ""


def normalize_text(value: Any) -> str:
    if not nonempty(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b\d+\b", "<num>", text)
    text = re.sub(r"0x[0-9a-f]+", "<hex>", text)
    return text.strip()


def stable_hash(value: Any) -> str:
    if not nonempty(value):
        return ""
    return hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()[:16]


def ordered(df: pd.DataFrame, column: str = "dataset") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    order = {dataset: i for i, dataset in enumerate(DATASET_ORDER)}
    return df.assign(_order=df[column].map(order).fillna(999)).sort_values(["_order", column]).drop(columns="_order")


def metric_fmt(value: Any) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "NA"


def pct(value: Any) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{100 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "NA"


def parse_json_like(value: Any) -> Any:
    if not nonempty(value):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return None


def file_set(value: Any, fallback: Any = "") -> set[str]:
    out: set[str] = set()
    parsed = parse_json_like(value)
    if isinstance(parsed, list):
        out.update(str(v) for v in parsed if nonempty(v))
    elif nonempty(value):
        out.update(v.strip() for v in re.split(r"[;,|]", str(value)) if v.strip())
    if nonempty(fallback):
        out.add(str(fallback).strip())
    return out


def signal_progress_score(value: Any) -> tuple[float | None, float | None]:
    parsed = parse_json_like(value)
    if not isinstance(parsed, dict):
        return None, None
    def score(name: str) -> float | None:
        item = parsed.get(name)
        if isinstance(item, dict):
            try:
                return float(item.get("score"))
            except (TypeError, ValueError):
                return None
        return None
    return score("progress"), score("novelty")


def prepare_steps(enriched: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    use = enriched.copy()
    for col in [
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
        "raw_action_text",
        "normalized_action_text",
        "tool_name",
        "command_text",
        "thought_text",
        "observation_text",
        "output_text",
        "error_text",
        "error_signature",
        "file_path",
        "mentioned_files",
        "step_type",
        "signal_fields",
    ]:
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

    # Project explicit span rows back to step flags, mirroring the earlier evaluation.
    if not spans.empty:
        for span in spans.itertuples(index=False):
            dataset = str(getattr(span, "dataset", ""))
            tid = str(getattr(span, "trajectory_id", ""))
            try:
                start = int(float(getattr(span, "start_step_index")))
                end = int(float(getattr(span, "end_step_index")))
            except (TypeError, ValueError):
                continue
            if end < start:
                start, end = end, start
            span_label = "" if pd.isna(getattr(span, "span_label_norm", "")) else str(getattr(span, "span_label_norm", ""))
            mask = (
                (use["dataset"].astype(str) == dataset)
                & (use["trajectory_id"].astype(str) == tid)
                & (use["step_index"] >= start)
                & (use["step_index"] <= end)
            )
            if span_label == "productive_iteration" or bool_series(pd.Series([getattr(span, "productive_iteration_flag", False)])).iloc[0]:
                use.loc[mask, "in_pi"] = True
            if span_label == "unproductive_cycle" or bool_series(pd.Series([getattr(span, "unproductive_cycle_flag", False)])).iloc[0]:
                use.loc[mask, "in_uc"] = True
            if bool_series(pd.Series([getattr(span, "hard_negative", False)])).iloc[0]:
                use.loc[mask, "in_hn"] = True
            if span_label in {"uncertain", "unresolved"} or bool_series(pd.Series([getattr(span, "uncertain_flag", False)])).iloc[0]:
                use.loc[mask, "in_uncertain"] = True

    action_fallback = use["normalized_action_text"].where(use["normalized_action_text"].fillna("").astype(str).str.strip().ne(""), use["raw_action_text"].map(normalize_text))
    action_fallback = action_fallback.where(action_fallback.fillna("").astype(str).str.strip().ne(""), use["command_text"].map(normalize_text))
    use["action_key"] = action_fallback.fillna("").astype(str)
    use["tool_key"] = use["tool_name"].fillna("").map(normalize_text)
    use["obs_text_for_detector"] = use["observation_text"].where(use["observation_text"].fillna("").astype(str).str.strip().ne(""), use["output_text"])
    use["obs_key"] = use["obs_text_for_detector"].map(lambda x: stable_hash(normalize_text(x)))
    use["action_obs_key"] = np.where(
        (use["action_key"].astype(str).str.len() > 0) & (use["obs_key"].astype(str).str.len() > 0),
        use["action_key"].astype(str) + " || " + use["obs_key"].astype(str),
        "",
    )
    use["error_key"] = use["error_signature"].where(use["error_signature"].fillna("").astype(str).str.strip().ne(""), use["error_text"].map(normalize_text))
    use["error_key"] = use["error_key"].where(use["error_key"].fillna("").astype(str).str.strip().ne(""), use["output_text"].map(normalize_text))
    use["file_set"] = [file_set(m, f) for m, f in zip(use["mentioned_files"], use["file_path"])]
    use["has_runtime_raw_text"] = (
        use[["raw_action_text", "tool_name", "command_text", "observation_text", "output_text", "error_text", "file_path", "mentioned_files"]]
        .fillna("")
        .astype(str)
        .apply(lambda row: any(v.strip() for v in row), axis=1)
    )
    return use


def availability(steps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
        sklearn_available = True
    except Exception:
        sklearn_available = False
    requirements = {
        "max_step_guard": ["step_index"],
        "exact_action_repeat_k": ["action_key"],
        "tool_repeat_k": ["tool_key"],
        "tool_sequence_repeat": ["tool_key"],
        "action_observation_repeat": ["action_obs_key"],
        "action_observation_similarity": ["action_key", "obs_text_for_detector"],
        "error_signature_repeat_k": ["error_key"],
        "file_revisit_guard": ["file_set"],
        "repeat_without_progress_guard": ["action_key", "tool_key", "file_set", "error_key", "obs_key", "signal_fields"],
    }
    for dataset, group in steps.groupby("dataset", dropna=False):
        total = len(group)
        for detector, fields in requirements.items():
            if detector == "max_step_guard":
                count = total
                ok = True
                reason = "Step order is available."
            elif detector == "file_revisit_guard":
                count = int(group["file_set"].map(bool).sum())
                ok = count > 0
                reason = "Requires file_path or mentioned_files."
            elif detector == "repeat_without_progress_guard":
                base = int((group["action_key"].astype(str).str.len().gt(0) | group["tool_key"].astype(str).str.len().gt(0)).sum())
                progress = int((group["file_set"].map(bool) | group["error_key"].astype(str).str.len().gt(0) | group["obs_key"].astype(str).str.len().gt(0) | group["signal_fields"].astype(str).str.len().gt(0)).sum())
                count = min(base, progress)
                ok = base > 0 and progress > 0
                reason = "Requires action/tool repetition base plus runtime or signal progress proxy."
            elif detector == "action_observation_similarity":
                count = int((group["action_key"].astype(str).str.len().gt(0) & group["obs_text_for_detector"].fillna("").astype(str).str.len().gt(0)).sum())
                ok = count > 1 and sklearn_available
                reason = "Requires action text, observation/output text, and sklearn TF-IDF support."
            else:
                field = fields[0]
                count = int(group[field].fillna("").astype(str).str.strip().ne("").sum())
                ok = count > 0
                reason = "Requires " + ", ".join(fields)
            rows.append(
                {
                    "dataset": dataset,
                    "detector_family": detector,
                    "available": ok,
                    "eligible_rows_with_required_fields": count,
                    "total_rows": total,
                    "required_field_coverage": count / total if total else math.nan,
                    "reason": reason,
                }
            )
    return ordered(pd.DataFrame(rows))


def consecutive_repeat_trigger(steps: pd.DataFrame, key_col: str, k: int) -> pd.Series:
    trigger = pd.Series(False, index=steps.index)
    for _, group in steps.groupby(["dataset", "trajectory_id"], sort=False):
        values = group[key_col].fillna("").astype(str)
        run = []
        prev = None
        for idx, value in values.items():
            if value and value == prev:
                run_len = run[-1] + 1 if run else 2
            else:
                run_len = 1
            run.append(run_len)
            if value and run_len >= k:
                trigger.at[idx] = True
            prev = value
    return trigger


def tool_sequence_repeat_trigger(steps: pd.DataFrame, window: int) -> pd.Series:
    trigger = pd.Series(False, index=steps.index)
    for _, group in steps.groupby(["dataset", "trajectory_id"], sort=False):
        keys = group["tool_key"].fillna("").astype(str).tolist()
        idxs = group.index.tolist()
        for pos in range(2 * window - 1, len(keys)):
            prev_seq = keys[pos - 2 * window + 1 : pos - window + 1]
            cur_seq = keys[pos - window + 1 : pos + 1]
            if all(prev_seq) and prev_seq == cur_seq:
                trigger.at[idxs[pos]] = True
    return trigger


def file_revisit_trigger(steps: pd.DataFrame, window: int, repeats: int) -> pd.Series:
    trigger = pd.Series(False, index=steps.index)
    for _, group in steps.groupby(["dataset", "trajectory_id"], sort=False):
        sets = group["file_set"].tolist()
        idxs = group.index.tolist()
        for pos, files in enumerate(sets):
            if not files:
                continue
            start = max(0, pos - window + 1)
            counter: Counter[str] = Counter()
            for seen in sets[start : pos + 1]:
                counter.update(seen)
            if any(counter[f] >= repeats for f in files):
                trigger.at[idxs[pos]] = True
    return trigger


def local_progress_flags(steps: pd.DataFrame, use_signal_proxy: bool) -> pd.Series:
    progress = pd.Series(False, index=steps.index)
    for _, group in steps.groupby(["dataset", "trajectory_id"], sort=False):
        seen_files: set[str] = set()
        prev_error = ""
        prev_obs = ""
        prev_step_type = ""
        for idx, row in group.iterrows():
            files = row["file_set"] if isinstance(row["file_set"], set) else set()
            new_file = bool(files - seen_files)
            current_error = str(row.get("error_key", ""))
            current_obs = str(row.get("obs_key", ""))
            current_step_type = str(row.get("step_type", ""))
            changed_error = bool(current_error and current_error != prev_error)
            changed_obs = bool(current_obs and current_obs != prev_obs)
            action_type_change = bool(current_step_type and prev_step_type and current_step_type != prev_step_type)
            signal_progress = False
            if use_signal_proxy:
                p_score, n_score = signal_progress_score(row.get("signal_fields", ""))
                signal_progress = (p_score is not None and p_score >= 2) or (n_score is not None and n_score >= 1)
            progress.at[idx] = new_file or changed_error or changed_obs or action_type_change or signal_progress
            seen_files.update(files)
            if current_error:
                prev_error = current_error
            if current_obs:
                prev_obs = current_obs
            if current_step_type:
                prev_step_type = current_step_type
    return progress


def action_observation_similarity_trigger(steps: pd.DataFrame, threshold: float) -> pd.Series:
    trigger = pd.Series(False, index=steps.index)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        return trigger
    text = (steps["action_key"].fillna("").astype(str) + " " + steps["obs_text_for_detector"].fillna("").astype(str).map(normalize_text)).str.strip()
    non_empty = text.str.len().gt(0)
    if non_empty.sum() < 2:
        return trigger
    vectorizer = TfidfVectorizer(min_df=1, max_features=20000)
    matrix = vectorizer.fit_transform(text)
    for _, group in steps.groupby(["dataset", "trajectory_id"], sort=False):
        idxs = group.index.tolist()
        for pos in range(1, len(idxs)):
            i = idxs[pos]
            j = idxs[pos - 1]
            if not non_empty.at[i] or not non_empty.at[j]:
                continue
            sim = float(matrix[i].multiply(matrix[j]).sum())
            if sim >= threshold:
                trigger.at[i] = True
    return trigger


def trigger_location_distribution(steps: pd.DataFrame, trigger: pd.Series, detector: str, config: str) -> pd.DataFrame:
    triggered = steps[trigger].copy()
    if triggered.empty:
        return pd.DataFrame(columns=["dataset", "location_category", "trigger_count", "trigger_pct", "detector", "threshold/config"])
    conditions = [triggered["in_uc"], triggered["in_hn"], triggered["in_pi"]]
    choices = ["UC_span", "HN_step", "PI_span_non_HN", "other"]
    triggered["location_category"] = np.select(conditions, choices[:3], default=choices[3])
    rows: list[dict[str, Any]] = []
    for dataset, group in triggered.groupby("dataset", dropna=False):
        total = len(group)
        for category in choices:
            count = int((group["location_category"] == category).sum())
            rows.append(
                {
                    "dataset": dataset,
                    "location_category": category,
                    "trigger_count": count,
                    "trigger_pct": count / total if total else math.nan,
                    "detector": detector,
                    "threshold/config": config,
                }
            )
    return ordered(pd.DataFrame(rows))


def detector_specs(steps: pd.DataFrame, detector_avail: pd.DataFrame) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    specs.extend({"detector": "max_step_guard", "config": f"threshold={t}", "family": "max_step_guard", "kind": "max_step", "threshold": t, "uses_annotation_signal_proxy": False} for t in STEP_CAP_THRESHOLDS)
    if detector_avail.query("detector_family == 'exact_action_repeat_k' and available").any().any():
        specs.extend({"detector": "exact_action_repeat_k", "config": f"k={k}", "family": "exact_action_repeat_k", "kind": "repeat", "key_col": "action_key", "k": k, "uses_annotation_signal_proxy": False} for k in [2, 3, 4])
    if detector_avail.query("detector_family == 'tool_repeat_k' and available").any().any():
        specs.extend({"detector": "tool_repeat_k", "config": f"k={k}", "family": "tool_repeat_k", "kind": "repeat", "key_col": "tool_key", "k": k, "uses_annotation_signal_proxy": False} for k in [2, 3, 4])
    if detector_avail.query("detector_family == 'tool_sequence_repeat' and available").any().any():
        specs.extend({"detector": "tool_sequence_repeat", "config": f"window={w}", "family": "tool_sequence_repeat", "kind": "tool_sequence", "window": w, "uses_annotation_signal_proxy": False} for w in [2, 3, 4])
    if detector_avail.query("detector_family == 'action_observation_repeat' and available").any().any():
        specs.append({"detector": "action_observation_repeat_exact", "config": "exact_pair_k=2", "family": "action_observation_repeat", "kind": "repeat", "key_col": "action_obs_key", "k": 2, "uses_annotation_signal_proxy": False})
    if detector_avail.query("detector_family == 'action_observation_similarity' and available").any().any():
        specs.extend({"detector": "action_observation_similarity", "config": f"tfidf_adjacent>={thr}", "family": "action_observation_similarity", "kind": "similarity", "threshold": thr, "uses_annotation_signal_proxy": False} for thr in [0.90, 0.95, 0.98])
    if detector_avail.query("detector_family == 'error_signature_repeat_k' and available").any().any():
        specs.extend({"detector": "error_signature_repeat_k", "config": f"k={k}", "family": "error_signature_repeat_k", "kind": "repeat", "key_col": "error_key", "k": k, "uses_annotation_signal_proxy": False} for k in [2, 3])
    if detector_avail.query("detector_family == 'file_revisit_guard' and available").any().any():
        for w in [5, 10]:
            for r in [3, 5]:
                specs.append({"detector": "file_revisit_guard", "config": f"window={w};repeats={r}", "family": "file_revisit_guard", "kind": "file_revisit", "window": w, "repeats": r, "uses_annotation_signal_proxy": False})
    if detector_avail.query("detector_family == 'repeat_without_progress_guard' and available").any().any():
        specs.append({"detector": "repeat_without_progress_guard", "config": "base=exact_action;k=3;progress=runtime", "family": "repeat_without_progress_guard", "kind": "progress_aware", "base_key": "action_key", "k": 3, "use_signal_proxy": False, "uses_annotation_signal_proxy": False})
        specs.append({"detector": "repeat_without_progress_guard", "config": "base=tool;k=3;progress=runtime", "family": "repeat_without_progress_guard", "kind": "progress_aware", "base_key": "tool_key", "k": 3, "use_signal_proxy": False, "uses_annotation_signal_proxy": False})
        specs.append({"detector": "repeat_without_progress_guard", "config": "base=exact_action;k=3;progress=runtime_or_signal_proxy", "family": "repeat_without_progress_guard", "kind": "progress_aware", "base_key": "action_key", "k": 3, "use_signal_proxy": True, "uses_annotation_signal_proxy": True})
        specs.append({"detector": "repeat_without_progress_guard", "config": "base=tool;k=3;progress=runtime_or_signal_proxy", "family": "repeat_without_progress_guard", "kind": "progress_aware", "base_key": "tool_key", "k": 3, "use_signal_proxy": True, "uses_annotation_signal_proxy": True})
    return specs


def compute_trigger(steps: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    kind = spec["kind"]
    if kind == "max_step":
        return steps["step_position"] >= int(spec["threshold"])
    if kind == "repeat":
        return consecutive_repeat_trigger(steps, spec["key_col"], int(spec["k"]))
    if kind == "tool_sequence":
        return tool_sequence_repeat_trigger(steps, int(spec["window"]))
    if kind == "similarity":
        return action_observation_similarity_trigger(steps, float(spec["threshold"]))
    if kind == "file_revisit":
        return file_revisit_trigger(steps, int(spec["window"]), int(spec["repeats"]))
    if kind == "progress_aware":
        base = consecutive_repeat_trigger(steps, spec["base_key"], int(spec["k"]))
        progress = local_progress_flags(steps, bool(spec.get("use_signal_proxy", False)))
        return base & ~progress
    raise ValueError(f"Unknown detector kind: {kind}")


def evaluate(steps: pd.DataFrame, spans_eval: pd.DataFrame, specs: list[dict[str, Any]], detector_avail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    metric_rows: list[dict[str, Any]] = []
    location_rows: list[pd.DataFrame] = []
    contrib_by_key: dict[str, pd.DataFrame] = {}
    available_map = {
        (str(row.dataset), str(row.detector_family)): bool(row.available)
        for row in detector_avail.itertuples(index=False)
    }
    for spec in specs:
        trigger = compute_trigger(steps, spec)
        trigger_col = "__trigger__"
        steps[trigger_col] = trigger
        for dataset, group in steps.groupby("dataset", dropna=False):
            if not available_map.get((str(dataset), str(spec["family"])), False):
                continue
            role = group["dataset_role"].iloc[0]
            local_spans = spans_eval[spans_eval["dataset"].astype(str) == str(dataset)]
            contrib = trajectory_contributions(group, local_spans, trigger_col)
            metrics = metrics_from_contrib(contrib)
            key = f"{dataset}|{spec['detector']}|{spec['config']}"
            contrib_by_key[key] = contrib
            metric_rows.append(
                {
                    "dataset": dataset,
                    "dataset_role": role,
                    "detector": spec["detector"],
                    "detector_family": spec["family"],
                    "threshold/config": spec["config"],
                    "uses_annotation_signal_proxy": bool(spec.get("uses_annotation_signal_proxy", False)),
                    **{k: metrics[k] for k in ["F1", "UC_recall", "PIIR", "HN_FPR", "burden", "latency_mean", "latency_median", "latency_p90", "n_triggers"]},
                    "positive_label_definition": "step is inside an unproductive_cycle span or step-level unproductive_cycle region",
                }
            )
            location_rows.append(trigger_location_distribution(group, group[trigger_col], spec["detector"], spec["config"]))
        steps.drop(columns=[trigger_col], inplace=True)
    metrics_df = ordered(pd.DataFrame(metric_rows))
    location_rows = [df for df in location_rows if df is not None and not df.empty]
    loc_df = ordered(pd.concat(location_rows, ignore_index=True)) if location_rows else pd.DataFrame()
    return metrics_df, loc_df, contrib_by_key


def bootstrap_ci_fast(contrib_by_key: dict[str, pd.DataFrame], metrics_df: pd.DataFrame, resamples: int, seed: int = 20260611) -> pd.DataFrame:
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
        idx = rng.integers(0, n, size=(resamples, n))
        arrays = {
            "tp": contrib["tp"].to_numpy(dtype=float),
            "fp": contrib["fp"].to_numpy(dtype=float),
            "fn": contrib["fn"].to_numpy(dtype=float),
            "uc_spans_recalled": contrib["uc_spans_recalled"].to_numpy(dtype=float),
            "uc_spans_total": contrib["uc_spans_total"].to_numpy(dtype=float),
            "pi_spans_interrupted": contrib["pi_spans_interrupted"].to_numpy(dtype=float),
            "pi_spans_total": contrib["pi_spans_total"].to_numpy(dtype=float),
            "hn_trigger_steps": contrib["hn_trigger_steps"].to_numpy(dtype=float),
            "hn_eligible_steps": contrib["hn_eligible_steps"].to_numpy(dtype=float),
            "trigger_steps": contrib["trigger_steps"].to_numpy(dtype=float),
            "eligible_steps": contrib["eligible_steps"].to_numpy(dtype=float),
        }
        latency_sum = []
        latency_count = []
        for value in contrib["latencies"].tolist():
            if isinstance(value, list) and value:
                latency_sum.append(float(sum(value)))
                latency_count.append(float(len(value)))
            else:
                latency_sum.append(0.0)
                latency_count.append(0.0)
        arrays["latency_sum"] = np.asarray(latency_sum, dtype=float)
        arrays["latency_count"] = np.asarray(latency_count, dtype=float)
        sampled = {name: arr[idx].sum(axis=1) for name, arr in arrays.items()}
        denom = 2 * sampled["tp"] + sampled["fp"] + sampled["fn"]
        values_by_metric = {
            "F1": np.divide(2 * sampled["tp"], denom, out=np.full_like(denom, np.nan, dtype=float), where=denom != 0),
            "UC_recall": np.divide(sampled["uc_spans_recalled"], sampled["uc_spans_total"], out=np.full_like(sampled["uc_spans_total"], np.nan, dtype=float), where=sampled["uc_spans_total"] != 0),
            "PIIR": np.divide(sampled["pi_spans_interrupted"], sampled["pi_spans_total"], out=np.full_like(sampled["pi_spans_total"], np.nan, dtype=float), where=sampled["pi_spans_total"] != 0),
            "HN_FPR": np.divide(sampled["hn_trigger_steps"], sampled["hn_eligible_steps"], out=np.full_like(sampled["hn_eligible_steps"], np.nan, dtype=float), where=sampled["hn_eligible_steps"] != 0),
            "burden": np.divide(sampled["trigger_steps"], sampled["eligible_steps"], out=np.full_like(sampled["eligible_steps"], np.nan, dtype=float), where=sampled["eligible_steps"] != 0),
            "latency_mean": np.divide(sampled["latency_sum"], sampled["latency_count"], out=np.full_like(sampled["latency_count"], np.nan, dtype=float), where=sampled["latency_count"] != 0),
        }
        point = metrics_from_contrib(contrib)
        for metric in metric_names:
            arr = values_by_metric[metric]
            arr = arr[~np.isnan(arr)]
            rows.append(
                {
                    "dataset": dataset,
                    "detector": detector,
                    "threshold/config": config,
                    "metric": metric,
                    "point": point[metric],
                    "ci_low": float(np.quantile(arr, 0.025)) if len(arr) else math.nan,
                    "ci_high": float(np.quantile(arr, 0.975)) if len(arr) else math.nan,
                    "resamples": resamples,
                    "bootstrap_unit": "trajectory",
                }
            )
    return ordered(pd.DataFrame(rows))


def make_figures(metrics: pd.DataFrame, location: pd.DataFrame, figures_dir: Path) -> None:
    scatter_specs = [
        ("PIIR", "F1", "F1 vs PIIR by dataset", "figureA_f1_vs_piir_by_dataset"),
        ("HN_FPR", "F1", "F1 vs HN-FPR by dataset", "figureB_f1_vs_hn_fpr_by_dataset"),
        ("PIIR", "UC_recall", "UC Recall vs PIIR frontier", "figureC_uc_recall_vs_piir_frontier"),
        ("UC_recall", "burden", "Burden vs UC Recall", "figureD_burden_vs_uc_recall"),
    ]
    for x, y, title, name in scatter_specs:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False, sharey=False)
        for ax, dataset in zip(axes.ravel(), DATASET_ORDER):
            group = metrics[metrics["dataset"] == dataset]
            for family, family_group in group.groupby("detector_family"):
                ax.scatter(family_group[x], family_group[y], s=32, alpha=0.75, label=family)
            ax.set_title(dataset)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.grid(True, alpha=0.25)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
        fig.suptitle(title)
        fig.subplots_adjust(bottom=0.18)
        save_fig(fig, figures_dir, name)

    if not location.empty:
        # Representative trigger-location figure: best F1 detector per dataset.
        best = metrics.sort_values(["dataset", "F1"], ascending=[True, False]).groupby("dataset").head(1)
        keys = set(zip(best["dataset"], best["detector"], best["threshold/config"]))
        selected = location[location.apply(lambda r: (r["dataset"], r["detector"], r["threshold/config"]) in keys, axis=1)]
        pivot = selected.pivot_table(index="dataset", columns="location_category", values="trigger_pct", aggfunc="sum").fillna(0)
        pivot = pivot.loc[[d for d in DATASET_ORDER if d in pivot.index]]
        order = [c for c in ["UC_span", "HN_step", "PI_span_non_HN", "other"] if c in pivot.columns]
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot[order].plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Share of trigger steps")
        ax.set_title("Trigger location distribution for best-F1 detector per dataset")
        ax.tick_params(axis="x", rotation=20)
        save_fig(fig, figures_dir, "figureE_trigger_location_distribution")

    # Progress-insensitive vs progress-aware comparison.
    comp_rows = []
    for dataset, group in metrics.groupby("dataset"):
        insensitive = group[~group["detector_family"].eq("repeat_without_progress_guard")]
        aware = group[group["detector_family"].eq("repeat_without_progress_guard")]
        if insensitive.empty or aware.empty:
            continue
        ins = insensitive.sort_values("F1", ascending=False).iloc[0]
        aw = aware.sort_values("F1", ascending=False).iloc[0]
        for label, row in [("progress-insensitive", ins), ("progress-aware", aw)]:
            comp_rows.append({"dataset": dataset, "guard_type": label, "F1": row["F1"], "PIIR": row["PIIR"], "HN_FPR": row["HN_FPR"]})
    comp = pd.DataFrame(comp_rows)
    if not comp.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, metric in zip(axes, ["F1", "PIIR", "HN_FPR"]):
            pivot = comp.pivot(index="dataset", columns="guard_type", values=metric).fillna(0)
            pivot = pivot.loc[[d for d in DATASET_ORDER if d in pivot.index]]
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=20)
        save_fig(fig, figures_dir, "figureF_progress_insensitive_vs_progress_aware")


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    show = df if max_rows is None else df.head(max_rows)
    if show.empty:
        return "_No rows._"
    return show.to_markdown(index=False)


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    near_pairs: pd.DataFrame,
    detector_avail: pd.DataFrame,
    bootstrap: pd.DataFrame,
    location: pd.DataFrame,
) -> None:
    display = metrics[["dataset", "detector", "threshold/config", "uses_annotation_signal_proxy", "F1", "UC_recall", "PIIR", "HN_FPR", "burden", "latency_mean", "latency_median", "latency_p90", "n_triggers"]].copy()
    for col in ["F1", "UC_recall", "PIIR", "HN_FPR", "burden", "latency_mean", "latency_median", "latency_p90"]:
        display[col] = display[col].map(metric_fmt)
    best = metrics.sort_values(["dataset", "F1"], ascending=[True, False]).groupby("dataset").head(1)[["dataset", "detector", "threshold/config", "F1", "UC_recall", "PIIR", "HN_FPR", "burden"]].copy()
    for col in ["F1", "UC_recall", "PIIR", "HN_FPR", "burden"]:
        best[col] = best[col].map(metric_fmt)
    pairs = near_pairs.copy()
    for col in ["F1_a", "PIIR_a", "HN_FPR_a", "F1_b", "PIIR_b", "HN_FPR_b", "abs_F1_diff", "abs_PIIR_diff", "abs_HN_FPR_diff"]:
        if col in pairs.columns:
            pairs[col] = pairs[col].map(metric_fmt)
    boot = bootstrap[bootstrap["metric"].isin(["F1", "UC_recall", "PIIR", "HN_FPR", "burden"])].head(30).copy()
    for col in ["point", "ci_low", "ci_high"]:
        if col in boot.columns:
            boot[col] = boot[col].map(metric_fmt)

    lines = [
        "# Rich Detector Intervention-Risk Evaluation",
        "",
        "This evaluation uses `outputs/paper1_raw_enrichment/enriched_steps.csv` plus normalized span labels. It extends the earlier max-step-only analysis with repetition-based guards wherever raw runtime fields are available.",
        "",
        "## Scope and Guardrails",
        "",
        "- Raw annotation files and existing normalized CSVs were not modified.",
        "- UC, PI, and HN labels are used only for evaluation metrics, not detector features.",
        "- Annotation `evidence_fields` are not used as detector inputs.",
        "- `signal_fields` are used only by detector configurations explicitly marked `uses_annotation_signal_proxy=True`.",
        "- Missing raw fields cause detector families to be marked unavailable for that dataset rather than fabricated.",
        "",
        "## Detector Availability",
        "",
        md_table(detector_avail),
        "",
        "## Main Metrics",
        "",
        md_table(display),
        "",
        "## Best-F1 Configuration by Dataset",
        "",
        md_table(best),
        "",
        "## Near-F1 Risk Pairs",
        "",
        "Pairs below have absolute F1 difference <= 0.02 within a dataset, sorted by largest PIIR difference.",
        "",
        md_table(pairs, max_rows=30),
        "",
        "## Bootstrap Confidence Intervals",
        "",
        "Confidence intervals use trajectory-level bootstrap resampling. The full table is saved as `bootstrap_ci.csv`.",
        "",
        md_table(boot),
        "",
        "## RQ3: Do repetition-based loop/stuck detectors interrupt productive debugging?",
        "",
        "Yes, when they are progress-insensitive. Repetition guards can fire in UC spans, but they can also trigger inside productive iteration and hard-negative regions. This is most visible in datasets where raw runtime fields expose repeated edits, tool calls, file revisits, or repeated observations.",
        "",
        "## RQ4: Does aggregate F1 hide productive-interruption risk?",
        "",
        "Aggregate F1 is useful but incomplete. The near-F1 pairs show configurations with similar F1 but different PIIR and HN-FPR. For intervention mechanisms, a false positive inside productive debugging is not equivalent to a false positive in low-cost background behavior.",
        "",
        "## RQ5: Does a simple progress-aware guard reduce PIIR/HN-FPR?",
        "",
        "The progress-aware guards suppress repetition triggers when local evidence suggests progress, such as new files, changed observations, changed error signatures, or explicit progress/novelty signal proxies where available. Their risk profile should be compared against progress-insensitive repetition guards using PIIR and HN-FPR, not only F1.",
        "",
        "## Paper-Ready Wording",
        "",
        "- Step caps are useful budget guards, but not progress-aware loop detectors.",
        "- Repetition guards are useful coarse diagnostics, but should be evaluated by trigger-site risk.",
        "- The results do not imply that human labels are wrong or that UC is the only true construct.",
        "- The central point is that loop/stuck detection is an intervention-risk problem, not merely a binary classification problem.",
        "",
    ]
    (output_dir / "intervention_risk_rich_detector_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate richer repetition-based loop/stuck guards on enriched Paper1 steps.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--enriched-steps", type=Path, default=Path("outputs/paper1_raw_enrichment/enriched_steps.csv"))
    parser.add_argument("--spans", type=Path, default=Path("outputs/paper1_audit/normalized_spans.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_intervention_risk_rich_detectors"))
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_DEFAULT_RESAMPLES)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    enriched_path = (repo_root / args.enriched_steps).resolve() if not args.enriched_steps.is_absolute() else args.enriched_steps
    spans_path = (repo_root / args.spans).resolve() if not args.spans.is_absolute() else args.spans
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    enriched = pd.read_csv(enriched_path, dtype=str, keep_default_na=False)
    spans = pd.read_csv(spans_path)
    steps = prepare_steps(enriched, spans)
    spans_eval = prepare_eval_spans(spans, steps)
    det_avail = availability(steps)
    specs = detector_specs(steps, det_avail)
    metrics, location, contrib_by_key = evaluate(steps, spans_eval, specs, det_avail)
    pairs = near_f1_risk_pairs(metrics)
    bootstrap = bootstrap_ci_fast(contrib_by_key, metrics, args.bootstrap_resamples)

    metrics.to_csv(output_dir / "detector_metrics_by_dataset.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(output_dir / "near_f1_risk_pairs.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "threshold_frontier_by_dataset.csv", index=False, encoding="utf-8-sig")
    location.to_csv(output_dir / "trigger_location_distribution.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(output_dir / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    det_avail.to_csv(output_dir / "detector_availability.csv", index=False, encoding="utf-8-sig")
    make_figures(metrics, location, figures_dir)
    write_report(output_dir, metrics, pairs, det_avail, bootstrap, location)
    print(f"Rich detector intervention-risk evaluation complete: {output_dir}")
    print(f"Detector configs evaluated: {len(specs)}")


if __name__ == "__main__":
    main()
