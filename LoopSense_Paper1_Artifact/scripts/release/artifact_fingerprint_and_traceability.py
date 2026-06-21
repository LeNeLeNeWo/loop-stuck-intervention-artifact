from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

POOLS = {
    "Natural500": {"trajectories": 500, "steps": 8550, "uc_prev": 0.268},
    "Final500": {"trajectories": 500, "steps": 26014, "uc_prev": 0.582},
    "StressFresh100": {"trajectories": 100, "steps": 3132, "uc_prev": 0.820},
    "OpenHandsExternal330": {"trajectories": 330, "steps": 51010, "uc_prev": 0.012},
}
DISPLAY_POOL = {"Final500": "Enriched500"}
NORMALIZED_COUNTS = {"normalized_steps.csv": 88706, "normalized_spans.csv": 4982}
CSV_GLOBS = [
    "outputs/expected/paper_tables/*.csv",
    "outputs/expected/construct_mismatch/*.csv",
    "outputs/expected/length_robustness/*.csv",
    "outputs/expected/intervention_risk_rich_detectors/*.csv",
    "outputs/expected/intervention_validation/*.csv",
    "outputs/expected/api_replan_intervention/*.csv",
    "outputs/expected/live_guard_experiment/*.csv",
    "outputs/expected/live_guard_official_oracle/*.csv",
    "outputs/expected/paper_findings/*.csv",
    "outputs/expected/provenance_audit/*.csv",
]
OLD_VERSION_TOKENS = [
    "v0.1", "v0.2", "v0.3", "v0.4", "v042", "released_natural500_source", "pseudo", "base100", "final400",
    "stresspool", "old", "backup", "previous", "draft", "shortlist", "review_packet", "manual_work",
    "trajectory_270_unified", "supplementary_270_final_review", "final_review_strict_hn_v3", "nonreleased_span_source",
]
ALLOWED_OLD_MENTION_PREFIXES = (
    "README.md", "DATASET_CARD.md", "ARTIFACT_EVALUATION.md", "anonymization_and_release_notes.md",
    "release_exclusion_report", "artifact_release_final_report.md", "fingerprint_report.md",
    "old_version_contamination_report.md", "reproducibility_audit_report.md", "clean_room_reproduction_report.md",
    "ARTIFACT_READY_FOR_RELEASE.md", "docs/", "manifest_sha256.csv",
    "scripts/release/", "outputs/expected/provenance_audit/",
    "scripts/api_replan_intervention/", "outputs/expected/api_replan_intervention/",
    "scripts/live_guard_experiment/", "outputs/expected/live_guard_experiment/",
    "scripts/live_guard_official_oracle/", "outputs/expected/live_guard_official_oracle/",
)
TEXT_SUFFIXES = {".md", ".txt", ".csv", ".tsv", ".json", ".jsonl", ".py", ".yml", ".yaml", ".cff", ".tex"}
DYNAMIC_REPORTS = {
    "manifest_sha256.csv",
    "fingerprint_report.md",
    "dataset_fingerprints.csv",
    "paper_number_traceability.csv",
    "clean_room_reproduction_report.md",
    "old_version_contamination_report.md",
    "ARTIFACT_READY_FOR_RELEASE.md",
    "artifact_release_final_report.md",
    "reproducibility_audit_report.md",
    "reproducibility_audit_results.csv",
}


def local_patterns() -> list[re.Pattern[str]]:
    return [
        re.compile(r"(?<![A-Za-z0-9_])D:[\\/][^\s,;]*", re.I),
        re.compile(r"\\\\" + "wsl" + r"\.localhost[^\s,;]*", re.I),
        re.compile(r"/home/[A-Za-z0-9_.-]+[^\s,;]*"),
    ]


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def display_pool(pool: str) -> str:
    return DISPLAY_POOL.get(pool, pool)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_csv_float(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def safe_float(value: Any) -> float | None:
    try:
        if value == "" or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def token_regex(token: str) -> re.Pattern[str]:
    if re.fullmatch(r"[A-Za-z0-9_.-]+", token):
        return re.compile(r"(?<![A-Za-z0-9])" + re.escape(token.lower()) + r"(?![A-Za-z0-9])")
    return re.compile(re.escape(token.lower()))


OLD_PATTERNS = {token: token_regex(token) for token in OLD_VERSION_TOKENS}


@dataclass
class CheckRow:
    item: str
    status: str
    detail: str


def status_fail(rows: list[dict[str, Any]]) -> bool:
    return any(str(r.get("status", "")).upper() == "FAIL" for r in rows)


def candidate_fingerprint_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pool in POOLS:
        files.extend([
            root / "data" / "final_adjudicated" / pool / "step_labels.csv",
            root / "data" / "final_adjudicated" / pool / "span_labels.csv",
        ])
    files.extend([
        root / "data" / "normalized" / "normalized_steps.csv",
        root / "data" / "normalized" / "normalized_spans.csv",
    ])
    for pattern in CSV_GLOBS:
        files.extend(sorted(root.glob(pattern)))
    seen: set[str] = set()
    out: list[Path] = []
    for path in files:
        if path.exists() and path.is_file():
            r = rel(path, root)
            if r not in seen:
                seen.add(r)
                out.append(path)
    return out


def fingerprint_file(path: Path, root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "artifact_relative_path": rel(path, root),
        "sha256": sha256_file(path),
        "row_count": "",
        "column_count": "",
        "dataset_if_known": "",
        "trajectory_count_if_derivable": "",
        "step_count_if_derivable": "",
        "span_count_if_derivable": "",
        "trajectory_id_set_sha256_if_derivable": "",
        "step_key_set_sha256_if_derivable": "",
        "created_by_script_if_known": "expected paper pipeline or release builder",
        "status": "PASS",
    }
    try:
        df = read_csv(path)
        row["row_count"] = len(df)
        row["column_count"] = len(df.columns)
        if "dataset" in df.columns:
            datasets = sorted(x for x in df["dataset"].astype(str).unique() if x)
            row["dataset_if_known"] = ";".join(datasets)
        else:
            for pool in POOLS:
                if f"/{pool}/" in row["artifact_relative_path"]:
                    row["dataset_if_known"] = pool
                    break
        tid_col = "public_trajectory_id" if "public_trajectory_id" in df.columns else ("trajectory_id" if "trajectory_id" in df.columns else "")
        if tid_col:
            tids = sorted(df[tid_col].astype(str).unique())
            row["trajectory_count_if_derivable"] = len(tids)
            row["trajectory_id_set_sha256_if_derivable"] = sha256_text("\n".join(tids))
        if "step_index" in df.columns:
            row["step_count_if_derivable"] = len(df)
            if tid_col:
                keys = sorted((df[tid_col].astype(str) + ":" + df["step_index"].astype(str)).unique())
                row["step_key_set_sha256_if_derivable"] = sha256_text("\n".join(keys))
        if "public_span_id" in df.columns or "span_label" in df.columns or "span_label_norm" in df.columns:
            row["span_count_if_derivable"] = len(df)
    except Exception as exc:
        row["status"] = "FAIL"
        row["created_by_script_if_known"] = f"could not parse CSV: {exc}"
    return row


def add_count_checks(root: Path, rows: list[dict[str, Any]]) -> list[CheckRow]:
    checks: list[CheckRow] = []
    for pool, expected in POOLS.items():
        path = root / "data" / "final_adjudicated" / pool / "step_labels.csv"
        if not path.exists():
            checks.append(CheckRow(f"{pool} step_labels.csv", "FAIL", "missing file"))
            continue
        df = read_csv(path)
        step_count = len(df)
        tid_col = "public_trajectory_id" if "public_trajectory_id" in df.columns else ""
        traj_count = df[tid_col].nunique() if tid_col else None
        label = display_pool(pool)
        checks.append(CheckRow(f"{label} trajectory count", "PASS" if traj_count == expected["trajectories"] else "FAIL", f"{traj_count} vs {expected['trajectories']}"))
        checks.append(CheckRow(f"{label} step count", "PASS" if step_count == expected["steps"] else "FAIL", f"{step_count} vs {expected['steps']}"))
    for fname, expected_count in NORMALIZED_COUNTS.items():
        path = root / "data" / "normalized" / fname
        if not path.exists():
            checks.append(CheckRow(fname, "FAIL", "missing file"))
        else:
            df = read_csv(path)
            checks.append(CheckRow(fname, "PASS" if len(df) == expected_count else "FAIL", f"{len(df)} vs {expected_count}"))
    for chk in checks:
        if chk.status == "FAIL":
            rows.append({
                "artifact_relative_path": chk.item,
                "sha256": "",
                "row_count": "",
                "column_count": "",
                "dataset_if_known": "",
                "trajectory_count_if_derivable": "",
                "step_count_if_derivable": "",
                "span_count_if_derivable": "",
                "trajectory_id_set_sha256_if_derivable": "",
                "step_key_set_sha256_if_derivable": "",
                "created_by_script_if_known": chk.detail,
                "status": "FAIL",
            })
    return checks


def write_fingerprints(root: Path) -> bool:
    rows = [fingerprint_file(path, root) for path in candidate_fingerprint_files(root)]
    count_checks = add_count_checks(root, rows)
    fields = [
        "artifact_relative_path", "sha256", "row_count", "column_count", "dataset_if_known",
        "trajectory_count_if_derivable", "step_count_if_derivable", "span_count_if_derivable",
        "trajectory_id_set_sha256_if_derivable", "step_key_set_sha256_if_derivable", "created_by_script_if_known", "status",
    ]
    with (root / "dataset_fingerprints.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fail = status_fail(rows) or any(c.status == "FAIL" for c in count_checks)
    lines = [
        "# Dataset Fingerprint Report", "",
        f"Fingerprint files checked: {len(rows)}", "",
        "## Required Count Checks", "",
    ]
    for c in count_checks:
        lines.append(f"- {c.status}: {c.item} ({c.detail})")
    lines.extend(["", f"Overall fingerprint status: {'FAIL' if fail else 'PASS'}", ""])
    (root / "fingerprint_report.md").write_text("\n".join(lines), encoding="utf-8")
    return not fail


def find_row(df: pd.DataFrame, **conds: str) -> pd.Series | None:
    if df.empty:
        return None
    mask = pd.Series([True] * len(df))
    for key, value in conds.items():
        if key not in df.columns:
            return None
        mask &= df[key].astype(str).eq(str(value))
    if not mask.any():
        return None
    return df[mask].iloc[0]


def add_trace(rows: list[dict[str, Any]], paper_number: str, context: str, expected: Any, reproduced: Any,
              tolerance: float | str, source_file: str, source_columns: str, method: str,
              status: str | None = None, notes: str = "") -> None:
    if status is None:
        if reproduced is None or reproduced == "":
            status = "FAIL"
        else:
            exp = safe_float(expected)
            rep = safe_float(reproduced)
            tol = safe_float(tolerance)
            status = "PASS" if exp is not None and rep is not None and tol is not None and abs(rep - exp) <= tol else ("PASS" if str(expected) == str(reproduced) else "FAIL")
    rows.append({
        "paper_number": paper_number,
        "paper_context": context,
        "expected_value": expected,
        "reproduced_value": reproduced if reproduced is not None else "",
        "tolerance": tolerance,
        "source_file": source_file,
        "source_columns": source_columns,
        "reproduction_method": method,
        "status": status,
        "notes": notes,
    })


def value_from_row(path: Path, conds: dict[str, str], column: str) -> Any:
    if not path.exists():
        return None
    df = read_csv_float(path)
    r = find_row(df, **conds)
    if r is None or column not in df.columns:
        return None
    return r[column]


def write_traceability(root: Path) -> bool:
    rows: list[dict[str, Any]] = []
    table1 = root / "outputs/expected/paper_tables/table1_evidence_pools.csv"
    t1 = read_csv_float(table1) if table1.exists() else pd.DataFrame()
    oh = find_row(t1, **{"Evidence pool": "OpenHandsExternal330"}) if not t1.empty else None
    add_trace(rows, "154.576", "OpenHandsExternal330 average steps", 154.576, None if oh is None else oh.get("Avg. steps"), 0.0005, rel(table1, root), "Evidence pool; Avg. steps", "row lookup")
    add_trace(rows, "0.012", "OpenHandsExternal330 UC prevalence", 0.012, None if oh is None else oh.get("UC prevalence"), 0.0015, rel(table1, root), "Evidence pool; UC prevalence", "row lookup")

    length = root / "outputs/expected/length_robustness/length_robustness_summary.csv"
    for paper, context, comp, expected in [
        ("0.083", "Global longest decile UC prevalence", "global_top_10pct_longest", 0.083333),
        ("0.388", "Global remaining 90% UC prevalence", "global_remaining_90pct", 0.388025),
        ("0.973", "Excluding OpenHands longest decile UC prevalence", "excluding_openhands_top_10pct_longest", 0.973451),
        ("0.402", "Excluding OpenHands remaining 90% UC prevalence", "excluding_openhands_remaining_90pct", 0.402229),
    ]:
        val = value_from_row(length, {"comparison": comp}, "contains_uc_pct")
        add_trace(rows, paper, context, expected, val, 0.0015, rel(length, root), "comparison; contains_uc_pct", "row lookup")

    metrics = root / "outputs/expected/intervention_risk_rich_detectors/detector_metrics_by_dataset.csv"
    met = read_csv_float(metrics) if metrics.exists() else pd.DataFrame()
    a = find_row(met, dataset="Enriched500", detector="max_step_guard", **{"threshold/config": "threshold=50"}) if not met.empty else None
    b = find_row(met, dataset="Enriched500", detector="file_revisit_guard", **{"threshold/config": "window=5;repeats=3"}) if not met.empty else None
    add_trace(rows, "0.719", "Enriched500 max_step_guard threshold=50 F1", 0.719, None if a is None else a.get("F1"), 0.0015, rel(metrics, root), "dataset; detector; threshold/config; F1", "row lookup")
    add_trace(rows, "0.729", "Enriched500 file_revisit_guard window=5,repeats=3 F1", 0.729, None if b is None else b.get("F1"), 0.0015, rel(metrics, root), "dataset; detector; threshold/config; F1", "row lookup")
    stress = find_row(met, dataset="StressFresh100", detector="file_revisit_guard", **{"threshold/config": "window=10;repeats=3"}) if not met.empty else None
    for col, expected in [("UC_recall", 0.958), ("PIIR", 0.868), ("HN_FPR", 0.797)]:
        add_trace(rows, str(expected), f"StressFresh100 file_revisit_guard window=10,repeats=3 {col}", expected, None if stress is None else stress.get(col), 0.0015, rel(metrics, root), f"dataset; detector; threshold/config; {col}", "row lookup")

    near = root / "outputs/expected/intervention_risk_rich_detectors/near_f1_risk_pairs.csv"
    ndf = read_csv_float(near) if near.exists() else pd.DataFrame()
    nr = find_row(ndf, dataset="Enriched500", detector_a="max_step_guard", config_a="threshold=50", detector_b="file_revisit_guard", config_b="window=5;repeats=3") if not ndf.empty else None
    for col, expected, label in [("abs_F1_diff", 0.009, "Delta F1"), ("abs_PIIR_diff", 0.461, "Delta PIIR"), ("abs_HN_FPR_diff", 0.230, "Delta HN-FPR")]:
        add_trace(rows, str(expected), f"Enriched500 near-F1 pair {label}", expected, None if nr is None else nr.get(col), 0.0015, rel(near, root), col, "row lookup")

    bootstrap = root / "outputs/expected/intervention_risk_rich_detectors/bootstrap_ci.csv"
    bdf = read_csv_float(bootstrap) if bootstrap.exists() else pd.DataFrame()
    ci_checks = [
        ("max_step_guard", "threshold=50", "F1", 0.690, 0.742),
        ("max_step_guard", "threshold=50", "PIIR", 0.077, 0.124),
        ("file_revisit_guard", "window=5;repeats=3", "F1", 0.699, 0.756),
        ("file_revisit_guard", "window=5;repeats=3", "PIIR", 0.535, 0.589),
    ]
    for detector, config, metric, lo, hi in ci_checks:
        r = find_row(bdf, dataset="Enriched500", detector=detector, **{"threshold/config": config, "metric": metric}) if not bdf.empty else None
        add_trace(rows, f"[{lo:.3f},{hi:.3f}]", f"Bootstrap CI Enriched500 {detector} {config} {metric}", lo, None if r is None else r.get("ci_low"), 0.0015, rel(bootstrap, root), "ci_low", "row lookup", notes="lower bound")
        add_trace(rows, f"[{lo:.3f},{hi:.3f}]", f"Bootstrap CI Enriched500 {detector} {config} {metric}", hi, None if r is None else r.get("ci_high"), 0.0015, rel(bootstrap, root), "ci_high", "row lookup", notes="upper bound")

    triggers = root / "outputs/expected/intervention_validation/stop_counterfactual_triggers.csv"
    trigger_count = None
    if triggers.exists():
        try:
            trigger_count = sum(1 for _ in triggers.open("r", encoding="utf-8-sig", errors="ignore")) - 1
        except Exception:
            trigger_count = None
    add_trace(rows, "246856", "Offline stop-counterfactual trigger-level events", 246856, trigger_count, 0, rel(triggers, root), "row count", "line count minus header")

    stop = root / "outputs/expected/intervention_validation/stop_counterfactual_summary_by_detector.csv"
    sdf = read_csv_float(stop) if stop.exists() else pd.DataFrame()
    stop_rows = [
        ("max_step_guard", "threshold=50", "productive_suffix_cut_mean", 11.8, "Enriched500 max-step T=50 PSL"),
        ("max_step_guard", "threshold=50", "UC_suffix_saved_mean", 48.1, "Enriched500 max-step T=50 UWS"),
        ("file_revisit_guard", "window=5;repeats=3", "productive_suffix_cut_mean", 24.8, "Enriched500 file-revisit w=5,r=3 PSL"),
        ("file_revisit_guard", "window=5;repeats=3", "UC_suffix_saved_mean", 36.2, "Enriched500 file-revisit w=5,r=3 UWS"),
    ]
    for det, cfg, col, expected, context in stop_rows:
        r = find_row(sdf, dataset="Enriched500", detector=det, detector_config=cfg) if not sdf.empty else None
        add_trace(rows, str(expected), context, expected, None if r is None else r.get(col), 0.1, rel(stop, root), col, "row lookup")

    api_sample = root / "outputs/expected/api_replan_intervention/prefix_replan_sample.csv"
    api_parsed = root / "outputs/expected/api_replan_intervention/api_replan_parsed_responses.csv"
    sample_count = len(read_csv_float(api_sample)) if api_sample.exists() else None
    parsed_count = len(read_csv_float(api_parsed)) if api_parsed.exists() else None
    add_trace(rows, "180", "API-assisted replan paired prefixes", 180, sample_count, 0, rel(api_sample, root), "row count", "CSV row count")
    add_trace(rows, "360", "API-assisted replan parsed JSON responses", 360, parsed_count, 0, rel(api_parsed, root), "row count", "CSV row count")

    api_effects = root / "outputs/expected/api_replan_intervention/api_replan_pairwise_effects.csv"
    adf = read_csv_float(api_effects) if api_effects.exists() else pd.DataFrame()
    api_checks = [
        ("all", "action_shift_rate", "warn_replan_mean", 0.7111111111111111, "API WARN_REPLAN action-shift rate"),
        ("all", "action_shift_rate", "ci_low", 0.6444444444444445, "API WARN_REPLAN action-shift CI lower"),
        ("all", "action_shift_rate", "ci_high", 0.7777777777777778, "API WARN_REPLAN action-shift CI upper"),
        ("all", "progress_seeking_score_mean", "neutral_mean", 2.0, "API neutral progress-seeking score"),
        ("all", "progress_seeking_score_mean", "warn_replan_mean", 2.1666666666666665, "API WARN_REPLAN progress-seeking score"),
        ("all", "progress_seeking_score_mean", "warn_minus_neutral", 0.16666666666666666, "API progress-seeking score paired effect"),
        ("all", "repeat_pattern_rate", "neutral_mean", 0.6388888888888888, "API neutral repeated-pattern rate"),
        ("all", "repeat_pattern_rate", "warn_replan_mean", 0.6555555555555556, "API WARN_REPLAN repeated-pattern rate"),
        ("productive_risk", "unsafe_overstop_rate", "neutral_mean", 0.027777777777777776, "API productive-risk neutral unsafe over-stop"),
        ("productive_risk", "unsafe_overstop_rate", "warn_replan_mean", 0.013888888888888888, "API productive-risk WARN_REPLAN unsafe over-stop"),
    ]
    for stratum, metric, col, expected, context in api_checks:
        r = find_row(adf, stratum=stratum, metric=metric) if not adf.empty else None
        add_trace(rows, f"{expected:.3f}", context, expected, None if r is None else r.get(col), 0.0015, rel(api_effects, root), col, "row lookup")

    live_summary = root / "outputs/expected/live_guard_official_oracle/live_guard_official_summary_by_arm.csv"
    ldf = read_csv_float(live_summary) if live_summary.exists() else pd.DataFrame()
    live_arm_checks = [
        ("NO_GUARD", "No Guard", 35, 48, 42.0, 758692),
        ("WARN_REPLAN_GUARD", "Warn/Replan Guard", 42, 54, 40.2, 744589),
        ("HARD_STOP_GUARD", "Hard Stop Guard", 8, 9, 8.0, 67225),
    ]
    for arm, label, resolved_count, patch_count, mean_steps, mean_tokens in live_arm_checks:
        r = find_row(ldf, arm=arm) if not ldf.empty else None
        n = int(round(float(r.get("N_official_oracle_outcomes")))) if r is not None and "N_official_oracle_outcomes" in ldf.columns else None
        attempted = int(round(float(r.get("N_attempted")))) if r is not None and "N_attempted" in ldf.columns else n
        reproduced_resolved = None
        reproduced_patch = None
        if r is not None and n:
            reproduced_resolved = f"{int(round(float(r.get('resolved_rate')) * n))}/{n}"
        if r is not None and attempted:
            reproduced_patch = f"{int(round(float(r.get('patch_generated_rate')) * attempted))}/{attempted}"
        add_trace(rows, f"{resolved_count}/60", f"Official-oracle live guard {label} resolved/test-pass", f"{resolved_count}/60", reproduced_resolved, "n/a", rel(live_summary, root), "arm; N_official_oracle_outcomes; resolved_rate", "row lookup", notes="60-task official-oracle manuscript value")
        add_trace(rows, f"{patch_count}/60", f"Official-oracle live guard {label} patch generation", f"{patch_count}/60", reproduced_patch, "n/a", rel(live_summary, root), "arm; N_attempted; patch_generated_rate", "row lookup", notes="60-task official-oracle manuscript value")
        add_trace(rows, f"{mean_steps:.1f}", f"Official-oracle live guard {label} mean steps", mean_steps, None if r is None else r.get("mean_steps"), 0.05, rel(live_summary, root), "arm; mean_steps", "row lookup")
        add_trace(rows, f"{mean_tokens:,}", f"Official-oracle live guard {label} mean tokens", mean_tokens, None if r is None else r.get("mean_tokens"), 0.5, rel(live_summary, root), "arm; mean_tokens", "row lookup")

    live_effects = root / "outputs/expected/live_guard_official_oracle/live_guard_official_pairwise_effects.csv"
    ledf = read_csv_float(live_effects) if live_effects.exists() else pd.DataFrame()
    live_effect_checks = [
        ("WARN_REPLAN_GUARD minus NO_GUARD", "resolved_num", "mean_difference", 0.11666666666666667, "Warn/Replan minus No Guard resolved/test-pass effect"),
        ("WARN_REPLAN_GUARD minus NO_GUARD", "resolved_num", "bootstrap95_low", 0.05, "Warn/Replan minus No Guard resolved/test-pass CI lower"),
        ("WARN_REPLAN_GUARD minus NO_GUARD", "resolved_num", "bootstrap95_high", 0.2, "Warn/Replan minus No Guard resolved/test-pass CI upper"),
        ("WARN_REPLAN_GUARD minus NO_GUARD", "patch_generated_num", "mean_difference", 0.1, "Warn/Replan minus No Guard patch-generation effect"),
        ("HARD_STOP_GUARD minus NO_GUARD", "resolved_num", "mean_difference", -0.45, "Hard Stop minus No Guard resolved/test-pass effect"),
        ("HARD_STOP_GUARD minus NO_GUARD", "steps", "mean_difference", -33.96666666666667, "Hard Stop minus No Guard mean-step effect"),
        ("HARD_STOP_GUARD minus NO_GUARD", "patch_generated_num", "mean_difference", -0.65, "Hard Stop minus No Guard patch-generation effect"),
    ]
    for comparison, metric, col, expected, context in live_effect_checks:
        r = find_row(ledf, comparison=comparison, metric=metric) if not ledf.empty else None
        add_trace(rows, f"{expected:.3f}", context, expected, None if r is None else r.get(col), 0.0015, rel(live_effects, root), "comparison; metric; " + col, "row lookup")

    readme = (root / "README.md").read_text(encoding="utf-8", errors="ignore") if (root / "README.md").exists() else ""
    card = (root / "DATASET_CARD.md").read_text(encoding="utf-8", errors="ignore") if (root / "DATASET_CARD.md").exists() else ""
    docs = readme + "\n" + card
    documented = "0.762" in docs and "independent" in docs.lower() and ("not included" in docs.lower() or "not release" in docs.lower())
    add_trace(rows, "0.762", "Annotation reliability Cohen's kappa", "documented", "documented" if documented else "missing documentation", "n/a", "README.md; DATASET_CARD.md", "documentation text", "documentation scan", "DOCUMENTED_NOT_REPRODUCED_BY_RELEASE" if documented else "FAIL", "Raw independent annotation sheets are intentionally not released, so kappa is documented rather than recomputed.")

    fields = ["paper_number", "paper_context", "expected_value", "reproduced_value", "tolerance", "source_file", "source_columns", "reproduction_method", "status", "notes"]
    with (root / "paper_number_traceability.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fail = any(r["status"] == "FAIL" for r in rows)
    return not fail


def is_allowed_mention(path: str) -> bool:
    return path.startswith(ALLOWED_OLD_MENTION_PREFIXES)


def scan_old_versions(root: Path) -> bool:
    blocked: list[str] = []
    allowed: list[str] = []
    for p in root.rglob("*"):
        r = rel(p, root)
        low = r.lower()
        hits = [tok for tok, pat in OLD_PATTERNS.items() if pat.search(low)]
        if hits:
            entry = f"{r}: {', '.join(hits)}"
            if is_allowed_mention(r):
                allowed.append(entry)
            else:
                blocked.append(entry)
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES or p.stat().st_size > 10_000_000:
            continue
        r = rel(p, root)
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        hits = [tok for tok, pat in OLD_PATTERNS.items() if pat.search(text.lower())]
        if hits:
            entry = f"{r}: {', '.join(sorted(set(hits)))}"
            if is_allowed_mention(r):
                allowed.append(entry)
            elif r.startswith("data/") or r.startswith("scripts/") or r.startswith("outputs/expected/"):
                blocked.append(entry)
            else:
                allowed.append(entry)
    lines = ["# Old-Version Contamination Report", "", "## Blocked actual inclusions", ""]
    lines.extend([f"- {x}" for x in blocked] or ["None."])
    lines.extend(["", "## Allowed explanatory mentions", ""])
    lines.extend([f"- {x}" for x in sorted(set(allowed))[:300]] or ["None."])
    lines.extend(["", f"Overall status: {'FAIL' if blocked else 'PASS'}", ""])
    (root / "old_version_contamination_report.md").write_text("\n".join(lines), encoding="utf-8")
    return not blocked


def sanitize(text: str) -> str:
    out = text
    for pattern in local_patterns():
        out = pattern.sub("<LOCAL_PATH>", out)
    return out


def run_command(cmd: list[str], cwd: Path) -> tuple[int, str, str, bool]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    combined = proc.stdout + proc.stderr
    external = any(pattern.search(combined) for pattern in local_patterns())
    return proc.returncode, sanitize(proc.stdout), sanitize(proc.stderr), external


def run_clean_room(root: Path, zip_path: Path, clean_room_dir: Path) -> bool:
    if clean_room_dir.exists():
        shutil.rmtree(clean_room_dir)
    clean_room_dir.mkdir(parents=True, exist_ok=True)
    extracted_root: Path | None = None
    commands: list[dict[str, Any]] = []
    status = "PASS"
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(clean_room_dir)
        dirs = [p for p in clean_room_dir.iterdir() if p.is_dir()]
        files = [p for p in clean_room_dir.iterdir() if p.is_file()]
        if len(dirs) != 1 or files or dirs[0].name != root.name:
            status = "FAIL"
        else:
            extracted_root = dirs[0]
            for cmd in [
                [sys.executable, "tests/test_artifact_integrity.py"],
                [sys.executable, "tests/test_reproduce_key_outputs.py"],
                [sys.executable, "scripts/release/artifact_fingerprint_and_traceability.py", "--artifact-root", ".", "--skip-clean-room"],
            ]:
                code, stdout, stderr, external = run_command(cmd, extracted_root)
                commands.append({"cmd": " ".join(cmd).replace(sys.executable, "python"), "returncode": code, "stdout": stdout[-2000:], "stderr": stderr[-2000:], "external_path_detected": external})
                if code != 0 or external:
                    status = "FAIL"
    except Exception as exc:
        status = "FAIL"
        commands.append({"cmd": "extract zip", "returncode": 1, "stdout": "", "stderr": sanitize(str(exc)), "external_path_detected": False})
    lines = [
        "# Clean-Room Reproduction Report", "",
        f"Zip: `{zip_path.name}`",
        f"Extraction directory label: `{clean_room_dir.name}`",
        f"Single top-level directory expected: `{root.name}/`",
        f"Overall status: {status}", "",
        "## Commands", "",
    ]
    for item in commands:
        lines.append(f"### `{item['cmd']}`")
        lines.append(f"- Return code: {item['returncode']}")
        lines.append(f"- External path detected: {item['external_path_detected']}")
        if item["stdout"]:
            lines.append("- Stdout excerpt:")
            lines.append("```text")
            lines.append(item["stdout"])
            lines.append("```")
        if item["stderr"]:
            lines.append("- Stderr excerpt:")
            lines.append("```text")
            lines.append(item["stderr"])
            lines.append("```")
        lines.append("")
    (root / "clean_room_reproduction_report.md").write_text("\n".join(lines), encoding="utf-8")
    return status == "PASS"


def write_manifest(root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        r = rel(p, root)
        if p.name == "manifest_sha256.csv" or p.name in DYNAMIC_REPORTS or "__pycache__" in p.parts or "/.git/" in r or "/.venv/" in r or r.startswith("_reproduction_tmp/"):
            continue
        rows.append({"path": r, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    with (root / "manifest_sha256.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def zip_artifact(root: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(x for x in root.rglob("*") if x.is_file()):
            r = rel(p, root)
            if "__pycache__" in p.parts or p.suffix.lower() == ".zip" or r.startswith("_reproduction_tmp/"):
                continue
            z.write(p, str(Path(root.name) / r).replace("\\", "/"))


def write_ready(root: Path, ready: bool, details: list[CheckRow]) -> None:
    lines = ["# Artifact Ready For Release", "", f"Status: {'READY' if ready else 'NOT READY'}", "", "## Gate checks", ""]
    for d in details:
        lines.append(f"- {d.status}: {d.item} - {d.detail}")
    (root / "ARTIFACT_READY_FOR_RELEASE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines2 = ["# Artifact Release Final Report", "", f"Status: {'READY' if ready else 'NOT READY'}", "", "## Summary", ""]
    lines2.extend(lines[5:])
    (root / "artifact_release_final_report.md").write_text("\n".join(lines2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate dataset fingerprints and paper-number traceability for the LoopSense Paper 1 artifact.")
    parser.add_argument("--artifact-root", type=Path, default=Path.cwd())
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--clean-room-dir", type=Path, default=None)
    parser.add_argument("--run-clean-room", action="store_true")
    parser.add_argument("--skip-clean-room", action="store_true")
    parser.add_argument("--rebuild-zip", action="store_true")
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    zip_path = (args.zip_path.resolve() if args.zip_path else root.parent / f"{root.name}.zip")
    clean_room_dir = (args.clean_room_dir.resolve() if args.clean_room_dir else root.parent / "clean_room_test")

    fp_ok = write_fingerprints(root)
    trace_ok = write_traceability(root)
    old_ok = scan_old_versions(root)

    clean_ok = True
    clean_note = "skipped"
    clean_report = root / "clean_room_reproduction_report.md"
    if args.run_clean_room and not args.skip_clean_room:
        clean_ok = run_clean_room(root, zip_path, clean_room_dir)
        clean_note = "passed" if clean_ok else "failed"
    elif clean_report.exists():
        existing = clean_report.read_text(encoding="utf-8", errors="ignore")
        clean_ok = "Overall status: PASS" in existing
        clean_note = "passed (existing report)" if clean_ok else "failed (existing report)"
    else:
        clean_report.write_text("# Clean-Room Reproduction Report\n\nClean-room run skipped for this invocation.\n", encoding="utf-8")

    gates = [
        CheckRow("final dataset fingerprints and counts", "PASS" if fp_ok else "FAIL", "dataset_fingerprints.csv and fingerprint_report.md"),
        CheckRow("paper-number traceability", "PASS" if trace_ok else "FAIL", "paper_number_traceability.csv"),
        CheckRow("old-version contamination", "PASS" if old_ok else "FAIL", "old_version_contamination_report.md"),
        CheckRow("clean-room reproduction", "PASS" if clean_ok else "FAIL", clean_note),
        CheckRow("intervention validation files", "PASS" if (root / "outputs/expected/intervention_validation/stop_counterfactual_summary_by_detector.csv").exists() else "FAIL", "expected intervention-validation outputs present"),
        CheckRow("API replan intervention files", "PASS" if (root / "outputs/expected/api_replan_intervention/api_replan_pairwise_effects.csv").exists() else "FAIL", "expected API-assisted replan outputs present"),
        CheckRow("live guard pilot files", "PASS" if (root / "outputs/expected/live_guard_experiment/live_guard_report.md").exists() else "FAIL", "sanitized pilot outputs present; no task-success claim without oracle"),
        CheckRow("official-oracle live guard files", "PASS" if (root / "outputs/expected/live_guard_official_oracle/live_guard_official_report.md").exists() else "FAIL", "sanitized official-oracle live guard outputs present"),
        CheckRow("README reproducibility levels", "PASS" if "Reproducibility levels" in (root / "README.md").read_text(encoding="utf-8", errors="ignore") else "FAIL", "README.md"),
        CheckRow("README version consistency", "PASS" if "Reproducibility and Version Consistency" in (root / "README.md").read_text(encoding="utf-8", errors="ignore") else "FAIL", "README.md"),
    ]
    ready = all(g.status == "PASS" for g in gates)
    write_ready(root, ready, gates)
    write_manifest(root)
    if args.rebuild_zip:
        zip_artifact(root, zip_path)
    print(f"Fingerprint status: {'PASS' if fp_ok else 'FAIL'}")
    print(f"Traceability status: {'PASS' if trace_ok else 'FAIL'}")
    print(f"Old-version contamination status: {'PASS' if old_ok else 'FAIL'}")
    print(f"Clean-room status: {clean_note}")
    print(f"Release readiness: {'READY' if ready else 'NOT READY'}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
