from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.paper1_audit.audit_dataset_integrity import ensure_normalized, length_summary
from scripts.paper1_audit.load_paper1_datasets import repo_root_from_script


def bool_series(series: pd.Series) -> pd.Series:
    def parse(value: object) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y", "t"}:
            return True
        if text in {"false", "0", "no", "n", "f", ""}:
            return False
        return bool(value)

    return series.map(parse)


def compute_trajectory_flags(steps: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    step_flags = (
        steps.groupby(["dataset", "dataset_role", "trajectory_id"], dropna=False)
        .agg(
            step_count=("step_index", "count"),
            pi_step_count=("productive_iteration_flag", lambda s: int(bool_series(s).sum())),
            hn_step_count=("hard_negative", lambda s: int(bool_series(s).sum())),
            uc_step_count=("unproductive_cycle_flag", lambda s: int(bool_series(s).sum())),
            uncertain_step_count=("uncertain_flag", lambda s: int(bool_series(s).sum())),
            unresolved_step_count=("unresolved_flag", lambda s: int(bool_series(s).sum())),
            productive_step_count=("label_norm", lambda s: int((s == "productive").sum())),
            stagnant_step_count=("label_norm", lambda s: int((s == "stagnant").sum())),
        )
        .reset_index()
    )
    if spans.empty:
        span_flags = pd.DataFrame(columns=["dataset", "trajectory_id"])
    else:
        span_flags = (
            spans.groupby(["dataset", "trajectory_id"], dropna=False)
            .agg(
                pi_span_count=("productive_iteration_flag", lambda s: int(bool_series(s).sum())),
                hn_span_count=("hard_negative", lambda s: int(bool_series(s).sum())),
                uc_span_count=("unproductive_cycle_flag", lambda s: int(bool_series(s).sum())),
                uncertain_span_count=("uncertain_flag", lambda s: int(bool_series(s).sum())),
                unresolved_span_count=("unresolved_flag", lambda s: int(bool_series(s).sum())),
            )
            .reset_index()
        )
    flags = step_flags.merge(span_flags, on=["dataset", "trajectory_id"], how="left")
    for col in ["pi_span_count", "hn_span_count", "uc_span_count", "uncertain_span_count", "unresolved_span_count"]:
        if col not in flags.columns:
            flags[col] = 0
        flags[col] = flags[col].fillna(0).astype(int)
    flags["has_pi"] = (flags["pi_step_count"] + flags["pi_span_count"]) > 0
    flags["has_hn"] = (flags["hn_step_count"] + flags["hn_span_count"]) > 0
    flags["has_uc"] = (flags["uc_step_count"] + flags["uc_span_count"]) > 0
    flags["has_uncertain_or_unresolved"] = (flags["uncertain_step_count"] + flags["unresolved_step_count"] + flags["uncertain_span_count"] + flags["unresolved_span_count"]) > 0
    flags["pi_density"] = (flags["pi_step_count"] / flags["step_count"]).fillna(0)
    flags["hn_density"] = (flags["hn_step_count"] / flags["step_count"]).fillna(0)
    flags["uc_density"] = (flags["uc_step_count"] / flags["step_count"]).fillna(0)
    flags["uncertain_unresolved_density"] = ((flags["uncertain_step_count"] + flags["unresolved_step_count"]) / flags["step_count"]).fillna(0)
    flags["hn_and_uc"] = flags["has_hn"] & flags["has_uc"]
    flags["pi_and_uc"] = flags["has_pi"] & flags["has_uc"]
    flags["pi_and_hn"] = flags["has_pi"] & flags["has_hn"]
    flags["hn_only"] = flags["has_hn"] & ~flags["has_uc"]
    flags["uc_only"] = flags["has_uc"] & ~flags["has_hn"]
    flags["neither_hn_nor_uc"] = ~flags["has_hn"] & ~flags["has_uc"]
    return flags


def prevalence(flags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, role), group in flags.groupby(["dataset", "dataset_role"], dropna=False):
        n = len(group)
        row = {"dataset": dataset, "dataset_role": role, "trajectory_count": n}
        for col in ["has_pi", "has_hn", "has_uc", "has_uncertain_or_unresolved", "hn_and_uc", "pi_and_uc", "pi_and_hn", "hn_only", "uc_only", "neither_hn_nor_uc"]:
            count = int(group[col].sum())
            row[f"{col}_count"] = count
            row[f"{col}_rate"] = count / n if n else 0
        rows.append(row)
    return pd.DataFrame(rows)


def overlap_table(flags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, role), group in flags.groupby(["dataset", "dataset_role"], dropna=False):
        categories = {
            "HN_and_UC": group["has_hn"] & group["has_uc"],
            "HN_only": group["has_hn"] & ~group["has_uc"],
            "UC_only": group["has_uc"] & ~group["has_hn"],
            "neither_HN_nor_UC": ~group["has_hn"] & ~group["has_uc"],
            "PI_and_UC": group["has_pi"] & group["has_uc"],
            "PI_and_HN": group["has_pi"] & group["has_hn"],
        }
        n = len(group)
        for name, mask in categories.items():
            count = int(mask.sum())
            rows.append({"dataset": dataset, "dataset_role": role, "overlap_category": name, "trajectory_count": count, "rate": count / n if n else 0})
    return pd.DataFrame(rows)


def distribution(df: pd.DataFrame, field: str, table_name: str) -> pd.DataFrame:
    if df.empty or field not in df.columns:
        return pd.DataFrame(columns=["table", "dataset", "label", "count", "rate"])
    rows = []
    for dataset, group in df.groupby("dataset"):
        total = len(group)
        labels = group[field].fillna("not_available").astype(str).replace({"": "not_available"})
        for label, count in labels.value_counts(dropna=False).sort_index().items():
            rows.append({"table": table_name, "dataset": dataset, "label": label, "count": int(count), "rate": count / total if total else 0})
    return pd.DataFrame(rows)


def density_summary(flags: pd.DataFrame) -> pd.DataFrame:
    return (
        flags.groupby(["dataset", "dataset_role"], dropna=False)[["pi_density", "hn_density", "uc_density", "uncertain_unresolved_density"]]
        .agg(["mean", "median", "max"])
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze construct distributions for Paper1 annotation datasets.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_audit"))
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    steps, spans = ensure_normalized(repo_root, output_dir)
    flags = compute_trajectory_flags(steps, spans)
    _, len_summary = length_summary(steps)
    outputs = {
        "construct_trajectory_flags": flags,
        "construct_prevalence_by_dataset": prevalence(flags),
        "construct_overlap_by_dataset": overlap_table(flags),
        "step_label_distribution": distribution(steps, "label_norm", "steps.label_norm"),
        "step_span_label_distribution": distribution(steps, "span_label_norm", "steps.span_label_norm"),
        "span_label_distribution": distribution(spans, "span_label_norm", "spans.span_label_norm"),
        "label_density_by_trajectory": flags,
        "label_density_summary": density_summary(flags),
        "dataset_size_for_construct_analysis": len_summary,
    }
    for name, df in outputs.items():
        df.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"Construct distribution analysis complete: {output_dir}")


if __name__ == "__main__":
    main()
