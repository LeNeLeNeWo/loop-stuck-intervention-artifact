from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.paper1_audit.load_paper1_datasets import LEGAL_SPAN_LABELS, LEGAL_STEP_LABELS, load_all_datasets, repo_root_from_script, save_table


def ensure_normalized(repo_root: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    steps_path = output_dir / "normalized_steps.csv"
    spans_path = output_dir / "normalized_spans.csv"
    if not steps_path.exists() or not spans_path.exists():
        steps, spans, schema = load_all_datasets(repo_root)
        save_table(steps, output_dir / "normalized_steps")
        save_table(spans, output_dir / "normalized_spans")
        schema.to_csv(output_dir / "schema_inventory.csv", index=False, encoding="utf-8-sig")
    return pd.read_csv(steps_path, low_memory=False), pd.read_csv(spans_path, low_memory=False)


def length_summary(steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lengths = (
        steps.groupby(["dataset", "dataset_role", "trajectory_id"], dropna=False)
        .agg(step_count=("step_index", "count"), min_step=("step_index", "min"), max_step=("step_index", "max"))
        .reset_index()
    )
    summary = (
        lengths.groupby(["dataset", "dataset_role"], dropna=False)["step_count"]
        .agg(["count", "sum", "mean", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "trajectory_count", "sum": "step_count", "mean": "avg_trajectory_length"})
    )
    p90 = lengths.groupby(["dataset", "dataset_role"], dropna=False)["step_count"].quantile(0.9).reset_index(name="p90_trajectory_length")
    summary = summary.merge(p90, on=["dataset", "dataset_role"], how="left")
    return lengths, summary


def overlap_matrix(steps: pd.DataFrame) -> pd.DataFrame:
    ids = {name: set(group["trajectory_id"].astype(str)) for name, group in steps.groupby("dataset")}
    rows = []
    for left, left_ids in ids.items():
        for right, right_ids in ids.items():
            rows.append({"dataset_a": left, "dataset_b": right, "overlap_trajectory_count": len(left_ids & right_ids)})
    return pd.DataFrame(rows)


def duplicate_steps(steps: pd.DataFrame) -> pd.DataFrame:
    dup = steps[steps.duplicated(["dataset", "trajectory_id", "step_index"], keep=False)]
    if dup.empty:
        return pd.DataFrame(columns=["dataset", "trajectory_id", "step_index", "duplicate_count"])
    return dup.groupby(["dataset", "trajectory_id", "step_index"]).size().reset_index(name="duplicate_count")


def missing_steps(steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, tid), group in steps.dropna(subset=["step_index"]).groupby(["dataset", "trajectory_id"]):
        vals = sorted({int(v) for v in group["step_index"].dropna()})
        if not vals:
            continue
        expected = set(range(vals[0], vals[-1] + 1))
        missing = sorted(expected - set(vals))
        if missing:
            rows.append(
                {
                    "dataset": dataset,
                    "trajectory_id": tid,
                    "min_step": vals[0],
                    "max_step": vals[-1],
                    "missing_count": len(missing),
                    "missing_preview": " ".join(map(str, missing[:20])),
                }
            )
    return pd.DataFrame(rows, columns=["dataset", "trajectory_id", "min_step", "max_step", "missing_count", "missing_preview"])


def impossible_labels(steps: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    def clean(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value or "")

    rows = []
    for _, row in steps.iterrows():
        label = clean(row.get("label_norm", ""))
        span_label = clean(row.get("span_label_norm", ""))
        if label and label not in LEGAL_STEP_LABELS:
            rows.append({"table": "steps", "dataset": row["dataset"], "trajectory_id": row["trajectory_id"], "index": row["step_index"], "field": "label_norm", "value": label})
        if span_label and span_label not in LEGAL_SPAN_LABELS:
            rows.append({"table": "steps", "dataset": row["dataset"], "trajectory_id": row["trajectory_id"], "index": row["step_index"], "field": "span_label_norm", "value": span_label})
    for _, row in spans.iterrows():
        span_label = clean(row.get("span_label_norm", ""))
        if span_label and span_label not in LEGAL_SPAN_LABELS:
            rows.append({"table": "spans", "dataset": row["dataset"], "trajectory_id": row["trajectory_id"], "index": row.get("span_id", ""), "field": "span_label_norm", "value": span_label})
    return pd.DataFrame(rows, columns=["table", "dataset", "trajectory_id", "index", "field", "value"])


def uncertain_unresolved_summary(steps: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in steps.groupby("dataset"):
        rows.append(
            {
                "dataset": dataset,
                "table": "steps",
                "row_count": len(group),
                "uncertain_count": int(group["uncertain_flag"].fillna(False).astype(bool).sum()),
                "unresolved_count": int(group["unresolved_flag"].fillna(False).astype(bool).sum()),
            }
        )
    for dataset, group in spans.groupby("dataset"):
        rows.append(
            {
                "dataset": dataset,
                "table": "spans",
                "row_count": len(group),
                "uncertain_count": int(group["uncertain_flag"].fillna(False).astype(bool).sum()),
                "unresolved_count": int(group["unresolved_flag"].fillna(False).astype(bool).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_audit_report(output_dir: Path, summary: pd.DataFrame, overlaps: pd.DataFrame, duplicates: pd.DataFrame, missing: pd.DataFrame, impossible: pd.DataFrame, uncertain: pd.DataFrame) -> None:
    lines = [
        "# Paper1 Dataset Integrity Audit",
        "",
        "## Dataset Size Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Integrity Checks",
        "",
        f"- Duplicate `(dataset, trajectory_id, step_index)` rows: `{len(duplicates)}` grouped duplicate keys.",
        f"- Trajectories with missing step indices: `{len(missing)}`.",
        f"- Impossible labels detected: `{len(impossible)}`.",
        "- Trajectory overlap matrix:",
        "",
        overlaps.pivot(index="dataset_a", columns="dataset_b", values="overlap_trajectory_count").to_markdown(),
        "",
        "## Uncertain / Unresolved Summary",
        "",
        uncertain.to_markdown(index=False),
        "",
    ]
    (output_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Paper1 normalized dataset integrity.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_audit"))
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    steps, spans = ensure_normalized(repo_root, output_dir)
    lengths, summary = length_summary(steps)
    overlaps = overlap_matrix(steps)
    duplicates = duplicate_steps(steps)
    missing = missing_steps(steps)
    impossible = impossible_labels(steps, spans)
    uncertain = uncertain_unresolved_summary(steps, spans)
    outputs = {
        "trajectory_lengths": lengths,
        "dataset_integrity_summary": summary,
        "trajectory_overlap_matrix": overlaps,
        "duplicate_step_ids": duplicates,
        "missing_step_indices": missing,
        "impossible_labels": impossible,
        "uncertain_unresolved_summary": uncertain,
    }
    for name, df in outputs.items():
        df.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_audit_report(output_dir, summary, overlaps, duplicates, missing, impossible, uncertain)
    print(f"Integrity audit complete: {output_dir}")


if __name__ == "__main__":
    main()
