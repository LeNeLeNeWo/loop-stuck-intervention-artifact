from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DATASET_CONFIGS = [
    {
        "dataset": "Final500",
        "role": "enriched difficult loop-heavy set",
        "directory": "data/final_adjudicated/Final500",
        "preferred_patterns": ["*manual_reviewed_labels*.jsonl", "*labels*.jsonl"],
    },
    {
        "dataset": "Natural500",
        "role": "natural-distribution control",
        "directory": "data/final_adjudicated/Natural500",
        "preferred_patterns": ["*manual_reviewed_labels*.jsonl", "*labels*.jsonl"],
    },
    {
        "dataset": "OpenHandsExternal330",
        "role": "external long-horizon generalization set",
        "directory": "data/annotations/paper1/openhands_external_330_original_labels",
        "preferred_patterns": ["final_reviewer_step_annotations.csv"],
    },
    {
        "dataset": "StressFresh100",
        "role": "clean stress challenge",
        "directory": "data/annotations/paper1/stressfresh_final_v1.0",
        "preferred_patterns": ["*semantic_manual_labels.jsonl", "*labels.jsonl"],
    },
]


FIELD_CANDIDATES = {
    "trajectory_id": ["trajectory_id", "pilot_id", "traj_id", "trajectory", "id"],
    "task_id": ["task_id", "task", "repo_task", "issue_id"],
    "step_index": ["step_index", "step_id", "turn_index", "index"],
    "label": ["label", "step_label", "final_reviewer_main_label", "adjudicated_label"],
    "span_label": ["span_label", "prior_span_label"],
    "hard_negative": ["hard_negative", "hard_negative_flag", "has_hard_negative", "is_hard_negative"],
    "confidence": ["confidence"],
    "evidence": ["evidence_summary", "evidence_note", "review_note", "manual_rationale", "adjudication_rationale"],
    "annotator": ["annotator_id", "team", "reviewer"],
    "span_id": ["span_id", "prior_span_id"],
    "span_start": ["span_start", "start_step_index", "span_start_step"],
    "span_end": ["span_end", "end_step_index", "span_end_step"],
}

LEGAL_STEP_LABELS = {"productive", "stagnant", "uncertain", "neutral"}
LEGAL_SPAN_LABELS = {"productive_iteration", "unproductive_cycle", "none", "uncertain", "unresolved"}


@dataclass
class SchemaDetection:
    source_file: str
    source_format: str
    fields: list[str]
    trajectory_id_field: str | None = None
    task_id_field: str | None = None
    step_index_field: str | None = None
    label_field: str | None = None
    span_label_field: str | None = None
    hard_negative_field: str | None = None
    confidence_field: str | None = None
    evidence_field: str | None = None
    annotator_field: str | None = None
    span_id_field: str | None = None
    span_start_field: str | None = None
    span_end_field: str | None = None
    notes: str = ""

    def as_dict(self, dataset: str, role: str) -> dict[str, Any]:
        out = {
            "dataset": dataset,
            "dataset_role": role,
            "source_file": self.source_file,
            "source_format": self.source_format,
            "fields_json": json.dumps(self.fields, ensure_ascii=False),
            "notes": self.notes,
        }
        for key, value in self.__dict__.items():
            if key not in {"source_file", "source_format", "fields", "notes"}:
                out[key] = value
        return out


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def first_present(fields: Iterable[str], candidates: Iterable[str]) -> str | None:
    field_set = set(fields)
    for candidate in candidates:
        if candidate in field_set:
            return candidate
    lowered = {f.lower(): f for f in fields}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def normalize_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lower = text.lower()
    aliases = {
        "productive_iteration_span": "productive_iteration",
        "productive_iteration": "productive_iteration",
        "unproductive_cycle_span": "unproductive_cycle",
        "unproductive": "stagnant",
        "stagnant": "stagnant",
        "productive": "productive",
        "uncertain": "uncertain",
        "neutral": "neutral",
        "none": "none",
        "no_span": "none",
        "unresolved": "unresolved",
    }
    return aliases.get(lower, lower)


def parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value):
            return None
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return None


def to_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def compact_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def discover_files(dataset_dir: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(dataset_dir.rglob(pattern))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in files:
        if path.is_file() and path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def read_first_jsonl(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    return {}


def detect_schema_from_fields(path: Path, source_format: str, fields: list[str]) -> SchemaDetection:
    return SchemaDetection(
        source_file=path.as_posix(),
        source_format=source_format,
        fields=fields,
        trajectory_id_field=first_present(fields, FIELD_CANDIDATES["trajectory_id"]),
        task_id_field=first_present(fields, FIELD_CANDIDATES["task_id"]),
        step_index_field=first_present(fields, FIELD_CANDIDATES["step_index"]),
        label_field=first_present(fields, FIELD_CANDIDATES["label"]),
        span_label_field=first_present(fields, FIELD_CANDIDATES["span_label"]),
        hard_negative_field=first_present(fields, FIELD_CANDIDATES["hard_negative"]),
        confidence_field=first_present(fields, FIELD_CANDIDATES["confidence"]),
        evidence_field=first_present(fields, FIELD_CANDIDATES["evidence"]),
        annotator_field=first_present(fields, FIELD_CANDIDATES["annotator"]),
        span_id_field=first_present(fields, FIELD_CANDIDATES["span_id"]),
        span_start_field=first_present(fields, FIELD_CANDIDATES["span_start"]),
        span_end_field=first_present(fields, FIELD_CANDIDATES["span_end"]),
    )


def detect_jsonl_schema(path: Path) -> SchemaDetection:
    first = read_first_jsonl(path)
    detection = detect_schema_from_fields(path, "jsonl", sorted(first.keys()))
    if not detection.trajectory_id_field or not detection.step_index_field:
        detection.notes = "JSONL did not expose an obvious trajectory/step schema in the first non-empty row."
    return detection


def detect_csv_schema(path: Path) -> SchemaDetection:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
    detection = detect_schema_from_fields(path, "csv", fields)
    if "span_annotations" in path.name and not detection.span_start_field:
        detection.notes = "CSV appears span-like but no start/end span fields were detected."
    return detection


def interesting_extra_fields(row: dict[str, Any]) -> dict[str, Any]:
    extras = {}
    for field in [
        "label_source",
        "decision_source",
        "semantic_review_version",
        "hard_negative_type",
        "loop_type",
        "secondary_loop_types",
        "intervention_needed",
        "suggested_intervention",
        "warning_flags",
        "signals",
        "review_basis",
        "schema_deviation_note",
    ]:
        if field in row:
            extras[field] = row.get(field)
    return extras


def normalize_step_record(
    *,
    dataset: str,
    role: str,
    subsource: str,
    source_file: Path,
    source_format: str,
    row_number: int,
    row: dict[str, Any],
    schema: SchemaDetection,
) -> dict[str, Any]:
    trajectory_id = row.get(schema.trajectory_id_field) if schema.trajectory_id_field else None
    task_id = row.get(schema.task_id_field) if schema.task_id_field else ""
    step_raw = row.get(schema.step_index_field) if schema.step_index_field else None
    label_raw = row.get(schema.label_field) if schema.label_field else ""
    span_label_raw = row.get(schema.span_label_field) if schema.span_label_field else ""
    hn_raw = row.get(schema.hard_negative_field) if schema.hard_negative_field else None
    label_norm = normalize_label(label_raw)
    span_label_norm = normalize_label(span_label_raw)
    hard_negative = parse_bool(hn_raw)
    evidence = row.get(schema.evidence_field) if schema.evidence_field else ""
    confidence = row.get(schema.confidence_field) if schema.confidence_field else ""
    annotator = row.get(schema.annotator_field) if schema.annotator_field else ""
    span_start = row.get(schema.span_start_field) if schema.span_start_field else ""
    span_end = row.get(schema.span_end_field) if schema.span_end_field else ""
    out = {
        "dataset": dataset,
        "dataset_role": role,
        "dataset_subsource": subsource,
        "source_file": source_file.as_posix(),
        "source_format": source_format,
        "row_number": row_number,
        "trajectory_id": str(trajectory_id) if trajectory_id is not None else "",
        "task_id": task_id,
        "step_id_raw": step_raw,
        "step_index": to_int_or_none(step_raw),
        "label_raw": label_raw,
        "label_norm": label_norm,
        "span_label_raw": span_label_raw,
        "span_label_norm": span_label_norm,
        "hard_negative_raw": hn_raw,
        "hard_negative": hard_negative,
        "productive_iteration_flag": span_label_norm == "productive_iteration",
        "unproductive_cycle_flag": span_label_norm == "unproductive_cycle",
        "uncertain_flag": label_norm == "uncertain" or span_label_norm == "uncertain",
        "unresolved_flag": span_label_norm == "unresolved",
        "confidence": confidence,
        "annotator_id": annotator,
        "evidence_text": evidence,
        "span_id_raw": row.get(schema.span_id_field) if schema.span_id_field else "",
        "span_start_raw": span_start,
        "span_end_raw": span_end,
        "raw_extra_json": json.dumps(interesting_extra_fields(row), ensure_ascii=False, sort_keys=True),
    }
    return out


def normalize_jsonl_steps(dataset: str, role: str, path: Path, subsource: str = "") -> tuple[list[dict[str, Any]], SchemaDetection]:
    schema = detect_jsonl_schema(path)
    steps: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            steps.append(
                normalize_step_record(
                    dataset=dataset,
                    role=role,
                    subsource=subsource,
                    source_file=path,
                    source_format="jsonl",
                    row_number=i,
                    row=row,
                    schema=schema,
                )
            )
    return steps, schema


def normalize_csv_steps(dataset: str, role: str, path: Path, subsource: str) -> tuple[list[dict[str, Any]], SchemaDetection]:
    schema = detect_csv_schema(path)
    steps: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader, start=1):
            steps.append(
                normalize_step_record(
                    dataset=dataset,
                    role=role,
                    subsource=subsource,
                    source_file=path,
                    source_format="csv",
                    row_number=i,
                    row=row,
                    schema=schema,
                )
            )
    return steps, schema


def normalize_csv_spans(dataset: str, role: str, path: Path, subsource: str) -> tuple[list[dict[str, Any]], SchemaDetection]:
    schema = detect_csv_schema(path)
    spans: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader, start=1):
            trajectory_id = row.get(schema.trajectory_id_field) if schema.trajectory_id_field else ""
            span_label_raw = row.get(schema.span_label_field) if schema.span_label_field else ""
            span_label_norm = normalize_label(span_label_raw)
            hn_raw = row.get(schema.hard_negative_field) if schema.hard_negative_field else None
            evidence = row.get(schema.evidence_field) if schema.evidence_field else ""
            confidence = row.get(schema.confidence_field) if schema.confidence_field else ""
            spans.append(
                {
                    "dataset": dataset,
                    "dataset_role": role,
                    "dataset_subsource": subsource,
                    "source_file": path.as_posix(),
                    "source_format": "csv",
                    "span_source": "explicit_span_csv",
                    "row_number": i,
                    "trajectory_id": str(trajectory_id),
                    "span_id": row.get(schema.span_id_field) if schema.span_id_field else "",
                    "start_step_index": to_int_or_none(row.get(schema.span_start_field)) if schema.span_start_field else None,
                    "end_step_index": to_int_or_none(row.get(schema.span_end_field)) if schema.span_end_field else None,
                    "span_label_raw": span_label_raw,
                    "span_label_norm": span_label_norm,
                    "hard_negative_raw": hn_raw,
                    "hard_negative": parse_bool(hn_raw),
                    "productive_iteration_flag": span_label_norm == "productive_iteration",
                    "unproductive_cycle_flag": span_label_norm == "unproductive_cycle",
                    "uncertain_flag": span_label_norm == "uncertain",
                    "unresolved_flag": span_label_norm == "unresolved",
                    "confidence": confidence,
                    "evidence_text": evidence,
                    "step_count": None,
                    "raw_extra_json": json.dumps(interesting_extra_fields(row), ensure_ascii=False, sort_keys=True),
                }
            )
    return spans, schema


def derive_spans_from_steps(steps: list[dict[str, Any]], source_note: str) -> list[dict[str, Any]]:
    if not steps:
        return []
    df = pd.DataFrame(steps)
    spans: list[dict[str, Any]] = []
    if "span_id_raw" in df.columns and df["span_id_raw"].fillna("").astype(str).str.len().gt(0).any():
        use = df[df["span_label_norm"].fillna("").astype(str).ne("")]
        use = use[~use["span_label_norm"].isin(["none", ""])]
        group_cols = ["dataset", "dataset_role", "dataset_subsource", "trajectory_id", "span_id_raw", "span_label_norm"]
        for idx, (_, g) in enumerate(use.groupby(group_cols, dropna=False), start=1):
            first = g.iloc[0]
            start_values = [to_int_or_none(v) for v in g.get("span_start_raw", pd.Series(dtype=object)).tolist()]
            end_values = [to_int_or_none(v) for v in g.get("span_end_raw", pd.Series(dtype=object)).tolist()]
            step_values = [v for v in g["step_index"].tolist() if pd.notna(v)]
            start = min([v for v in start_values if v is not None] or step_values or [None])
            end = max([v for v in end_values if v is not None] or step_values or [None])
            hn = bool(g["hard_negative"].fillna(False).astype(bool).any())
            spans.append(
                {
                    "dataset": first["dataset"],
                    "dataset_role": first["dataset_role"],
                    "dataset_subsource": first["dataset_subsource"],
                    "source_file": first["source_file"],
                    "source_format": first["source_format"],
                    "span_source": source_note,
                    "row_number": idx,
                    "trajectory_id": first["trajectory_id"],
                    "span_id": first["span_id_raw"] or f"{first['trajectory_id']}_derived_{idx:04d}",
                    "start_step_index": start,
                    "end_step_index": end,
                    "span_label_raw": first["span_label_raw"],
                    "span_label_norm": first["span_label_norm"],
                    "hard_negative_raw": hn,
                    "hard_negative": hn,
                    "productive_iteration_flag": first["span_label_norm"] == "productive_iteration",
                    "unproductive_cycle_flag": first["span_label_norm"] == "unproductive_cycle",
                    "uncertain_flag": first["span_label_norm"] == "uncertain",
                    "unresolved_flag": first["span_label_norm"] == "unresolved",
                    "confidence": "",
                    "evidence_text": "Derived from repeated step-level span_id/span_label fields.",
                    "step_count": int(len(g)),
                    "raw_extra_json": json.dumps({"derivation": source_note}, sort_keys=True),
                }
            )
        return spans

    # Fallback: contiguous runs of the same non-none step span label.
    for (dataset, role, subsource, trajectory_id), g in df.sort_values(["dataset", "trajectory_id", "step_index"]).groupby(
        ["dataset", "dataset_role", "dataset_subsource", "trajectory_id"], dropna=False
    ):
        current_label = None
        current_rows: list[pd.Series] = []
        local_idx = 0
        for _, row in g.iterrows():
            label = row.get("span_label_norm") or ""
            if label in {"", "none"}:
                label = ""
            if label != current_label:
                if current_label:
                    local_idx += 1
                    spans.append(make_contiguous_span(current_rows, current_label, local_idx, source_note))
                current_label = label
                current_rows = [row] if label else []
            elif label:
                current_rows.append(row)
        if current_label:
            local_idx += 1
            spans.append(make_contiguous_span(current_rows, current_label, local_idx, source_note))
    return spans


def make_contiguous_span(rows: list[pd.Series], label: str, idx: int, source_note: str) -> dict[str, Any]:
    first = rows[0]
    step_values = [r.get("step_index") for r in rows if pd.notna(r.get("step_index"))]
    hn = any(bool(r.get("hard_negative")) for r in rows)
    return {
        "dataset": first["dataset"],
        "dataset_role": first["dataset_role"],
        "dataset_subsource": first["dataset_subsource"],
        "source_file": first["source_file"],
        "source_format": first["source_format"],
        "span_source": source_note,
        "row_number": idx,
        "trajectory_id": first["trajectory_id"],
        "span_id": f"{first['trajectory_id']}_contiguous_{idx:04d}",
        "start_step_index": min(step_values) if step_values else None,
        "end_step_index": max(step_values) if step_values else None,
        "span_label_raw": label,
        "span_label_norm": label,
        "hard_negative_raw": hn,
        "hard_negative": hn,
        "productive_iteration_flag": label == "productive_iteration",
        "unproductive_cycle_flag": label == "unproductive_cycle",
        "uncertain_flag": label == "uncertain",
        "unresolved_flag": label == "unresolved",
        "confidence": "",
        "evidence_text": "Derived from contiguous step-level span_label values.",
        "step_count": len(rows),
        "raw_extra_json": json.dumps({"derivation": source_note}, sort_keys=True),
    }


def choose_jsonl_label_file(dataset_dir: Path, preferred_patterns: list[str]) -> Path | None:
    candidates = discover_files(dataset_dir, preferred_patterns)
    filtered = []
    for path in candidates:
        name = path.name.lower()
        if any(skip in name for skip in ["diff", "decision", "disagreement", "queue", "worklist", "issues"]):
            continue
        filtered.append(path)
    if not filtered:
        filtered = [p for p in dataset_dir.rglob("*.jsonl") if p.is_file()]
    if not filtered:
        return None
    preferred_terms = ["manual_reviewed_labels", "semantic_manual_labels", "reviewed_labels", "labels"]
    return sorted(filtered, key=lambda p: (min([preferred_terms.index(t) for t in preferred_terms if t in p.name.lower()] or [99]), len(p.parts)))[0]


def load_all_datasets(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_steps: list[dict[str, Any]] = []
    all_spans: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    for config in DATASET_CONFIGS:
        dataset = config["dataset"]
        role = config["role"]
        dataset_dir = repo_root / config["directory"]
        if not dataset_dir.exists():
            schema_rows.append({"dataset": dataset, "dataset_role": role, "source_file": str(dataset_dir), "notes": "directory_missing"})
            continue
        if dataset == "OpenHandsExternal330":
            step_files = sorted(dataset_dir.rglob("final_reviewer_step_annotations.csv"))
            span_files = sorted(dataset_dir.rglob("final_reviewer_span_annotations.csv"))
            for path in step_files:
                subsource = path.parent.name
                steps, schema = normalize_csv_steps(dataset, role, path.relative_to(repo_root), subsource)
                all_steps.extend(steps)
                schema_rows.append(schema.as_dict(dataset, role))
            for path in span_files:
                subsource = path.parent.name
                spans, schema = normalize_csv_spans(dataset, role, path.relative_to(repo_root), subsource)
                all_spans.extend(spans)
                schema_rows.append(schema.as_dict(dataset, role))
            continue
        label_file = choose_jsonl_label_file(dataset_dir, config["preferred_patterns"])
        if label_file is None:
            schema_rows.append({"dataset": dataset, "dataset_role": role, "source_file": str(dataset_dir), "notes": "no_jsonl_label_file_detected"})
            continue
        steps, schema = normalize_jsonl_steps(dataset, role, label_file.relative_to(repo_root), "")
        all_steps.extend(steps)
        all_spans.extend(derive_spans_from_steps(steps, "derived_from_step_level_span_fields"))
        schema_rows.append(schema.as_dict(dataset, role))
    steps_df = pd.DataFrame(all_steps)
    spans_df = pd.DataFrame(all_spans)
    schema_df = pd.DataFrame(schema_rows)
    return steps_df, spans_df, schema_df


def save_table(df: pd.DataFrame, path_stem: Path) -> list[str]:
    written = []
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path_stem.with_suffix(".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    written.append(str(csv_path))
    try:
        parquet_path = path_stem.with_suffix(".parquet")
        parquet_df = df.copy()
        for col in parquet_df.columns:
            if parquet_df[col].dtype == "object":
                parquet_df[col] = parquet_df[col].map(lambda value: "" if pd.isna(value) else str(value))
        parquet_df.to_parquet(parquet_path, index=False)
        skipped_path = path_stem.with_suffix(".parquet.SKIPPED.txt")
        if skipped_path.exists():
            skipped_path.unlink()
        written.append(str(parquet_path))
    except Exception as exc:  # pragma: no cover - depends on optional engines
        note_path = path_stem.with_suffix(".parquet.SKIPPED.txt")
        note_path.write_text(f"Parquet export skipped: {exc}\n", encoding="utf-8")
        written.append(str(note_path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and normalize active Paper1 annotation datasets.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_audit"))
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    steps_df, spans_df, schema_df = load_all_datasets(repo_root)
    written = []
    written += save_table(steps_df, output_dir / "normalized_steps")
    written += save_table(spans_df, output_dir / "normalized_spans")
    schema_df.to_csv(output_dir / "schema_inventory.csv", index=False, encoding="utf-8-sig")
    (output_dir / "schema_inventory.json").write_text(schema_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    written += [str(output_dir / "schema_inventory.csv"), str(output_dir / "schema_inventory.json")]
    print("Loaded datasets")
    print(f"steps={len(steps_df)} spans={len(spans_df)} schema_files={len(schema_df)}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
