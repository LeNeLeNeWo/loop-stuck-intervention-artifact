from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.paper1_audit.load_paper1_datasets import repo_root_from_script


DATASET_CONFIGS = [
    {
        "dataset": "Final500",
        "directory": "data/final_adjudicated/Final500",
    },
    {
        "dataset": "Natural500",
        "directory": "data/final_adjudicated/Natural500",
    },
    {
        "dataset": "OpenHandsExternal330",
        "directory": "data/annotations/paper1/openhands_external_330_original_labels",
    },
    {
        "dataset": "StressFresh100",
        "directory": "data/annotations/paper1/stressfresh_final_v1.0",
    },
]

RAW_FIELD_COLUMNS = [
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
    "evidence_fields",
    "raw_field_source",
]

LABEL_FIELD_NAMES = {
    "label",
    "step_label",
    "span_label",
    "hard_negative",
    "hard_negative_flag",
    "manual_core",
    "team_a_core",
    "team_b_core",
    "adjudicated_label",
}

TEXT_FIELD_ALIASES = {
    "raw_action_text": ["action", "raw_action", "action_text", "tool_action"],
    "tool_name": ["tool_name", "tool", "function_name"],
    "command_text": ["command", "cmd", "command_text", "shell_command", "tool_input"],
    "thought_text": ["thought", "thought_text", "reasoning"],
    "observation_text": ["observation", "observation_text", "tool_output"],
    "output_text": ["output", "stdout", "result"],
    "error_text": ["error", "stderr", "exception", "traceback"],
    "file_path": ["file_path", "path", "filename", "file"],
    "step_type": ["step_type", "event_type", "action_type"],
}


def compact_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip() != ""


def first_nonempty(*values: Any) -> str:
    for value in values:
        if nonempty(value):
            return str(value).strip()
    return ""


def normalize_text(value: Any) -> str:
    if not nonempty(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b\d+\b", "<num>", text)
    text = re.sub(r"0x[0-9a-f]+", "<hex>", text)
    return text.strip()


def infer_step_type(action: str, tool_name: str = "") -> str:
    source = first_nonempty(tool_name, action)
    if not source:
        return ""
    token = re.split(r"\s+", source.strip(), maxsplit=1)[0].lower()
    aliases = {
        "run": "run",
        "bash": "run",
        "python": "run",
        "pytest": "run",
        "create": "create",
        "edit": "edit",
        "open": "open",
        "read": "open",
        "search": "search",
        "grep": "search",
        "find": "search",
        "submit": "submit",
    }
    return aliases.get(token, token)


def infer_command_text(action: str, tool_name: str, tool_input: str = "") -> str:
    tool = normalize_text(tool_name)
    action_norm = normalize_text(action)
    if tool in {"bash", "run", "shell", "terminal"}:
        return first_nonempty(tool_input, action)
    if action_norm.startswith(("pytest ", "python ", "pip ", "git ", "ls ", "grep ", "find ", "rg ", "sed ", "cat ")):
        return action
    return ""


FILE_PATTERN = re.compile(
    r"(?:/|\\)?(?:[\w.-]+(?:/|\\))+[\w.@%+=:,~-]+\.[A-Za-z0-9_]+|[\w.-]+\.(?:py|js|ts|tsx|jsx|json|yaml|yml|toml|ini|cfg|md|txt|csv|sh|sql|java|go|rs|cpp|c|h|hpp)",
    flags=re.IGNORECASE,
)


def extract_files(*texts: Any) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for value in texts:
        if not nonempty(value):
            continue
        for match in FILE_PATTERN.findall(str(value)):
            clean = match.strip("`'\"()[]{}.,;:")
            if clean and clean not in seen:
                seen.add(clean)
                found.append(clean)
    return found


ERROR_PATTERNS = [
    re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\b(?::\s*[^\n\r]+)?"),
    re.compile(r"\bAssertionError\b(?::\s*[^\n\r]+)?"),
    re.compile(r"\bFAILED\b[^\n\r]*"),
    re.compile(r"\bERRORS?:[^\n\r]*", flags=re.IGNORECASE),
    re.compile(r"\bTraceback \(most recent call last\)", flags=re.IGNORECASE),
]


def derive_error_signature(error_text: str, output_text: str = "", observation_text: str = "") -> str:
    text = first_nonempty(error_text, output_text, observation_text)
    if not text:
        return ""
    for pattern in ERROR_PATTERNS:
        match = pattern.search(text)
        if match:
            sig = match.group(0)
            sig = re.sub(r"\b\d+\b", "<num>", sig)
            sig = re.sub(r"line <num>", "line <num>", sig, flags=re.IGNORECASE)
            return normalize_text(sig)[:240]
    lowered = text.lower()
    if any(token in lowered for token in ["error", "failed", "exception", "traceback"]):
        snippet = re.sub(r"\s+", " ", text).strip()[:240]
        return normalize_text(snippet)
    return ""


def record_key(dataset: str, trajectory_id: Any, step_id: Any) -> tuple[str, str, int] | None:
    try:
        return (dataset, str(trajectory_id), int(float(str(step_id))))
    except (TypeError, ValueError):
        return None


def merge_record(target: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for col in RAW_FIELD_COLUMNS:
        incoming = update.get(col, "")
        if col in {"signal_fields", "evidence_fields", "mentioned_files", "raw_field_source"}:
            if nonempty(incoming):
                existing = target.get(col, "")
                if not nonempty(existing):
                    target[col] = incoming
                elif str(incoming) not in str(existing):
                    target[col] = f"{existing} || {incoming}"
        elif nonempty(incoming) and not nonempty(target.get(col, "")):
            target[col] = incoming
    if not nonempty(target.get("normalized_action_text", "")):
        target["normalized_action_text"] = normalize_text(target.get("raw_action_text", ""))
    if not nonempty(target.get("step_type", "")):
        target["step_type"] = infer_step_type(str(target.get("raw_action_text", "")), str(target.get("tool_name", "")))
    if not nonempty(target.get("command_text", "")):
        target["command_text"] = infer_command_text(
            str(target.get("raw_action_text", "")),
            str(target.get("tool_name", "")),
            "",
        )
    if not nonempty(target.get("error_signature", "")):
        target["error_signature"] = derive_error_signature(
            str(target.get("error_text", "")),
            str(target.get("output_text", "")),
            str(target.get("observation_text", "")),
        )
    if not nonempty(target.get("mentioned_files", "")):
        files = extract_files(
            target.get("raw_action_text", ""),
            target.get("command_text", ""),
            target.get("observation_text", ""),
            target.get("output_text", ""),
            target.get("error_text", ""),
            target.get("file_path", ""),
        )
        if files:
            target["mentioned_files"] = json.dumps(files, ensure_ascii=False)
            if not nonempty(target.get("file_path", "")):
                target["file_path"] = files[0]
    return target


def row_to_record(dataset: str, trajectory_id: Any, step_id: Any, source: str, fields: dict[str, Any]) -> tuple[tuple[str, str, int] | None, dict[str, Any]]:
    action = first_nonempty(fields.get("raw_action_text"), fields.get("action"), fields.get("tool_action"))
    tool = first_nonempty(fields.get("tool_name"), fields.get("tool"), fields.get("function_name"))
    command = first_nonempty(
        fields.get("command_text"),
        fields.get("command"),
        fields.get("cmd"),
        infer_command_text(action, tool, str(fields.get("tool_input", ""))),
    )
    thought = first_nonempty(fields.get("thought_text"), fields.get("thought"), fields.get("reasoning"))
    observation = first_nonempty(fields.get("observation_text"), fields.get("observation"), fields.get("tool_output"))
    output = first_nonempty(fields.get("output_text"), fields.get("output"), fields.get("stdout"), fields.get("result"))
    error = first_nonempty(fields.get("error_text"), fields.get("error"), fields.get("stderr"), fields.get("exception"), fields.get("traceback"))
    file_path = first_nonempty(fields.get("file_path"), fields.get("path"), fields.get("filename"), fields.get("file"))
    mentioned = extract_files(action, command, observation, output, error, file_path, fields.get("tool_input", ""))
    signal = fields.get("signal_fields", "")
    evidence = fields.get("evidence_fields", "")
    rec = {
        "raw_action_text": action,
        "normalized_action_text": normalize_text(action),
        "tool_name": tool,
        "command_text": command,
        "thought_text": thought,
        "observation_text": observation,
        "output_text": output,
        "error_text": error,
        "error_signature": first_nonempty(fields.get("error_signature"), derive_error_signature(error, output, observation)),
        "file_path": file_path or (mentioned[0] if mentioned else ""),
        "mentioned_files": json.dumps(mentioned, ensure_ascii=False) if mentioned else "",
        "step_type": first_nonempty(fields.get("step_type"), infer_step_type(action, tool)),
        "signal_fields": compact_json(signal),
        "evidence_fields": compact_json(evidence),
        "raw_field_source": source,
    }
    return record_key(dataset, trajectory_id, step_id), rec


def parse_stress_review_packets(root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    packet_dir = root / "manual_read_adjudication_v1.1" / "review_packets"
    for path in packet_dir.glob("*.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tid = obj.get("trajectory_id") or path.stem
        for step in obj.get("steps", []):
            visible = step.get("visible_step") or {}
            if not isinstance(visible, dict):
                continue
            key, rec = row_to_record(
                "StressFresh100",
                tid,
                step.get("step_id"),
                path.as_posix(),
                {
                    "thought": visible.get("thought"),
                    "action": visible.get("action"),
                    "tool_name": visible.get("tool_name"),
                    "tool_input": visible.get("tool_input"),
                    "observation": visible.get("observation"),
                    "error": visible.get("error"),
                },
            )
            if key:
                records[key] = merge_record(records.get(key, {}), rec)
    return records


def parse_final_markdown_packets(root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in (root / "batches").glob("*.review.md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"# Manual Review Packet:\s*([^\n\r]+)", text)
        tid = match.group(1).strip().strip("`") if match else path.name.replace(".review.md", "")
        step_matches = list(re.finditer(r"^### Step\s+(\d+)\s*$", text, flags=re.MULTILINE))
        for i, sm in enumerate(step_matches):
            step_id = sm.group(1)
            start = sm.end()
            end = step_matches[i + 1].start() if i + 1 < len(step_matches) else len(text)
            block = text[start:end]
            fields: dict[str, str] = {}
            for name in ["thought", "action", "observation", "error"]:
                m = re.search(rf"^- {name}:\s*(.*?)(?=\n- [A-Za-z_]+:|\n### Step|\Z)", block, flags=re.MULTILINE | re.DOTALL)
                if m:
                    fields[name] = re.sub(r"\s+", " ", m.group(1)).strip()
            team_a = re.search(r"^- Team A:\s*`(.+?)`\s*$", block, flags=re.MULTILINE)
            team_b = re.search(r"^- Team B:\s*`(.+?)`\s*$", block, flags=re.MULTILINE)
            evidence_bits = []
            for match_obj in [team_a, team_b]:
                if match_obj:
                    try:
                        team = json.loads(match_obj.group(1))
                        evidence_bits.append(
                            {
                                "evidence_quotes": team.get("evidence_quotes"),
                                "evidence_summary": team.get("evidence_summary"),
                            }
                        )
                    except json.JSONDecodeError:
                        pass
            if evidence_bits:
                fields["evidence_fields"] = evidence_bits
            key, rec = row_to_record("Final500", tid, step_id, path.as_posix(), fields)
            if key:
                records[key] = merge_record(records.get(key, {}), rec)
    return records


def parse_natural_review_batches(root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in (root / "review_batches").glob("*.md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        item_matches = list(re.finditer(r"^# Review Item\s+\d+:\s*`([^`]+)`\s*$", text, flags=re.MULTILINE))
        for i, im in enumerate(item_matches):
            tid = im.group(1)
            start = im.end()
            end = item_matches[i + 1].start() if i + 1 < len(item_matches) else len(text)
            item = text[start:end]
            step_matches = list(re.finditer(r"^### Step\s+(\d+)\s*$", item, flags=re.MULTILINE))
            for j, sm in enumerate(step_matches):
                block_start = sm.end()
                block_end = step_matches[j + 1].start() if j + 1 < len(step_matches) else len(item)
                block = item[block_start:block_end]
                json_match = re.search(r"```json\s*(.*?)\s*```", block, flags=re.DOTALL)
                evidence = ""
                trajectory_id = tid
                step_id = sm.group(1)
                if json_match:
                    try:
                        obj = json.loads(json_match.group(1))
                        trajectory_id = obj.get("trajectory_id") or tid
                        step_id = obj.get("step_id", step_id)
                        evidence = {
                            "team_a_evidence_summary": obj.get("team_a_evidence_summary"),
                            "team_b_evidence_summary": obj.get("team_b_evidence_summary"),
                        }
                    except json.JSONDecodeError:
                        pass
                key, rec = row_to_record(
                    "Natural500",
                    trajectory_id,
                    step_id,
                    path.as_posix(),
                    {"evidence_fields": evidence},
                )
                if key:
                    records[key] = merge_record(records.get(key, {}), rec)
    return records


def parse_openhands_reviewer_csv(root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in root.rglob("final_reviewer_step_annotations.csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key, rec = row_to_record(
                    "OpenHandsExternal330",
                    row.get("pilot_id") or row.get("trajectory_id"),
                    row.get("step_index") or row.get("step_id"),
                    path.as_posix(),
                    {
                        "evidence_fields": {
                            "evidence_note": row.get("evidence_note"),
                            "schema_deviation_note": row.get("schema_deviation_note"),
                        }
                    },
                )
                if key:
                    records[key] = merge_record(records.get(key, {}), rec)
    return records


def parse_label_sidecars(dataset: str, root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in root.glob("*reviewed_labels*.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                evidence = {
                    "evidence_summary": obj.get("evidence_summary"),
                    "evidence_quotes": obj.get("evidence_quotes"),
                    "manual_rationale": obj.get("manual_rationale"),
                    "adjudication_rationale": obj.get("adjudication_rationale"),
                    "notes": obj.get("notes"),
                }
                key, rec = row_to_record(
                    dataset,
                    obj.get("trajectory_id") or obj.get("pilot_id"),
                    obj.get("step_id") or obj.get("step_index"),
                    path.as_posix(),
                    {
                        "signal_fields": obj.get("signals"),
                        "evidence_fields": evidence,
                    },
                )
                if key:
                    records[key] = merge_record(records.get(key, {}), rec)
    return records


def collect_raw_records(repo_root: Path) -> tuple[dict[tuple[str, str, int], dict[str, Any]], list[dict[str, Any]]]:
    combined: dict[tuple[str, str, int], dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    for config in DATASET_CONFIGS:
        dataset = config["dataset"]
        root = repo_root / config["directory"]
        parsers = []
        if dataset == "Final500":
            parsers = [parse_final_markdown_packets, lambda r, d=dataset: parse_label_sidecars(d, r)]
        elif dataset == "Natural500":
            parsers = [parse_natural_review_batches, lambda r, d=dataset: parse_label_sidecars(d, r)]
        elif dataset == "OpenHandsExternal330":
            parsers = [parse_openhands_reviewer_csv]
        elif dataset == "StressFresh100":
            parsers = [parse_stress_review_packets]
        for parser in parsers:
            before = len(combined)
            parsed = parser(root)
            for key, rec in parsed.items():
                combined[key] = merge_record(combined.get(key, {}), rec)
            inventory.append(
                {
                    "dataset": dataset,
                    "parser": getattr(parser, "__name__", "label_sidecar_parser"),
                    "records_parsed": len(parsed),
                    "new_or_merged_records_after_parser": len(combined) - before,
                }
            )
    return combined, inventory


def availability(enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in enriched.groupby("dataset", dropna=False):
        total = len(group)
        for field in RAW_FIELD_COLUMNS:
            count = int(group[field].fillna("").astype(str).str.strip().ne("").sum())
            rows.append(
                {
                    "dataset": dataset,
                    "field": field,
                    "available_step_count": count,
                    "total_step_count": total,
                    "availability_rate": count / total if total else 0.0,
                    "available": count > 0,
                }
            )
    return pd.DataFrame(rows)


def detector_availability(field_avail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    field_map = {
        (row.dataset, row.field): bool(row.available)
        for row in field_avail.itertuples(index=False)
    }
    detectors = {
        "max_step_guard": ["step_index"],
        "exact_action_repeat_k": ["normalized_action_text", "raw_action_text", "command_text"],
        "tool_repeat_k": ["tool_name"],
        "tool_sequence_repeat": ["tool_name"],
        "action_observation_repeat": ["normalized_action_text", "observation_text", "output_text"],
        "error_signature_repeat_k": ["error_signature", "error_text", "output_text"],
        "file_revisit_guard": ["file_path", "mentioned_files"],
        "repeat_without_progress_guard": ["normalized_action_text", "raw_action_text", "command_text", "tool_name"],
    }
    for dataset in sorted(field_avail["dataset"].dropna().unique()):
        for detector, fields in detectors.items():
            if detector == "max_step_guard":
                available = True
                reason = "Step order is available in normalized rows."
            elif detector == "repeat_without_progress_guard":
                repetition_base = any(field_map.get((dataset, field), False) for field in ["normalized_action_text", "raw_action_text", "command_text", "tool_name"])
                progress_signal = any(field_map.get((dataset, field), False) for field in ["file_path", "mentioned_files", "error_signature", "observation_text", "signal_fields"])
                available = repetition_base and progress_signal
                reason = "Requires a repetition base plus at least one progress proxy: file_path/mentioned_files, error_signature, observation_text, or signal_fields."
            else:
                available = any(field_map.get((dataset, field), False) for field in fields)
                reason = "Requires any of: " + ", ".join(fields)
            rows.append({"dataset": dataset, "detector_family": detector, "available": available, "reason": reason})
    return pd.DataFrame(rows)


def write_report(output_dir: Path, enriched: pd.DataFrame, avail: pd.DataFrame, detector_avail: pd.DataFrame, inventory: list[dict[str, Any]]) -> None:
    match_rows = []
    for dataset, group in enriched.groupby("dataset", dropna=False):
        total = len(group)
        matched = int(group["raw_field_source"].fillna("").astype(str).str.strip().ne("").sum())
        runtime_matched = int(
            group[["raw_action_text", "tool_name", "command_text", "thought_text", "observation_text", "output_text", "error_text"]]
            .fillna("")
            .astype(str)
            .apply(lambda row: any(v.strip() for v in row), axis=1)
            .sum()
        )
        match_rows.append(
            {
                "dataset": dataset,
                "total_step_rows": total,
                "matched_any_raw_or_sidecar": matched,
                "match_rate_any_raw_or_sidecar": matched / total if total else 0,
                "matched_runtime_text_fields": runtime_matched,
                "match_rate_runtime_text_fields": runtime_matched / total if total else 0,
            }
        )
    match_df = pd.DataFrame(match_rows)
    lines = [
        "# Paper1 Raw Field Enrichment Report",
        "",
        "This report describes enrichment of `outputs/paper1_audit/normalized_steps.csv` with raw runtime fields where those fields were recoverable from active dataset directories under `data/annotations/paper1`.",
        "",
        "Raw annotation files and existing normalized CSVs were not modified. The enriched table is a new derived artifact.",
        "",
        "## Match Rates",
        "",
        match_df.to_markdown(index=False),
        "",
        "## Raw Field Availability",
        "",
        avail.pivot(index="dataset", columns="field", values="availability_rate").fillna(0).round(3).to_markdown(),
        "",
        "## Detector Family Availability",
        "",
        detector_avail.to_markdown(index=False),
        "",
        "## Parser Inventory",
        "",
        pd.DataFrame(inventory).to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- `evidence_fields` and `signal_fields` are preserved for auditability, but label fields are not used as detector inputs.",
        "- `signal_fields` can be used only by explicitly marked progress-aware proxy guards; they are not raw runtime logs.",
        "- Final500 raw runtime text is mostly recoverable from manual review packets for disagreement steps.",
        "- Natural500 active paper1 directories expose evidence summaries but not raw action/tool/observation fields.",
        "- StressFresh100 review packets expose visible step fields including thought, action, tool, observation, and error.",
        "- OpenHandsExternal330 active label directories do not expose raw action/tool/observation fields; only reviewer evidence sidecars are available.",
    ]
    (output_dir / "raw_enrichment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich normalized Paper1 step rows with recoverable raw runtime fields.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--normalized-steps", type=Path, default=Path("outputs/paper1_audit/normalized_steps.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_raw_enrichment"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    normalized_path = (repo_root / args.normalized_steps).resolve() if not args.normalized_steps.is_absolute() else args.normalized_steps
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = pd.read_csv(normalized_path, dtype=str, keep_default_na=False)
    for col in RAW_FIELD_COLUMNS:
        steps[col] = ""

    raw_records, inventory = collect_raw_records(repo_root)
    matched = 0
    for idx, row in steps.iterrows():
        key = record_key(str(row["dataset"]), row["trajectory_id"], row["step_index"])
        if key and key in raw_records:
            matched += 1
            rec = raw_records[key]
            for col in RAW_FIELD_COLUMNS:
                steps.at[idx, col] = rec.get(col, "")
        # Always preserve existing normalized evidence/proxy signal sidecars in explicit fields.
        evidence_bits = []
        if nonempty(row.get("evidence_text", "")):
            evidence_bits.append({"normalized_evidence_text": row.get("evidence_text")})
        extra = {}
        if nonempty(row.get("raw_extra_json", "")):
            try:
                extra = json.loads(row.get("raw_extra_json"))
            except json.JSONDecodeError:
                extra = {}
        if isinstance(extra, dict) and extra.get("signals") and not nonempty(steps.at[idx, "signal_fields"]):
            steps.at[idx, "signal_fields"] = compact_json(extra.get("signals"))
        if evidence_bits and not nonempty(steps.at[idx, "evidence_fields"]):
            steps.at[idx, "evidence_fields"] = compact_json(evidence_bits)
        if not nonempty(steps.at[idx, "normalized_action_text"]):
            steps.at[idx, "normalized_action_text"] = normalize_text(steps.at[idx, "raw_action_text"])
        if not nonempty(steps.at[idx, "error_signature"]):
            steps.at[idx, "error_signature"] = derive_error_signature(
                steps.at[idx, "error_text"],
                steps.at[idx, "output_text"],
                steps.at[idx, "observation_text"],
            )
    steps.to_csv(output_dir / "enriched_steps.csv", index=False, encoding="utf-8-sig")
    avail = availability(steps)
    avail.to_csv(output_dir / "raw_field_availability_by_dataset.csv", index=False, encoding="utf-8-sig")
    det_avail = detector_availability(avail)
    det_avail.to_csv(output_dir / "detector_family_availability_by_dataset.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inventory).to_csv(output_dir / "raw_enrichment_parser_inventory.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir, steps, avail, det_avail, inventory)
    print(f"Raw enrichment complete: {output_dir}")
    print(f"Matched normalized rows with recoverable sidecars: {matched}/{len(steps)}")


if __name__ == "__main__":
    main()
