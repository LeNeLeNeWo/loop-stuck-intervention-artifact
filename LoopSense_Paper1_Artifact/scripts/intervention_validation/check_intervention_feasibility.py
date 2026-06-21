from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


OFFLINE_FILES = {
    "normalized_steps": Path("outputs/paper1_audit/normalized_steps.csv"),
    "normalized_spans": Path("outputs/paper1_audit/normalized_spans.csv"),
    "detector_metrics": Path("outputs/paper1_intervention_risk_rich_detectors/detector_metrics_by_dataset.csv"),
    "trigger_location_distribution": Path("outputs/paper1_intervention_risk_rich_detectors/trigger_location_distribution.csv"),
    "near_f1_risk_pairs": Path("outputs/paper1_intervention_risk_rich_detectors/near_f1_risk_pairs.csv"),
    "threshold_frontier": Path("outputs/paper1_intervention_risk_rich_detectors/threshold_frontier_by_dataset.csv"),
    "enriched_steps": Path("outputs/paper1_raw_enrichment/enriched_steps.csv"),
    "length_vs_constructs": Path("outputs/paper1_construct_mismatch/length_vs_constructs.csv"),
}

LIVE_HINTS = {
    "raw_trajectory_logs": [Path("data/trajectories"), Path("data/raw_trajectories"), Path("logs"), Path("outputs/raw_trajectories")],
    "task_metadata": [Path("data/tasks"), Path("data/benchmark_tasks"), Path("data/swebench"), Path("metadata")],
    "repo_checkouts": [Path("repos"), Path("repositories"), Path("checkouts"), Path("data/repos")],
    "agent_runner": [Path("scripts/run_agent.py"), Path("scripts/agents/run_agent.py"), Path("scripts/paper1/run_agent.py")],
    "model_config": [Path("configs"), Path("config"), Path(".env.example")],
    "test_oracle": [Path("scripts/evaluate_patch.py"), Path("scripts/run_tests.py"), Path("scripts/evaluation")],
}

SUCCESS_KEYWORDS = ["success", "resolved", "tests_passed", "passed", "pass", "outcome", "final_status"]
VALIDATION_KEYWORDS = ["validation", "patch", "test", "tests", "oracle", "verified"]
PROGRESS_COLUMNS = ["signal_fields", "raw_action_text", "normalized_action_text", "tool_name", "observation_text", "output_text", "error_signature", "file_path", "mentioned_files", "step_type"]


@dataclass
class FileStatus:
    key: str
    path: Path
    exists: bool
    size: int | None
    columns: list[str]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def read_header(path: Path) -> list[str]:
    if not path.exists() or path.suffix.lower() not in {".csv", ".tsv"}:
        return []
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            return next(csv.reader(f, delimiter=delimiter), [])
    except Exception:
        return []


def status_for_files(repo_root: Path) -> list[FileStatus]:
    rows: list[FileStatus] = []
    for key, rel in OFFLINE_FILES.items():
        path = repo_root / rel
        rows.append(FileStatus(key, path, path.exists(), path.stat().st_size if path.exists() else None, read_header(path)))
    return rows


def columns_matching(columns: list[str], keywords: list[str]) -> list[str]:
    return [c for c in columns if any(k in c.lower() for k in keywords)]


def scan_csv_headers(repo_root: Path, max_files: int = 2500) -> dict[str, list[dict[str, Any]]]:
    out = {"success_like": [], "validation_like": []}
    count = 0
    for base in [repo_root / "outputs", repo_root / "data"]:
        if not base.exists():
            continue
        for path in base.rglob("*.csv"):
            count += 1
            if count > max_files:
                break
            header = read_header(path)
            success_hits = columns_matching(header, SUCCESS_KEYWORDS)
            validation_hits = columns_matching(header, VALIDATION_KEYWORDS)
            if success_hits:
                out["success_like"].append({"path": str(path.relative_to(repo_root)), "columns": success_hits})
            if validation_hits:
                out["validation_like"].append({"path": str(path.relative_to(repo_root)), "columns": validation_hits})
    return out


def live_status(repo_root: Path) -> list[dict[str, Any]]:
    rows = []
    for key, candidates in LIVE_HINTS.items():
        hits = []
        for rel in candidates:
            p = repo_root / rel
            if p.exists():
                hits.append(str(rel))
        rows.append({"requirement": key, "available": bool(hits), "evidence": hits})
    return rows


def md_bool(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit feasibility for Paper 1 intervention validation.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_intervention_validation"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    statuses = status_for_files(repo_root)
    status_by_key = {s.key: s for s in statuses}
    headers = scan_csv_headers(repo_root)
    live = live_status(repo_root)

    normalized_steps_cols = status_by_key["normalized_steps"].columns
    normalized_spans_cols = status_by_key["normalized_spans"].columns
    enriched_cols = status_by_key["enriched_steps"].columns
    detector_metrics_cols = status_by_key["detector_metrics"].columns

    offline_fields = {
        "normalized steps": status_by_key["normalized_steps"].exists,
        "normalized spans": status_by_key["normalized_spans"].exists,
        "detector metrics": status_by_key["detector_metrics"].exists,
        "trigger location distribution": status_by_key["trigger_location_distribution"].exists,
        "step indices and trajectory ids": all(c in normalized_steps_cols for c in ["trajectory_id", "step_index"]),
        "span labels PI/HN/UC": any(c in normalized_spans_cols for c in ["span_label_norm", "productive_iteration_flag", "unproductive_cycle_flag", "hard_negative"]),
        "step-level progress signals": any(c in enriched_cols for c in PROGRESS_COLUMNS),
    }
    final_success_columns = columns_matching(normalized_steps_cols + normalized_spans_cols + enriched_cols + detector_metrics_cols, SUCCESS_KEYWORDS)
    validation_columns = columns_matching(normalized_steps_cols + normalized_spans_cols + enriched_cols + detector_metrics_cols, VALIDATION_KEYWORDS)

    live_possible = all(row["available"] for row in live)
    offline_possible = all([offline_fields["normalized steps"], offline_fields["normalized spans"], offline_fields["step indices and trajectory ids"], offline_fields["span labels PI/HN/UC"]])
    recommendation = "offline + live prefix-branch experiment" if live_possible else "offline stop-counterfactual only"

    report_lines = [
        "# Intervention Validation Feasibility Report",
        "",
        f"Repository root: `{repo_root}`",
        "",
        "## Offline Counterfactual Inputs",
        "",
        "| Input | Exists | Size bytes | Key columns present |",
        "|---|---:|---:|---|",
    ]
    for s in statuses:
        key_cols = ", ".join(s.columns[:10]) + (" ..." if len(s.columns) > 10 else "")
        report_lines.append(f"| {s.key} | {md_bool(s.exists)} | {s.size if s.size is not None else 'NA'} | {key_cols} |")
    report_lines.extend([
        "",
        "## Offline Field Availability",
        "",
        "| Field group | Available |",
        "|---|---:|",
    ])
    for k, v in offline_fields.items():
        report_lines.append(f"| {k} | {md_bool(v)} |")
    report_lines.extend([
        "",
        "## Outcome and Validation Availability",
        "",
        f"Final task success fields in primary inputs: `{', '.join(final_success_columns) if final_success_columns else 'none detected'}`.",
        f"Post-trigger validation fields in primary inputs: `{', '.join(validation_columns) if validation_columns else 'none detected as structured fields'}`.",
        "",
        "CSV header scan found the following success-like fields elsewhere in the repository outputs/data. These are reported for audit only and are not assumed to be usable without checking semantics:",
        "",
    ])
    if headers["success_like"]:
        for hit in headers["success_like"][:30]:
            report_lines.append(f"- `{hit['path']}`: {', '.join(hit['columns'])}")
    else:
        report_lines.append("- none")
    report_lines.extend([
        "",
        "Validation-like fields elsewhere in repository outputs/data:",
        "",
    ])
    if headers["validation_like"]:
        for hit in headers["validation_like"][:30]:
            report_lines.append(f"- `{hit['path']}`: {', '.join(hit['columns'])}")
    else:
        report_lines.append("- none")

    report_lines.extend([
        "",
        "## Live Prefix-Branch Rerun Requirements",
        "",
        "| Requirement | Available | Evidence |",
        "|---|---:|---|",
    ])
    for row in live:
        report_lines.append(f"| {row['requirement']} | {md_bool(row['available'])} | {', '.join(row['evidence']) if row['evidence'] else 'not found'} |")

    missing_live = [row["requirement"] for row in live if not row["available"]]
    report_lines.extend([
        "",
        "## Decision",
        "",
        f"Offline stop-counterfactual feasible: **{md_bool(offline_possible)}**.",
        f"Can identify final task success in primary inputs: **{md_bool(bool(final_success_columns))}**.",
        f"Can identify structured post-trigger successful validation in primary inputs: **{md_bool(bool(validation_columns))}**.",
        "Can restore prefix state: **no**; no complete replayable prefix-state artifact was found by this audit.",
        f"Live prefix-branch rerun feasible: **{md_bool(live_possible)}**.",
        f"Recommended execution: **{recommendation}**.",
        "",
        "## Caution",
        "",
        "The offline validation can evaluate hard-stop consequences on observed traces by measuring productive suffix cut and UC suffix saved. It cannot estimate stochastic rerun outcomes, warn/replan/handoff behavior, or causal effects on final success when final success and prefix-state replay are unavailable.",
    ])
    if missing_live:
        report_lines.extend([
            "",
            "Missing live rerun requirements:",
            "",
        ])
        for item in missing_live:
            report_lines.append(f"- {item}")

    report = "\n".join(report_lines) + "\n"
    (out_dir / "feasibility_report.md").write_text(report, encoding="utf-8")

    data = {
        "offline_possible": offline_possible,
        "live_possible": live_possible,
        "recommendation": recommendation,
        "final_success_columns_primary": final_success_columns,
        "validation_columns_primary": validation_columns,
        "missing_live_requirements": missing_live,
        "offline_fields": offline_fields,
    }
    (out_dir / "feasibility_report.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    if not live_possible:
        (out_dir / "prefix_branch_feasibility_only.md").write_text(
            "# Prefix-Branch Feasibility Only\n\n"
            "Live prefix-branch reruns were not executed. The audit did not find the complete set of artifacts needed to restore prefix state and rerun an agent: raw replayable trajectory logs, task metadata, repository checkouts, an agent runner, model configuration, and test/oracle harness were not all available.\n\n"
            "This file is intentionally a feasibility statement, not a live-intervention result.\n",
            encoding="utf-8",
        )

    print(f"Wrote feasibility report to {out_dir / 'feasibility_report.md'}")
    print(f"Recommended execution: {recommendation}")


if __name__ == "__main__":
    main()
