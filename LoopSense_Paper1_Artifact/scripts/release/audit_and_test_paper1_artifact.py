from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
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

REQUIRED_ROOT = [
    "README.md", "LICENSE", "CITATION.cff", "DATASET_CARD.md", "ARTIFACT_EVALUATION.md",
    "anonymization_and_release_notes.md", "requirements.txt", "environment.yml", "manifest_sha256.csv",
    "artifact_integrity_report.md", "release_exclusion_report.md",
]

REQUIRED_EXPECTED = {
    "outputs/expected/paper_tables": [
        "table1_evidence_pools.csv", "table2_construct_non_equivalence.csv",
        "table3_focused_detector_risk_comparison.csv", "table4_near_f1_risk_divergence.csv",
    ],
    "outputs/expected/paper_figures": [
        "fig1.pdf", "fig1.png", "fig1.svg", "fig2.pdf", "fig2.png", "fig2.svg",
        "fig3.pdf", "fig3.png", "fig3.svg", "fig4.pdf", "fig4.png", "fig4.svg",
        "fig5_stop_counterfactual_tradeoff.pdf", "fig5_stop_counterfactual_tradeoff.png", "fig5_stop_counterfactual_tradeoff.svg",
    ],
    "outputs/expected/construct_mismatch": [
        "construct_prevalence_by_dataset.csv", "length_vs_constructs.csv", "construct_mismatch_report.md",
    ],
    "outputs/expected/length_robustness": [
        "length_robustness_summary.csv", "within_dataset_longest_decile.csv", "excluding_openhands_longest_decile.csv",
        "length_construct_correlations.csv", "length_robustness_report.md",
    ],
    "outputs/expected/intervention_risk_rich_detectors": [
        "detector_metrics_by_dataset.csv", "near_f1_risk_pairs.csv", "trigger_location_distribution.csv",
        "bootstrap_ci.csv", "detector_availability.csv", "intervention_risk_rich_detector_report.md",
    ],
    "outputs/expected/intervention_validation": [
        "feasibility_report.md", "stop_counterfactual_triggers.csv", "stop_counterfactual_summary_by_detector.csv",
        "stop_counterfactual_summary_by_location.csv", "stop_counterfactual_near_f1_pairs.csv",
        "stop_counterfactual_report.md", "intervention_validation_master_report.md", "prefix_branch_feasibility_only.md",
    ],
    "outputs/expected/paper_findings": [
        "headline_findings.csv", "headline_findings.md", "strongest_near_f1_risk_pairs.csv", "strongest_near_f1_risk_pairs.md",
        "detector_family_summary.csv", "detector_family_summary.md", "dataset_role_summary.csv", "dataset_role_summary.md",
    ],
    "outputs/expected/live_intervention": [
        "feasibility_report.md", "feasibility_report.json", "live_intervention_not_feasible.md",
        "prefix_branch_sample.csv", "prefix_branch_results.csv", "prefix_branch_summary.csv",
        "live_intervention_report.md", "sampling_report.md", "run_report.md",
    ],
    "outputs/expected/provenance_audit": ["provenance_summary_by_pool.csv", "provenance_audit_report.md"],
    "outputs/expected/api_replan_intervention": [
        "api_env_check.md", "prefix_replan_sample.csv", "prefix_replan_prompts.jsonl",
        "api_replan_parsed_responses.csv", "api_replan_summary_by_group.csv",
        "api_replan_pairwise_effects.csv", "api_replan_report.md",
        "api_call_log.csv",
        "fig6_api_replan_intervention.pdf", "fig6_api_replan_intervention.png",
        "fig6_api_replan_intervention.svg", "table_api_replan_summary.csv",
        "table_api_replan_pairwise_effects.csv", "api_replan_manifest.json",
    ],
}

REQUIRED_SCRIPT_FILES = [
    "scripts/audit/audit_dataset_integrity.py",
    "scripts/audit/load_paper1_datasets.py",
    "scripts/construct_mismatch/build_construct_mismatch_analysis.py",
    "scripts/construct_mismatch/length_robustness_checks.py",
    "scripts/raw_enrichment/enrich_normalized_steps_with_raw_fields.py",
    "scripts/intervention_risk/build_rich_detector_evaluation.py",
    "scripts/intervention_validation/check_intervention_feasibility.py",
    "scripts/intervention_validation/build_stop_counterfactual.py",
    "scripts/intervention_validation/build_prefix_branch_experiment.py",
    "scripts/intervention_validation/run_prefix_branch_experiment.py",
    "scripts/intervention_validation/analyze_prefix_branch_results.py",
    "scripts/live_intervention/check_live_replay_feasibility.py",
    "scripts/live_intervention/select_prefix_branch_sample.py",
    "scripts/live_intervention/run_prefix_branch_experiment.py",
    "scripts/live_intervention/analyze_prefix_branch_results.py",
    "scripts/api_replan_intervention/check_deepseek_api_env.py",
    "scripts/api_replan_intervention/build_api_replan_prefix_dataset.py",
    "scripts/api_replan_intervention/run_deepseek_api_replan_experiment.py",
    "scripts/api_replan_intervention/analyze_api_replan_results.py",
    "scripts/api_replan_intervention/build_api_replan_figures.py",
    "scripts/synthesis/extract_paper_findings.py",
    "scripts/synthesis/build_main_paper_tables.py",
    "scripts/synthesis/build_publication_figures.py",
    "scripts/release/build_paper1_artifact.py",
    "scripts/release/audit_and_test_paper1_artifact.py",
]

FORBIDDEN_TOKENS = {
    "annotator1", "annotator2", "rater1", "rater2", "independent", "pre_adjudication",
    "raw_annotation", "disagreement", "conflict", "negotiation", "private", "notes", "draft", "old",
    "backup", "pseudo", "shortlist", "review_packet", "manual_work", "raw_human", "human_raw",
}
ALLOWED_EXPLANATORY = {
    "README.md", "DATASET_CARD.md", "ARTIFACT_EVALUATION.md", "anonymization_and_release_notes.md",
    "release_exclusion_report.md", "reproducibility_audit_report.md", "artifact_release_final_report.md", "old_version_contamination_report.md",
    "docs/reproduction_guide.md", "docs/provenance.md", "docs/intervention_validation.md",
}
ALLOWED_TOKEN_PATHS = {
    "anonymization_and_release_notes.md",
    "old_version_contamination_report.md",
    "outputs/expected/live_guard_experiment/LIVE_GUARD_RELEASE_NOTES.md",
    "outputs/expected/live_guard_official_oracle/LIVE_GUARD_OFFICIAL_RELEASE_NOTES.md",
}

LOCAL_PATH_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9_])D:[\\/][^\s,;]*", re.I),
    re.compile(r"\\\\" + "wsl" + r"\.localhost[^\s,;]*", re.I),
    re.compile(r"/home/[A-Za-z0-9_.-]+[^\s,;]*"),
]
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[abprs]-[A-Za-z0-9-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{30,}", re.I),
]
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


@dataclass
class Result:
    check: str
    status: str
    path: str
    detail: str


def add(results: list[Result], check: str, status: str, path: str, detail: str) -> None:
    results.append(Result(check, status, path.replace("\\", "/"), detail))


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def token_hits(path: str) -> list[str]:
    parts = re.split(r"[^A-Za-z0-9]+", path.lower())
    joined = path.lower()
    hits = [t for t in FORBIDDEN_TOKENS if t in parts or ("_" in t and t in joined)]
    return sorted(set(hits))


def text_files(root: Path):
    suffixes = {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".py", ".yml", ".yaml", ".cff"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes and path.stat().st_size < 80_000_000:
            yield path


def check_structure(root: Path, results: list[Result]) -> None:
    for name in REQUIRED_ROOT:
        p = root / name
        add(results, "root_required", "PASS" if p.exists() and p.is_file() else "FAIL", name, "required root file")
    allowed_pools = set(POOLS)
    final_dir = root / "data" / "final_adjudicated"
    actual = {p.name for p in final_dir.iterdir() if p.is_dir()} if final_dir.exists() else set()
    for extra in sorted(actual - allowed_pools):
        add(results, "dataset_whitelist", "FAIL", f"data/final_adjudicated/{extra}", "unexpected dataset directory")
    for pool in sorted(allowed_pools):
        pool_dir = final_dir / pool
        if not pool_dir.exists():
            add(results, "dataset_whitelist", "FAIL", f"data/final_adjudicated/{pool}", "missing final pool directory")
            continue
        allowed_files = {"README.md", "step_labels.csv", "span_labels.csv"}
        actual_files = {p.name for p in pool_dir.iterdir() if p.is_file()}
        for fname in sorted(allowed_files):
            p = pool_dir / fname
            add(results, "dataset_files", "PASS" if p.exists() and p.stat().st_size > 0 else "FAIL", rel(p, root), "required nonempty pool file")
        for extra in sorted(actual_files - allowed_files):
            add(results, "dataset_files", "FAIL", rel(pool_dir / extra, root), "unexpected file in final pool directory")


def check_counts(root: Path, results: list[Result]) -> None:
    for pool, expected in POOLS.items():
        p = root / "data" / "final_adjudicated" / pool / "step_labels.csv"
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            tid_col = "public_trajectory_id" if "public_trajectory_id" in df.columns else None
            steps = len(df)
            traj = df[tid_col].nunique() if tid_col else None
            add(results, "pool_steps", "PASS" if steps == expected["steps"] else "FAIL", rel(p, root), f"steps={steps}, expected={expected['steps']}")
            add(results, "pool_trajectories", "PASS" if traj == expected["trajectories"] else "FAIL", rel(p, root), f"trajectories={traj}, expected={expected['trajectories']}")
            uc_col = "unproductive_cycle_flag" if "unproductive_cycle_flag" in df.columns else None
            if tid_col and uc_col:
                prev = df.groupby(tid_col)[uc_col].max().mean()
                status = "PASS" if abs(float(prev) - expected["uc_prev"]) < 0.0015 else "FAIL"
                add(results, "pool_uc_prevalence", status, rel(p, root), f"uc_prevalence={prev:.3f}, expected={expected['uc_prev']:.3f}")
            else:
                add(results, "pool_uc_prevalence", "WARN", rel(p, root), "not derivable from step labels")
        except Exception as exc:
            add(results, "pool_counts", "FAIL", rel(p, root), f"could not read: {exc}")


def check_forbidden_paths(root: Path, results: list[Result]) -> None:
    for p in root.rglob("*"):
        r = rel(p, root)
        if any(part in {"__pycache__", ".git", ".venv"} for part in Path(r).parts):
            add(results, "forbidden_cache", "FAIL", r, "cache or hidden environment directory included")
        if p.is_file() and p.suffix.lower() == ".zip":
            add(results, "forbidden_zip", "FAIL", r, "nested zip included")
        hits = token_hits(r)
        if hits and r not in ALLOWED_TOKEN_PATHS and not r.startswith("docs/") and not r.startswith("release_exclusion_report"):
            add(results, "forbidden_path_token", "FAIL", r, "forbidden path token(s): " + ", ".join(hits))


def check_text_privacy(root: Path, results: list[Result]) -> None:
    for p in text_files(root):
        r = rel(p, root)
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as exc:
            add(results, "text_scan", "WARN", r, f"could not read: {exc}")
            continue
        for i, line in enumerate(lines, 1):
            bad = []
            if any(pat.search(line) for pat in LOCAL_PATH_PATTERNS):
                bad.append("local path")
            if EMAIL_RE.search(line):
                bad.append("email-like string")
            if any(pat.search(line) for pat in SECRET_PATTERNS):
                bad.append("secret/token-like string")
            if bad:
                add(results, "privacy_text", "FAIL", f"{r}:{i}", "; ".join(bad) + " :: " + line[:160])
                break


def check_scripts_and_outputs(root: Path, results: list[Result]) -> None:
    for script in REQUIRED_SCRIPT_FILES:
        p = root / script
        add(results, "required_script", "PASS" if p.exists() and p.stat().st_size > 0 else "FAIL", script, "required script")
    for folder, files in REQUIRED_EXPECTED.items():
        for fname in files:
            p = root / folder / fname
            add(results, "required_expected_output", "PASS" if p.exists() and p.stat().st_size > 0 else "FAIL", f"{folder}/{fname}", "required expected output")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_manifest(root: Path, results: list[Result]) -> None:
    manifest = root / "manifest_sha256.csv"
    if not manifest.exists():
        add(results, "manifest", "FAIL", "manifest_sha256.csv", "missing manifest")
        return
    try:
        rows = list(csv.DictReader(manifest.open("r", encoding="utf-8-sig", newline="")))
    except Exception as exc:
        add(results, "manifest", "FAIL", "manifest_sha256.csv", f"cannot read: {exc}")
        return
    for row in rows:
        p = root / row.get("path", "")
        if not p.exists():
            add(results, "manifest_file", "FAIL", row.get("path", ""), "manifest entry missing from artifact")
            continue
        actual = sha256(p)
        expected = row.get("sha256", "")
        if actual != expected:
            add(results, "manifest_hash", "FAIL", row.get("path", ""), "sha256 mismatch")
    add(results, "manifest", "PASS", "manifest_sha256.csv", f"checked {len(rows)} entries")


def run_reproduction_test(root: Path, results: list[Result]) -> None:
    test = root / "tests" / "test_reproduce_key_outputs.py"
    if not test.exists():
        add(results, "reproduction_test", "FAIL", rel(test, root), "missing reproduction test")
        return
    proc = subprocess.run([sys.executable, str(test)], cwd=root, capture_output=True, text=True)
    status = "PASS" if proc.returncode == 0 else "FAIL"
    detail = (proc.stdout + proc.stderr).strip().replace("\n", " | ")[:1000]
    add(results, "reproduction_test", status, rel(test, root), detail or "completed")


def write_reports(root: Path, results: list[Result]) -> None:
    out_csv = root / "reproducibility_audit_results.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "path", "detail"])
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)
    counts = pd.DataFrame([r.__dict__ for r in results])["status"].value_counts().to_dict() if results else {}
    hard_failures = [r for r in results if r.status == "FAIL"]
    lines = [
        "# Reproducibility Audit Report",
        "",
        f"Artifact root: `{root.name}`",
        "",
        "## Summary",
        "",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- WARN: {counts.get('WARN', 0)}",
        f"- FAIL: {counts.get('FAIL', 0)}",
        f"- Release status: {'READY' if not hard_failures else 'NOT READY'}",
        "",
        "## Failures",
        "",
    ]
    if hard_failures:
        for r in hard_failures[:200]:
            lines.append(f"- **{r.check}** `{r.path}`: {r.detail}")
    else:
        lines.append("No hard failures detected.")
    lines.extend(["", "## Notes", "", "Warnings and full machine-readable results are in `reproducibility_audit_results.csv`."])
    (root / "reproducibility_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict audit for the LoopSense Paper 1 release artifact.")
    parser.add_argument("--artifact-root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-reproduction-test", action="store_true")
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    results: list[Result] = []
    check_structure(root, results)
    check_counts(root, results)
    check_forbidden_paths(root, results)
    check_text_privacy(root, results)
    check_scripts_and_outputs(root, results)
    check_manifest(root, results)
    if not args.skip_reproduction_test:
        run_reproduction_test(root, results)
    write_reports(root, results)
    failures = [r for r in results if r.status == "FAIL"]
    print(f"Audit complete: {len(failures)} failure(s). Report: {root / 'reproducibility_audit_report.md'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
