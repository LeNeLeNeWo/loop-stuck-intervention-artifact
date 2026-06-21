from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from official_common import copy_sanitized, ensure_output, redactor, repo_root


ARTIFACT_FILES = [
    "OFFICIAL_ORACLE_PREREGISTRATION.md",
    "oracle_install_report.md",
    "extend_to_60_sample_validation.md",
    "official_oracle_task_sample_sanitized.csv",
    "official_oracle_run_plan.json",
    "official_oracle_run_plan_60.json",
    "official_oracle_extension_tasks_31_60.csv",
    "agent_run_log_sanitized.csv",
    "patch_manifest_sanitized.csv",
    "oracle_eval_results.csv",
    "live_guard_official_results_sanitized.csv",
    "live_guard_official_summary_by_arm.csv",
    "live_guard_official_pairwise_effects.csv",
    "live_guard_official_report.md",
    "fig6_live_guard_official_oracle.pdf",
    "fig6_live_guard_official_oracle.png",
    "fig6_live_guard_official_oracle.svg",
    "table_live_guard_official_summary.csv",
    "table_live_guard_official_pairwise_effects.csv",
    "LIVE_GUARD_OFFICIAL_FINAL_REPORT.md",
    "LIVE_GUARD_60TASK_FINAL_REPORT.md",
]


def make_sanitized(out: Path) -> None:
    path_columns = ["raw_log_path_local", "patch_path_local", "error_summary"]
    run_log = out / "agent_run_log.csv"
    if run_log.exists():
        df = pd.read_csv(run_log)
        df = df.drop(columns=[c for c in path_columns if c in df.columns])
        df.to_csv(out / "agent_run_log_sanitized.csv", index=False)
    patch = out / "patch_manifest.csv"
    if patch.exists():
        df = pd.read_csv(patch)
        df = df.drop(columns=[c for c in ["patch_path_local"] if c in df.columns])
        df.to_csv(out / "patch_manifest_sanitized.csv", index=False)
    results = out / "live_guard_official_results.csv"
    if results.exists():
        df = pd.read_csv(results)
        df = df.drop(columns=[c for c in path_columns if c in df.columns])
        df.to_csv(out / "live_guard_official_results_sanitized.csv", index=False)


def paper_threshold(out: Path) -> bool:
    p = out / "live_guard_official_summary_by_arm.csv"
    if not p.exists():
        return False
    df = pd.read_csv(p)
    vals = {}
    for _, row in df.iterrows():
        if "N_official_oracle_outcomes" in row and pd.notna(row.get("N_official_oracle_outcomes")):
            vals[row["arm"]] = int(row.get("N_official_oracle_outcomes") or 0)
        else:
            vals[row["arm"]] = int(row.get("N_oracle_evaluated") or 0)
    return vals.get("NO_GUARD", 0) >= 60 and vals.get("WARN_REPLAN_GUARD", 0) >= 60


def update_docs(artifact: Path, integrated: bool) -> None:
    readme = artifact / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8", errors="ignore")
        if "official-oracle live guard outputs" not in text.lower():
            text += (
                "\n## Official-oracle live guard outputs\n"
                "`outputs/expected/live_guard_official_oracle/` contains sanitized outputs for the 60-task official-oracle live guard experiment. "
                "The current manuscript uses the 60-task official outcomes; empty-patch rows are reported separately and counted as unresolved.\n"
            )
            readme.write_text(text, encoding="utf-8")
    doc = artifact / "docs/intervention_validation.md"
    if doc.exists():
        text = doc.read_text(encoding="utf-8", errors="ignore")
        if "Official-oracle live guard experiment" not in text:
            text += (
                "\n## Official-oracle live guard experiment\n"
                "`outputs/expected/live_guard_official_oracle/` contains preregistered official-oracle live guard outputs. "
                + (
                    "The manuscript was updated because the preregistered oracle threshold was met.\n"
                    if integrated
                    else "The manuscript was not updated with live task-success claims because the preregistered oracle threshold was not met.\n"
                )
            )
            doc.write_text(text, encoding="utf-8")
    guide = artifact / "docs/reproduction_guide.md"
    if guide.exists():
        text = guide.read_text(encoding="utf-8", errors="ignore")
        if "live_guard_official_oracle" not in text:
            text += (
                "\n## Official-oracle live guard rerun\n"
                "Rerunning `scripts/live_guard_official_oracle/` requires Docker, public SWE-bench downloads, a SWE-bench harness, and user-provided DeepSeek credentials. Raw run logs and benchmark workspaces are not included in the release artifact.\n"
            )
            guide.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Update artifact and optionally paper for official-oracle live guard experiment.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)
    artifact = args.artifact_root or (root / "release_artifacts/LoopSense_Paper1_Artifact")
    dest = artifact / "outputs/expected/live_guard_official_oracle"
    dest.mkdir(parents=True, exist_ok=True)
    make_sanitized(out)
    for name in ARTIFACT_FILES:
        src = out / name
        if src.exists():
            copy_sanitized(src, dest / name)
    notes = [
        "# Official-Oracle Live Guard Release Notes",
        "",
        "- Raw live trajectories, raw repo workspaces, Docker caches, and API credentials are not included.",
        "- Patch files are not included by default; sanitized patch manifests and oracle summaries are included.",
        "- Manuscript task-success claims require the preregistered oracle threshold.",
    ]
    (dest / "LIVE_GUARD_OFFICIAL_RELEASE_NOTES.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    script_dest = artifact / "scripts/live_guard_official_oracle"
    script_dest.mkdir(parents=True, exist_ok=True)
    for script in (root / "scripts/paper1_live_guard_official_oracle").glob("*.py"):
        copy_sanitized(script, script_dest / script.name)
    integrated = paper_threshold(out)
    (out / "paper_integration_decision.md").write_text(
        f"# Paper Integration Decision\n\nIntegrated into paper: **{'yes' if integrated else 'no'}**\n\n"
        + ("The 60-task official-oracle threshold was met.\n" if integrated else "The 60-task official-oracle threshold was not met; no live task-success claim was added.\n"),
        encoding="utf-8",
    )
    update_docs(artifact, integrated)
    print(f"Official-oracle artifact updated: {dest}")
    print(f"Paper integration threshold met: {'yes' if integrated else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
