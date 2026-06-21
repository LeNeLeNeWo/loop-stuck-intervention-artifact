from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from live_guard_common import ensure_output, redactor, repo_root


FILES_TO_COPY = [
    "LIVE_GUARD_PREREGISTRATION.md",
    "live_stack_check.md",
    "runner_install_report.md",
    "live_task_sample_sanitized.csv",
    "live_guard_run_plan.json",
    "live_guard_results.csv",
    "live_guard_summary_by_arm.csv",
    "live_guard_pairwise_effects.csv",
    "live_guard_report.md",
    "fig6_live_guard_policy.pdf",
    "fig6_live_guard_policy.png",
    "fig6_live_guard_policy.svg",
    "table_live_guard_summary.csv",
    "table_live_guard_pairwise_effects.csv",
    "LIVE_GUARD_FINAL_REPORT.md",
]


def copy_text_sanitized(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8", errors="replace")
    dst.write_text(redactor()(text), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy sanitized live guard outputs into release artifact.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)
    artifact = args.artifact_root or (root / "release_artifacts/LoopSense_Paper1_Artifact")
    dest = artifact / "outputs/expected/live_guard_experiment"
    dest.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in FILES_TO_COPY:
        src = out / name
        if not src.exists():
            continue
        dst = dest / name
        if src.suffix.lower() in {".png", ".pdf"}:
            shutil.copy2(src, dst)
        else:
            copy_text_sanitized(src, dst)
        copied.append(name)

    run_log = out / "live_guard_run_log.csv"
    if run_log.exists():
        df = pd.read_csv(run_log)
        df = df.drop(columns=[c for c in ["raw_log_path_local", "error_summary"] if c in df.columns])
        df.to_csv(dest / "live_guard_run_log_sanitized.csv", index=False)
        copied.append("live_guard_run_log_sanitized.csv")

    notes = [
        "# Live Guard Release Notes",
        "",
        "This directory contains sanitized outputs for the `paper1_live_guard_experiment`.",
        "",
        "- Raw trajectory logs are not included because they can contain full model context, task text, local paths, and tool outputs.",
        "- API keys and endpoint values are not printed or released.",
        "- Task identifiers are represented by released IDs in public-facing files where possible.",
        "- If `oracle_available` is false, these outputs support tool-level live policy observations only, not task-success claims.",
        "",
        "Copied files:",
    ]
    notes.extend(f"- `{name}`" for name in copied)
    (dest / "LIVE_GUARD_RELEASE_NOTES.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    script_dest = artifact / "scripts/live_guard_experiment"
    script_dest.mkdir(parents=True, exist_ok=True)
    for script in (root / "scripts/paper1_live_guard_experiment").glob("*.py"):
        copy_text_sanitized(script, script_dest / script.name)

    print(f"Artifact live guard outputs updated: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
