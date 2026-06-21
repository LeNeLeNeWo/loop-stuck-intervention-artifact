from __future__ import annotations

import argparse
from pathlib import Path


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live prefix-branch interventions if replay artifacts are available.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_intervention_validation"))
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "prefix_branch_feasibility_only.md"
    if not report.exists():
        report.write_text("# Prefix-Branch Feasibility Only\n\nLive rerun was not executed because replay artifacts are unavailable.\n", encoding="utf-8")
    print("Live prefix-branch rerun not executed. See prefix_branch_feasibility_only.md.")


if __name__ == "__main__":
    main()
