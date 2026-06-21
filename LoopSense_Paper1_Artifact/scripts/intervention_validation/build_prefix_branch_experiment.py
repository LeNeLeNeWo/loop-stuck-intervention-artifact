from __future__ import annotations

import argparse
from pathlib import Path


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a prefix-branch sample if live rerun artifacts are available.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_intervention_validation"))
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "prefix_branch_feasibility_only.md"
    report.write_text(
        "# Prefix-Branch Feasibility Only\n\n"
        "No live prefix-branch sample was built. The feasibility audit did not find the complete replay stack needed to restore prefix state: raw replayable trajectories, task metadata, repository checkouts, agent runner, model configuration, and tests/oracle harness.\n\n"
        "This script intentionally refuses to fabricate prefix samples without those artifacts.\n",
        encoding="utf-8",
    )
    print(f"Live prefix-branch sample not built; wrote {report}")


if __name__ == "__main__":
    main()
