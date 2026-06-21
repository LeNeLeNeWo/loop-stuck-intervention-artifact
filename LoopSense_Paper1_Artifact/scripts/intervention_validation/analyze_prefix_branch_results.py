from __future__ import annotations

import argparse
from pathlib import Path


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze live prefix-branch results if they exist.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_intervention_validation"))
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    results = out_dir / "prefix_branch_results.csv"
    if not results.exists():
        report = out_dir / "prefix_branch_feasibility_only.md"
        if not report.exists():
            report.write_text("# Prefix-Branch Feasibility Only\n\nNo live prefix-branch results exist to analyze.\n", encoding="utf-8")
        print("No live prefix-branch results found. Analysis skipped; see prefix_branch_feasibility_only.md.")
        return
    raise SystemExit("prefix_branch_results.csv exists, but live-result analysis is intentionally not implemented until the replay schema is available.")


if __name__ == "__main__":
    main()
