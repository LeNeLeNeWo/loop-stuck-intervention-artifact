from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from live_guard_common import ensure_output, repo_root


LABELS = {
    "NO_GUARD": "No Guard",
    "WARN_REPLAN_GUARD": "Warn/Replan",
    "HARD_STOP_GUARD": "Hard Stop",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build live guard figures.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)
    summary_path = out / "live_guard_summary_by_arm.csv"
    if not summary_path.exists():
        print("Missing live_guard_summary_by_arm.csv")
        return 2
    df = pd.read_csv(summary_path)
    if df.empty:
        print("No summary rows.")
        return 2
    arms = list(df["arm"])
    x = range(len(arms))
    labels = [LABELS.get(a, a) for a in arms]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    axes[0].bar(x, df["patch_generated_rate"], color=["#4c78a8", "#f58518", "#54a24b"][: len(arms)])
    axes[0].set_title("Patch generated")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[0].set_ylabel("Rate")
    axes[1].bar(x, df["mean_steps"], color=["#4c78a8", "#f58518", "#54a24b"][: len(arms)])
    axes[1].set_title("Mean steps")
    axes[1].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[2].bar(x, df["mean_repeated_file_rate"], color=["#4c78a8", "#f58518", "#54a24b"][: len(arms)])
    axes[2].set_title("Repeated-file rate")
    axes[2].set_ylim(0, max(0.1, float(df["mean_repeated_file_rate"].max()) * 1.2))
    axes[2].set_xticks(list(x), labels, rotation=20, ha="right")
    fig.suptitle("Live guard-policy experiment")
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(out / f"fig6_live_guard_policy.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"Figure written to {out / 'fig6_live_guard_policy.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
