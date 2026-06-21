from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from official_common import ensure_output, repo_root


LABELS = {"NO_GUARD": "No Guard", "WARN_REPLAN_GUARD": "Warn/Replan", "HARD_STOP_GUARD": "Hard Stop"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build official-oracle live guard figure.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-tasks", type=int, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)
    path = out / "live_guard_official_summary_by_arm.csv"
    if not path.exists():
        print("Missing summary CSV.")
        return 2
    df = pd.read_csv(path)
    if df.empty:
        print("Empty summary.")
        return 2
    arms = list(df["arm"])
    labels = [LABELS.get(a, a) for a in arms]
    x = range(len(arms))
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.4))
    axes = axes.flatten()
    colors = ["#4c78a8", "#f58518", "#54a24b"][: len(arms)]
    resolved = pd.to_numeric(df.get("resolved_rate", 0), errors="coerce").fillna(0)
    axes[0].bar(x, resolved, color=colors)
    axes[0].set_title("Resolved/test-pass")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[0].set_ylabel("Rate")
    patch = pd.to_numeric(df.get("patch_generated_rate", 0), errors="coerce").fillna(0)
    axes[1].bar(x, patch, color=colors)
    axes[1].set_title("Patch generated")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[2].bar(x, pd.to_numeric(df["mean_steps"], errors="coerce"), color=colors)
    axes[2].set_title("Mean steps")
    axes[2].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[3].bar(x, pd.to_numeric(df["mean_tokens"], errors="coerce"), color=colors)
    axes[3].set_title("Mean tokens")
    axes[3].set_xticks(list(x), labels, rotation=20, ha="right")
    repetition = pd.to_numeric(df["mean_post_trigger_repetition_rate"], errors="coerce")
    axes[4].bar(x, repetition, color=colors)
    axes[4].set_title("Post-trigger repetition")
    axes[4].set_ylim(0, max(0.1, float(repetition.max()) * 1.2))
    axes[4].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[4].set_ylabel("Rate")
    axes[5].axis("off")
    title_n = f" on {args.n_tasks} frozen SWE-bench Verified tasks" if args.n_tasks else ""
    fig.suptitle(f"Official-oracle live guard experiment{title_n}")
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(out / f"fig6_live_guard_official_oracle.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"Figure: {out / 'fig6_live_guard_official_oracle.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
