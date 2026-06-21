from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_SUBDIR = Path("outputs/paper1_api_replan_intervention")
GROUP_ORDER = ["productive_risk", "uc", "matched_control"]
GROUP_LABEL = {
    "productive_risk": "PI/HN\nproductive-risk",
    "uc": "UC",
    "matched_control": "Control",
}
ARM_LABEL = {"NEUTRAL_CONTINUE": "Neutral", "WARN_REPLAN": "Warn/replan"}
ARM_COLOR = {"NEUTRAL_CONTINUE": "#7a8793", "WARN_REPLAN": "#1f77b4"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def bar_panel(ax, summary: pd.DataFrame, metric: str, ylabel: str, title: str, percent: bool = True) -> None:
    x = np.arange(len(GROUP_ORDER))
    width = 0.34
    for i, arm in enumerate(["NEUTRAL_CONTINUE", "WARN_REPLAN"]):
        vals = []
        for group in GROUP_ORDER:
            row = summary[(summary["stratum"] == group) & (summary["arm"] == arm)]
            vals.append(float(row[metric].iloc[0]) if not row.empty and pd.notna(row[metric].iloc[0]) else np.nan)
        xpos = x + (-width / 2 if arm == "NEUTRAL_CONTINUE" else width / 2)
        bars = ax.bar(xpos, vals, width=width, label=ARM_LABEL[arm], color=ARM_COLOR[arm], alpha=0.9)
        for rect, val in zip(bars, vals):
            if np.isnan(val):
                continue
            label = f"{val*100:.0f}%" if percent else f"{val:.2f}"
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + (0.02 if percent else 0.05), label, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABEL[g] for g in GROUP_ORDER], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    if percent:
        ax.set_ylim(0, 1.05)
    else:
        ax.set_ylim(0, 3.25)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build API-assisted prefix replan intervention figures.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = repo_root()
    out = args.output_dir or root / OUTPUT_SUBDIR
    summary_path = out / "api_replan_summary_by_group.csv"
    effects_path = out / "api_replan_pairwise_effects.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    effects = pd.read_csv(effects_path) if effects_path.exists() else pd.DataFrame()

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    bar_panel(axes[0], summary, "repeat_pattern_rate", "Rate", "A. Immediate repeated-pattern plan", percent=True)
    bar_panel(axes[1], summary, "progress_seeking_score_mean", "Score (0-3)", "B. Progress-seeking score", percent=False)
    bar_panel(axes[2], summary, "continue_rate", "Rate", "C. Continue decision", percent=True)

    prod_warn = summary[(summary["stratum"] == "productive_risk") & (summary["arm"] == "WARN_REPLAN")]
    if not prod_warn.empty and pd.notna(prod_warn["unsafe_overstop_rate"].iloc[0]):
        axes[2].text(
            0.02,
            0.08,
            f"PI/HN unsafe over-stop: {100*float(prod_warn['unsafe_overstop_rate'].iloc[0]):.1f}%",
            transform=axes[2].transAxes,
            fontsize=7.5,
            color="#7a2f2f",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fff5f5", "edgecolor": "#e0b0b0", "linewidth": 0.6},
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("API-Assisted Warn/Replan Intervention on Real Debugging Prefixes", fontsize=11, weight="bold", y=1.08)

    for suffix in ["pdf", "png", "svg"]:
        fig.savefig(out / f"fig6_api_replan_intervention.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if not effects.empty:
        effects.to_csv(out / "table_api_replan_pairwise_effects.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "table_api_replan_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Figure written: {out / 'fig6_api_replan_intervention.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
