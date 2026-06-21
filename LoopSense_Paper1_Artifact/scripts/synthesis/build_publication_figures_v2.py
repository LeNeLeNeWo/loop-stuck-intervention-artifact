from __future__ import annotations

import argparse
import math
import re
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd


DATASET_ORDER = ["Natural500", "Final500", "StressFresh100", "OpenHandsExternal330"]
DATASET_LABEL = {
    "Natural500": "Natural500",
    "Final500": "Final500",
    "StressFresh100": "StressFresh100",
    "OpenHandsExternal330": "OpenHands\nExternal330",
}
DATASET_COLOR = {
    "Natural500": "#0072B2",
    "Final500": "#009E73",
    "StressFresh100": "#E69F00",
    "OpenHandsExternal330": "#CC79A7",
}
OKABE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#8A8F98",
    "light_gray": "#E9ECEF",
}
CONSTRUCT_COLOR = {
    "HN-only": "#0072B2",
    "UC-only": "#E69F00",
    "HN+UC mixed": "#CC79A7",
    "Neither": "#8A8F98",
}
CONSTRUCT_HATCH = {
    "HN-only": "",
    "UC-only": "///",
    "HN+UC mixed": "\\\\\\",
    "Neither": "...",
}
FAMILY_MARKERS = {
    "max_step_guard": "o",
    "file_revisit_guard": "s",
    "exact_action_repeat_k": "^",
    "tool_repeat_k": "D",
    "tool_sequence_repeat": "v",
    "action_observation_repeat": "P",
    "action_observation_similarity": "X",
    "error_signature_repeat_k": "h",
    "repeat_without_progress_guard": "*",
}
FAMILY_LABEL = {
    "max_step_guard": "max-step",
    "file_revisit_guard": "file-revisit",
    "exact_action_repeat_k": "exact-action",
    "tool_repeat_k": "tool-repeat",
    "tool_sequence_repeat": "tool-seq",
    "action_observation_repeat": "action-observation",
    "action_observation_similarity": "action-observation sim",
    "error_signature_repeat_k": "error-signature",
    "repeat_without_progress_guard": "progress-aware",
}

PAPER_REPO_DEFAULT = Path(
    r"<WSL_PATH>\Ubuntu\home\yang\ICSE 2027\LoopBench-Agent-ICSE2027-anonymous"
)


@dataclass(frozen=True)
class SourceRecord:
    key: str
    requested: str
    actual: Path
    status: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.35,
            "lines.linewidth": 1.0,
        }
    )


def pct(value: Any, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def f3(value: Any) -> str:
    return f"{float(value):.3f}"


def count(value: Any) -> str:
    return f"{int(round(float(value))):,}"


def wrapped(text: str, width: int) -> str:
    return textwrap.fill(str(text), width=width, break_long_words=False)


def slug_rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_csv(root: Path, key: str, requested_rel: str) -> tuple[pd.DataFrame, SourceRecord]:
    requested = root / requested_rel
    if requested.exists():
        return pd.read_csv(requested), SourceRecord(key, requested_rel, requested, "preferred")

    requested_path = Path(requested_rel)
    search_dir = root / requested_path.parent
    if not search_dir.exists():
        search_dir = root / "outputs"

    stem = requested_path.stem.lower()
    candidates = sorted(search_dir.rglob("*.csv"))
    exactish = [p for p in candidates if stem in p.stem.lower()]
    if not exactish:
        tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if len(t) > 2]

        def score(path: Path) -> int:
            name = path.stem.lower()
            return sum(1 for token in tokens if token in name)

        scored = sorted(((score(p), p) for p in candidates), key=lambda item: (-item[0], str(item[1])))
        exactish = [p for s, p in scored if s > 0]

    if not exactish:
        raise FileNotFoundError(f"Could not find requested CSV or close fallback for {requested_rel}")

    actual = exactish[0]
    return pd.read_csv(actual), SourceRecord(key, requested_rel, actual, "fallback")


def ordered_by_dataset(df: pd.DataFrame, column: str) -> pd.DataFrame:
    order = {dataset: i for i, dataset in enumerate(DATASET_ORDER)}
    return (
        df.assign(_dataset_order=df[column].map(order).fillna(999))
        .sort_values(["_dataset_order", column])
        .drop(columns=["_dataset_order"])
        .reset_index(drop=True)
    )


def parse_count_rate(value: Any) -> tuple[int, float]:
    match = re.match(r"\s*(\d+)\s*\(([-+0-9.]+)\)\s*", str(value))
    if not match:
        raise ValueError(f"Expected 'count (rate)' cell, got {value!r}")
    return int(match.group(1)), float(match.group(2))


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.08", fc="#4B3F72", ec="none"),
        clip_on=False,
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"{stem}.{ext}"
        kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": 0.06,
            "facecolor": "white",
            "metadata": {"Creator": "build_publication_figures_v2.py"},
        }
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    return to_rgba(hex_color, alpha)


def add_ring_gauge(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    value: float,
    color: str,
    label: str,
    value_label: str,
    hatch: str | None = None,
) -> None:
    ax = fig.add_axes(rect)
    ax.set_aspect("equal")
    ax.set_axis_off()
    bg = patches.Wedge((0.5, 0.5), 0.48, 0, 360, width=0.14, facecolor="#ECEFF3", edgecolor="none")
    fg = patches.Wedge((0.5, 0.5), 0.48, 90, 90 - 360 * value, width=0.14, facecolor=color, edgecolor="white", linewidth=0.35)
    if hatch:
        fg.set_hatch(hatch)
    ax.add_patch(bg)
    ax.add_patch(fg)
    ax.text(0.5, 0.56, value_label, ha="center", va="center", fontsize=7.2, fontweight="bold", color="#111111")
    ax.text(0.5, 0.33, label, ha="center", va="center", fontsize=6.7, color="#4B5563")


def draw_mosaic_tile(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    hatch: str,
    label: str,
    count_value: int,
    rate: float,
    text_color: str = "#111111",
) -> None:
    if w <= 0 or h <= 0:
        return
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.002,rounding_size=0.008",
            facecolor=rgba(color, 0.82),
            edgecolor="white",
            linewidth=0.9,
            hatch=hatch,
        )
    )
    area = w * h
    if area > 0.012:
        ax.text(
            x + w / 2,
            y + h / 2,
            f"{label}\n{pct(rate)}\n({count_value})",
            ha="center",
            va="center",
            fontsize=6.6,
            color=text_color,
            fontweight="bold" if rate >= 0.25 else "normal",
            linespacing=0.96,
        )
    elif area > 0.0045:
        ax.text(
            x + w / 2,
            y + h / 2,
            f"{pct(rate)}\n({count_value})",
            ha="center",
            va="center",
            fontsize=5.6,
            color=text_color,
            linespacing=0.92,
        )


def draw_fig1(table1: pd.DataFrame, out_dir: Path) -> list[Path]:
    table1 = ordered_by_dataset(table1, "Evidence pool")
    fig = plt.figure(figsize=(7.1, 4.15), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.text(0.5, 0.978, "Evidence Pool Atlas", ha="center", va="top", fontsize=11.4, fontweight="bold")
    fig.text(
        0.5,
        0.938,
        "Distinct roles, lengths, and PI/HN/UC prevalence across four frozen evidence pools.",
        ha="center",
        va="top",
        fontsize=7.8,
        color="#4B5563",
    )

    role_badge = {
        "Natural500": "natural control",
        "Final500": "enriched difficult",
        "StressFresh100": "stress challenge",
        "OpenHandsExternal330": "external horizon",
    }
    marker_map = {"PI": "o", "HN": "s", "UC": "^"}
    linestyle_map = {"PI": "-", "HN": "--", "UC": ":"}
    construct_columns = [
        ("PI", "PI prevalence"),
        ("HN", "HN prevalence"),
        ("UC", "UC prevalence"),
    ]
    row_y = {
        "Natural500": 0.690,
        "Final500": 0.510,
        "StressFresh100": 0.330,
        "OpenHandsExternal330": 0.150,
    }
    row_h = 0.150
    row_x = 0.058
    row_w = 0.884
    metric_x = [0.368, 0.468, 0.568]
    metric_labels = ["Traj.", "Steps", "Avg."]
    track_x0, track_x1 = 0.665, 0.890

    ax.text(row_x + 0.010, 0.862, "POOL AND SAMPLING ROLE", ha="left", va="center", fontsize=6.2, color="#6B7280", fontweight="bold")
    ax.text(0.348, 0.862, "SIZE", ha="left", va="center", fontsize=6.2, color="#6B7280", fontweight="bold")
    ax.text(track_x0 - 0.050, 0.862, "CONSTRUCT PREVALENCE", ha="left", va="center", fontsize=6.2, color="#6B7280", fontweight="bold")
    ax.plot([row_x, row_x + row_w], [0.846, 0.846], color="#DDE3EA", lw=0.65)

    for idx, dataset in enumerate(DATASET_ORDER):
        row = table1.loc[table1["Evidence pool"].eq(dataset)].iloc[0]
        color = DATASET_COLOR[dataset]
        y = row_y[dataset]
        bg = "#FFFFFF" if idx % 2 == 0 else "#FBFCFD"
        ax.add_patch(
            patches.FancyBboxPatch(
                (row_x, y),
                row_w,
                row_h,
                boxstyle="round,pad=0.006,rounding_size=0.010",
                fc=bg,
                ec="#DDE3EA",
                lw=0.55,
            )
        )
        ax.add_patch(patches.Rectangle((row_x, y), 0.006, row_h, facecolor=color, edgecolor="none", alpha=0.88))
        ax.plot([0.330, 0.330], [y + 0.018, y + row_h - 0.018], color="#E5EAF0", lw=0.50)
        ax.plot([0.625, 0.625], [y + 0.018, y + row_h - 0.018], color="#E5EAF0", lw=0.50)

        name_size = 8.8 if dataset == "OpenHandsExternal330" else 9.7
        ax.text(row_x + 0.025, y + row_h - 0.038, dataset, ha="left", va="center", fontsize=name_size, fontweight="bold", color=color)
        ax.text(
            row_x + 0.025,
            y + row_h - 0.069,
            role_badge[dataset],
            ha="left",
            va="center",
            fontsize=6.1,
            color=color,
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.04", fc=rgba(color, 0.07), ec=color, lw=0.45),
        )
        ax.text(row_x + 0.025, y + 0.016, wrapped(row["Main use"], 42), ha="left", va="bottom", fontsize=5.6, color="#374151", linespacing=1.02)

        metrics = [
            count(row["Trajectories"]),
            count(row["Steps"]),
            f"{float(row['Avg. steps']):.3f}" if dataset == "OpenHandsExternal330" else f"{float(row['Avg. steps']):.1f}",
        ]
        for k, (label, value) in enumerate(zip(metric_labels, metrics)):
            weight = "bold" if dataset == "OpenHandsExternal330" and label == "Avg." else "normal"
            ax.text(metric_x[k], y + row_h - 0.055, value, ha="center", va="center", fontsize=8.2, color="#111111", fontweight=weight)
            ax.text(metric_x[k], y + 0.047, label, ha="center", va="center", fontsize=5.9, color="#6B7280")
        if dataset == "OpenHandsExternal330":
            ax.text(
                0.500,
                y + 0.022,
                f"{float(row['Avg. steps']):.3f} steps, UC {pct(row['UC prevalence'])}",
                ha="center",
                va="center",
                fontsize=5.9,
                color=color,
                bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.04", fc=rgba(color, 0.06), ec=rgba(color, 0.65), lw=0.45),
            )

        for k, (short, column) in enumerate(construct_columns):
            value = float(row[column])
            yy = y + row_h - 0.038 - k * 0.040
            ax.plot([track_x0, track_x1], [yy, yy], color="#E5EAF0", lw=1.05, solid_capstyle="round")
            ax.plot(
                [track_x0, track_x0 + (track_x1 - track_x0) * value],
                [yy, yy],
                color=color,
                lw=1.25,
                alpha=0.90,
                linestyle=linestyle_map[short],
                solid_capstyle="round",
            )
            xv = track_x0 + (track_x1 - track_x0) * value
            ax.scatter([xv], [yy], s=26, marker=marker_map[short], facecolor="#FFFFFF", edgecolor=color, linewidth=0.95, zorder=4)
            ax.text(track_x0 - 0.026, yy, short, ha="right", va="center", fontsize=5.9, color="#4B5563", fontweight="bold")
            ax.text(track_x1 + 0.014, yy, pct(value), ha="left", va="center", fontsize=6.0, color="#111111")

    ax.text(
        row_x,
        0.070,
        "Design implication:",
        ha="left",
        va="center",
        fontsize=6.9,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        row_x + 0.150,
        0.070,
        "pool separation is part of construct validity; roles, lengths, and PI/HN/UC prevalence differ sharply.",
        ha="left",
        va="center",
        fontsize=6.9,
        color="#374151",
    )
    return save_figure(fig, out_dir, "fig1")


def draw_fig2(
    length_rows: pd.DataFrame,
    table1: pd.DataFrame,
    robustness: pd.DataFrame,
    within: pd.DataFrame,
    excluding_openhands: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    fig = plt.figure(figsize=(7.1, 4.60), facecolor="white")
    fig.text(0.5, 0.975, "Length Is Distribution-Sensitive", ha="center", va="top", fontsize=11.5, fontweight="bold")
    fig.text(
        0.5,
        0.94,
        "Trajectory length is informative in some pools, but not a standalone UC proxy.",
        ha="center",
        va="top",
        fontsize=8.4,
        style="italic",
        color="#333333",
    )
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.985, top=0.845, bottom=0.128, width_ratios=[1.10, 1.0], wspace=0.31)
    ax_a = fig.add_subplot(gs[0, 0])
    right = gs[0, 1].subgridspec(2, 1, height_ratios=[0.98, 1.03], hspace=0.62)
    ax_b = fig.add_subplot(right[0])
    ax_in = fig.add_subplot(right[1])

    panel_label(ax_a, "A", x=-0.13, y=1.04)
    panel_label(ax_b, "B", x=-0.10, y=1.12)

    positions = np.arange(1, len(DATASET_ORDER) + 1)
    data_by_dataset = [length_rows.loc[length_rows["dataset"].eq(ds), "trajectory_length"].astype(float).to_numpy() for ds in DATASET_ORDER]
    violins = ax_a.violinplot(
        data_by_dataset,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, ds in zip(violins["bodies"], DATASET_ORDER):
        body.set_facecolor(DATASET_COLOR[ds])
        body.set_edgecolor(DATASET_COLOR[ds])
        body.set_linewidth(0.70)
        body.set_alpha(0.16)

    for pos, ds in zip(positions, DATASET_ORDER):
        sub = length_rows.loc[length_rows["dataset"].eq(ds)].copy()
        values = sub["trajectory_length"].astype(float)
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        p10, p90 = np.percentile(values, [10, 90])
        ax_a.plot([pos, pos], [p10, p90], color=rgba(DATASET_COLOR[ds], 0.42), lw=0.85, solid_capstyle="round", zorder=2)
        ax_a.plot([pos, pos], [q1, q3], color=DATASET_COLOR[ds], lw=2.8, alpha=0.88, solid_capstyle="round", zorder=3)
        ax_a.plot([pos - 0.145, pos + 0.145], [med, med], color="#111111", lw=0.90, solid_capstyle="round", zorder=4)

        stats = table1.loc[table1["Evidence pool"].eq(ds)].iloc[0]
        mean_label = f"{float(stats['Avg. steps']):.1f}"
        ax_a.text(
            pos,
            288,
            f"N={int(stats['Trajectories'])}\nmean={mean_label}\nUC={pct(stats['UC prevalence'])}",
            ha="center",
            va="top",
            fontsize=5.7,
            color=DATASET_COLOR[ds],
            fontweight="bold" if ds == "OpenHandsExternal330" else "normal",
            bbox=dict(boxstyle="round,pad=0.16,rounding_size=0.03", fc="#FFFFFF", ec=rgba(DATASET_COLOR[ds], 0.25), lw=0.35),
        )

    ax_a.set_yscale("log")
    ax_a.set_ylim(0.75, 360)
    ax_a.set_xticks(positions)
    ax_a.set_xticklabels([DATASET_LABEL[ds] for ds in DATASET_ORDER])
    ax_a.set_ylabel("Trajectory length (steps)")
    ax_a.set_title("Length distributions and construct prevalence", loc="left", pad=8, fontsize=8.8, fontweight="bold")
    ax_a.grid(axis="y", which="major", color="#D1D5DB", linestyle="-", alpha=0.8)
    ax_a.grid(axis="y", which="minor", color="#ECEFF3", linestyle="-", alpha=0.55)
    ax_a.spines[["top", "right"]].set_visible(False)
    range_handle = mlines.Line2D([], [], color=OKABE["gray"], linestyle="-", lw=0.85, alpha=0.65, label="10-90% range")
    q_handle = mlines.Line2D([], [], color="#111111", linestyle="-", lw=1.0, label="median; thick line = IQR")
    ax_a.legend(handles=[q_handle, range_handle], loc="lower left", frameon=False, ncol=1, borderpad=0.2, handletextpad=0.45)

    global_top = robustness.loc[robustness["comparison"].eq("global_top_10pct_longest")].iloc[0]
    global_rest = robustness.loc[robustness["comparison"].eq("global_remaining_90pct")].iloc[0]
    excl_top = excluding_openhands.loc[excluding_openhands["comparison"].eq("excluding_openhands_top_10pct_longest")].iloc[0]
    excl_rest = excluding_openhands.loc[excluding_openhands["comparison"].eq("excluding_openhands_remaining_90pct")].iloc[0]

    top_color = OKABE["vermillion"]
    rest_color = OKABE["sky"]
    global_rows = [
        ("Global\n(all pools)", 100 * float(global_top["contains_uc_pct"]), 100 * float(global_rest["contains_uc_pct"])),
        ("Excluding\nOpenHands", 100 * float(excl_top["contains_uc_pct"]), 100 * float(excl_rest["contains_uc_pct"])),
    ]
    yvals = np.arange(len(global_rows))[::-1]
    for y, (label, top_v, rest_v) in zip(yvals, global_rows):
        lo, hi = sorted([top_v, rest_v])
        ax_b.plot([lo, hi], [y, y], color="#9CA3AF", lw=1.15, zorder=1)
        ax_b.scatter([rest_v], [y], s=54, marker="o", color=rest_color, edgecolor="#1D4ED8", linewidth=0.55, zorder=3, label="remaining 90%" if y == yvals[0] else None)
        ax_b.scatter([top_v], [y], s=62, marker="D", color=top_color, edgecolor="#7F1D1D", linewidth=0.55, zorder=4, label="longest decile" if y == yvals[0] else None)
        ax_b.text(top_v + (3.0 if top_v <= rest_v else -3.0), y + 0.18, f"{top_v:.1f}%", ha="left" if top_v <= rest_v else "right", va="bottom", fontsize=7.0, color="#111111", fontweight="bold")
        ax_b.text(rest_v + (3.0 if rest_v < 86 else -3.0), y - 0.18, f"{rest_v:.1f}%", ha="left" if rest_v < 86 else "right", va="top", fontsize=7.0, color="#111111")
    ax_b.set_yticks(yvals)
    ax_b.set_yticklabels([row[0] for row in global_rows])
    ax_b.set_xlim(0, 103)
    ax_b.set_ylim(-0.65, 1.55)
    ax_b.set_xlabel("UC prevalence (%)")
    ax_b.set_title("Longest-decile robustness", loc="left", pad=8, fontsize=9.0, fontweight="bold")
    ax_b.grid(axis="x", color="#D9DDE2", alpha=0.75)
    ax_b.spines[["top", "right", "left"]].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    ax_b.legend(loc="upper left", bbox_to_anchor=(0.0, 1.04), frameon=False, ncol=2, borderaxespad=0.0, columnspacing=1.0)
    ax_b.text(
        0.00,
        1.205,
        "Removing the external long-horizon pool flips the longest-decile contrast.",
        transform=ax_b.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.6,
        fontweight="bold",
        color="#4B3F72",
        clip_on=False,
    )

    within_values: list[tuple[str, float, float]] = []
    for ds in DATASET_ORDER:
        top = within.loc[within["comparison"].eq(f"{ds}_within_dataset_top_10pct_longest")].iloc[0]
        rest = within.loc[within["comparison"].eq(f"{ds}_within_dataset_remaining")].iloc[0]
        within_values.append((ds, 100 * float(top["contains_uc_pct"]), 100 * float(rest["contains_uc_pct"])))
    yw = np.arange(len(within_values))[::-1]
    for y, (ds, top_v, rest_v) in zip(yw, within_values):
        line_color = rgba(DATASET_COLOR[ds], 0.52)
        ax_in.plot([min(top_v, rest_v), max(top_v, rest_v)], [y, y], color=line_color, lw=1.2, zorder=1)
        ax_in.scatter([rest_v], [y], s=35, marker="o", color=rest_color, edgecolor="#1D4ED8", linewidth=0.45, zorder=3)
        ax_in.scatter([top_v], [y], s=41, marker="D", color=top_color, edgecolor="#7F1D1D", linewidth=0.45, zorder=4)
        label_x = max(top_v, rest_v) + 2.0
        if label_x > 102:
            label_x = 102
        ax_in.text(label_x, y, f"{top_v:.1f} / {rest_v:.1f}", ha="left", va="center", fontsize=6.0, color="#333333")
    ax_in.set_xlim(0, 112)
    ax_in.set_ylim(-0.7, len(within_values) - 0.3)
    ax_in.set_yticks(yw)
    ax_in.set_yticklabels([DATASET_LABEL[v[0]] for v in within_values], fontsize=6.5)
    ax_in.set_xlabel("UC prevalence (%)", fontsize=7.0)
    ax_in.set_title("Within-pool contrast: top decile / rest", loc="left", pad=5, fontsize=8.0, fontweight="bold")
    ax_in.grid(axis="x", color="#E5E7EB", alpha=0.8)
    ax_in.spines[["top", "right", "left"]].set_visible(False)
    ax_in.tick_params(axis="y", length=0)

    return save_figure(fig, out_dir, "fig2")


def build_construct_rows(table1: pd.DataFrame, table2: pd.DataFrame) -> pd.DataFrame:
    n_by_dataset = dict(zip(table1["Evidence pool"], table1["Trajectories"]))
    rows: list[dict[str, Any]] = []
    for _, row in ordered_by_dataset(table2, "Dataset").iterrows():
        dataset = row["Dataset"]
        n = int(n_by_dataset[dataset])
        hn_count, hn_rate = parse_count_rate(row["HN-only"])
        uc_count, uc_rate = parse_count_rate(row["UC-only"])
        mixed_count, mixed_rate = parse_count_rate(row["HN+UC mixed"])
        neither_count = n - hn_count - uc_count - mixed_count
        neither_rate = neither_count / n
        rows.append(
            {
                "Dataset": dataset,
                "N": n,
                "HN-only count": hn_count,
                "HN-only rate": hn_rate,
                "UC-only count": uc_count,
                "UC-only rate": uc_rate,
                "HN+UC mixed count": mixed_count,
                "HN+UC mixed rate": mixed_rate,
                "Neither count": neither_count,
                "Neither rate": neither_rate,
            }
        )
    return pd.DataFrame(rows)


def draw_fig3(table1: pd.DataFrame, table2: pd.DataFrame, out_dir: Path) -> list[Path]:
    construct = build_construct_rows(table1, table2)
    fig = plt.figure(figsize=(7.1, 3.70), facecolor="white")
    fig.text(0.5, 0.975, "HN/UC Construct Non-equivalence", ha="center", va="top", fontsize=11.5, fontweight="bold")
    fig.text(
        0.5,
        0.94,
        "HN-only, UC-only, and mixed trajectories coexist across evidence pools.",
        ha="center",
        va="top",
        fontsize=8.4,
        style="italic",
        color="#333333",
    )
    category_specs = [
        ("HN-only", "HN-only count", "HN-only rate"),
        ("UC-only", "UC-only count", "UC-only rate"),
        ("HN+UC mixed", "HN+UC mixed count", "HN+UC mixed rate"),
        ("Neither", "Neither count", "Neither rate"),
    ]

    ax = fig.add_axes([0.125, 0.330, 0.805, 0.455])
    y_positions = np.arange(len(DATASET_ORDER))[::-1]
    for y, dataset in zip(y_positions, DATASET_ORDER):
        row = construct.loc[construct["Dataset"].eq(dataset)].iloc[0]
        left = 0.0
        small_count = 0
        for category, count_col, rate_col in category_specs:
            rate = 100.0 * float(row[rate_col])
            n_count = int(row[count_col])
            color = CONSTRUCT_COLOR[category]
            ax.barh(
                y,
                rate,
                left=left,
                height=0.56,
                color=rgba(color, 0.80 if category != "Neither" else 0.62),
                edgecolor="white",
                linewidth=0.85,
                zorder=3,
            )
            center = left + rate / 2.0
            if rate >= 7.0:
                ax.text(center, y, f"{rate:.1f}%\n({n_count})", ha="center", va="center", fontsize=6.3, color="#111111", linespacing=0.95)
            elif n_count > 0:
                base_offset = 0.60 if center < 25 else 0.22
                offset = base_offset - small_count * (0.14 if center < 25 else 0.18)
                text_x = 101.5 if center > 90 else 22.0
                ax.annotate(
                    f"{category.replace(' mixed', '')}: {n_count} ({rate:.1f}%)",
                    xy=(center, y),
                    xycoords="data",
                    xytext=(text_x, y + offset),
                    textcoords="data",
                    ha="left",
                    va="center",
                    fontsize=5.8,
                    color="#333333",
                    bbox=dict(boxstyle="round,pad=0.10,rounding_size=0.02", fc="#FFFFFF", ec="none", alpha=0.82),
                    arrowprops=dict(arrowstyle="-", color="#7A7F87", lw=0.45, shrinkA=0, shrinkB=2),
                    clip_on=False,
                )
                small_count += 1
            left += rate

    ax.set_xlim(0, 112)
    ax.set_ylim(-0.65, len(DATASET_ORDER) - 0.35)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{DATASET_LABEL[ds]}\nN={int(construct.loc[construct['Dataset'].eq(ds), 'N'].iloc[0])}" for ds in DATASET_ORDER])
    ax.set_xlabel("Trajectory share (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", color="#E5E7EB", alpha=0.85, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="y", labelsize=7.0)

    legend_handles = [
        patches.Patch(facecolor=rgba(CONSTRUCT_COLOR[cat], 0.80 if cat != "Neither" else 0.62), edgecolor="none", label=cat)
        for cat in ("HN-only", "UC-only", "HN+UC mixed", "Neither")
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=4, frameon=False, handlelength=1.35, columnspacing=1.2)

    final = construct.loc[construct["Dataset"].eq("Final500")].iloc[0]
    fig.text(0.125, 0.155, "Final500 headline:", ha="left", va="center", fontsize=7.6, fontweight="bold")
    fig.text(
        0.285,
        0.155,
        f"HN-only {int(final['HN-only count'])}/500   UC-only {int(final['UC-only count'])}/500   HN+UC {int(final['HN+UC mixed count'])}/500",
        ha="left",
        va="center",
        fontsize=7.0,
        color="#333333",
    )
    fig.text(0.125, 0.095, "Takeaway:", ha="left", va="center", fontsize=7.3, fontweight="bold")
    fig.text(
        0.225,
        0.095,
        "HN-only, UC-only, and mixed cases coexist; HN and UC are related, not interchangeable.",
        ha="left",
        va="center",
        fontsize=6.8,
        color="#333333",
    )
    fig.text(0.125, 0.050, "HN = hard negative; UC = unproductive cycle.", ha="left", va="center", fontsize=6.2, color="#555555")

    return save_figure(fig, out_dir, "fig3")


def metric_row(metrics: pd.DataFrame, dataset: str, detector: str, config: str) -> pd.Series:
    rows = metrics.loc[
        metrics["dataset"].eq(dataset)
        & metrics["detector"].eq(detector)
        & metrics["threshold/config"].eq(config)
    ]
    if rows.empty:
        raise KeyError(f"No detector metric row for {dataset}, {detector}, {config}")
    return rows.iloc[0]


def get_final500_pair(metrics: pd.DataFrame, near_f1: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    rows = near_f1.loc[
        near_f1["dataset"].eq("Final500")
        & near_f1["detector_a"].eq("max_step_guard")
        & near_f1["config_a"].eq("threshold=50")
        & near_f1["detector_b"].eq("file_revisit_guard")
        & near_f1["config_b"].eq("window=5;repeats=3")
    ]
    if rows.empty:
        raise KeyError("Missing required Final500 near-F1 pair in near_f1_risk_pairs.csv")
    pair = rows.iloc[0]
    row_a = metric_row(metrics, "Final500", "max_step_guard", "threshold=50")
    row_b = metric_row(metrics, "Final500", "file_revisit_guard", "window=5;repeats=3")
    return row_a, row_b, pair


def draw_fig4(metrics: pd.DataFrame, near_f1: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig = plt.figure(figsize=(7.1, 4.35), facecolor="white")
    fig.text(0.5, 0.975, "Aggregate F1 Can Hide Trigger-Site Risk", ha="center", va="top", fontsize=12, fontweight="bold")
    fig.text(
        0.5,
        0.94,
        "Near-F1 detector settings can have sharply different productive-interruption risk.",
        ha="center",
        va="top",
        fontsize=8.4,
        style="italic",
        color="#333333",
    )
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.985, top=0.835, bottom=0.230, wspace=0.23)
    ax_piir = fig.add_subplot(gs[0, 0])
    ax_hn = fig.add_subplot(gs[0, 1])
    axes = [(ax_piir, "PIIR", "PIIR"), (ax_hn, "HN_FPR", "HN-FPR")]

    for ax, metric_col, metric_label in axes:
        panel_label(ax, "A" if metric_col == "PIIR" else "B", x=-0.12, y=1.07)
        ax.axhspan(0.6, 1.02, color=OKABE["vermillion"], alpha=0.045, zorder=0)
        ax.text(0.98, 0.97, "high intervention risk", transform=ax.transAxes, ha="right", va="top", fontsize=6.7, color="#7F1D1D", style="italic")
        for dataset in DATASET_ORDER:
            for family, marker in FAMILY_MARKERS.items():
                sub = metrics.loc[metrics["dataset"].eq(dataset) & metrics["detector_family"].eq(family)]
                if sub.empty:
                    continue
                alpha = 0.10 if dataset == "OpenHandsExternal330" else 0.20
                size = 9 if dataset == "OpenHandsExternal330" else 11
                ax.scatter(
                    sub["F1"],
                    sub[metric_col],
                    s=size,
                    marker=marker,
                    color=DATASET_COLOR[dataset],
                    alpha=alpha,
                    edgecolors="none",
                    linewidths=0.0,
                    zorder=3,
                )
        ax.set_xlim(-0.015, 0.86)
        ax.set_ylim(-0.02, 1.03)
        ax.set_xlabel("F1 score")
        ax.set_ylabel(metric_label)
        ax.grid(color="#D9DDE2", alpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"F1 vs. {metric_label}", loc="left", fontsize=9.5, fontweight="bold", pad=7)

    row_a, row_b, pair = get_final500_pair(metrics, near_f1)
    pair_color = "#003F8C"
    for ax, metric_col, metric_label in axes:
        ya = float(row_a[metric_col])
        yb = float(row_b[metric_col])
        xa = float(row_a["F1"])
        xb = float(row_b["F1"])
        ax.plot([xa, xb], [ya, yb], color=pair_color, lw=1.25, alpha=0.90, zorder=5)
        for label, x0, y0, marker in [("A", xa, ya, "o"), ("B", xb, yb, "s")]:
            ax.scatter([x0], [y0], s=118, marker="o", facecolors=rgba(pair_color, 0.09), edgecolors="none", zorder=4)
            ax.scatter([x0], [y0], s=66, marker=marker, color=DATASET_COLOR["Final500"], edgecolors=pair_color, linewidths=0.95, zorder=6)
            ax.text(
                x0 + 0.018,
                y0 + (0.034 if label == "B" else -0.048),
                label,
                color=pair_color,
                fontsize=8.0,
                fontweight="bold",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.08,rounding_size=0.025", fc="#FFFFFF", ec="none", alpha=0.88),
            )
        delta_f1 = float(pair["abs_F1_diff"])
        delta_y = abs(yb - ya)
        xy = ((xa + xb) / 2, (ya + yb) / 2)
        text_xy = (0.56, 0.43) if metric_col == "PIIR" else (0.51, 0.39)
        ax.annotate(
            f"$\\Delta$F1 = {delta_f1:.3f}\n$\\Delta${metric_label} = {delta_y:.3f}",
            xy=xy,
            xycoords="data",
            xytext=text_xy,
            textcoords="axes fraction",
            ha="left",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color="#111111",
            arrowprops=dict(arrowstyle="-", color=pair_color, lw=0.75),
            bbox=dict(boxstyle="round,pad=0.20,rounding_size=0.035", fc="#FFFFFF", ec=rgba(pair_color, 0.35), lw=0.55, alpha=0.96),
        )

    dataset_legend_label = {
        "Natural500": "Natural500",
        "Final500": "Final500",
        "StressFresh100": "StressFresh100",
        "OpenHandsExternal330": "OpenHandsExt.",
    }
    dataset_handles = [
        mlines.Line2D([], [], color=DATASET_COLOR[ds], marker="o", linestyle="None", markersize=4.7, label=dataset_legend_label[ds])
        for ds in DATASET_ORDER
    ]
    fig.text(
        0.50,
        0.132,
        "Final500 near-F1 pair: A = max-step T=50; B = file-revisit w=5,r=3.",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#333333",
    )
    fig.legend(
        handles=dataset_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=False,
        title="Dataset color",
        title_fontsize=7.4,
        columnspacing=1.0,
        handletextpad=0.35,
    )

    return save_figure(fig, out_dir, "fig4")


def write_captions(out_dir: Path) -> Path:
    text = r"""% Generated by scripts/paper1_synthesis/build_publication_figures_v2.py.
% Captions are intentionally concise and IEEE-style; figure files are PDF-first.

% Fig. 1
\caption{Evidence Pool Atlas. The four evidence pools differ in sampling role, trajectory length, and PI/HN/UC prevalence, so they are analyzed as distinct empirical roles rather than one pooled natural distribution.}

% Fig. 2
\caption{Length Is Distribution-Sensitive. Trajectory length carries useful information within some SWE-agent pools, but the global longest-decile contrast reverses after excluding OpenHandsExternal330; length is therefore not a standalone semantic proxy for unproductive cycling.}

% Fig. 3
\caption{HN/UC Construct Non-equivalence. HN-only, UC-only, and mixed HN+UC trajectories coexist across evidence pools, showing that hard-negative productive behavior and unproductive cycling are related but non-equivalent constructs.}

% Fig. 4
\caption{Aggregate F1 Can Hide Trigger-Site Risk. Detector configurations with near-identical F1 can differ sharply in \piir{} and \hnfpr{}, so F1 remains useful but incomplete for intervention-oriented evaluation.}
"""
    path = out_dir / "figure_captions.tex"
    path.write_text(text, encoding="utf-8")
    return path


def write_report(
    root: Path,
    out_dir: Path,
    sources: list[SourceRecord],
    table1: pd.DataFrame,
    robustness: pd.DataFrame,
    excluding_openhands: pd.DataFrame,
    construct_rows: pd.DataFrame,
    metrics: pd.DataFrame,
    near_f1: pd.DataFrame,
    copied: list[Path],
    generated: list[Path],
) -> Path:
    table1 = ordered_by_dataset(table1, "Evidence pool")
    row_a, row_b, pair = get_final500_pair(metrics, near_f1)
    global_top = robustness.loc[robustness["comparison"].eq("global_top_10pct_longest")].iloc[0]
    global_rest = robustness.loc[robustness["comparison"].eq("global_remaining_90pct")].iloc[0]
    excl_top = excluding_openhands.loc[excluding_openhands["comparison"].eq("excluding_openhands_top_10pct_longest")].iloc[0]
    excl_rest = excluding_openhands.loc[excluding_openhands["comparison"].eq("excluding_openhands_remaining_90pct")].iloc[0]
    final = construct_rows.loc[construct_rows["Dataset"].eq("Final500")].iloc[0]
    oh = table1.loc[table1["Evidence pool"].eq("OpenHandsExternal330")].iloc[0]

    csv_dirs = [
        "outputs/paper1_main_tables",
        "outputs/paper1_paper_findings",
        "outputs/paper1_length_robustness",
        "outputs/paper1_construct_mismatch",
        "outputs/paper1_intervention_risk_rich_detectors",
        "outputs/paper1_audit",
    ]
    lines: list[str] = []
    lines.append("# Figure Generation Report")
    lines.append("")
    lines.append("Generated by `scripts/paper1_synthesis/build_publication_figures_v2.py` using existing CSV/report artifacts only. Raw annotation files were not read, experiments were not rerun, and no numeric values were hand-entered into the figure drawing code.")
    lines.append("")
    lines.append("## Available CSV Inventory")
    for rel_dir in csv_dirs:
        path = root / rel_dir
        csvs = sorted(p.name for p in path.glob("*.csv")) if path.exists() else []
        lines.append(f"- `{rel_dir}`: {', '.join(csvs) if csvs else 'no CSV files found'}")
    lines.append("")
    lines.append("## Data Sources Used")
    lines.append("| Key | Requested file | Actual file | Status |")
    lines.append("|---|---|---|---|")
    for source in sources:
        lines.append(
            f"| {source.key} | `{source.requested}` | `{slug_rel(root, source.actual)}` | {source.status} |"
        )
    lines.append("")
    lines.append("## Generated Outputs")
    for path in sorted(generated):
        lines.append(f"- `{slug_rel(root, path)}`")
    lines.append("")
    lines.append("## Copied To Paper Repository")
    for path in sorted(copied):
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Numeric Checks")
    lines.append("- Fig. 1 OpenHandsExternal330 contrast: average length "
                 f"{float(oh['Avg. steps']):.3f} steps, UC prevalence {pct(oh['UC prevalence'])}.")
    lines.append("- Fig. 2 global longest-decile UC prevalence: "
                 f"{pct(global_top['contains_uc_pct'])} vs remaining 90% {pct(global_rest['contains_uc_pct'])}.")
    lines.append("- Fig. 2 excluding OpenHandsExternal330: longest decile "
                 f"{pct(excl_top['contains_uc_pct'])} vs remaining 90% {pct(excl_rest['contains_uc_pct'])}.")
    lines.append("- Fig. 3 Final500 headline: HN-only "
                 f"{int(final['HN-only count'])}/500, UC-only {int(final['UC-only count'])}/500, "
                 f"HN+UC mixed {int(final['HN+UC mixed count'])}/500.")
    lines.append("- Fig. 4 Final500 pair A max_step_guard threshold=50: "
                 f"F1 {f3(row_a['F1'])}, PIIR {f3(row_a['PIIR'])}, HN-FPR {f3(row_a['HN_FPR'])}.")
    lines.append("- Fig. 4 Final500 pair B file_revisit_guard window=5;repeats=3: "
                 f"F1 {f3(row_b['F1'])}, PIIR {f3(row_b['PIIR'])}, HN-FPR {f3(row_b['HN_FPR'])}.")
    lines.append("- Fig. 4 near-F1 deltas from `near_f1_risk_pairs.csv`: "
                 f"Delta F1 {float(pair['abs_F1_diff']):.3f}, Delta PIIR {float(pair['abs_PIIR_diff']):.3f}, "
                 f"Delta HN-FPR {float(pair['abs_HN_FPR_diff']):.3f}.")
    lines.append("")
    lines.append("## Quality Notes")
    lines.append("- Matplotlib only; seaborn is not used.")
    lines.append("- SVG files are the editable vector sources for post-processing in tools such as Inkscape, Illustrator, or Figma.")
    lines.append("- PDF files are vector-friendly and intended as the LaTeX-first figure format; PNG files are saved at 300 dpi for preview/compatibility.")
    lines.append("- Dataset colors are consistent across figures and use a restrained color-blind-friendly palette.")
    lines.append("- Detector family is encoded by marker shape in Fig. 4; dataset is encoded by color.")
    lines.append("- High-risk regions and near-F1 callouts are intentionally sparse to preserve IEEE double-column readability.")
    path = out_dir / "figure_generation_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def copy_figures(out_dir: Path, paper_repo: Path) -> list[Path]:
    figures_dir = paper_repo / "figures"
    if not figures_dir.exists():
        raise FileNotFoundError(f"Paper figures directory does not exist: {figures_dir}")
    copied: list[Path] = []
    for stem in ("fig1", "fig2", "fig3", "fig4"):
        for ext in ("png", "pdf", "svg"):
            src = out_dir / f"{stem}.{ext}"
            dst = figures_dir / f"{stem}.{ext}"
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def assert_key_values(table1: pd.DataFrame, robustness: pd.DataFrame, excluding: pd.DataFrame, metrics: pd.DataFrame, near_f1: pd.DataFrame) -> None:
    oh = table1.loc[table1["Evidence pool"].eq("OpenHandsExternal330")].iloc[0]
    if not math.isclose(float(oh["Avg. steps"]), 154.576, rel_tol=0, abs_tol=0.001):
        raise AssertionError("OpenHandsExternal330 average length no longer matches expected CSV value near 154.576")
    if not math.isclose(float(oh["UC prevalence"]), 0.012, rel_tol=0, abs_tol=0.001):
        raise AssertionError("OpenHandsExternal330 UC prevalence no longer matches expected CSV value near 0.012")

    global_top = robustness.loc[robustness["comparison"].eq("global_top_10pct_longest")].iloc[0]
    global_rest = robustness.loc[robustness["comparison"].eq("global_remaining_90pct")].iloc[0]
    excl_top = excluding.loc[excluding["comparison"].eq("excluding_openhands_top_10pct_longest")].iloc[0]
    excl_rest = excluding.loc[excluding["comparison"].eq("excluding_openhands_remaining_90pct")].iloc[0]
    checks = [
        (global_top["contains_uc_pct"], 0.083, "global longest-decile UC prevalence"),
        (global_rest["contains_uc_pct"], 0.388, "global remaining UC prevalence"),
        (excl_top["contains_uc_pct"], 0.973, "excluding-OpenHands longest-decile UC prevalence"),
        (excl_rest["contains_uc_pct"], 0.402, "excluding-OpenHands remaining UC prevalence"),
    ]
    for actual, expected, label in checks:
        if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=0.001):
            raise AssertionError(f"{label} no longer matches expected rounded CSV value {expected}")

    row_a, row_b, pair = get_final500_pair(metrics, near_f1)
    metric_checks = [
        (row_a["F1"], 0.719, "Final500 max-step F1"),
        (row_a["PIIR"], 0.101, "Final500 max-step PIIR"),
        (row_a["HN_FPR"], 0.118, "Final500 max-step HN-FPR"),
        (row_b["F1"], 0.729, "Final500 file-revisit F1"),
        (row_b["PIIR"], 0.562, "Final500 file-revisit PIIR"),
        (row_b["HN_FPR"], 0.349, "Final500 file-revisit HN-FPR"),
        (pair["abs_F1_diff"], 0.009, "Final500 pair Delta F1"),
        (pair["abs_PIIR_diff"], 0.461, "Final500 pair Delta PIIR"),
        (pair["abs_HN_FPR_diff"], 0.230, "Final500 pair Delta HN-FPR"),
    ]
    for actual, expected, label in metric_checks:
        if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=0.001):
            raise AssertionError(f"{label} no longer matches expected rounded CSV value {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication-ready Paper 1 main figures.")
    parser.add_argument("--paper-repo", type=Path, default=PAPER_REPO_DEFAULT, help="Paper repository path whose figures directory receives fig1-fig4.")
    parser.add_argument("--no-copy", action="store_true", help="Generate outputs but do not copy into the paper repository.")
    args = parser.parse_args()

    setup_matplotlib()
    root = repo_root()
    out_dir = root / "outputs" / "paper1_publication_figures_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, pd.DataFrame] = {}
    records: list[SourceRecord] = []

    def load(key: str, rel: str) -> pd.DataFrame:
        df, record = resolve_csv(root, key, rel)
        loaded[key] = df
        records.append(record)
        return df

    table1 = load("table1_evidence_pools", "outputs/paper1_main_tables/table1_evidence_pools.csv")
    load("dataset_role_summary", "outputs/paper1_paper_findings/dataset_role_summary.csv")
    table2 = load("table2_construct_non_equivalence", "outputs/paper1_main_tables/table2_construct_non_equivalence.csv")
    load("construct_prevalence_by_dataset", "outputs/paper1_construct_mismatch/construct_prevalence_by_dataset.csv")
    length_rows = load("length_vs_constructs", "outputs/paper1_construct_mismatch/length_vs_constructs.csv")
    robustness = load("length_robustness_summary", "outputs/paper1_length_robustness/length_robustness_summary.csv")
    within = load("within_dataset_longest_decile", "outputs/paper1_length_robustness/within_dataset_longest_decile.csv")
    excluding = load("excluding_openhands_longest_decile", "outputs/paper1_length_robustness/excluding_openhands_longest_decile.csv")
    metrics = load("detector_metrics_by_dataset", "outputs/paper1_intervention_risk_rich_detectors/detector_metrics_by_dataset.csv")
    near_f1 = load("near_f1_risk_pairs", "outputs/paper1_intervention_risk_rich_detectors/near_f1_risk_pairs.csv")
    load("strongest_near_f1_risk_pairs", "outputs/paper1_paper_findings/strongest_near_f1_risk_pairs.csv")
    load("table4_near_f1_risk_divergence", "outputs/paper1_main_tables/table4_near_f1_risk_divergence.csv")

    assert_key_values(table1, robustness, excluding, metrics, near_f1)

    generated: list[Path] = []
    generated.extend(draw_fig1(table1, out_dir))
    generated.extend(draw_fig2(length_rows, table1, robustness, within, excluding, out_dir))
    generated.extend(draw_fig3(table1, table2, out_dir))
    generated.extend(draw_fig4(metrics, near_f1, out_dir))
    captions_path = write_captions(out_dir)
    generated.append(captions_path)

    copied: list[Path] = []
    if not args.no_copy:
        copied = copy_figures(out_dir, args.paper_repo)

    report_path = write_report(
        root=root,
        out_dir=out_dir,
        sources=records,
        table1=table1,
        robustness=robustness,
        excluding_openhands=excluding,
        construct_rows=build_construct_rows(table1, table2),
        metrics=metrics,
        near_f1=near_f1,
        copied=copied,
        generated=generated,
    )
    generated.append(report_path)

    print(f"Generated {len(generated)} files in {out_dir}")
    if copied:
        print(f"Copied {len(copied)} figure files into {args.paper_repo / 'figures'}")
    print(f"Report: {report_path}")
    print(f"Captions: {captions_path}")


if __name__ == "__main__":
    main()
