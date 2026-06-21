from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.paper1_audit.audit_dataset_integrity import ensure_normalized, length_summary
from scripts.paper1_audit.construct_distribution_analysis import compute_trajectory_flags, density_summary, distribution, overlap_table, prevalence
from scripts.paper1_audit.load_paper1_datasets import repo_root_from_script


ROLE_NOTES = {
    "Natural500": "Natural-distribution control.",
    "Final500": "Enriched difficult loop-heavy set.",
    "StressFresh100": "Clean stress challenge.",
    "OpenHandsExternal330": "External long-horizon generalization set.",
}


def save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    png = output_dir / f"{name}.png"
    pdf = output_dir / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)


def make_table1(summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    table = summary.copy()
    table["role_note"] = table["dataset"].map(ROLE_NOTES).fillna(table["dataset_role"])
    table = table[
        [
            "dataset",
            "role_note",
            "trajectory_count",
            "step_count",
            "avg_trajectory_length",
            "median",
            "p90_trajectory_length",
            "min",
            "max",
        ]
    ].rename(
        columns={
            "role_note": "dataset_role",
            "median": "median_trajectory_length",
            "min": "min_trajectory_length",
            "max": "max_trajectory_length",
        }
    )
    table.to_csv(output_dir / "table1_dataset_role_size_summary.csv", index=False, encoding="utf-8-sig")
    return table


def make_table2(prev: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    cols = [
        "dataset",
        "trajectory_count",
        "has_pi_count",
        "has_pi_rate",
        "has_hn_count",
        "has_hn_rate",
        "has_uc_count",
        "has_uc_rate",
        "has_uncertain_or_unresolved_count",
        "has_uncertain_or_unresolved_rate",
    ]
    table = prev[cols].copy()
    table.to_csv(output_dir / "table2_trajectory_pi_hn_uc_prevalence.csv", index=False, encoding="utf-8-sig")
    return table


def make_table3(step_dist: pd.DataFrame, step_span_dist: pd.DataFrame, span_dist: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    table = pd.concat([step_dist, step_span_dist, span_dist], ignore_index=True)
    table.to_csv(output_dir / "table3_step_and_span_label_distribution.csv", index=False, encoding="utf-8-sig")
    return table


def figure1(lengths: pd.DataFrame, output_dir: Path) -> None:
    datasets = list(lengths["dataset"].drop_duplicates())
    data = [lengths[lengths["dataset"] == d]["step_count"].tolist() for d in datasets]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, tick_labels=datasets, showfliers=False)
    ax.set_ylabel("Steps per trajectory")
    ax.set_title("Figure 1. Trajectory length distribution by dataset")
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, output_dir, "figure1_trajectory_length_distribution")


def figure2(prev: pd.DataFrame, output_dir: Path) -> None:
    plot = prev.set_index("dataset")[["has_pi_rate", "has_hn_rate", "has_uc_rate"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    plot.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Trajectory-level prevalence")
    ax.set_title("Figure 2. PI/HN/UC prevalence by dataset")
    ax.legend(["PI", "HN", "UC"])
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, output_dir, "figure2_pi_hn_uc_prevalence")


def figure3(overlaps: pd.DataFrame, output_dir: Path) -> None:
    keep = overlaps[overlaps["overlap_category"].isin(["HN_and_UC", "HN_only", "UC_only", "neither_HN_nor_UC"])]
    pivot = keep.pivot(index="dataset", columns="overlap_category", values="rate").fillna(0)
    order = [c for c in ["HN_and_UC", "HN_only", "UC_only", "neither_HN_nor_UC"] if c in pivot.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot[order].plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of trajectories")
    ax.set_title("Figure 3. HN/UC overlap by dataset")
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, output_dir, "figure3_hn_uc_overlap")


def figure4(flags: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=False, sharey=True)
    for ax, field, title in zip(axes, ["pi_density", "hn_density", "uc_density"], ["PI density", "HN density", "UC density"]):
        for dataset, group in flags.groupby("dataset"):
            ax.scatter(group["step_count"], group[field], s=12, alpha=0.55, label=dataset)
        ax.set_xlabel("Trajectory length")
        ax.set_title(title)
    axes[0].set_ylabel("Step-level density")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.subplots_adjust(bottom=0.28)
    save_fig(fig, output_dir, "figure4_length_vs_construct_density")


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def write_dataset_card(output_dir: Path, table1: pd.DataFrame, table2: pd.DataFrame, table3: pd.DataFrame, audit_summary: pd.DataFrame, overlaps: pd.DataFrame, impossible: pd.DataFrame, missing: pd.DataFrame, duplicates: pd.DataFrame) -> None:
    lines = [
        "# Paper1 Annotation Dataset Card",
        "",
        "This card describes the four active Paper1 annotation datasets under `data/annotations/paper1` after normalizing them into step and span tables.",
        "",
        "## Dataset Roles",
        "",
        "- **Natural500**: natural-distribution control; shorter and less enriched for hard-negative or unproductive-cycle cases.",
        "- **Final500**: enriched difficult loop-heavy set; designed to contain many productive-iteration and unproductive-cycle examples.",
        "- **StressFresh100**: clean stress challenge; frozen 100-trajectory pressure set with many UC/HN boundary cases.",
        "- **OpenHandsExternal330**: external long-horizon generalization set; combines 270 strict-HN V3 trajectories and 60 U-span expansion trajectories.",
        "",
        "**Important**: these four datasets should not be pooled as one natural distribution. They encode different sampling mechanisms and construct stress levels.",
        "",
        "## Table 1. Dataset Role and Size Summary",
        "",
        md_table(table1),
        "",
        "## Table 2. Trajectory-Level PI/HN/UC Prevalence",
        "",
        md_table(table2),
        "",
        "## Table 3. Step-Level and Span-Level Label Distribution",
        "",
        md_table(table3),
        "",
        "## Integrity Results",
        "",
        md_table(audit_summary),
        "",
        f"- Duplicate step keys: `{len(duplicates)}` grouped keys.",
        f"- Trajectories with missing step indices: `{len(missing)}`.",
        f"- Impossible labels detected: `{len(impossible)}`.",
        "- Pairwise trajectory overlap matrix:",
        "",
        overlaps.pivot(index="dataset_a", columns="dataset_b", values="overlap_trajectory_count").to_markdown(),
        "",
        "## Paper-Ready Findings",
        "",
        "- Productive iteration, hard negatives, and unproductive cycles are related but not interchangeable constructs.",
        "- Natural500 has much shorter trajectories and lower HN/UC prevalence than Final500, supporting its role as a natural-distribution control.",
        "- Final500 is substantially more loop-heavy and is better interpreted as an enriched difficult set than as a natural sample.",
        "- StressFresh100 intentionally concentrates challenge cases: UC prevalence is high while PI and HN remain present, making it suitable for stress testing construct boundaries.",
        "- OpenHandsExternal330 is dominated by long-horizon productive iteration; UC is rare, so it mainly probes external generalization and long-context behavior rather than natural UC frequency.",
        "- HN should not be treated as a synonym for productive iteration: datasets can have high PI prevalence with much narrower HN prevalence.",
        "- UC should not be inferred from trajectory length alone; OpenHands trajectories are longest but have very low UC prevalence under the original strict semantic review.",
        "- Density-normalized reporting is necessary because the datasets differ sharply in average trajectory length and sampling purpose.",
        "",
    ]
    (output_dir / "dataset_card.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_audit_report(output_dir: Path, table1: pd.DataFrame, table2: pd.DataFrame, impossible: pd.DataFrame, missing: pd.DataFrame, duplicates: pd.DataFrame) -> None:
    lines = [
        "# Paper1 Audit Report",
        "",
        "## Summary",
        "",
        "The pipeline loaded the four active Paper1 annotation datasets, inferred file schemas, normalized step/span records, and generated construct-distribution analyses.",
        "",
        "## Dataset Size",
        "",
        md_table(table1),
        "",
        "## Construct Prevalence",
        "",
        md_table(table2),
        "",
        "## Validation Status",
        "",
        f"- Duplicate step keys: `{len(duplicates)}`.",
        f"- Missing step-index trajectories: `{len(missing)}`.",
        f"- Impossible labels: `{len(impossible)}`.",
        "",
        "## Paper-Ready Findings",
        "",
        "- The datasets represent different construct regimes and should be reported separately.",
        "- Natural500 is the natural-distribution control, while Final500 and StressFresh100 are deliberately enriched/challenging.",
        "- OpenHandsExternal330 is an external long-horizon set; its length distribution differs sharply from the 500/100-trajectory internal sets.",
        "- Productive iteration is common in all datasets, but hard-negative marking is narrower and policy-sensitive.",
        "- Unproductive-cycle prevalence is dataset-dependent and should not be approximated by raw trajectory length.",
        "- HN/UC overlap is a distinct construct slice, not the default case.",
        "",
    ]
    (output_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Paper1 audit tables and figures.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_audit"))
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    steps, spans = ensure_normalized(repo_root, output_dir)
    lengths, summary = length_summary(steps)
    flags = compute_trajectory_flags(steps, spans)
    prev = prevalence(flags)
    ov = overlap_table(flags)
    step_dist = distribution(steps, "label_norm", "steps.label_norm")
    step_span_dist = distribution(steps, "span_label_norm", "steps.span_label_norm")
    span_dist = distribution(spans, "span_label_norm", "spans.span_label_norm")
    dens = density_summary(flags)
    table1 = make_table1(summary, output_dir)
    table2 = make_table2(prev, output_dir)
    table3 = make_table3(step_dist, step_span_dist, span_dist, output_dir)
    # Keep refreshed summary files next to the publication tables.
    lengths.to_csv(output_dir / "trajectory_lengths.csv", index=False, encoding="utf-8-sig")
    prev.to_csv(output_dir / "construct_prevalence_by_dataset.csv", index=False, encoding="utf-8-sig")
    ov.to_csv(output_dir / "construct_overlap_by_dataset.csv", index=False, encoding="utf-8-sig")
    dens.to_csv(output_dir / "label_density_summary.csv", index=False, encoding="utf-8-sig")
    figure1(lengths, output_dir)
    figure2(prev, output_dir)
    figure3(ov, output_dir)
    figure4(flags, output_dir)
    impossible_path = output_dir / "impossible_labels.csv"
    missing_path = output_dir / "missing_step_indices.csv"
    duplicates_path = output_dir / "duplicate_step_ids.csv"
    overlaps_path = output_dir / "trajectory_overlap_matrix.csv"
    impossible = pd.read_csv(impossible_path) if impossible_path.exists() else pd.DataFrame()
    missing = pd.read_csv(missing_path) if missing_path.exists() else pd.DataFrame()
    duplicates = pd.read_csv(duplicates_path) if duplicates_path.exists() else pd.DataFrame()
    overlaps = pd.read_csv(overlaps_path) if overlaps_path.exists() else pd.DataFrame()
    if overlaps.empty:
        ids = {name: set(group["trajectory_id"].astype(str)) for name, group in steps.groupby("dataset")}
        overlaps = pd.DataFrame(
            [{"dataset_a": a, "dataset_b": b, "overlap_trajectory_count": len(ids[a] & ids[b])} for a in ids for b in ids]
        )
    write_dataset_card(output_dir, table1, table2, table3, summary, overlaps, impossible, missing, duplicates)
    write_final_audit_report(output_dir, table1, table2, impossible, missing, duplicates)
    print(f"Tables and figures complete: {output_dir}")


if __name__ == "__main__":
    main()
