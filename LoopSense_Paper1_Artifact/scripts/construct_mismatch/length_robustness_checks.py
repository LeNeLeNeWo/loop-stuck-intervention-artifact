from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


DATASET_ORDER = ["Natural500", "Final500", "StressFresh100", "OpenHandsExternal330"]
OPENHANDS_DATASET = "OpenHandsExternal330"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def output_root(root: Path) -> Path:
    return root / "outputs" / "paper1_length_robustness"


def bool_series(series: pd.Series) -> pd.Series:
    def parse(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}

    return series.map(parse)


def ordered(df: pd.DataFrame, dataset_col: str = "dataset") -> pd.DataFrame:
    if dataset_col not in df.columns:
        return df
    order = {name: i for i, name in enumerate(DATASET_ORDER)}
    return (
        df.assign(_dataset_order=df[dataset_col].map(order).fillna(999))
        .sort_values(["_dataset_order", dataset_col])
        .drop(columns="_dataset_order")
    )


def pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100.0 * value:.1f}%"


def save_fig(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{name}.png", dpi=240)
    fig.savefig(figures_dir / f"{name}.pdf")
    plt.close(fig)


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    length_path = root / "outputs" / "paper1_construct_mismatch" / "length_vs_constructs.csv"
    prevalence_path = root / "outputs" / "paper1_construct_mismatch" / "construct_prevalence_by_dataset.csv"
    steps_path = root / "outputs" / "paper1_audit" / "normalized_steps.csv"
    spans_path = root / "outputs" / "paper1_audit" / "normalized_spans.csv"

    length_df = pd.read_csv(length_path)
    prevalence_df = pd.read_csv(prevalence_path)
    steps_df = pd.read_csv(steps_path, low_memory=False)
    spans_df = pd.read_csv(spans_path, low_memory=False)

    require_columns(
        length_df,
        [
            "dataset",
            "trajectory_id",
            "trajectory_length",
            "has_pi",
            "has_hn",
            "has_uc",
            "pi_density",
            "hn_density",
            "uc_density",
        ],
        "length_vs_constructs.csv",
    )
    for column in ["has_pi", "has_hn", "has_uc"]:
        length_df[column] = bool_series(length_df[column])
    if "top_10pct_longest_global" in length_df.columns:
        length_df["top_10pct_longest_global"] = bool_series(length_df["top_10pct_longest_global"])
    if "top_10pct_longest_within_dataset" in length_df.columns:
        length_df["top_10pct_longest_within_dataset"] = bool_series(
            length_df["top_10pct_longest_within_dataset"]
        )

    return length_df, prevalence_df, steps_df, spans_df


def longest_decile_mask(df: pd.DataFrame) -> tuple[pd.Series, float]:
    if df.empty:
        return pd.Series([], dtype=bool), math.nan
    threshold = float(df["trajectory_length"].quantile(0.90))
    mask = df["trajectory_length"] >= threshold
    return mask, threshold


def summarize_subset(
    df: pd.DataFrame,
    mask: pd.Series,
    comparison: str,
    selection_scope: str,
    selection_rule: str,
    threshold: float | None = None,
) -> dict[str, object]:
    subset = df.loc[mask].copy()
    n = len(subset)

    def prevalence(flag: str) -> tuple[int, float]:
        count = int(subset[flag].sum()) if n else 0
        return count, count / n if n else math.nan

    uc_count, uc_pct = prevalence("has_uc")
    hn_count, hn_pct = prevalence("has_hn")
    pi_count, pi_pct = prevalence("has_pi")
    return {
        "comparison": comparison,
        "selection_scope": selection_scope,
        "selection_rule": selection_rule,
        "length_threshold": threshold,
        "trajectory_count": n,
        "mean_length": subset["trajectory_length"].mean() if n else math.nan,
        "median_length": subset["trajectory_length"].median() if n else math.nan,
        "p90_length": subset["trajectory_length"].quantile(0.90) if n else math.nan,
        "min_length": subset["trajectory_length"].min() if n else math.nan,
        "max_length": subset["trajectory_length"].max() if n else math.nan,
        "contains_uc_count": uc_count,
        "contains_uc_pct": uc_pct,
        "contains_hn_count": hn_count,
        "contains_hn_pct": hn_pct,
        "contains_pi_count": pi_count,
        "contains_pi_pct": pi_pct,
        "mean_uc_density": subset["uc_density"].mean() if n else math.nan,
        "mean_hn_density": subset["hn_density"].mean() if n else math.nan,
        "mean_pi_density": subset["pi_density"].mean() if n else math.nan,
    }


def global_longest_decile(length_df: pd.DataFrame) -> pd.DataFrame:
    if "top_10pct_longest_global" in length_df.columns:
        mask = length_df["top_10pct_longest_global"]
        threshold = float(length_df.loc[mask, "trajectory_length"].min()) if mask.any() else math.nan
        rule = "existing top_10pct_longest_global flag from construct-mismatch pipeline"
    else:
        mask, threshold = longest_decile_mask(length_df)
        rule = "trajectory_length >= global 90th percentile, ties retained"
    rows = [
        summarize_subset(
            length_df,
            mask,
            "global_top_10pct_longest",
            "all_datasets",
            rule,
            threshold,
        ),
        summarize_subset(
            length_df,
            ~mask,
            "global_remaining_90pct",
            "all_datasets",
            rule,
            threshold,
        ),
    ]
    return pd.DataFrame(rows)


def within_dataset_longest_decile(length_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    use_existing = "top_10pct_longest_within_dataset" in length_df.columns
    for dataset, group in ordered(length_df).groupby("dataset", sort=False):
        if use_existing:
            mask = group["top_10pct_longest_within_dataset"]
            threshold = float(group.loc[mask, "trajectory_length"].min()) if mask.any() else math.nan
            rule = "existing top_10pct_longest_within_dataset flag from construct-mismatch pipeline"
        else:
            mask, threshold = longest_decile_mask(group)
            rule = "trajectory_length >= within-dataset 90th percentile, ties retained"
        rows.append(
            summarize_subset(
                group,
                mask,
                f"{dataset}_within_dataset_top_10pct_longest",
                dataset,
                rule,
                threshold,
            )
        )
        rows.append(
            summarize_subset(
                group,
                ~mask,
                f"{dataset}_within_dataset_remaining",
                dataset,
                rule,
                threshold,
            )
        )
    return pd.DataFrame(rows)


def excluding_openhands_longest_decile(length_df: pd.DataFrame) -> pd.DataFrame:
    filtered = length_df[length_df["dataset"] != OPENHANDS_DATASET].copy()
    mask, threshold = longest_decile_mask(filtered)
    rule = "trajectory_length >= non-OpenHands 90th percentile, ties retained"
    rows = [
        summarize_subset(
            filtered,
            mask,
            "excluding_openhands_top_10pct_longest",
            "all_except_OpenHandsExternal330",
            rule,
            threshold,
        ),
        summarize_subset(
            filtered,
            ~mask,
            "excluding_openhands_remaining_90pct",
            "all_except_OpenHandsExternal330",
            rule,
            threshold,
        ),
    ]
    return pd.DataFrame(rows)


def length_correlations(length_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_group(name: str, group: pd.DataFrame) -> None:
        for density_col, construct in [
            ("uc_density", "UC"),
            ("hn_density", "HN"),
            ("pi_density", "PI"),
        ]:
            valid = group[["trajectory_length", density_col]].dropna()
            rho = (
                valid["trajectory_length"].corr(valid[density_col], method="spearman")
                if len(valid) >= 2
                else math.nan
            )
            rows.append(
                {
                    "dataset": name,
                    "construct": construct,
                    "density_column": density_col,
                    "spearman_rho": rho,
                    "trajectory_count": len(valid),
                }
            )

    for dataset, group in ordered(length_df).groupby("dataset", sort=False):
        add_group(dataset, group)
    add_group("GlobalAllDatasets", length_df)
    add_group("GlobalExcludingOpenHands", length_df[length_df["dataset"] != OPENHANDS_DATASET])
    return pd.DataFrame(rows)


def write_summary_csv(
    out_dir: Path,
    global_df: pd.DataFrame,
    within_df: pd.DataFrame,
    excluding_df: pd.DataFrame,
) -> pd.DataFrame:
    summary = pd.concat(
        [
            global_df.assign(analysis="global_longest_decile"),
            excluding_df.assign(analysis="excluding_openhands_longest_decile"),
            within_df.assign(analysis="within_dataset_longest_decile"),
        ],
        ignore_index=True,
    )
    summary.to_csv(out_dir / "length_robustness_summary.csv", index=False)
    return summary


def plot_within_uc_prevalence(within_df: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = within_df.copy()
    plot_df["group"] = plot_df["comparison"].map(
        lambda text: "Longest decile" if "top_10pct" in str(text) else "Remaining"
    )
    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(plot_df["selection_scope"])]
    x = range(len(datasets))
    width = 0.36
    top_vals = []
    rest_vals = []
    for dataset in datasets:
        group = plot_df[plot_df["selection_scope"] == dataset]
        top_vals.append(float(group.loc[group["group"] == "Longest decile", "contains_uc_pct"].iloc[0]))
        rest_vals.append(float(group.loc[group["group"] == "Remaining", "contains_uc_pct"].iloc[0]))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar([i - width / 2 for i in x], top_vals, width, label="Longest decile")
    ax.bar([i + width / 2 for i in x], rest_vals, width, label="Remaining")
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("UC prevalence")
    ax.set_title("Within-dataset UC prevalence: longest decile vs remaining trajectories")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, figures_dir, "within_dataset_uc_prevalence_longest_decile_vs_rest")


def plot_length_vs_uc_density(length_df: pd.DataFrame, figures_dir: Path) -> None:
    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(length_df["dataset"])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=False, sharey=True)
    axes_flat = axes.flatten()
    for ax, dataset in zip(axes_flat, datasets):
        group = length_df[length_df["dataset"] == dataset]
        colors = group["has_uc"].map({True: "#b23a48", False: "#3d6fb6"})
        ax.scatter(group["trajectory_length"], group["uc_density"], c=colors, s=18, alpha=0.65, linewidths=0)
        ax.set_title(dataset)
        ax.set_xlabel("Trajectory length")
        ax.set_ylabel("UC density")
        ax.grid(alpha=0.25)
    for ax in axes_flat[len(datasets) :]:
        ax.axis("off")
    fig.suptitle("Trajectory length vs UC density by dataset", y=1.02)
    save_fig(fig, figures_dir, "length_vs_uc_density_by_dataset")


def plot_correlation_heatmap(correlations: pd.DataFrame, figures_dir: Path) -> None:
    row_order = DATASET_ORDER + ["GlobalAllDatasets", "GlobalExcludingOpenHands"]
    col_order = ["UC", "HN", "PI"]
    matrix = (
        correlations.pivot(index="dataset", columns="construct", values="spearman_rho")
        .reindex(index=row_order, columns=col_order)
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    image = ax.imshow(matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Spearman correlation: length vs construct density")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            label = "NA" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="black", fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho")
    save_fig(fig, figures_dir, "length_construct_correlation_heatmap")


def sentence_for_decile(name: str, df: pd.DataFrame) -> str:
    top = df[df["comparison"].str.contains("top_10pct", regex=False)].iloc[0]
    rest = df[~df["comparison"].str.contains("top_10pct", regex=False)].iloc[0]
    return (
        f"- {name}: longest-decile trajectories have UC prevalence {pct(top['contains_uc_pct'])} "
        f"({int(top['contains_uc_count'])}/{int(top['trajectory_count'])}) versus "
        f"{pct(rest['contains_uc_pct'])} ({int(rest['contains_uc_count'])}/"
        f"{int(rest['trajectory_count'])}) in the comparison set. "
        f"Mean lengths are {top['mean_length']:.3f} versus {rest['mean_length']:.3f}."
    )


def write_report(
    out_dir: Path,
    global_df: pd.DataFrame,
    within_df: pd.DataFrame,
    excluding_df: pd.DataFrame,
    correlations: pd.DataFrame,
    prevalence_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    spans_df: pd.DataFrame,
) -> None:
    global_top = global_df.iloc[0]
    global_rest = global_df.iloc[1]
    excl_top = excluding_df.iloc[0]
    excl_rest = excluding_df.iloc[1]

    global_driven_by_openhands = (
        global_top["contains_uc_pct"] < global_rest["contains_uc_pct"]
        and excl_top["contains_uc_pct"] >= excl_rest["contains_uc_pct"]
    )
    drive_sentence = (
        "The global longest-decile contrast is therefore strongly distribution-sensitive: "
        "after excluding OpenHandsExternal330, the direction reverses rather than remaining lower. "
        "This indicates that the global result is driven in large part by the external long-horizon pool."
        if global_driven_by_openhands
        else "The excluding-OpenHands contrast does not fully reverse the global pattern, but the magnitude still differs across evidence pools."
    )

    lines = [
        "# Length Robustness Checks",
        "",
        "This robustness check evaluates the claim that long trajectories are not sufficient evidence of unproductive cycling. It uses the completed Paper1 normalized and construct-mismatch outputs without modifying the raw annotation files.",
        "",
        "## Inputs",
        "",
        "- `outputs/paper1_construct_mismatch/length_vs_constructs.csv`",
        "- `outputs/paper1_construct_mismatch/construct_prevalence_by_dataset.csv`",
        "- `outputs/paper1_audit/normalized_steps.csv`",
        "- `outputs/paper1_audit/normalized_spans.csv`",
        "",
        "## Sanity Counts",
        "",
        f"- Normalized step rows loaded: `{len(steps_df)}`.",
        f"- Normalized span rows loaded: `{len(spans_df)}`.",
        f"- Construct prevalence rows loaded: `{len(prevalence_df)}`.",
        "",
        "## Main Results",
        "",
        sentence_for_decile("Global all-dataset contrast", global_df),
        sentence_for_decile("Excluding OpenHandsExternal330", excluding_df),
        "",
        drive_sentence,
        "",
        "Within-dataset contrasts show why length should be treated as an ambiguous and distribution-sensitive signal rather than as a direct loop/stuck proxy:",
    ]

    for dataset in DATASET_ORDER:
        group = within_df[within_df["selection_scope"] == dataset]
        if len(group) == 2:
            lines.append(sentence_for_decile(dataset, group))

    lines.extend(
        [
            "",
            "## Correlation Summary",
            "",
            "Spearman correlations between trajectory length and construct density are reported separately by dataset and globally. Positive correlations indicate that longer trajectories tend to have higher density for the construct in that evidence pool; negative correlations indicate the opposite. These correlations should not be interpreted causally.",
            "",
            "| dataset | length vs UC density | length vs HN density | length vs PI density |",
            "|---|---:|---:|---:|",
        ]
    )
    corr_table = correlations.pivot(index="dataset", columns="construct", values="spearman_rho")
    for dataset in DATASET_ORDER + ["GlobalAllDatasets", "GlobalExcludingOpenHands"]:
        if dataset in corr_table.index:
            uc = corr_table.loc[dataset, "UC"]
            hn = corr_table.loc[dataset, "HN"]
            pi = corr_table.loc[dataset, "PI"]
            lines.append(f"| {dataset} | {uc:.3f} | {hn:.3f} | {pi:.3f} |")

    lines.extend(
        [
            "",
            "## Paper-Ready Interpretation",
            "",
            "- These checks support the narrower claim that length alone is insufficient as a loop/stuck proxy; they do not imply that length never matters.",
            "- The global longest-decile result is distribution-sensitive, especially because OpenHandsExternal330 is much longer while having rare UC labels in this evidence pool.",
            "- Within SWE-agent evidence pools, the longest decile can have high UC prevalence, which is consistent with length being informative in some settings but not diagnostic across settings.",
            "- The construct signal depends on both dataset source and local process labels. This motivates evaluating detector triggers by intervention site rather than using trajectory length as a standalone stuckness criterion.",
            "",
            "## Generated Outputs",
            "",
            "- `length_robustness_summary.csv`",
            "- `within_dataset_longest_decile.csv`",
            "- `excluding_openhands_longest_decile.csv`",
            "- `length_construct_correlations.csv`",
            "- `figures/within_dataset_uc_prevalence_longest_decile_vs_rest.png` and `.pdf`",
            "- `figures/length_vs_uc_density_by_dataset.png` and `.pdf`",
            "- `figures/length_construct_correlation_heatmap.png` and `.pdf`",
        ]
    )
    (out_dir / "length_robustness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = repo_root()
    out_dir = output_root(root)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    length_df, prevalence_df, steps_df, spans_df = load_inputs(root)

    global_df = global_longest_decile(length_df)
    within_df = within_dataset_longest_decile(length_df)
    excluding_df = excluding_openhands_longest_decile(length_df)
    correlations = length_correlations(length_df)

    write_summary_csv(out_dir, global_df, within_df, excluding_df)
    within_df.to_csv(out_dir / "within_dataset_longest_decile.csv", index=False)
    excluding_df.to_csv(out_dir / "excluding_openhands_longest_decile.csv", index=False)
    correlations.to_csv(out_dir / "length_construct_correlations.csv", index=False)

    plot_within_uc_prevalence(within_df, figures_dir)
    plot_length_vs_uc_density(length_df, figures_dir)
    plot_correlation_heatmap(correlations, figures_dir)
    write_report(out_dir, global_df, within_df, excluding_df, correlations, prevalence_df, steps_df, spans_df)

    required = [
        out_dir / "length_robustness_report.md",
        out_dir / "length_robustness_summary.csv",
        out_dir / "within_dataset_longest_decile.csv",
        out_dir / "excluding_openhands_longest_decile.csv",
        out_dir / "length_construct_correlations.csv",
        figures_dir / "within_dataset_uc_prevalence_longest_decile_vs_rest.png",
        figures_dir / "within_dataset_uc_prevalence_longest_decile_vs_rest.pdf",
        figures_dir / "length_vs_uc_density_by_dataset.png",
        figures_dir / "length_vs_uc_density_by_dataset.pdf",
        figures_dir / "length_construct_correlation_heatmap.png",
        figures_dir / "length_construct_correlation_heatmap.pdf",
    ]
    empty = [path for path in required if not path.exists() or path.stat().st_size == 0]
    if empty:
        raise RuntimeError(f"Expected non-empty outputs were not created: {empty}")
    print(f"Length robustness checks written to {out_dir}")


if __name__ == "__main__":
    main()
