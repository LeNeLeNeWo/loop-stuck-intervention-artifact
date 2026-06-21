# Length Robustness Checks

This robustness check evaluates the claim that long trajectories are not sufficient evidence of unproductive cycling. It uses the completed Paper1 normalized and construct-mismatch outputs without modifying the raw annotation files.

## Inputs

- `outputs/paper1_construct_mismatch/length_vs_constructs.csv`
- `outputs/paper1_construct_mismatch/construct_prevalence_by_dataset.csv`
- `outputs/paper1_audit/normalized_steps.csv`
- `outputs/paper1_audit/normalized_spans.csv`

## Sanity Counts

- Normalized step rows loaded: `88706`.
- Normalized span rows loaded: `4982`.
- Construct prevalence rows loaded: `4`.

## Main Results

- Global all-dataset contrast: longest-decile trajectories have UC prevalence 8.3% (12/144) versus 38.8% (499/1286) in the comparison set. Mean lengths are 195.542 versus 47.082.
- Excluding OpenHandsExternal330: longest-decile trajectories have UC prevalence 97.3% (110/113) versus 40.2% (397/987) in the comparison set. Mean lengths are 119.947 versus 24.460.

The global longest-decile contrast is therefore strongly distribution-sensitive: after excluding OpenHandsExternal330, the direction reverses rather than remaining lower. This indicates that the global result is driven in large part by the external long-horizon pool.

Within-dataset contrasts show why length should be treated as an ambiguous and distribution-sensitive signal rather than as a direct loop/stuck proxy:
- Natural500: longest-decile trajectories have UC prevalence 96.0% (48/50) versus 19.1% (86/450) in the comparison set. Mean lengths are 54.840 versus 12.907.
- Enriched500: longest-decile trajectories have UC prevalence 100.0% (52/52) versus 53.3% (239/448) in the comparison set. Mean lengths are 145.327 versus 41.199.
- StressFresh100: longest-decile trajectories have UC prevalence 100.0% (10/10) versus 80.0% (72/90) in the comparison set. Mean lengths are 83.600 versus 25.511.
- OpenHandsExternal330: longest-decile trajectories have UC prevalence 1.2% (1/85) versus 1.2% (3/245) in the comparison set. Mean lengths are 201.000 versus 138.469.

## Correlation Summary

Spearman correlations between trajectory length and construct density are reported separately by dataset and globally. Positive correlations indicate that longer trajectories tend to have higher density for the construct in that evidence pool; negative correlations indicate the opposite. These correlations should not be interpreted causally.

| dataset | length vs UC density | length vs HN density | length vs PI density |
|---|---:|---:|---:|
| Natural500 | 0.537 | 0.318 | -0.244 |
| Enriched500 | 0.689 | -0.018 | -0.591 |
| StressFresh100 | 0.513 | -0.101 | -0.006 |
| OpenHandsExternal330 | -0.003 | 0.448 | 0.620 |
| GlobalAllDatasets | 0.169 | 0.025 | -0.404 |
| GlobalExcludingOpenHands | 0.667 | 0.399 | -0.281 |

## Paper-Ready Interpretation

- These checks support the narrower claim that length alone is insufficient as a loop/stuck proxy; they do not imply that length never matters.
- The global longest-decile result is distribution-sensitive, especially because OpenHandsExternal330 is much longer while having rare UC labels in this evidence pool.
- Within SWE-agent evidence pools, the longest decile can have high UC prevalence, which is consistent with length being informative in some settings but not diagnostic across settings.
- The construct signal depends on both dataset source and local process labels. This motivates evaluating detector triggers by intervention site rather than using trajectory length as a standalone stuckness criterion.

## Generated Outputs

- `length_robustness_summary.csv`
- `within_dataset_longest_decile.csv`
- `excluding_openhands_longest_decile.csv`
- `length_construct_correlations.csv`
- `figures/within_dataset_uc_prevalence_longest_decile_vs_rest.png` and `.pdf`
- `figures/length_vs_uc_density_by_dataset.png` and `.pdf`
- `figures/length_construct_correlation_heatmap.png` and `.pdf`
