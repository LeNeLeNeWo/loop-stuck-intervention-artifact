# Construct Mismatch Analysis

This analysis uses the normalized step and span tables from `outputs/paper1_audit/` and writes derived construct tables and figures to `outputs/paper1_construct_mismatch/`.

## Core Claim

In software-engineering agent debugging, repetition and long trajectories are ambiguous evidence. Productive iteration, hard negatives, and unproductive cycles are related but distinct constructs. These results motivate evaluation designs that measure intervention risk rather than treating loop or stuck detection as a simple binary classification task.

## RQ1: How different are the four evidence pools?

The four datasets differ in sampling role and construct prevalence. Natural500 is the natural-distribution control, Enriched500 is enriched for difficult loop-heavy behavior, StressFresh100 concentrates clean challenge cases, and OpenHandsExternal330 is an external long-horizon generalization set. They should therefore be reported separately rather than pooled as a single natural distribution.

| dataset              | dataset_role                             |   trajectory_count |   contains_pi_count | contains_pi_pct   |   contains_hn_count | contains_hn_pct   |   contains_uc_count | contains_uc_pct   |   contains_both_hn_and_uc_count | contains_both_hn_and_uc_pct   |   contains_pi_but_no_uc_count | contains_pi_but_no_uc_pct   |   contains_hn_but_no_uc_count | contains_hn_but_no_uc_pct   |
|:---------------------|:-----------------------------------------|-------------------:|--------------------:|:------------------|--------------------:|:------------------|--------------------:|:------------------|--------------------------------:|:------------------------------|------------------------------:|:----------------------------|------------------------------:|:----------------------------|
| Natural500           | natural-distribution control             |                500 |                 492 | 98.4%             |                 165 | 33.0%             |                 134 | 26.8%             |                              71 | 14.2%                         |                           363 | 72.6%                       |                            94 | 18.8%                       |
| Enriched500             | enriched difficult loop-heavy set        |                500 |                 499 | 99.8%             |                 387 | 77.4%             |                 291 | 58.2%             |                             241 | 48.2%                         |                           209 | 41.8%                       |                           146 | 29.2%                       |
| StressFresh100       | clean stress challenge                   |                100 |                  92 | 92.0%             |                  46 | 46.0%             |                  82 | 82.0%             |                              30 | 30.0%                         |                            18 | 18.0%                       |                            16 | 16.0%                       |
| OpenHandsExternal330 | external long-horizon generalization set |                330 |                 329 | 99.7%             |                  50 | 15.2%             |                   4 | 1.2%              |                               3 | 0.9%                          |                           326 | 98.8%                       |                            47 | 14.2%                       |

## RQ2: Does long or repetitive behavior imply unproductive cycling?

The global top decile of longest trajectories has UC prevalence `8.3%`, compared with `38.8%` for the remaining trajectories. OpenHands external trajectories are much longer on average, but their UC prevalence is `1.2%`, compared with `46.1%` for the SWE-agent datasets. This supports the paper's caution that length is not a direct substitute for unproductive cycling.

| comparison               |   trajectory_count |   mean_length |   median_length |   p90_length |   contains_uc_count | contains_uc_pct   |   contains_hn_count | contains_hn_pct   |   mean_pi_density |   mean_hn_density |   mean_uc_density |
|:-------------------------|-------------------:|--------------:|----------------:|-------------:|--------------------:|:------------------|--------------------:|:------------------|------------------:|------------------:|------------------:|
| global_top_10pct_longest |                144 |       195.542 |             201 |          201 |                  12 | 8.3%              |                  55 | 38.2%             |             0.67  |             0.06  |             0.065 |
| global_remaining_90pct   |               1286 |        47.082 |              26 |          125 |                 499 | 38.8%             |                 593 | 46.1%             |             0.713 |             0.248 |             0.198 |
| OpenHands_external       |                330 |       154.576 |             155 |          201 |                   4 | 1.2%              |                  50 | 15.2%             |             0.607 |             0.025 |             0.001 |
| SWE_agent_datasets       |               1100 |        34.269 |              20 |           87 |                 507 | 46.1%             |                 598 | 54.4%             |             0.739 |             0.29  |             0.24  |

## RQ3: How often do hard negatives and unproductive cycles overlap?

HN and UC overlap at the trajectory level in some datasets, but neither construct contains the other. HN-only trajectories are intervention-risk cases because they contain behavior that may look repetitive while remaining productive. UC-only trajectories capture unproductive cycling without the hard-negative caveat. Mixed HN+UC trajectories show that local labels can coexist within the same long debugging run.

| dataset              | dataset_role                             | category          |   trajectory_count | trajectory_pct   |
|:---------------------|:-----------------------------------------|:------------------|-------------------:|:-----------------|
| Natural500           | natural-distribution control             | HN_only           |                 94 | 18.8%            |
| Natural500           | natural-distribution control             | UC_only           |                 63 | 12.6%            |
| Natural500           | natural-distribution control             | mixed_HN_and_UC   |                 71 | 14.2%            |
| Natural500           | natural-distribution control             | neither_HN_nor_UC |                272 | 54.4%            |
| Enriched500             | enriched difficult loop-heavy set        | HN_only           |                146 | 29.2%            |
| Enriched500             | enriched difficult loop-heavy set        | UC_only           |                 50 | 10.0%            |
| Enriched500             | enriched difficult loop-heavy set        | mixed_HN_and_UC   |                241 | 48.2%            |
| Enriched500             | enriched difficult loop-heavy set        | neither_HN_nor_UC |                 63 | 12.6%            |
| StressFresh100       | clean stress challenge                   | HN_only           |                 16 | 16.0%            |
| StressFresh100       | clean stress challenge                   | UC_only           |                 52 | 52.0%            |
| StressFresh100       | clean stress challenge                   | mixed_HN_and_UC   |                 30 | 30.0%            |
| StressFresh100       | clean stress challenge                   | neither_HN_nor_UC |                  2 | 2.0%             |
| OpenHandsExternal330 | external long-horizon generalization set | HN_only           |                 47 | 14.2%            |
| OpenHandsExternal330 | external long-horizon generalization set | UC_only           |                  1 | 0.3%             |
| OpenHandsExternal330 | external long-horizon generalization set | mixed_HN_and_UC   |                  3 | 0.9%             |
| OpenHandsExternal330 | external long-horizon generalization set | neither_HN_nor_UC |                279 | 84.5%            |

### Span-Level HN/UC Overlap

| dataset              | dataset_role                             |   hn_span_count |   uc_span_count |   overlapping_hn_uc_span_pairs |   hn_spans_overlapping_uc_count |   uc_spans_overlapping_hn_count |   mean_iou_for_overlapping_pairs |
|:---------------------|:-----------------------------------------|----------------:|----------------:|-------------------------------:|--------------------------------:|--------------------------------:|---------------------------------:|
| Natural500           | natural-distribution control             |             233 |             235 |                             38 |                              34 |                              35 |                            0.503 |
| Enriched500             | enriched difficult loop-heavy set        |             832 |            1036 |                            240 |                             192 |                             205 |                            0.61  |
| StressFresh100       | clean stress challenge                   |              68 |             191 |                              0 |                               0 |                               0 |                            0     |
| OpenHandsExternal330 | external long-horizon generalization set |              59 |               5 |                              0 |                               0 |                               0 |                            0     |

## RQ4: Do productive-looking and unproductive regions differ in local process signals?

The normalized tables do not retain enough raw action/tool/file/error text to compute every requested local feature directly. For Enriched500 and Natural500, historical release-derived signal scores provide partial proxies for progress, novelty, recurrence, and error persistence. For StressFresh100 and OpenHandsExternal330, these raw local process signals are not available in the normalized audit tables, so the analysis documents the limitation instead of inventing features.

| feature                      | status                                          | reason                                                                                                                      |
|:-----------------------------|:------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------|
| repeated same action         | unavailable                                     | The normalized audit tables do not preserve raw action text or canonical action identifiers.                                |
| repeated tool                | unavailable                                     | The normalized audit tables do not preserve tool_name/tool_call fields.                                                     |
| repeated file                | unavailable                                     | The normalized audit tables do not preserve file paths or edited/read file fields.                                          |
| repeated error signature     | partially_available_for_Enriched500_and_Natural500 | Historical release-derived signal scores include error_persistence and recurrence, but normalized tables do not preserve raw stack traces. |
| new file/function touched    | unavailable                                     | The normalized audit tables do not preserve file/function identifiers.                                                      |
| action change after feedback | unavailable                                     | The normalized audit tables do not preserve consecutive action semantics.                                                   |
| new observation or evidence  | partially_available_for_Enriched500_and_Natural500 | Historical release-derived signal scores include novelty and progress proxies, but not the raw observations themselves.                    |
| same-error persistence       | partially_available_for_Enriched500_and_Natural500 | Historical release-derived signal scores include error_persistence proxies.                                                                |

### Local Signal Proxy Summary

| dataset    | region_type             |   step_count |   rows_with_legacy_signal_scores |   legacy_signal_coverage_pct |   progress_score_mean |   progress_score_median |   recurrence_score_mean |   recurrence_score_median |   novelty_score_mean |   novelty_score_median |   error_persistence_score_mean |   error_persistence_score_median |   budget_pressure_score_mean |   budget_pressure_score_median |
|:-----------|:------------------------|-------------:|---------------------------------:|-----------------------------:|----------------------:|------------------------:|------------------------:|--------------------------:|---------------------:|-----------------------:|-------------------------------:|---------------------------------:|-----------------------------:|-------------------------------:|
| Natural500 | HN                      |          524 |                              524 |                            1 |                 1.324 |                       1 |                   1.086 |                         1 |                1.071 |                      1 |                          0.672 |                                0 |                        0.866 |                              1 |
| Natural500 | PI_non_HN               |         4689 |                             4689 |                            1 |                 2.345 |                       2 |                   0.09  |                         0 |                1.823 |                      2 |                          0.04  |                                0 |                        0.799 |                              0 |
| Natural500 | UC                      |         2089 |                             2089 |                            1 |                 0.487 |                       0 |                   2.429 |                         3 |                0.209 |                      0 |                          1.755 |                                2 |                        1.241 |                              1 |
| Natural500 | other                   |         1227 |                             1227 |                            1 |                 1.399 |                       1 |                   0.509 |                         0 |                0.923 |                      1 |                          0.274 |                                0 |                        0.995 |                              1 |
| Natural500 | uncertain_or_unresolved |           21 |                               21 |                            1 |                 1.286 |                       1 |                   0.524 |                         0 |                0.857 |                      1 |                          0.286 |                                0 |                        1.81  |                              3 |
| Enriched500   | HN                      |         1757 |                             1757 |                            1 |                 1.365 |                       1 |                   1.453 |                         2 |                1.199 |                      1 |                          0.894 |                                0 |                        0.861 |                              1 |
| Enriched500   | PI_non_HN               |         9693 |                             9693 |                            1 |                 1.976 |                       2 |                   0.446 |                         0 |                1.629 |                      2 |                          0.201 |                                0 |                        0.722 |                              0 |
| Enriched500   | UC                      |        12397 |                            12397 |                            1 |                 0.461 |                       0 |                   2.58  |                         3 |                0.186 |                      0 |                          1.642 |                                2 |                        1.279 |                              1 |
| Enriched500   | other                   |         2039 |                             2039 |                            1 |                 1.378 |                       1 |                   0.795 |                         0 |                0.965 |                      1 |                          0.493 |                                0 |                        0.734 |                              0 |
| Enriched500   | uncertain_or_unresolved |          128 |                              128 |                            1 |                 0.875 |                       1 |                   1.688 |                         2 |                0.414 |                      0 |                          1.359 |                                2 |                        1.25  |                              1 |

## Paper-Ready Findings

- The four evidence pools occupy different construct regimes and should not be pooled as one natural distribution.
- Productive iteration is common across datasets, but hard negatives are a narrower construct tied to intervention risk.
- Unproductive cycles are dataset-dependent: they are prominent in Enriched500 and StressFresh100 but rare in OpenHandsExternal330 despite its much longer trajectories.
- Long trajectories do not necessarily imply unproductive cycling; length and repetition need local semantic interpretation.
- HN-only and UC-only trajectories both exist, showing that hard negatives and unproductive cycles are related but non-equivalent.
- Mixed HN+UC trajectories show that a single debugging run can contain productive repetition and unproductive cycling in different local regions.
- The available local signal proxies support construct separation, but full action/tool/file/error feature analysis requires retaining raw process fields in the normalized tables.
- These findings motivate intervention-risk evaluation: a detector should distinguish when to interrupt unproductive cycling from when to allow productive iteration to continue.

## Reproducibility Notes

- Inputs: `outputs/paper1_audit/normalized_steps.csv` and `outputs/paper1_audit/normalized_spans.csv`.
- No raw annotation files were modified.
- The analysis treats UC, HN, and productive iteration as annotated constructs, not as claims that one label family is the only ground truth.