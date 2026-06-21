# Headline Empirical Findings

These findings are extracted from the completed Paper1 audit, construct-mismatch, raw-enrichment, and intervention-risk outputs. They use exact values from the generated CSV files and avoid pooling datasets unless the source analysis explicitly reports a diagnostic cross-pool comparison.

## F01. Length is not UC

- **Dataset / contrast**: OpenHandsExternal330 vs SWE-agent datasets
- **Metric 1**: OpenHands mean length = `154.576`
- **Metric 2**: OpenHands UC prevalence = `1.2%`
- **Comparison**: OpenHandsExternal330 is much longer on average than SWE-agent datasets (154.576 vs 34.269 steps), but has lower UC prevalence (1.2% vs 46.1%).
- **Paper claim supported**: Long-horizon trajectories are not sufficient evidence of unproductive cycling.
- **Source**: `outputs/paper1_construct_mismatch/tables/length_comparison_summary.csv`

Cautious interpretation: this finding suggests a construct or intervention-risk pattern in the corresponding evidence pool; it should not be read as a claim that human labels are wrong or that UC is the only valid construct.

## F02. Length is not UC

- **Dataset / contrast**: All four evidence pools, reported as a diagnostic contrast
- **Metric 1**: UC prevalence in global top 10% longest = `8.3%`
- **Metric 2**: UC prevalence in remaining 90% = `38.8%`
- **Comparison**: The global top decile of longest trajectories has lower UC prevalence than the remaining trajectories (8.3% vs 38.8%).
- **Paper claim supported**: Length alone is an ambiguous signal for loop/stuck behavior.
- **Source**: `outputs/paper1_construct_mismatch/tables/length_comparison_summary.csv`

Cautious interpretation: this finding suggests a construct or intervention-risk pattern in the corresponding evidence pool; it should not be read as a claim that human labels are wrong or that UC is the only valid construct.

## F03. Evidence-pool contrast

- **Dataset / contrast**: StressFresh100
- **Metric 1**: UC prevalence = `82.0%`
- **Metric 2**: HN prevalence = `46.0%`
- **Comparison**: StressFresh100 is a clean stress challenge with high UC prevalence (82.0%) while still containing HN cases (46.0%).
- **Paper claim supported**: Stress evidence pools expose intervention-risk cases that are less visible in natural-distribution controls.
- **Source**: `outputs/paper1_construct_mismatch/construct_prevalence_by_dataset.csv`

Cautious interpretation: this finding suggests a construct or intervention-risk pattern in the corresponding evidence pool; it should not be read as a claim that human labels are wrong or that UC is the only valid construct.

## F04. Construct mismatch

- **Dataset / contrast**: OpenHandsExternal330
- **Metric 1**: PI prevalence = `99.7%`
- **Metric 2**: UC prevalence = `1.2%`
- **Comparison**: OpenHandsExternal330 has near-universal PI (329/330) but very rare UC (4/330).
- **Paper claim supported**: Productive iteration and unproductive cycling are related but non-equivalent constructs.
- **Source**: `outputs/paper1_construct_mismatch/construct_prevalence_by_dataset.csv`

Cautious interpretation: this finding suggests a construct or intervention-risk pattern in the corresponding evidence pool; it should not be read as a claim that human labels are wrong or that UC is the only valid construct.

## F05. HN/UC non-equivalence

- **Dataset / contrast**: Enriched500
- **Metric 1**: HN-only trajectories = `146 (29.2%)`
- **Metric 2**: UC-only trajectories = `50 (10.0%)`
- **Comparison**: Enriched500 contains HN-only, UC-only, and mixed HN+UC trajectories; mixed HN+UC is 241 trajectories (48.2%).
- **Paper claim supported**: HN and UC should not be collapsed into a single binary stuck construct.
- **Source**: `outputs/paper1_construct_mismatch/tables/hn_uc_overlap_trajectory_categories.csv`

Cautious interpretation: this finding suggests a construct or intervention-risk pattern in the corresponding evidence pool; it should not be read as a claim that human labels are wrong or that UC is the only valid construct.

## F06. F1 hides trigger-site risk

- **Dataset / contrast**: Enriched500
- **Metric 1**: max_step_guard threshold=50 F1 / PIIR = `0.719 / 0.101`
- **Metric 2**: file_revisit_guard window=5;repeats=3 F1 / PIIR = `0.729 / 0.562`
- **Comparison**: These two detectors differ by only 0.009 F1, but PIIR differs by 0.461 and HN-FPR by 0.230.
- **Paper claim supported**: Similar aggregate F1 can mask substantially different productive-interruption risk.
- **Source**: `outputs/paper1_intervention_risk_rich_detectors/near_f1_risk_pairs.csv`

Cautious interpretation: this finding suggests a construct or intervention-risk pattern in the corresponding evidence pool; it should not be read as a claim that human labels are wrong or that UC is the only valid construct.

## F07. Progress-insensitive repetition risk

- **Dataset / contrast**: Enriched500
- **Metric 1**: file_revisit_guard window=10;repeats=3 UC recall = `88.9%`
- **Metric 2**: PIIR / HN-FPR = `0.623 / 0.401`
- **Comparison**: The file revisit guard recalls 88.9% of UC spans, but also interrupts 62.3% of PI spans and triggers on 40.1% of HN-eligible steps.
- **Paper claim supported**: Repetition-based guards can be useful coarse diagnostics but require trigger-site risk evaluation.
- **Source**: `outputs/paper1_intervention_risk_rich_detectors/detector_metrics_by_dataset.csv`

Cautious interpretation: this finding suggests a construct or intervention-risk pattern in the corresponding evidence pool; it should not be read as a claim that human labels are wrong or that UC is the only valid construct.

## F08. Stress challenge exposes risk

- **Dataset / contrast**: StressFresh100
- **Metric 1**: file_revisit_guard window=10;repeats=3 UC recall = `95.8%`
- **Metric 2**: PIIR / HN-FPR = `0.868 / 0.797`
- **Comparison**: In StressFresh100, the same family reaches very high UC recall (95.8%) but also very high PIIR and HN-FPR.
- **Paper claim supported**: StressFresh100 surfaces intervention risks that a natural-control-only evaluation could understate.
- **Source**: `outputs/paper1_intervention_risk_rich_detectors/detector_metrics_by_dataset.csv`

Cautious interpretation: this finding suggests a construct or intervention-risk pattern in the corresponding evidence pool; it should not be read as a claim that human labels are wrong or that UC is the only valid construct.

## F09. Step caps are budget guards

- **Dataset / contrast**: OpenHandsExternal330
- **Metric 1**: max_step_guard threshold=25 burden / PIIR = `0.845 / 1.000`
- **Metric 2**: max_step_guard threshold=150 F1 / PIIR = `0.008 / 0.536`
- **Comparison**: On long OpenHands trajectories with rare UC, low step caps trigger broadly (threshold=25 burden 84.5%, PIIR 100.0%); even threshold=150 has F1 0.008 with PIIR 53.6%.
- **Paper claim supported**: Step caps can control budget but should not be interpreted as progress-aware loop detectors.
- **Source**: `outputs/paper1_intervention_risk_rich_detectors/detector_metrics_by_dataset.csv`

Cautious interpretation: this finding suggests a construct or intervention-risk pattern in the corresponding evidence pool; it should not be read as a claim that human labels are wrong or that UC is the only valid construct.

## F10. Progress-aware suppression

- **Dataset / contrast**: Enriched500
- **Metric 1**: exact_action_repeat_k k=3 F1 / PIIR / HN-FPR = `0.205 / 0.028 / 0.004`
- **Metric 2**: repeat_without_progress_guard runtime F1 / PIIR / HN-FPR = `0.197 / 0.019 / 0.003`
- **Comparison**: The progress-aware guard lowers PIIR from 0.028 to 0.019 and HN-FPR from 0.004 to 0.003, with F1 changing from 0.205 to 0.197.
- **Paper claim supported**: Simple progress-aware suppression is consistent with lower interruption risk, although utility tradeoffs remain.
- **Source**: `outputs/paper1_intervention_risk_rich_detectors/detector_metrics_by_dataset.csv`

Cautious interpretation: this finding suggests a construct or intervention-risk pattern in the corresponding evidence pool; it should not be read as a claim that human labels are wrong or that UC is the only valid construct.

## F11. Progress-aware suppression

- **Dataset / contrast**: StressFresh100
- **Metric 1**: tool_repeat_k k=3 F1 / PIIR / HN-FPR = `0.446 / 0.283 / 0.137`
- **Metric 2**: repeat_without_progress_guard tool-runtime F1 / PIIR / HN-FPR = `0.273 / 0.004 / 0.000`
- **Comparison**: On StressFresh100, progress-aware suppression reduces PIIR from 0.283 to 0.004 and HN-FPR from 0.137 to 0.000, while F1 decreases from 0.446 to 0.273.
- **Paper claim supported**: Progress awareness can reduce trigger-site risk, but the paper should report the corresponding utility tradeoff.
- **Source**: `outputs/paper1_intervention_risk_rich_detectors/detector_metrics_by_dataset.csv`

Cautious interpretation: this finding suggests a construct or intervention-risk pattern in the corresponding evidence pool; it should not be read as a claim that human labels are wrong or that UC is the only valid construct.

## F12. Raw-field availability caveat

- **Dataset / contrast**: Enriched500, StressFresh100, OpenHandsExternal330
- **Metric 1**: raw action coverage Enriched500 / StressFresh100 = `64.7% / 100.0%`
- **Metric 2**: raw action coverage OpenHandsExternal330 = `0.0%`
- **Comparison**: Richer repetition detectors are enabled mainly where raw runtime fields are recoverable; OpenHandsExternal330 active labels expose no raw action text in the current artifact.
- **Paper claim supported**: Detector-family comparisons should be interpreted with dataset-specific raw-field availability in mind.
- **Source**: `outputs/paper1_raw_enrichment/raw_field_availability_by_dataset.csv`

Cautious interpretation: this finding suggests a construct or intervention-risk pattern in the corresponding evidence pool; it should not be read as a claim that human labels are wrong or that UC is the only valid construct.
