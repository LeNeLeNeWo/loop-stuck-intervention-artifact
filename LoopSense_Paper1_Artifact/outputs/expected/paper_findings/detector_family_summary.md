# Detector Family Summary

Best risk-utility setting is selected by a descriptive score `F1 - 0.5*PIIR - 0.5*HN_FPR`. This score is used only to summarize configurations; it is not proposed as a paper metric.

## max_step_guard

- **Type**: progress-insensitive
- **Available in**: Natural500, Enriched500, StressFresh100, OpenHandsExternal330
- **Best risk-utility setting**: Enriched500 `threshold=50` (F1 0.719, PIIR 0.101, HN-FPR 0.118)
- **Worst PIIR/HN-FPR setting**: worst PIIR: OpenHandsExternal330 `threshold=25` = 1.000; worst HN-FPR: OpenHandsExternal330 `threshold=25` = 1.000
- **Interpretation**: Use this family as a detector-site risk profile, not as evidence that the underlying labels are wrong.

## exact_action_repeat_k

- **Type**: progress-insensitive
- **Available in**: Enriched500, StressFresh100
- **Best risk-utility setting**: StressFresh100 `k=4` (F1 0.287, PIIR 0.039, HN-FPR 0.038)
- **Worst PIIR/HN-FPR setting**: worst PIIR: StressFresh100 `k=2` = 0.116; worst HN-FPR: StressFresh100 `k=2` = 0.094
- **Interpretation**: Use this family as a detector-site risk profile, not as evidence that the underlying labels are wrong.

## tool_repeat_k

- **Type**: progress-insensitive
- **Available in**: StressFresh100
- **Best risk-utility setting**: StressFresh100 `k=4` (F1 0.418, PIIR 0.159, HN-FPR 0.062)
- **Worst PIIR/HN-FPR setting**: worst PIIR: StressFresh100 `k=2` = 0.516; worst HN-FPR: StressFresh100 `k=2` = 0.248
- **Interpretation**: Use this family as a detector-site risk profile, not as evidence that the underlying labels are wrong.

## tool_sequence_repeat

- **Type**: progress-insensitive
- **Available in**: StressFresh100
- **Best risk-utility setting**: StressFresh100 `window=3` (F1 0.496, PIIR 0.097, HN-FPR 0.060)
- **Worst PIIR/HN-FPR setting**: worst PIIR: StressFresh100 `window=2` = 0.349; worst HN-FPR: StressFresh100 `window=2` = 0.229
- **Interpretation**: Use this family as a detector-site risk profile, not as evidence that the underlying labels are wrong.

## action_observation_repeat

- **Type**: progress-insensitive
- **Available in**: Enriched500, StressFresh100
- **Best risk-utility setting**: StressFresh100 `exact_pair_k=2` (F1 0.277, PIIR 0.000, HN-FPR 0.000)
- **Worst PIIR/HN-FPR setting**: worst PIIR: Enriched500 `exact_pair_k=2` = 0.040; worst HN-FPR: Enriched500 `exact_pair_k=2` = 0.007
- **Interpretation**: Use this family as a detector-site risk profile, not as evidence that the underlying labels are wrong.

## action_observation_similarity

- **Type**: progress-insensitive
- **Available in**: Enriched500, StressFresh100
- **Best risk-utility setting**: StressFresh100 `tfidf_adjacent>=0.98` (F1 0.316, PIIR 0.120, HN-FPR 0.043)
- **Worst PIIR/HN-FPR setting**: worst PIIR: StressFresh100 `tfidf_adjacent>=0.9` = 0.283; worst HN-FPR: StressFresh100 `tfidf_adjacent>=0.9` = 0.100
- **Interpretation**: Use this family as a detector-site risk profile, not as evidence that the underlying labels are wrong.

## error_signature_repeat_k

- **Type**: progress-insensitive
- **Available in**: Enriched500, StressFresh100
- **Best risk-utility setting**: StressFresh100 `k=3` (F1 0.264, PIIR 0.023, HN-FPR 0.013)
- **Worst PIIR/HN-FPR setting**: worst PIIR: StressFresh100 `k=2` = 0.143; worst HN-FPR: StressFresh100 `k=2` = 0.049
- **Interpretation**: Use this family as a detector-site risk profile, not as evidence that the underlying labels are wrong.

## file_revisit_guard

- **Type**: progress-insensitive
- **Available in**: Enriched500, StressFresh100
- **Best risk-utility setting**: Enriched500 `window=5;repeats=5` (F1 0.639, PIIR 0.253, HN-FPR 0.169)
- **Worst PIIR/HN-FPR setting**: worst PIIR: StressFresh100 `window=10;repeats=3` = 0.868; worst HN-FPR: StressFresh100 `window=10;repeats=3` = 0.797
- **Interpretation**: Use this family as a detector-site risk profile, not as evidence that the underlying labels are wrong.

## repeat_without_progress_guard

- **Type**: progress-aware
- **Available in**: Enriched500, StressFresh100
- **Best risk-utility setting**: StressFresh100 `base=tool;k=3;progress=runtime` (F1 0.273, PIIR 0.004, HN-FPR 0.000)
- **Worst PIIR/HN-FPR setting**: worst PIIR: Enriched500 `base=exact_action;k=3;progress=runtime` = 0.019; worst HN-FPR: Enriched500 `base=exact_action;k=3;progress=runtime` = 0.003
- **Interpretation**: Use this family as a detector-site risk profile, not as evidence that the underlying labels are wrong.
