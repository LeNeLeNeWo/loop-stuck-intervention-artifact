# API Replan Prefix Sampling Report

Target prefixes: 180
Selected prefixes: 180
Prompt arms per prefix: 2
Prompt rows: 360
Seed: 20270614

## Group Composition

| stratum | dataset_display | n |
| --- | --- | --- |
| matched_control | Enriched500 | 11 |
| matched_control | StressFresh100 | 25 |
| productive_risk | Enriched500 | 59 |
| productive_risk | StressFresh100 | 13 |
| uc | Enriched500 | 59 |
| uc | StressFresh100 | 13 |

## Detector Composition

| stratum | detector_family | detector_config | n |
| --- | --- | --- | --- |
| matched_control | file_revisit_guard | window=5;repeats=3 | 36 |
| productive_risk | file_revisit_guard | window=5;repeats=3 | 72 |
| uc | file_revisit_guard | window=5;repeats=3 | 72 |

## Privacy Boundary

- `prefix_replan_sample.csv` and `prefix_replan_prompts.jsonl` use released prefix ids only.
- `prefix_replan_sample_private.csv` contains raw trajectory ids for local audit and must not be released.
- Prompts do not include PI/HN/UC labels in the model-visible message text.
