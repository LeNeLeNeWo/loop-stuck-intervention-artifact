# Extend To 60 Sample Validation

Generated: 2026-06-16T14:25:41Z
Status: **PASS**

## Checks

- required_files_present: PASS
- sample_has_at_least_60_tasks: PASS
- selected_order_unique_1_60: PASS
- selected_for_30_prefix_true: PASS
- selected_for_60_prefix_true: PASS
- first30_completed_rows_match_prefix: PASS
- completed_90_rows_for_first30_x_3_arms: PASS
- tasks31_60_not_completed: PASS
- task_source_ok: PASS
- traceability_level_ok: PASS
- arms_ok: PASS

## Counts

- sample_rows: 60
- first30_tasks: 30
- next30_tasks: 30
- existing_run_rows: 90
- next30_existing_task_arm_rows: 0
- required_missing: none

## Decision

Proceed with tasks 31-60 only.
