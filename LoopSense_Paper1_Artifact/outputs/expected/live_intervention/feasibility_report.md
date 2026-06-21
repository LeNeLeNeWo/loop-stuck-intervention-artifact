# Live / Replay Prefix-Branch Feasibility Report

This audit checks whether a true prefix-branch intervention experiment can be executed from the current repository state. It does not run agents.

Conclusion: **NOT_FEASIBLE**

## Primary Analysis Inputs

| Input | Exists | Size bytes | Columns (prefix) |
|---|---:|---:|---|
| normalized_steps | yes | 78897881 | dataset, dataset_role, dataset_subsource, source_file, source_format, row_number, trajectory_id, task_id, step_id_raw, step_index, label_raw, label_norm, span_label_raw, span_label_norm, hard_negative_raw, hard_negative |
| normalized_spans | yes | 2838876 | dataset, dataset_role, dataset_subsource, source_file, source_format, span_source, row_number, trajectory_id, span_id, start_step_index, end_step_index, span_label_raw, span_label_norm, hard_negative_raw, hard_negative, productive_iteration_flag |
| stop_counterfactual_triggers | yes | 62693661 | trigger_id, dataset, trajectory_id, step_index, detector, detector_family, detector_config, trigger_location_type, trajectory_final_success, current_prefix_success, post_trigger_success, post_trigger_validation_event, remaining_steps_after_trigger, remaining_productive_steps_after_trigger, remaining_PI_steps_after_trigger, remaining_HN_steps_after_trigger |
| detector_metrics | yes | 19633 | dataset, dataset_role, detector, detector_family, threshold/config, uses_annotation_signal_proxy, F1, UC_recall, PIIR, HN_FPR, burden, latency_mean, latency_median, latency_p90, n_triggers, positive_label_definition |
| enriched_steps | yes | 192473603 | dataset, dataset_role, dataset_subsource, source_file, source_format, row_number, trajectory_id, task_id, step_id_raw, step_index, label_raw, label_norm, span_label_raw, span_label_norm, hard_negative_raw, hard_negative |

## Replay Requirements

| Requirement | Available | Evidence |
|---|---:|---|
| raw_trajectory_logs | yes | data/raw, logs |
| task_metadata | yes | data/raw/swe |
| repo_checkouts | no | not found |
| action_replay | no | not found |
| swe_runner | no | not found |
| openhands_runner | no | not found |
| model_config | yes | configs, .env.example |
| test_oracle | yes | tests |
| intervention_injection | yes | scripts/paper1_live_intervention/run_prefix_branch_experiment.py |
| model_credentials | yes | credential environment variable detected; name withheld |

## Decision Details

- Can restore prefix state: no.
- Can run an agent with intervention prompt injection: no.
- Can evaluate resolved/tests-pass outcomes: yes.
- Model credentials detected in current environment: yes; variable name withheld in the release artifact.

## Missing Pieces

- repo_checkouts
- action_replay
- swe_runner
- openhands_runner

## Consequence

Because the current repository does not expose a complete prefix-state replay stack, no live/replay intervention results are produced. The existing offline hard-stop counterfactual remains the strongest executable intervention-validation evidence in this artifact.
