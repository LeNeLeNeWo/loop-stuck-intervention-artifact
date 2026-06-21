# Intervention Validation Feasibility Report

Repository root: `<LOCAL_REPO_ROOT>`

## Offline Counterfactual Inputs

| Input | Exists | Size bytes | Key columns present |
|---|---:|---:|---|
| normalized_steps | yes | 78897881 | dataset, dataset_role, dataset_subsource, source_file, source_format, row_number, trajectory_id, task_id, step_id_raw, step_index ... |
| normalized_spans | yes | 2838876 | dataset, dataset_role, dataset_subsource, source_file, source_format, span_source, row_number, trajectory_id, span_id, start_step_index ... |
| detector_metrics | yes | 19633 | dataset, dataset_role, detector, detector_family, threshold/config, uses_annotation_signal_proxy, F1, UC_recall, PIIR, HN_FPR ... |
| trigger_location_distribution | yes | 18826 | dataset, location_category, trigger_count, trigger_pct, detector, threshold/config |
| near_f1_risk_pairs | yes | 17500 | dataset, within_0_01, within_0_02, detector_a, config_a, F1_a, PIIR_a, HN_FPR_a, detector_b, config_b ... |
| threshold_frontier | yes | 19633 | dataset, dataset_role, detector, detector_family, threshold/config, uses_annotation_signal_proxy, F1, UC_recall, PIIR, HN_FPR ... |
| enriched_steps | yes | 192473603 | dataset, dataset_role, dataset_subsource, source_file, source_format, row_number, trajectory_id, task_id, step_id_raw, step_index ... |
| length_vs_constructs | yes | 284872 | dataset, dataset_role, trajectory_id, source_family, trajectory_length, top_10pct_longest_global, top_10pct_longest_within_dataset, pi_density, hn_density, uc_density ... |

## Offline Field Availability

| Field group | Available |
|---|---:|
| normalized steps | yes |
| normalized spans | yes |
| detector metrics | yes |
| trigger location distribution | yes |
| step indices and trajectory ids | yes |
| span labels PI/HN/UC | yes |
| step-level progress signals | yes |

## Outcome and Validation Availability

Final task success fields in primary inputs: `unresolved_flag, unresolved_flag, unresolved_flag`.
Post-trigger validation fields in primary inputs: `none detected as structured fields`.

CSV header scan found the following success-like fields elsewhere in the repository outputs/data. These are reported for audit only and are not assumed to be usable without checking semantics:

- `outputs\paper1_audit\construct_prevalence_by_dataset.csv`: has_uncertain_or_unresolved_count, has_uncertain_or_unresolved_rate
- `outputs\paper1_audit\construct_trajectory_flags.csv`: unresolved_step_count, unresolved_span_count, has_uncertain_or_unresolved, uncertain_unresolved_density
- `outputs\paper1_audit\label_density_by_trajectory.csv`: unresolved_step_count, unresolved_span_count, has_uncertain_or_unresolved, uncertain_unresolved_density
- `outputs\paper1_audit\label_density_summary.csv`: uncertain_unresolved_density, uncertain_unresolved_density, uncertain_unresolved_density
- `outputs\paper1_audit\normalized_spans.csv`: unresolved_flag
- `outputs\paper1_audit\normalized_steps.csv`: unresolved_flag
- `outputs\paper1_audit\table2_trajectory_pi_hn_uc_prevalence.csv`: has_uncertain_or_unresolved_count, has_uncertain_or_unresolved_rate
- `outputs\paper1_audit\uncertain_unresolved_summary.csv`: unresolved_count
- `outputs\paper1_raw_enrichment\enriched_steps.csv`: unresolved_flag
- `outputs\paper1_construct_mismatch\tables\construct_trajectory_flags.csv`: unresolved_step_count, unresolved_span_count, has_uncertain_or_unresolved, uncertain_unresolved_density
- `nonreleased outcome-association artifact`: outcome_field, outcome_true_rate
- `<NONRELEASED_SOURCE_OUTCOME_ASSOCIATION>`: outcome_field, outcome_true_rate
- `<NONRELEASED_OPENHANDS_LABEL_SOURCE>`: first_pass_summary
- `nonreleased OpenHands span-source artifact`: first_pass_summary
- `nonreleased Natural500 outcome-association artifact`: outcome_field, outcome_true_rate

Validation-like fields elsewhere in repository outputs/data:

- none

## Live Prefix-Branch Rerun Requirements

| Requirement | Available | Evidence |
|---|---:|---|
| raw_trajectory_logs | yes | logs |
| task_metadata | no | not found |
| repo_checkouts | no | not found |
| agent_runner | no | not found |
| model_config | yes | configs, .env.example |
| test_oracle | no | not found |

## Decision

Offline stop-counterfactual feasible: **yes**.
Can identify final task success in primary inputs: **yes**.
Can identify structured post-trigger successful validation in primary inputs: **no**.
Can restore prefix state: **no**; no complete replayable prefix-state artifact was found by this audit.
Live prefix-branch rerun feasible: **no**.
Recommended execution: **offline stop-counterfactual only**.

## Caution

The offline validation can evaluate hard-stop consequences on observed traces by measuring productive suffix cut and UC suffix saved. It cannot estimate stochastic rerun outcomes, warn/replan/handoff behavior, or causal effects on final success when final success and prefix-state replay are unavailable.

Missing live rerun requirements:

- task_metadata
- repo_checkouts
- agent_runner
- test_oracle
