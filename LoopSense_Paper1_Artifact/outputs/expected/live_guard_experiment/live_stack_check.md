# Live Guard Stack Check

Decision: **LIVE_POLICY_EXPERIMENT_FEASIBLE_WITH_PUBLIC_BENCHMARK**

Preregistration: `outputs\paper1_live_guard_experiment\LIVE_GUARD_PREREGISTRATION.md`

## DeepSeek API

- `DEEPSEEK_BASE_URL`: present
- `DEEPSEEK_API_KEY`: present; value not printed
- `DEEPSEEK_MODEL`: deepseek-v4-flash
- Endpoint: `https://api.deepseek.com/<masked-path>`
- Test call: pass
- Latency seconds: 1.0994792999699712
- Token usage keys: completion_tokens, completion_tokens_details, prompt_cache_hit_tokens, prompt_cache_miss_tokens, prompt_tokens, prompt_tokens_details, total_tokens

## System

- Python: 3.13.9 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 19:09:58) [MSC v.1929 64 bit (AMD64)]
- Platform: Windows-11-10.0.26200-SP0
- Disk free GB: 127.03
- Git: pass
- Docker daemon: pass
- Network: pass (github.com:443)

## Python Modules

- `datasets`: present
- `swebench`: missing
- `docker`: missing
- `litellm`: present
- `pandas`: present
- `matplotlib`: present

## Existing Data Inputs

| Input | Exists | Size bytes | Columns prefix |
|---|---:|---:|---|
| normalized_steps | yes | 78897881 | dataset, dataset_role, dataset_subsource, source_file, source_format, row_number, trajectory_id, task_id, step_id_raw, step_index, label_raw, label_norm, span_label_raw, span_label_norm, hard_negative_raw, hard_negative, productive_iteration_flag, unproductive_cycle_flag, uncertain_flag, unresolved_flag, confidence, annotator_id, evidence_text, span_id_raw |
| normalized_spans | yes | 2838876 | dataset, dataset_role, dataset_subsource, source_file, source_format, span_source, row_number, trajectory_id, span_id, start_step_index, end_step_index, span_label_raw, span_label_norm, hard_negative_raw, hard_negative, productive_iteration_flag, unproductive_cycle_flag, uncertain_flag, unresolved_flag, confidence, evidence_text, step_count, raw_extra_json |
| stop_counterfactual_triggers | yes | 62693661 | trigger_id, dataset, trajectory_id, step_index, detector, detector_family, detector_config, trigger_location_type, trajectory_final_success, current_prefix_success, post_trigger_success, post_trigger_validation_event, remaining_steps_after_trigger, remaining_productive_steps_after_trigger, remaining_PI_steps_after_trigger, remaining_HN_steps_after_trigger, remaining_UC_steps_after_trigger, remaining_productive_spans_after_trigger, remaining_UC_spans_after_trigger, would_stop_before_observed_success, would_cut_productive_suffix, would_save_UC_suffix, is_first_trigger_for_config_trajectory, evidence_available_flag |
| provenance_summary | yes | 6756 | pool, source_framework, model_name, model_provider, benchmark_or_task_source, task_count, trajectory_log_source, confidence, evidence_files, notes |

## Runner Candidates

- `mini_swe_agent`: available
- `swe_agent`: not available
- `openhands`: not available

## Interpretation

- Full task-success claims require both a runnable agent/tool stack and an available task oracle.
- If the SWE-bench oracle package remains unavailable, live runs may still support tool-level repetition/cost evidence, but not resolved/tests-pass claims.
- No raw API key, authorization header, or local path is written to this report.
