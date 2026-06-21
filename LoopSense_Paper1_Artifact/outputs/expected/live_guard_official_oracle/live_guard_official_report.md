# Official-Oracle Live Guard Report

Generated: 2026-06-16T19:02:14Z

Status: **READY_WITH_LIVE_TASK_SUCCESS_EVIDENCE**

## Completion

- `NO_GUARD`: attempted=60, completed=60, official_oracle_outcomes=60, oracle_test_runs=48, official_empty_patch=12, patch_generated_rate=0.800
- `WARN_REPLAN_GUARD`: attempted=60, completed=60, official_oracle_outcomes=60, oracle_test_runs=54, official_empty_patch=6, patch_generated_rate=0.900
- `HARD_STOP_GUARD`: attempted=60, completed=60, official_oracle_outcomes=60, oracle_test_runs=9, official_empty_patch=51, patch_generated_rate=0.150

## Pairwise Effects

| Comparison | Metric | N | Mean diff | 95% CI |
|---|---|---:|---:|---|
| WARN_REPLAN_GUARD minus NO_GUARD | resolved_num | 60 | 0.1167 | [0.0500, 0.2000] |
| WARN_REPLAN_GUARD minus NO_GUARD | tests_pass_num | 60 | 0.1167 | [0.0500, 0.2000] |
| WARN_REPLAN_GUARD minus NO_GUARD | steps | 60 | -1.7500 | [-3.7833, 0.4167] |
| WARN_REPLAN_GUARD minus NO_GUARD | tokens_total | 60 | -14103.0000 | [-89226.1000, 69239.5000] |
| WARN_REPLAN_GUARD minus NO_GUARD | post_trigger_repetition_rate | 60 | 0.0269 | [-0.0159, 0.0750] |
| WARN_REPLAN_GUARD minus NO_GUARD | patch_generated_num | 60 | 0.1000 | [0.0333, 0.1833] |
| HARD_STOP_GUARD minus NO_GUARD | resolved_num | 60 | -0.4500 | [-0.5833, -0.3167] |
| HARD_STOP_GUARD minus NO_GUARD | tests_pass_num | 60 | -0.4500 | [-0.5833, -0.3167] |
| HARD_STOP_GUARD minus NO_GUARD | steps | 60 | -33.9667 | [-37.2500, -30.7833] |
| HARD_STOP_GUARD minus NO_GUARD | tokens_total | 60 | -691467.2333 | [-803718.0500, -589787.0500] |
| HARD_STOP_GUARD minus NO_GUARD | post_trigger_repetition_rate | 60 | -0.0370 | [-0.0952, 0.0198] |
| HARD_STOP_GUARD minus NO_GUARD | patch_generated_num | 60 | -0.6500 | [-0.7667, -0.5167] |

## Interpretation Boundary

- Live task-success claims require at least 60 official oracle evaluations for NO_GUARD and WARN_REPLAN_GUARD.
- Empty-patch rows submitted to the official harness are counted as official outcomes with resolved/tests_pass=false and are reported separately from test-run rows.
- Oracle infrastructure errors are reported separately from resolved outcomes.
