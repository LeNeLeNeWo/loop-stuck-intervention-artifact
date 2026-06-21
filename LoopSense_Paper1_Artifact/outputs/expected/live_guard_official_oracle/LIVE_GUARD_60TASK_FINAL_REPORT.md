# Live Guard 60-Task Final Report

Generated: 2026-06-16T19:04:59Z

Final status: **READY_WITH_60TASK_LIVE_EVIDENCE**

## Frozen Sample Validation

- The initial frozen prefix matched the preregistered sample order.
- The extension tasks were selected from the existing frozen sample order; no replacement or cherry-picking was performed.
- Guard, prompt, budget, runner, model, arms, and oracle were unchanged.

## Completion and Oracle Rows

- Expected total task-arm rows for 60 tasks: 180.
- Latest completed task-arm rows: 180/180.
- Expected official oracle outcomes: 180.
- Official oracle outcome rows: 180/180.
- Raw infrastructure_failed rows during extension/retry: 25; final latest infrastructure failures: 0.
- Final latest timeouts/errors: 0/0.

## 60-Task Results

- `NO_GUARD`: resolved/test-pass=35/60, patch_generated=48/60, mean_steps=42.0, mean_tokens=758,692, post_trigger_repetition=0.504
- `WARN_REPLAN_GUARD`: resolved/test-pass=42/60, patch_generated=54/60, mean_steps=40.2, mean_tokens=744,589, post_trigger_repetition=0.530
- `HARD_STOP_GUARD`: resolved/test-pass=8/60, patch_generated=9/60, mean_steps=8.0, mean_tokens=67,225, post_trigger_repetition=0.467

## Pairwise Effects, 60 Tasks

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

## Interpretation

- WARN_REPLAN helped outcomes in this 60-task sample: resolved/test-pass increased from 35/60 to 42/60, with paired bootstrap CI above zero.
- WARN_REPLAN increased patch generation from 48/60 to 54/60; steps and tokens decreased slightly but CIs cross zero.
- HARD_STOP still saves execution and sharply hurts outcomes: resolved/test-pass dropped to 8/60 and patch generation to 9/60 while mean steps and tokens fell strongly.
- Post-trigger repetition is not uniformly reduced by WARN_REPLAN.

## Paper and Artifact Status

- Paper updated to 60-task live evidence: yes.
- Artifact updated and clean-room status: passed.
- zip_path: `release_artifacts/LoopSense_Paper1_Artifact.zip`; size_bytes=7192087
- Current PDF page count is recorded by the final release audit.
