# Live Guard Experiment Report

Generated: 2026-06-15T03:25:49Z

Status: **LIVE_RESULTS_AVAILABLE**

## Scope

- This experiment executes a real mini-swe-agent runner on SWE-bench-style Docker tasks with DeepSeek API calls.
- Official resolved/tests-pass oracle was not run by this analyzer unless `oracle_available=true` appears in the run log.
- Therefore task-success claims are allowed only if oracle availability is confirmed.

## Completion

- Total task-arm attempts logged: 34
- Infrastructure-failed attempts retained in audit log: 4
- Latest task-arm rows used for arm summaries: 30
- `NO_GUARD` completed tasks: 10
- `WARN_REPLAN_GUARD` completed tasks: 10
- `HARD_STOP_GUARD` completed tasks: 10
- Oracle success available: no
- Paper integration threshold met for NO_GUARD and WARN_REPLAN_GUARD: no

## Summary By Arm

| Arm | N rows | Completed | Guard trigger rate | Patch generated rate | Mean steps | Mean repeated file rate | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| NO_GUARD | 10 | 10 | 1.000 | 0.000 | 10.00 | 0.474 | 56256.0 |
| WARN_REPLAN_GUARD | 10 | 10 | 0.800 | 0.000 | 10.00 | 0.374 | 60433.3 |
| HARD_STOP_GUARD | 10 | 10 | 1.000 | 0.000 | 4.80 | 0.459 | 19714.9 |

## Pairwise Effects

| Comparison | Metric | N | Mean diff | 95% CI |
|---|---|---:|---:|---|
| WARN_REPLAN_GUARD minus NO_GUARD | steps | 10 | 0.0000 | [0.0000, 0.0000] |
| WARN_REPLAN_GUARD minus NO_GUARD | tokens_total | 10 | 4177.3000 | [-6738.7000, 16462.6000] |
| WARN_REPLAN_GUARD minus NO_GUARD | repeated_file_rate | 10 | -0.0998 | [-0.2314, 0.0262] |
| WARN_REPLAN_GUARD minus NO_GUARD | repeated_tool_rate | 10 | 0.0605 | [-0.0402, 0.1616] |
| WARN_REPLAN_GUARD minus NO_GUARD | patch_generated_num | 10 | 0.0000 | [0.0000, 0.0000] |
| HARD_STOP_GUARD minus NO_GUARD | steps | 10 | -5.2000 | [-6.0000, -4.2000] |
| HARD_STOP_GUARD minus NO_GUARD | tokens_total | 10 | -36541.1000 | [-45589.7000, -27035.4000] |
| HARD_STOP_GUARD minus NO_GUARD | repeated_file_rate | 10 | -0.0150 | [-0.1201, 0.0995] |
| HARD_STOP_GUARD minus NO_GUARD | repeated_tool_rate | 10 | -0.0652 | [-0.1752, 0.0595] |
| HARD_STOP_GUARD minus NO_GUARD | patch_generated_num | 10 | 0.0000 | [0.0000, 0.0000] |

## Interpretation Boundary

- If oracle availability is `no`, the evidence is live tool-level policy evidence, not task-success evidence.
- Negative, mixed, or underpowered effects must be reported as such; no selective exclusion is applied here.
