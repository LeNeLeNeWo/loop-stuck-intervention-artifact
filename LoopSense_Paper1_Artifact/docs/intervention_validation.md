# Intervention Validation

The artifact contains three manuscript-level intervention-validation layers plus a small supplementary live pilot: offline hard-stop counterfactuals, API-assisted warning/replan probes, and a 60-task official-oracle live guard experiment.

## Offline hard-stop counterfactual

Offline stop-counterfactual validation asks what observed suffix would be cut off if the first detector/guard trigger became a hard stop. It reports Productive Suffix Loss (PSL) and UC Waste Saved (UWS). This analysis targets observed hard-stop consequences on recorded traces; it does not estimate live causal success effects.

Expected outputs are in `outputs/expected/intervention_validation/`, including anonymized trigger-level rows, detector/location summaries, near-F1 pair summaries, and the master report.

## API-assisted warning/replan experiment

`outputs/expected/api_replan_intervention/` contains the sanitized outputs of a real DeepSeek API-assisted prefix replan experiment. The experiment samples 180 real debugging prefixes from Enriched500 and StressFresh100, runs each prefix under a neutral continuation prompt and a WARN_REPLAN prompt, and parses 360 structured JSON responses.

This is a prompt-level behavioral experiment. It does not restore repository state, execute tools, run tests, or estimate final task success. It tests whether a generic warning/replan intervention changes the proposed next-step plan on real prefixes.

Headline results:

- 180 paired prefixes; 360 parsed JSON responses.
- Overall action-shift rate under WARN_REPLAN: 0.711 with paired bootstrap CI [0.644, 0.778].
- Progress-seeking score increases from 2.00 to 2.17; difference 0.167 with CI [0.033, 0.300].
- Overall repeated-pattern rate is 0.639 under neutral continuation and 0.656 under WARN_REPLAN, so repetition does not fall uniformly.
- Productive-risk unsafe over-stop decreases from 0.028 to 0.014.

The public artifact includes sanitized prompts, a sanitized API call log, parsed responses, summary tables, pairwise effects, and API-assisted figure source outputs. Raw API responses and private prefix-to-trajectory mappings are intentionally excluded.

## Live/replay feasibility

`outputs/expected/live_intervention/` contains a feasibility-only report. No live/replay outcome claims are included because the full replay stack is not available in the public artifact.

## Live guard-policy pilot

`outputs/expected/live_guard_experiment/` contains a sanitized 10-task live guard-policy pilot on public SWE-Bench Verified fallback tasks using mini-swe-agent and DeepSeek. The pilot compares `NO_GUARD`, `WARN_REPLAN_GUARD`, and `HARD_STOP_GUARD` under a fixed 10-step budget.

This is real tool-level execution evidence: Docker task environments were started, the model issued tool calls, and guard decisions were logged. It is not a task-success experiment because no official resolved/tests-pass oracle was available in this release run. The paper integration threshold was not met, so the pilot is included as supplementary artifact evidence rather than a manuscript headline result.

Headline pilot results:

- 10 completed tasks per arm; 4 earlier infrastructure-failed dry-run attempts are retained in the audit log.
- WARN_REPLAN mean repeated-file rate was 0.374 versus 0.474 for NO_GUARD; paired difference -0.100 with 95% CI [-0.231, 0.026].
- HARD_STOP reduced mean steps by 5.2 and mean tokens by about 36.5k per task, but success loss cannot be evaluated without an oracle.

Raw live trajectories are not released because they contain full model context, task text, tool outputs, and local paths. The artifact includes sanitized run logs, summaries, figures, preregistration, and release notes.

## Official-oracle live guard experiment
`outputs/expected/live_guard_official_oracle/` contains preregistered official-oracle live guard outputs. The manuscript was updated because the 60-task official-outcome threshold was met for NO_GUARD and WARN_REPLAN_GUARD.

Headline official-oracle results:

- 60 frozen SWE-bench Verified tasks; 180 completed task-arm runs.
- Resolved/test-pass outcomes: NO_GUARD 35/60, WARN_REPLAN_GUARD 42/60, HARD_STOP_GUARD 8/60.
- Patch generation: NO_GUARD 48/60, WARN_REPLAN_GUARD 54/60, HARD_STOP_GUARD 9/60.
- Mean steps: NO_GUARD 42.0, WARN_REPLAN_GUARD 40.2, HARD_STOP_GUARD 8.0.
- Mean tokens: NO_GUARD 758,692, WARN_REPLAN_GUARD 744,589, HARD_STOP_GUARD 67,225.
- WARN_REPLAN_GUARD minus NO_GUARD resolved/test-pass difference is 0.117 with paired bootstrap 95% CI [0.050, 0.200].
- HARD_STOP_GUARD minus NO_GUARD resolved/test-pass difference is -0.450 with paired bootstrap 95% CI [-0.583, -0.317].
- Empty-patch rows are official harness empty-patch outcomes and are counted as unresolved; raw trajectories and benchmark workspaces are not released.
- Boundary: the live experiment uses one model, one runner, public benchmark fallback tasks, and a fixed 50-step budget.
