# Official-Oracle Live Guard Experiment Preregistration

Experiment: `paper1_live_guard_official_oracle`

Generated: 2026-06-15T04:10:23Z

This preregistration is written before any full official-oracle live guard run.

## Research Question

Do guard-policy interventions in a real SE-agent loop reduce repetition/cost while preserving task success under an official SWE-bench-style oracle?

## Arms

1. `NO_GUARD`: normal mini-swe-agent runner behavior.
2. `WARN_REPLAN_GUARD`: when the guard fires, inject the frozen warning/replan instruction once.
3. `HARD_STOP_GUARD`: when the guard fires, stop and evaluate the current patch/state.

## Frozen Guard

- Primary guard: `file_revisit_guard window=5,repeats=3`.
- Warning/replan injection cap: at most once per run.
- Secondary safety cap: max steps and token/cost budget are the same for all arms.
- The prompt, guard, sample, metrics, and stopping rules must not be changed after outcomes are inspected.

## Sample

- Primary target: 60 tasks x 3 arms.
- Minimum manuscript-usable threshold: 30 tasks x 3 arms with official oracle results.
- If cost/runtime is high: complete the first 30 frozen tasks, then continue to 60 only if feasible.
- Task source fixed before outcomes: `princeton-nlp/SWE-Bench_Verified:test`.
- Fixed seed: `20270614`.
- Do not exclude hard tasks after seeing outcomes.
- Infrastructure failures are recorded separately as `infrastructure_failed`.

## Primary Metrics

- Official resolved / tests pass.
- Patch generated.
- Steps/actions.
- Tokens/cost when available.
- Guard triggered and trigger step.
- Post-trigger repetition rate.
- Repeated-file/action/tool/error counts.
- Timeout/failure mode.

## Statistical Plan

- Pair arms by task.
- Use paired bootstrap with 1000 resamples and report 95% confidence intervals.
- For paired binary resolved/tests-pass, report paired differences; McNemar may be added if straightforward.
- Report all tasks and all failed arms.
- No selective exclusion after outcomes.

## Manuscript Integration Rule

Add live task-success claims only if:

- at least 30 tasks have official oracle results for `NO_GUARD` and `WARN_REPLAN_GUARD`;
- `oracle_eval_results.csv` contains resolved/tests-pass fields from a real oracle run;
- no major privacy issue is detected; and
- results are interpretable.

If these conditions are not met, keep the run as artifact-only pilot/feasibility evidence.
