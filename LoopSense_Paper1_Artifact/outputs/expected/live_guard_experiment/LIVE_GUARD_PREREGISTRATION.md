# Live Guard Policy Experiment Preregistration

Experiment: `paper1_live_guard_experiment`

Generated: 2026-06-15T02:27:06Z

This preregistration is written before any full live guard-policy run. Dry-run and stack-validation commands may be used only to verify feasibility, logging, and safety.

## Research Questions

1. Does a warning/replan guard reduce repeated behavior and execution cost while preserving task success compared with no guard?
2. Does a hard-stop guard save execution cost but risk success loss when the trigger occurs in productive-risk settings?

## Arms

- `NO_GUARD`: run the agent normally under the fixed budget.
- `WARN_REPLAN_GUARD`: when the primary guard fires, inject one generic warning/replan message and continue.
- `HARD_STOP_GUARD`: when the primary guard fires, stop the run and evaluate the current patch/state if an oracle is available.
- `PROGRESS_AWARE_GUARD`: optional only if deterministic progress signals are straightforward; it must remain separate and cannot replace the primary arms.

## Frozen Guard Rules

- Primary guard: `file_revisit_guard window=5,repeats=3`.
- Secondary guard: `max_step_guard threshold=50`, used only as a secondary budget/safety signal if integration is straightforward.
- The guard rule, prompt, sample, stopping rules, and metrics must not be changed after seeing outcomes.

## Sample

- Prefer 60 runnable tasks; target 90 if budget permits; optionally 120 if runtime/cost remains reasonable.
- Minimum acceptable live task experiment: 30 tasks x 3 arms.
- The sample is fixed before outcome inspection.
- Prefer tasks traceable to Enriched500 and StressFresh100 with public benchmark/oracle availability.
- If historical task IDs cannot be mapped to runnable public tasks, use a public SWE-bench-compatible task subset and report the mismatch as `public_benchmark_fallback`.

## Metrics

Primary metrics:

- Resolved / tests pass / task success when an oracle is available.
- Steps/actions.
- Repeated action/file/tool/error rate.
- Guard trigger count.
- Cost/tokens.
- Wall-clock time.

Secondary metrics:

- Patch generated.
- Failure mode.
- No-trigger fraction.
- Intervention accepted.
- Post-trigger repetition rate.
- Post-trigger new-file/new-test/new-error-change count when measurable.

## Statistical Plan

- Compare arms on the same sampled tasks.
- Use paired bootstrap over tasks.
- For binary success, report paired difference and 95% CI; McNemar test if straightforward.
- For steps/cost/repetition, report mean/median difference and paired bootstrap 95% CI.
- Report all tasks, all arms, and all failures.
- No selective exclusion after outcomes.

## Interpretation

- Strong evidence: `WARN_REPLAN_GUARD` reduces repetition/cost with no large success drop.
- Hard-stop risk evidence: `HARD_STOP_GUARD` saves steps/cost but reduces success or stops productive tasks.
- Negative or mixed results must be reported honestly.
- If no full tool-level runner and oracle are available, the experiment must be reported as replay-lite or infeasible, not as live task-success evidence.
