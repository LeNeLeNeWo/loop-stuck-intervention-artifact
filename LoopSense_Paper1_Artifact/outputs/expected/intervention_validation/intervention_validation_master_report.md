# Intervention Validation Master Report

## 1. Feasibility Audit Result

The audit recommends offline stop-counterfactual validation only. Live prefix-branch reruns were not executed because the repository does not expose the complete replay stack needed to restore prefix state, rerun an agent, and evaluate task success under controlled arms.

## 2. Offline Stop-Counterfactual Metrics

Detector/config rows summarized: 58.
Trigger-location rows summarized: 20.
Stop-Harm Rate is unavailable because final success and post-trigger successful validation are not structured in the primary inputs.

## 3. Live Prefix-Branch Rerun Feasibility

Feasibility-only note: `prefix_branch_feasibility_only.md`.

## 4. Figures Generated

- `fig_stop_counterfactual_tradeoff.pdf/png/svg`
- `fig_stop_harm_by_location.pdf/png/svg`
- `fig_near_f1_stop_counterfactual.pdf/png/svg`

## 5. Tables Generated

- `stop_counterfactual_triggers.csv`
- `stop_counterfactual_summary_by_detector.csv`
- `stop_counterfactual_summary_by_location.csv`
- `stop_counterfactual_near_f1_pairs.csv`

## 6. Remaining Limitations

The offline analysis evaluates hard-stop consequences on observed suffixes. It cannot estimate stochastic rerun behavior, warn/replan/handoff effects, or final task success deltas without a replayable live experiment.
