# Live Guard Final Report

Generated: 2026-06-15T03:25:49Z

Final feasibility level: public benchmark live policy
- Runner used: mini-swe-agent local package.
- Model: `deepseek-v4-flash`
- Task source: `princeton-nlp/SWE-Bench_Verified:test` public benchmark fallback.
- Traceability level: public_benchmark_fallback
- Task arms run: HARD_STOP_GUARD, NO_GUARD, WARN_REPLAN_GUARD
- Latest completed tasks per arm: NO_GUARD=10, WARN_REPLAN_GUARD=10, HARD_STOP_GUARD=10
- Infrastructure-failed attempts retained: 4
- Oracle availability: no
- Paper integration threshold met: no

## Primary Outcomes

- NO_GUARD mean steps: 10.0000; mean repeated-file rate: 0.4740; mean tokens: 56256.0000.
- WARN_REPLAN_GUARD mean steps: 10.0000; mean repeated-file rate: 0.3743; mean tokens: 60433.3000.
- HARD_STOP_GUARD mean steps: 4.8000; mean repeated-file rate: 0.4590; mean tokens: 19714.9000.
- WARN_REPLAN minus NO_GUARD repeated-file-rate effect: -0.0998 [-0.2314, 0.0262].
- WARN_REPLAN minus NO_GUARD token effect: 4177.3000 [-6738.7000, 16462.6000].
- HARD_STOP minus NO_GUARD step effect: -5.2000 [-6.0000, -4.2000].
- HARD_STOP minus NO_GUARD token effect: -36541.1000 [-45589.7000, -27035.4000].

## Interpretation

- The pilot provides real tool-level live-policy evidence, not task-success evidence.
- Warning/replanning shows a negative repeated-file-rate point estimate, but the bootstrap CI crosses zero at N=10.
- Hard stop saves steps and tokens in this short-budget pilot, but no oracle is available to evaluate success loss.
- No abstract or main-result success claim should be added from this pilot alone.

Final status: **NOT READY for live task-success manuscript claims; READY as artifact pilot evidence**.
