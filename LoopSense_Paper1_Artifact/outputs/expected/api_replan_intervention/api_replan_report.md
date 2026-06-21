# API-Assisted Prefix Replan Intervention Report

This experiment calls a real DeepSeek OpenAI-compatible chat-completions endpoint on sanitized debugging prefixes. It is a prompt-level prefix replan experiment: it does not restore repository state, execute tools, run tests, or estimate final task success.

## Run Summary

- Prompt rows parsed: 360
- JSON parsed OK rows: 360
- JSON parse success rate: 100.0%
- Prefixes with complete paired arms: 180

## Group-by-Arm Summary

| stratum | arm | N | repeat_pattern_rate | new_evidence_action_rate | action_shift_rate | progress_seeking_score_mean | continue_rate | unsafe_overstop_rate | json_parse_success_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_control | NEUTRAL_CONTINUE | 36 | 0.6667 | 0.9722 |  | 1.9444 | 0.9722 |  | 1.0000 |
| matched_control | WARN_REPLAN | 36 | 0.5556 | 0.9722 | 0.7500 | 2.2778 | 0.9722 |  | 1.0000 |
| productive_risk | NEUTRAL_CONTINUE | 72 | 0.5972 | 1.0000 |  | 2.1250 | 0.9722 | 0.0278 | 1.0000 |
| productive_risk | WARN_REPLAN | 72 | 0.6667 | 0.9861 | 0.6806 | 2.2083 | 0.9861 | 0.0139 | 1.0000 |
| uc | NEUTRAL_CONTINUE | 72 | 0.6667 | 0.9722 |  | 1.9028 | 0.9861 |  | 1.0000 |
| uc | WARN_REPLAN | 72 | 0.6944 | 1.0000 | 0.7222 | 2.0694 | 1.0000 |  | 1.0000 |

## Paired Effects

| stratum | metric | N_pairs | neutral_mean | warn_replan_mean | warn_minus_neutral | ci_low | ci_high | mcnemar_b_neutral_yes_warn_no | mcnemar_c_neutral_no_warn_yes | mcnemar_chi2 | mcnemar_p_approx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_control | repeat_pattern_rate | 36 | 0.6667 | 0.5556 | -0.1111 | -0.3333 | 0.1111 | 10 | 6 | 0.5625 | 0.45325470475373647 |
| matched_control | new_evidence_action_rate | 36 | 0.9722 | 0.9722 | 0.0000 | 0.0000 | 0.0000 |  |  |  |  |
| matched_control | progress_seeking_score_mean | 36 | 1.9444 | 2.2778 | 0.3333 | 0.0000 | 0.6667 |  |  |  |  |
| matched_control | continue_rate | 36 | 0.9722 | 0.9722 | 0.0000 | 0.0000 | 0.0000 |  |  |  |  |
| matched_control | action_shift_rate | 36 | 0.0000 | 0.7500 | 0.7500 | 0.6111 | 0.8889 |  |  |  |  |
| productive_risk | repeat_pattern_rate | 72 | 0.5972 | 0.6667 | 0.0694 | -0.0417 | 0.1944 | 7 | 12 | 0.8421052631578947 | 0.3587953578869416 |
| productive_risk | new_evidence_action_rate | 72 | 1.0000 | 0.9861 | -0.0139 | -0.0417 | 0.0000 |  |  |  |  |
| productive_risk | progress_seeking_score_mean | 72 | 2.1250 | 2.2083 | 0.0833 | -0.1111 | 0.2639 |  |  |  |  |
| productive_risk | continue_rate | 72 | 0.9722 | 0.9861 | 0.0139 | 0.0000 | 0.0417 |  |  |  |  |
| productive_risk | unsafe_overstop_rate | 72 | 0.0278 | 0.0139 | -0.0139 | -0.0417 | 0.0000 |  |  |  |  |
| productive_risk | action_shift_rate | 72 | 0.0000 | 0.6806 | 0.6806 | 0.5694 | 0.7778 |  |  |  |  |
| uc | repeat_pattern_rate | 72 | 0.6667 | 0.6944 | 0.0278 | -0.0972 | 0.1528 | 10 | 12 | 0.045454545454545456 | 0.8311704095417624 |
| uc | new_evidence_action_rate | 72 | 0.9722 | 1.0000 | 0.0278 | 0.0000 | 0.0694 |  |  |  |  |
| uc | progress_seeking_score_mean | 72 | 1.9028 | 2.0694 | 0.1667 | -0.0694 | 0.3889 |  |  |  |  |
| uc | continue_rate | 72 | 0.9861 | 1.0000 | 0.0139 | 0.0000 | 0.0417 |  |  |  |  |
| uc | action_shift_rate | 72 | 0.0000 | 0.7222 | 0.7222 | 0.6111 | 0.8194 |  |  |  |  |
| all | repeat_pattern_rate | 180 | 0.6389 | 0.6556 | 0.0167 | -0.0667 | 0.1000 | 27 | 30 | 0.07017543859649122 | 0.7910815129207817 |
| all | new_evidence_action_rate | 180 | 0.9833 | 0.9889 | 0.0056 | -0.0111 | 0.0222 |  |  |  |  |
| all | progress_seeking_score_mean | 180 | 2.0000 | 2.1667 | 0.1667 | 0.0333 | 0.3000 |  |  |  |  |
| all | continue_rate | 180 | 0.9778 | 0.9889 | 0.0111 | 0.0000 | 0.0278 |  |  |  |  |
| all | action_shift_rate | 180 | 0.0000 | 0.7111 | 0.7111 | 0.6444 | 0.7778 |  |  |  |  |

## Interpretation Boundary

- The experiment measures whether a generic warning/replan prompt changes proposed next-step plans on real debugging prefixes.
- It does not claim task success improvement, tests-passed improvement, or full-agent deployment effects.
- Productive-risk over-stop is tracked as `unsafe_overstop_rate` so that a warning is not rewarded for simply stopping productive prefixes.
