# Offline Stop-Counterfactual Intervention Validation

This analysis asks: if a detector/guard trigger were converted into a hard stop at step `t`, what observed suffix would that stop cut off? It is an offline hard-stop counterfactual over observed traces, not a full causal rerun and not evidence about warn/replan/handoff policies.

## Feasibility

Feasibility audit: `<LOCAL_REPO_ROOT>\研1\科研\ccfa\LoopSense-LPR\outputs\paper1_intervention_validation\feasibility_report.md`.
Final task success and structured post-trigger successful validation were not available in the primary normalized/rich inputs. Therefore Stop-Harm Rate and success-destroying trigger counts are marked unavailable rather than fabricated.

## Metric Definitions

- Productive Suffix Loss (PSL): remaining productive steps after a first trigger, where productive means PI or HN.
- UC Waste Saved (UWS): remaining UC steps after a first trigger.
- Stop-Harm Rate (SHR): unavailable in this artifact because observed final success/post-trigger validation fields are not structured in the primary inputs.
- Detector summaries use the first trigger per dataset/trajectory/detector/config, matching a hard-stop policy.

## Focused Detector Configurations

| dataset | detector | detector_config | F1 | PIIR | HN_FPR | productive_suffix_cut_mean | UC_suffix_saved_mean | would_cut_productive_suffix_rate | would_save_UC_suffix_rate |
|---|---|---|---|---|---|---|---|---|---|
| Enriched500 | file_revisit_guard | window=5;repeats=3 | 0.729 | 0.562 | 0.349 | 24.849 | 36.235 | 0.961 | 0.668 |
| Enriched500 | max_step_guard | threshold=50 | 0.719 | 0.101 | 0.118 | 11.802 | 48.064 | 0.483 | 0.866 |
| Enriched500 | repeat_without_progress_guard | base=exact_action;k=3;progress=runtime | 0.197 | 0.019 | 0.003 | 10.533 | 46.880 | 0.533 | 0.853 |
| Enriched500 | repeat_without_progress_guard | base=exact_action;k=3;progress=runtime_or_signal_proxy | 0.193 | 0.019 | 0.003 | 10.972 | 48.736 | 0.556 | 0.861 |
| StressFresh100 | file_revisit_guard | window=10;repeats=3 | 0.652 | 0.868 | 0.797 | 10.170 | 13.610 | 0.880 | 0.820 |
| StressFresh100 | repeat_without_progress_guard | base=exact_action;k=3;progress=runtime | 0.256 | 0.004 | 0.000 | 3.263 | 16.263 | 0.421 | 0.632 |
| StressFresh100 | repeat_without_progress_guard | base=exact_action;k=3;progress=runtime_or_signal_proxy | 0.256 | 0.004 | 0.000 | 3.263 | 16.263 | 0.421 | 0.632 |
| StressFresh100 | repeat_without_progress_guard | base=tool;k=3;progress=runtime | 0.273 | 0.004 | 0.000 | 3.480 | 14.080 | 0.440 | 0.640 |
| StressFresh100 | repeat_without_progress_guard | base=tool;k=3;progress=runtime_or_signal_proxy | 0.273 | 0.004 | 0.000 | 3.480 | 14.080 | 0.440 | 0.640 |

## By Trigger Location

| dataset | trigger_location_type | first_trigger_count | productive_suffix_cut_mean | UC_suffix_saved_mean | would_cut_productive_suffix_rate | would_save_UC_suffix_rate |
|---|---|---|---|---|---|---|
| Natural500 | UC | 80.000 | 3.125 | 24.775 | 0.200 | 0.938 |
| Natural500 | HN | 19.000 | 11.263 | 9.789 | 0.947 | 0.579 |
| Natural500 | PI | 7.000 | 2.857 | 11.000 | 0.857 | 0.429 |
| Natural500 | other | 4.000 | 3.250 | 3.250 | 0.500 | 0.500 |
| Natural500 | uncertain | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Enriched500 | UC | 1927.000 | 13.480 | 48.366 | 0.634 | 0.950 |
| Enriched500 | HN | 1238.000 | 24.404 | 26.001 | 0.989 | 0.513 |
| Enriched500 | PI | 395.000 | 20.937 | 33.858 | 0.967 | 0.552 |
| Enriched500 | other | 110.000 | 18.536 | 27.945 | 0.800 | 0.473 |
| Enriched500 | uncertain | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| StressFresh100 | UC | 561.000 | 4.975 | 17.857 | 0.576 | 0.879 |
| StressFresh100 | HN | 224.000 | 10.335 | 6.929 | 0.964 | 0.469 |
| StressFresh100 | PI | 358.000 | 10.402 | 15.034 | 0.919 | 0.827 |
| StressFresh100 | other | 204.000 | 4.637 | 7.157 | 0.686 | 0.711 |
| StressFresh100 | uncertain | 22.000 | 6.818 | 14.864 | 0.727 | 0.909 |
| OpenHandsExternal330 | UC | 2.000 | 16.000 | 8.000 | 0.500 | 1.000 |
| OpenHandsExternal330 | HN | 36.000 | 42.000 | 0.528 | 0.889 | 0.028 |
| OpenHandsExternal330 | PI | 923.000 | 63.488 | 0.126 | 0.950 | 0.010 |
| OpenHandsExternal330 | other | 576.000 | 83.623 | 0.160 | 0.925 | 0.009 |
| OpenHandsExternal330 | uncertain | 12.000 | 4.583 | 0.000 | 0.083 | 0.000 |

## Near-F1 Stop-Counterfactual Pairs

| dataset | detector_a | config_a | detector_b | config_b | abs_F1_diff | abs_PIIR_diff | abs_HN_FPR_diff | productive_suffix_cut_mean_a | productive_suffix_cut_mean_b | UC_suffix_saved_mean_a | UC_suffix_saved_mean_b |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Enriched500 | max_step_guard | threshold=50 | file_revisit_guard | window=5;repeats=3 | 0.009 | 0.461 | 0.230 | 11.802 | 24.849 | 48.064 | 36.235 |
| Enriched500 | max_step_guard | threshold=100 | action_observation_similarity | tfidf_adjacent>=0.98 | 0.013 | 0.078 | 0.006 | 4.600 | 16.770 | 36.646 | 39.492 |
| Enriched500 | max_step_guard | threshold=50 | file_revisit_guard | window=10;repeats=5 | 0.002 | 0.313 | 0.153 | 11.802 | 22.204 | 48.064 | 39.189 |
| Enriched500 | exact_action_repeat_k | k=4 | action_observation_repeat_exact | exact_pair_k=2 | 0.012 | 0.028 | 0.006 | 8.500 | 16.248 | 50.089 | 44.991 |
| Enriched500 | action_observation_repeat_exact | exact_pair_k=2 | repeat_without_progress_guard | base=exact_action;k=3;progress=runtime | 0.000 | 0.021 | 0.004 | 16.248 | 10.533 | 44.991 | 46.880 |
| Enriched500 | exact_action_repeat_k | k=3 | action_observation_repeat_exact | exact_pair_k=2 | 0.007 | 0.012 | 0.003 | 10.944 | 16.248 | 41.045 | 44.991 |
| Enriched500 | action_observation_repeat_exact | exact_pair_k=2 | repeat_without_progress_guard | base=exact_action;k=3;progress=runtime_or_signal_proxy | 0.004 | 0.021 | 0.005 | 16.248 | 10.972 | 44.991 | 48.736 |
| Enriched500 | file_revisit_guard | window=5;repeats=3 | file_revisit_guard | window=10;repeats=5 | 0.007 | 0.147 | 0.078 | 24.849 | 22.204 | 36.235 | 39.189 |
| Enriched500 | file_revisit_guard | window=10;repeats=3 | file_revisit_guard | window=10;repeats=5 | 0.019 | 0.209 | 0.129 | 24.820 | 22.204 | 35.322 | 39.189 |
| Enriched500 | exact_action_repeat_k | k=4 | repeat_without_progress_guard | base=exact_action;k=3;progress=runtime_or_signal_proxy | 0.007 | 0.006 | 0.001 | 8.500 | 10.972 | 50.089 | 48.736 |
| Enriched500 | exact_action_repeat_k | k=3 | exact_action_repeat_k | k=4 | 0.019 | 0.015 | 0.002 | 10.944 | 8.500 | 41.045 | 50.089 |
| Enriched500 | exact_action_repeat_k | k=4 | repeat_without_progress_guard | base=exact_action;k=3;progress=runtime | 0.011 | 0.007 | 0.001 | 8.500 | 10.533 | 50.089 | 46.880 |
| Enriched500 | repeat_without_progress_guard | base=exact_action;k=3;progress=runtime | repeat_without_progress_guard | base=exact_action;k=3;progress=runtime_or_signal_proxy | 0.004 | 0.001 | 0.000 | 10.533 | 10.972 | 46.880 | 48.736 |
| Enriched500 | exact_action_repeat_k | k=3 | repeat_without_progress_guard | base=exact_action;k=3;progress=runtime | 0.008 | 0.009 | 0.001 | 10.944 | 10.533 | 41.045 | 46.880 |
| Enriched500 | file_revisit_guard | window=5;repeats=3 | file_revisit_guard | window=10;repeats=3 | 0.012 | 0.061 | 0.052 | 24.849 | 24.820 | 36.235 | 35.322 |
| Enriched500 | exact_action_repeat_k | k=3 | repeat_without_progress_guard | base=exact_action;k=3;progress=runtime_or_signal_proxy | 0.012 | 0.009 | 0.001 | 10.944 | 10.972 | 41.045 | 48.736 |
| OpenHandsExternal330 | max_step_guard | threshold=25 | max_step_guard | threshold=200 | 0.003 | 0.894 | 0.996 | 96.567 | 0.011 | 0.188 | 0.000 |
| OpenHandsExternal330 | max_step_guard | threshold=50 | max_step_guard | threshold=200 | 0.004 | 0.891 | 0.996 | 90.688 | 0.011 | 0.188 | 0.000 |
| OpenHandsExternal330 | max_step_guard | threshold=75 | max_step_guard | threshold=200 | 0.005 | 0.870 | 0.990 | 74.101 | 0.011 | 0.187 | 0.000 |
| OpenHandsExternal330 | max_step_guard | threshold=25 | max_step_guard | threshold=150 | 0.005 | 0.464 | 0.552 | 96.567 | 30.831 | 0.188 | 0.153 |

## Interpretation

PIIR and HN-FPR remain trigger-site risk proxies. This offline analysis strengthens the intervention interpretation by showing whether high-risk triggers also cut off observed productive suffixes under a hard-stop policy. It still does not estimate stochastic live rerun outcomes or softer interventions such as warn, summarize, replan, escalate, or handoff.
