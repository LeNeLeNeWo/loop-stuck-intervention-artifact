# LoopSense Paper 1 Artifact

## Paper
**When Is an Agent Really Stuck? Construct Validity and Productive-Interruption Risk in Software-Engineering Agents**

This artifact supports reproduction of the main audit tables, construct-mismatch analyses, detector-risk analyses, publication figures, offline stop-counterfactual intervention validation, the sanitized outputs of an API-assisted prefix replan intervention experiment, a small sanitized live guard-policy pilot, and the 60-task official-oracle live guard experiment used in the current manuscript.

## What this artifact contains
- Final released/adjudicated labels only.
- Normalized step and span tables with anonymized public trajectory identifiers.
- Analysis scripts, release scripts, expected paper outputs, paper figures/tables, intervention-validation outputs, API-assisted replan summaries, live guard pilot summaries, 60-task official-oracle live guard summaries, provenance summaries, and integrity reports.

## What this artifact does NOT contain
- Raw independent double-annotation sheets.
- Disagreement, conflict, negotiation, or private annotation records.
- Raw un-anonymized trajectories, local machine paths, API keys, tokens, or full model execution credentials.
- Raw API responses or private prefix-to-trajectory mappings for the API-assisted replan experiment.
- Raw live guard trajectory logs or full benchmark working directories.
- A live prefix-branch rerun stack; original agent trajectory generation is out of scope.
- Inputs required to recompute Cohen's kappa; only final adjudicated labels are released.

## Evidence pools
| Pool | Framework/source | Trajectories | Steps | Avg. steps | UC prevalence | Role | Label source |
|---|---|---:|---:|---:|---:|---|---|
| Natural500 | SWE-agent / LoopBench-Agent | 500 | 8550 | 17.100 | 0.268 | natural-distribution control | final adjudicated labels |
| Enriched500 | SWE-agent / LoopBench-Agent | 500 | 26014 | 52.028 | 0.582 | enriched difficult loop-heavy set | final adjudicated labels |
| StressFresh100 | SWE-agent / LoopBench-Agent | 100 | 3132 | 31.320 | 0.820 | clean stress challenge | final adjudicated labels |
| OpenHandsExternal330 | OpenHands | 330 | 51010 | 154.576 | 0.012 | external long-horizon pool | original external labels normalized into the shared schema |

## Reproducibility levels
- **Level 1:** regenerate/check paper tables and figures from included expected CSVs.
- **Level 2:** verify analysis results from released normalized data and included intermediate tables.
- **Level 3:** raw trajectory rerun or live prefix-branch intervention rerun; not included.

The artifact supports Level 1 and Level 2. Raw-runtime-dependent detector reruns are partial by design when non-released raw trajectory logs would be required. The offline stop-counterfactual analysis is included and reproducible from included trigger/intermediate tables.


## Reproducibility and Version Consistency
This artifact contains only the final released dataset used for the paper. Historical annotation versions, pre-adjudication materials, disagreement sheets, private notes, and raw independent annotation files are excluded. The fingerprint report records SHA256 hashes and row counts for released datasets and key expected outputs. The paper-number traceability audit maps each headline result in the paper to a source CSV or report inside this artifact. The clean-room test verifies that key outputs can be checked from the zip without accessing the original local repository. Cohen's kappa is a step-level reliability result for the double-annotated manually reviewed construct-labeling task; OpenHandsExternal330 uses original external labels normalized into the shared schema and is outside that kappa claim. Raw independent annotation sheets are not released, so kappa is documented here but not recomputed from the public artifact.

## Quick start
```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_artifact_integrity.py
python tests/test_reproduce_key_outputs.py
python scripts/release/audit_and_test_paper1_artifact.py --artifact-root .
```

## Expected headline values
- OpenHandsExternal330 average steps: 154.576; UC prevalence: 0.012.
- Natural500 steps: 8,550; Enriched500 steps: 26,014; StressFresh100 steps: 3,132; OpenHandsExternal330 steps: 51,010.
- Enriched500 near-F1 pair: F1 0.719 vs 0.729, delta F1 0.009, delta PIIR 0.461, delta HN-FPR 0.230.
- StressFresh100 file-revisit window 10 repeats 3: UC recall 0.958, PIIR 0.868, HN-FPR 0.797.
- Offline stop-counterfactual trigger rows: 246,856.
- Enriched500 max-step T=50 PSL/UWS: 11.8 / 48.1; file-revisit w=5,r=3 PSL/UWS: 24.8 / 36.2.
- API-assisted WARN_REPLAN experiment: 180 paired prefixes, 360 parsed responses, action-shift rate 0.711, progress-seeking score 2.00 to 2.17, overall repeated-pattern rate 0.639 to 0.656, productive-risk unsafe over-stop 0.028 to 0.014.
- Official-oracle live guard experiment: 60 frozen SWE-bench Verified tasks per arm; resolved/test-pass outcomes are No Guard 35/60, Warn/Replan Guard 42/60, and Hard Stop Guard 8/60.
- Official-oracle patch generation: No Guard 48/60, Warn/Replan Guard 54/60, and Hard Stop Guard 9/60.
- Official-oracle mean steps: No Guard 42.0, Warn/Replan Guard 40.2, and Hard Stop Guard 8.0.
- Official-oracle mean tokens: No Guard 758,692, Warn/Replan Guard 744,589, and Hard Stop Guard 67,225.
- Enriched500 corresponds to the internal `final500_label_adjudication_v042_manual_review` source directory; the released reader-facing dataset name is Enriched500.

## Directory structure
`data/final_adjudicated/` contains released labels. `data/normalized/` contains normalized tables. `scripts/` contains artifact-relative analysis scripts. `outputs/expected/` contains expected tables, figures, metrics, intervention-validation outputs, and provenance summaries. `docs/` contains schema and reproduction documentation. `tests/` contains integrity and headline reproducibility tests.

## Annotation release policy
Labels are semantic local-process annotations over trajectory, step, and span units. For the double-annotated manually reviewed construct-labeling task, the paper reports two trained annotators and step-level Cohen's kappa 0.762 before adjudication. OpenHandsExternal330 uses original external labels normalized into the shared schema and is reported separately from that reliability claim. Uncertain and unresolved states are preserved when applicable. Detector outputs and final task outcomes are not annotation evidence.

## Intervention validation
Live/replay feasibility-only artifact: `outputs/expected/live_intervention/` records that a reproducible historical prefix-branch replay is not feasible from the public artifact because prefix-state recovery, runner, credentials, and oracle pieces are incomplete. No live success/cost/repetition effects are claimed from that feasibility-only report.

PIIR and HN-FPR are trigger-site risk metrics. The offline stop-counterfactual evaluates what observed suffix would be cut off if the first detector/guard trigger were converted into a hard stop. This is not a live causal rerun and does not evaluate warn, replan, handoff, or escalation policies.

API-assisted replan outputs are in `outputs/expected/api_replan_intervention/`. This experiment uses real sanitized prefixes and real DeepSeek API calls, but it is prompt-level only: it does not restore repositories, execute tools, run tests, or estimate final task success. The artifact includes sanitized prompts, a sanitized API call log, parsed responses, summaries, and figure source outputs. Raw API responses and private prefix mappings are not released.

Live guard pilot outputs are in `outputs/expected/live_guard_experiment/`. This pilot ran mini-swe-agent on 10 public SWE-Bench Verified fallback tasks with DeepSeek, comparing No Guard, Warn/Replan Guard, and Hard Stop Guard under a 10-step budget. It is real tool-level live-policy evidence, but it is underpowered for manuscript claims and has no official resolved/tests-pass oracle. It should not be read as task-success evidence.

## Provenance
Pool-level provenance summaries are in `outputs/expected/provenance_audit/` and `docs/provenance.md`. Detailed raw provenance hits are not included because they are large and can contain raw source identifiers.

## Known limitations
Raw-field availability differs by pool. Live prefix-branch reruns are not included. Full raw trajectory generation is not reproduced. Independent annotation sheets and disagreement records are intentionally excluded.

## Citation
```bibtex
@inproceedings{anonymous2027loopsense,
  title={When Is an Agent Really Stuck? Construct Validity and Productive-Interruption Risk in Software-Engineering Agents},
  author={Anonymous Authors},
  booktitle={Anonymous submission},
  year={2027}
}
```

## License
License to be finalized before public release.

## Official-oracle live guard outputs
`outputs/expected/live_guard_official_oracle/` contains sanitized outputs for the preregistered 60-task official-oracle live guard experiment. The first 30 tasks match the frozen prefix, and tasks 31-60 follow the same frozen sample order. The manuscript uses the 60-task official outcomes: WARN_REPLAN_GUARD resolves 42/60 tasks versus 35/60 for NO_GUARD, while HARD_STOP_GUARD resolves 8/60 and sharply reduces execution. Patch generation is 54/60, 48/60, and 9/60 for Warn/Replan, No Guard, and Hard Stop respectively. Empty-patch rows are reported separately and counted as unresolved. Raw live trajectories, full benchmark workspaces, Docker caches, and credentials are not released.
