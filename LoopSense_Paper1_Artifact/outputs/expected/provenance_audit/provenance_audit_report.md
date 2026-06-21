# Paper1 Trajectory Provenance Audit

- Hits written: 1031769
- Missing/low-confidence field count: 0

## Summary By Pool

### Natural500
- Source framework: SWE-agent / LoopBench-Agent
- Model: swe-agent-llama-70b (460); swe-agent-llama-8b (9); swe-agent-llama-405b (5)
- Model provider: SWE-agent source model_name namespace
- Benchmark/task source: nebius/swe-agent-trajectories over nebius/SWE-bench-extra and the dev split of princeton-nlp/SWE-bench
- Confidence: medium
- Evidence categories: released final-label metadata; public source dataset metadata; source parquet metadata; normalization script
- Notes: Raw parquet row-token validation matched 474 trajectories and mismatched 26; source README confirms model_name is a trajectory-generation field; active-pool row-token model coverage is partial. Annotation-assistance metadata is not used as trajectory-generation provenance.

### Enriched500
- Source framework: SWE-agent / LoopBench-Agent
- Model: swe-agent-llama-70b (449); swe-agent-llama-8b (22); swe-agent-llama-405b (1)
- Model provider: SWE-agent source model_name namespace
- Benchmark/task source: nebius/swe-agent-trajectories over nebius/SWE-bench-extra and the dev split of princeton-nlp/SWE-bench
- Confidence: medium
- Evidence categories: released final-label metadata; public source dataset metadata; source parquet metadata; normalization script
- Notes: Raw parquet row-token validation matched 472 trajectories and mismatched 28; source README confirms model_name is a trajectory-generation field; active-pool row-token model coverage is partial. Annotation-assistance metadata is not used as trajectory-generation provenance.

### StressFresh100
- Source framework: SWE-agent / LoopBench-Agent
- Model: swe-agent-llama-70b (88); swe-agent-llama-8b (7)
- Model provider: SWE-agent source model_name namespace
- Benchmark/task source: nebius/swe-agent-trajectories over nebius/SWE-bench-extra and the dev split of princeton-nlp/SWE-bench
- Confidence: medium
- Evidence categories: released final-label metadata; public source dataset metadata; source parquet metadata; normalization script
- Notes: Raw parquet row-token validation matched 95 trajectories and mismatched 5; source README confirms model_name is a trajectory-generation field; active-pool row-token model coverage is partial. Annotation-assistance metadata is not used as trajectory-generation provenance.

### OpenHandsExternal330
- Source framework: OpenHands external
- Model: Qwen/Qwen3-Coder-480B-A35B-Instruct
- Model provider: Qwen model namespace in source dataset/model card
- Benchmark/task source: nebius/SWE-rebench real GitHub issues
- Confidence: high
- Evidence categories: released pool index; external-source feasibility and freeze metadata; normalization compatibility metadata; preparation script
- Notes: OpenHandsExternal330 uses original external labels normalized into the shared schema and is outside the kappa claim.

## Skipped Directories

- outputs/paper1_audit/construct_trajectory_flags.csv skipped derived large table
- outputs/paper1_audit/label_density_by_trajectory.csv skipped derived large table
- outputs/paper1_audit/normalized_steps.csv skipped derived large table
- outputs/paper1_intervention_validation/stop_counterfactual_triggers.csv skipped derived large table
- outputs/paper1_raw_enrichment/enriched_steps.csv skipped derived large table
- outputs/paper1_construct_mismatch/tables/construct_trajectory_flags.csv skipped derived large table
- release_artifacts/LoopSense_Paper1_Artifact/data/normalized/normalized_steps.csv skipped derived large table

## Safe Paper Updates

- Report source framework per evidence pool rather than pooling datasets.
- Report OpenHandsExternal330 as original external labels normalized into the shared schema and outside the step-level kappa claim.
- Report kappa as step-level when describing the double-annotated manually reviewed construct-labeling task.

## Unsafe Claims

- Do not infer SWE-bench Lite or SWE-bench Verified unless an evidence file states that exact source.
- Do not infer trajectory-generation model from annotation-assistance model fields unless tied to raw trajectory metadata.

