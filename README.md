# Loop/Stuck Intervention Artifact

This repository contains the anonymized artifact for the paper:

**When Is an Agent Really Stuck? Trigger-Site Risk in Software-Engineering Agent Guards**

The artifact supports reproduction checks for construct-validity analyses, detector-risk analyses, offline intervention validation, API-assisted replan evidence, and the 60-task official-oracle live guard experiment.

## Independent annotation supplement

The two annotators' anonymized pre-adjudication label tables and agreement-reproduction script are available in [`labels/independent_annotations/`](labels/independent_annotations/README.md). They cover 1,100 trajectories and 37,696 paired steps, with 4,945 and 4,827 spans. The reported step-level kappa is 0.762; 4,298 matched span pairs have label agreement 94.95% and kappa 0.885.

This is an additional release. The submitted paper's statement that independent annotation sheets were excluded describes the original artifact release. The supplement is separate from the final adjudicated labels used for detector evaluation and is not inside the original `LoopSense_Paper1_Artifact.zip`. The original artifact directory, ZIP, and their recorded checksums retain their original scope.

## Included

- Final adjudicated labels and normalized step/span tables.
- Expected paper tables, figures, summaries, and traceability files.
- Analysis, audit, release, and reproduction scripts.
- Sanitized outputs for offline stop-counterfactual, API-assisted replan, live guard pilot, and 60-task official-oracle live guard experiments.
- Integrity, fingerprint, traceability, and clean-room reports.
- `LoopSense_Paper1_Artifact.zip`, with one top-level directory: `LoopSense_Paper1_Artifact/`.

## Excluded from the original artifact snapshot

- Raw independent annotation sheets.
- Disagreement sheets, private notes, raw reviewer notes, and pre-adjudication records.
- Raw trajectories, raw live run logs, benchmark workspaces, Docker caches, and repository checkouts.
- API keys, credentials, `.env` files, local paths, and private issue URLs.

The independent label tables are now available separately in the supplement above. Raw trajectory logs and private adjudication working records remain excluded.

## Quickstart

```bash
unzip LoopSense_Paper1_Artifact.zip
cd LoopSense_Paper1_Artifact
python tests/test_artifact_integrity.py
python tests/test_reproduce_key_outputs.py
python scripts/release/artifact_fingerprint_and_traceability.py --artifact-root . --skip-clean-room
```

Expected key values:

- Pool sizes: Natural500 = 500, Enriched500 = 500, StressFresh100 = 100, OpenHandsExternal330 = 330.
- Normalized rows: `normalized_steps.csv` = 88,706; `normalized_spans.csv` = 4,982.
- Official-oracle live guard resolved/test-pass: No Guard 35/60, Warn/Replan Guard 42/60, Hard Stop Guard 8/60.
- Official-oracle patch generation: No Guard 48/60, Warn/Replan Guard 54/60, Hard Stop Guard 9/60.
- Official-oracle mean steps: No Guard 42.0, Warn/Replan Guard 40.2, Hard Stop Guard 8.0.
- Official-oracle mean tokens: No Guard 758,692, Warn/Replan Guard 744,589, Hard Stop Guard 67,225.

`Enriched500` is the reader-facing name. It corresponds to an internal provenance source directory retained only as a compatibility note inside the artifact documentation.
