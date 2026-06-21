# Clean-Room Reproduction Report

Zip: `LoopSense_Paper1_Artifact.zip`
Extraction directory label: `clean_room_test`
Single top-level directory expected: `LoopSense_Paper1_Artifact/`
Overall status: PASS

## Commands

### `python tests/test_artifact_integrity.py`
- Return code: 0
- External path detected: False
- Stdout excerpt:
```text
artifact integrity test passed

```

### `python tests/test_reproduce_key_outputs.py`
- Return code: 0
- External path detected: False
- Stdout excerpt:
```text
headline reproducibility test passed; raw-runtime-dependent reruns are partial by design

```

### `python scripts/release/artifact_fingerprint_and_traceability.py --artifact-root . --skip-clean-room`
- Return code: 0
- External path detected: False
- Stdout excerpt:
```text
Fingerprint status: PASS
Traceability status: PASS
Old-version contamination status: PASS
Clean-room status: passed (existing report)
Release readiness: READY

```
