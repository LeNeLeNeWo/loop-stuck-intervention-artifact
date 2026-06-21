# Dataset Card

This artifact contains four released evidence pools: Natural500, Enriched500, StressFresh100, and OpenHandsExternal330. Natural500 is a natural-distribution control, Enriched500 is an enriched difficult loop-heavy set, StressFresh100 is a clean stress challenge, and OpenHandsExternal330 is an external long-horizon pool.

Enriched500 is the reader-facing name. It corresponds to the internal `final500_label_adjudication_v042_manual_review` source directory retained in provenance records.

Enriched500 and StressFresh100 are not natural prevalence samples. Raw-field availability differs by pool. For the double-annotated manually reviewed construct-labeling task, two software-engineering master's students familiar with agent-based debugging independently reviewed trajectory-local evidence at step level. Cohen's kappa was 0.762 before adjudication. OpenHandsExternal330 uses original external labels normalized into the shared schema and is outside the kappa claim. Only final released labels are included; raw independent annotation sheets are excluded.

Labels: productive iteration (PI), hard negative (HN), unproductive cycle (UC), uncertain, and unresolved.
