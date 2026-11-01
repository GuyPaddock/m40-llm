# Causal selector analyzer ready

The CPU-only `channel_predictability` example implements the fixed
[protocol](2026-10-31-selector-predictability-protocol.md). It validates
complete captured positions, paired outputs/counts, clean release provenance,
the original model/corpus hashes, the whole source weight region and every
captured Q8 tensor before computing metrics. Down-column norms use the
original signed Q8 payloads and FP16 scales.

The signed CountSketch has 16 dimensions and fixed seed
`0x4d34305f53454c31`; its SplitMix64 finalizer and channel-to-bucket mapping
are committed before analysis. Each causal history accepts labels only at
scheduled dense refreshes. The nearest-state bank holds at most eight
refresh records. Predicted rows record the exact source rows and selected
channels. All histories reset at task boundaries, and a previous-layer
signal is reported only where that layer exists.

Dense refresh successes are excluded from sparse recall/coverage gates.
Bootstrap and refresh work are charged in a labeled estimate using the
rounded historical Q8 and selected-Q4 medians from `c16559e`. This is not a
measured selector cost. Hidden-vector oracle reuse remains a diagnostic
control; it is not classified as a causal sparse policy. Features still
come from dense trajectories, so this is not recursive sparse generation.

Nine CPU behavioral tests and warning-denied Clippy pass. Tests cover rank
ties, malformed values, signed sketches, scheduled label access, task reset,
bounded retrieval, independent Q8 norms, contribution metrics, zero mass,
refresh-aware cost arithmetic and report rejection. An end-to-end mutation
of skipped-token hidden values changes the previous-oracle diagnostic while
leaving every causal policy's selections and source rows unchanged. A task
with no consumed generated inputs still produces an explicit empty summary.

The initial build tried a hashing dependency available only with CUDA.
Streaming coreutils hashing now uses the project's existing dependency and
the no-default-features build passes. The initial failure and final checks
are preserved in `2026-10-31-selector-analyzer-*.log.gz`. No real capture has
been analyzed at this implementation checkpoint.
