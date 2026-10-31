# Quantized KV selection failure diagnostics

The [standalone release failure](2026-10-31-quantized-kv-release-failure.md)
remains unresolved. A release test executable passes the failing 8K Q8/head64
case when run alone. A second test replays every case in layer zero, using
the standalone runner's ordering and explicit dense configuration; all twelve
cases pass. These passes do not invalidate the original failure or establish
which executable, data or execution-state difference caused it.

The [isolated log](2026-10-31-quantized-kv-selection-isolated.log.gz) and
[ordered replay log](2026-10-31-quantized-kv-selection-sequence.log.gz) retain
both outcomes. The ignored regression requires `KV_CAPTURE_REPORT` and
`KV_INPUTS`, admits their complete hashes and fails when artifacts are missing.
It checks the original independent selected positions, reconstructed values,
attention references and tracked cleanup. CUDA/server all-target
[Clippy](2026-10-31-quantized-kv-selection-clippy.log.gz) passes.

The validator now reports the actual and expected old block IDs and their
independent FP64 scores on a selection mismatch. It keeps the original
position-order comparison and all numerical requirements. No selector,
quantizer, input, tolerance or runtime setting changes. The next action is a
clean standalone release replay with these diagnostics; timing remains blocked
by the unresolved admission failure.
