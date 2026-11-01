# Preliminary Direct Verifier Correctness

Clean implementation `9aff60ad522ca4c1c14806b06479333b90cc345c`, executable
`a25a248d42b378b12b28dac70c54a622883b9708662ca1b819b29a88db3ebb56`,
passes all 512 actual IDs, 511 full-vocabulary vectors with zero difference,
10,616,832 valid KV words bit-identically, all rejection/partial/drop cases
and twelve frozen task outputs. Tracked teardown is zero.

The [raw report](2026-10-28-direct-verifier-preliminary-correctness.jsonl)
contains no primary performance trials. Before timing, audit coverage was
expanded from actual joint-pair hashes plus prepared identities to actual
before/after hashes for all 37 prepared down/head owners as well. Target
execution, CUDA kernels, scheduling and acceptance gates did not change.
Retain this passing report separately and rerun the full correctness gate
with the new executable; do not reuse its hash admission or pool timings.
