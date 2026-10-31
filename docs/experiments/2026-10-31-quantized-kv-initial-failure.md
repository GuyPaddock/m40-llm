# Quantized KV initial hardware admission failure

The first release M40 execution of the staged reference fails exact value
reconstruction in `test_mixed_staging_preserves_selected_cache_and_attention`.
The build succeeds, then the test reports a mismatch between staged FP16
value bytes and the independent CPU reconstruction. No timings run.
The original assertion does not identify the token or bit patterns; diagnosis
must add those details before choosing a correction.

The [initial log](2026-10-31-quantized-kv-initial-failure.log.gz) retains the
failure. The [executed script](2026-10-31-quantized-kv-initial-qualification.sh)
stops immediately; the example runner, affected regressions and cuBLAS-disabled
checks have not executed. The source matches the
[frozen file hashes](2026-10-31-quantized-kv-initial-source.sha256).
The paired runner remains a dependent uncommitted unit.

Preserve the failing staged implementation and its hardware regression before
any correction. The stage supports head64 and head128 through compile-time
specializations, while runtime backend admission stays unchanged. Earlier
CPU, Clippy and compile/link checks remain valid within their stated scope;
they do not establish GPU correctness. No default, numerical tolerance or
exact reconstructed-byte assertion is changed to admit this result.
