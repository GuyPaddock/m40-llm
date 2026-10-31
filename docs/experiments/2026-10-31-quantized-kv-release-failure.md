# Quantized KV release stops at selection admission

The clean `c9f6066` real-input release passes nine cases, then fails at layer
0, context 8,192, Q8 backing and head dimension 64. The runner reports
`selected positions differ sequence=0 head=0` and exits before any timing.
The failing case input SHA-256 is
`e64ab70140a0382368f8973f116a8d198a6d622ae6054facfab52c9453aeb634`.
Executable SHA-256 is
`801c65ae6a97e4164c56bb8b81f363b5e780d07ef18a3cbd0486db9bf56dcda3`.

The [raw records](2026-10-31-quantized-kv-release-failure.jsonl.gz) contain
one manifest, ten setup records, nine passing validations, nine completed
cases and one validation failure. There is no full-grid completion or timing
record. The [run log](2026-10-31-quantized-kv-release-failure.log.gz),
[build log](2026-10-31-quantized-kv-release-failure-build.log.gz) and
[executed script](2026-10-31-quantized-kv-release-failure.sh) preserve the result.
The script exits before its final idle query; inspect M40 again before the
next hardware run.

The independent CPU reference ranks mathematical block-mean dot products in
FP64. The GPU reduces them in FP32. The current error does not identify the
selected block IDs or establish the cause. Add diagnostics and compare those
scores before changing the selector, input bytes or admission requirements.
Preserve the original exact-position gate and the passing synthetic coverage
within their recorded scope. This is a correctness admission failure, not a
measured speed result or a model-quality assessment.
