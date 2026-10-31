# Quantized KV fixture admission

The `kv_attention_bench` example now admits completed lossless-capture inputs
on CPU. It verifies the original Qwen model and documentation source, eight
distinct disjoint input spans, all 96 stream identities and offsets, exact
artifact length, whole-file and per-stream SHA-256, sequence completion and
tracked cleanup. A running or partial capture is rejected. It does not infer
completion from the existence of a file or a manifest.

The reader reconstructs token/head/channel order from the raw per-head FP16
streams for layers 0/17/35 and contexts 1024/2048/4096/8192. Head128 keeps
the original head dimensions. Head64 takes the first 64 channels of each head
as a diagnostic fixture; it does not represent Qwen execution. Queries are
deterministic synthetic FP32 vectors, distinct by layer, sequence and query
head. They are not captured model queries or learned predictions.

Inspection emits 24 input identities with host payload sizes and explicit
scope. It generates no text and measures no GPU work or quality. Non-finite
KV values are rejected before they enter an attention timing case. Full-file
and per-stream hashes are checked even when inspecting a short prefix.

Four CPU behavioral tests pass. They cover real-file admission with a sparse
201326592-byte fixture, head ordering and signed zero, the head64 slice,
tail corruption, invalid stream hashes, missing/duplicate records, offsets,
source/input identity, cleanup, non-finite values and deterministic queries.
CPU Clippy and formatting pass. The synthetic test artifact is deleted after
the test; no real-model fixture has yet passed this inspector because the
full eight-sequence capture is still running.

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/quantized-kv/target
RUSTFLAGS="-D warnings" cargo test --no-default-features --locked \
  --example kv_attention_bench -- --nocapture --test-threads=1
cargo clippy --no-default-features --example kv_attention_bench -- -D warnings
cargo run --release --no-default-features --example kv_attention_bench -- \
  --capture-report COMPLETE_CAPTURE.jsonl --kv-inputs COMPLETE_CAPTURE.bin \
  --output NEW_INSPECTION.jsonl
```

## Fixed benchmark inputs and comparisons

The next implementation uses these admitted bytes with 16 query heads and two
KV heads. It tests each captured layer/context at batches 1/2/4/8 using the
first B distinct sequences, recent window1024, block size32, plain top8,
score order and no anchors or masks. The 1K case has no old tokens and is a
useful no-quantization control. No runtime defaults change.

- Head64: Q8 staged versus Q8 direct, and mixed FP16-K/Q4-V staged versus
  mixed direct.
- Head128: mixed staged research reference versus the existing mixed direct
  consumer. Head128 Q8 remains unsupported and is not silently substituted.
- Mixed formats: measure serial rows and the existing batched direct consumer
  separately so batching gains are not attributed solely to dequantization.
- Dense FP16: retain a full-context reference on the same inputs for numerical
  and cost context. Selection and quantization differences remain separate
  from staged/direct parity.

Each pair shares one encoded cache allocation, queries, lengths and selection
configuration. Include selection, gather/reconstruction and attention inside
the operation interval. Report upload, append/quantization, allocation and
scratch residency separately. Preserve FP16 reconstruction rounding. Validate
staged positions/values and outputs before timing; reuse the existing paired
benchmark statistics and extension/confirmation rules.

This is a complete attention-operator study using real KV and synthetic
queries. It cannot establish whole-model throughput, task quality or the
distribution of queries encountered during generation. The
[staged reference](2026-10-31-quantized-kv-staging.md) still needs its hardware
checks before it can serve as a control. Paired timing is not implemented by
this fixture-admission commit.
