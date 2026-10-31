# FP16 codec qualification

The CPU implementation in [`src/lossless_kv`](../../src/lossless_kv/mod.rs)
passes five behavioral tests and the complete 225-test non-CUDA suite.
CUDA/server all-target Clippy, formatting and whitespace checks pass. The
[source and format audit](2026-10-31-lossless-kv-audit.md) fixes the candidate
before any real KV compression measurement.

The encoder owns a serialized stream with a fixed directory. The decoder
reads fields independently by bit position; it shares no streaming bit-pack
accumulator with the encoder. Reconstruction uses integer operations only.
`from_bytes` validates header/count bounds, directory continuity, tail counts,
payload bounds, packed eligibility, exponent codes and padding before
returning an owner. It cannot allocate from an unchecked serialized count.

Tests cover all 65,536 FP16 bit patterns in ordered and permuted streams,
offset widths zero through four, wide-range fallback, exceptional patterns,
tail lengths and malformed input. They check every reconstructed word, known
code bits and independently calculated byte sizes. The first two runs exposed
incorrect tail-size expectations in the test, not a codec failure. The exact
strict boundary at width zero is 14 raw values versus 15 packed values:
14 values need 20 packed bytes plus eight directory bytes, equal to 28 raw
bytes; 15 need 21 plus eight, less than 30. The global header can still make
a tiny complete stream larger. The final tests preserve this distinction.

Raw initial failures and final validation logs use prefix
`2026-10-31-lossless-kv-codec-`; file hashes are in the adjacent SHA manifest.
No GPU execution, real-cache storage saving or attention performance follows
from these CPU checks. The next unit extends the existing model-quality
runner to capture real dense FP16 cache bits and evaluate this fixed format.

Validation, after sourcing `scripts/dev-env.sh` and selecting this worktree's
disk-backed `CARGO_TARGET_DIR`:

```bash
RUSTFLAGS='-D warnings' cargo test --no-default-features --locked
M40LLM_ENABLE_CUBLAS=1 cargo clippy --features cuda,server --all-targets -- -D warnings
cargo fmt --all -- --check
```

Runtime KV allocation, attention dispatch and defaults are unchanged.
