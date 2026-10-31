# Quantized attention CPU reference

The probe now has an independent CPU reference for local Q8 and mixed
FP16-K/Q4-V execution. Five new behavioral tests pass, alongside the fixture
reader's four integrity tests. This verifies the reference mathematics; GPU
consumer qualification and paired timings remain pending.

The reference separates three operations that must not be conflated:

- Old K/V quantization uses a per-token, per-KV-head FP32 absmax scale, signed
  codes with ties-to-even rounding, and FP16 reconstruction for attention.
  Mixed backing preserves FP16 K and quantizes only old V to signed Q4.
  Recent K/V retain FP16 values.
- Block selection ranks mean-key dot products and appends the complete recent
  window. Q8 selectors use `code * FP32 scale` before FP16 rounding. Their
  attention consumers round the reconstructed value to FP16. The reference
  stores these key representations separately.
- FP64 attention computes stable softmax over explicit selected positions.
  A full-context FP16 reference and a selected-original-FP16 reference isolate
  selection effects from quantization at fixed positions.

The current direct kernels still scan all old keys to choose blocks. Avoiding
staged KV buffers does not eliminate those selection reads. The benchmark must
include selection in both arms and distinguish logical reads from DRAM traffic.

The existing explicit-position append API evaluates an adjacent-pair RoPE
expression even when frequency is zero. That preserves numeric values but can
change the sign of zero. A separate reference models that expression before
comparing staged FP16 bits; it does not silently alter the captured artifact.
Quantized integer zero reconstructs as positive zero, while recent V preserves
its original zero sign. Lossless-KV reconstruction has a separate bit-exact
contract and is not weakened by this attention reference.

Tests use hand-computable block order and arithmetic means, including GQA head
mapping, partial old blocks, exact score ties, recent-only contexts, signed
Q4 ties, Q8 pre/post-FP16 distinctions, signed zero and malformed/non-finite
inputs. The CUDA benchmark will require the selected positions to match this
ranking, then check every staged K/V word and FP64 output before timing. A
selection mismatch is a failed numerical admission, not a performance result.

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/quantized-kv/target
RUSTFLAGS="-D warnings" cargo test --no-default-features --locked \
  --example kv_attention_bench
```

The [benchmark implementation checkpoint](2026-10-31-quantized-kv-benchmark.md)
describes the dependent uncommitted GPU work. The
[checksum manifest](2026-10-31-quantized-kv-benchmark.sha256) preserves CPU tests,
Clippy and compile-only GPU results. The original Clippy type-complexity failure
is retained; a named staging-readback record resolves it. Compiled hardware
tests have not run while the lossless capture owns the M40.
