# SA4_128 Maxwell consumer qualification

The experiment-only FP32 sign-add consumer passes its numerical, stream,
guard and ownership checks on the Tesla M40. This qualifies the implementation
for controlled projection measurements. No model-weight fit, decoding speed
or model-quality result follows from these small correctness fixtures.

## Implementation

[`shiftadd.cuh`](../../cuda/shiftadd.cuh) assigns one warp to each output
channel and four warps to a block. Each lane reads four FP32 activations per
128-weight group, accumulates four signed plane sums, and reduces them within
the warp. Lane zero applies the four stored amplitudes and accumulates the
output. It handles rows 1, 2, 4 and 8 and incomplete final groups of four
output channels. Input widths must be multiples of 256.

The kernel reads the 138-byte SA4_128 blocks directly. It does not quantize
activations or write a dequantized weight matrix. FP16 scale reads require
only two-byte alignment; sign-plane payload reads use bytes. There is no
scratch allocation. The context method in
[`src/cuda/shiftadd.rs`](../../src/cuda/shiftadd.rs) validates dimensions,
pointer alignment, range overflow and output overlap, then enqueues on the
requested stream. The C ABI repeats these checks. Callers retain RAII buffer
ownership, validate encoded scales and establish producer dependencies.

The ABI symbol is `m40llm_gemv_f32xsa4_128_async`. CUDA exports stay behind
`cuda`; the non-CUDA C stub returns an error. No inference dispatch or default
changes. The format notice points to the retained upstream MIT license.

## Behavioral evidence

Three ignored hardware tests in [`shiftadd_cuda.rs`](../../tests/shiftadd_cuda.rs)
run explicitly with one test thread and an actual M40 context. The direct
grid covers input widths 256, 512, 2048 and 11008; output widths 1, 5 and 17;
rows 1, 2, 4 and 8; and ordinary, cancellation-heavy and zero activations.
It checks 4,140 output values against FP64 decoded-weight dots. All input and
weight bytes remain unchanged, all guards survive, and every case returns
tracked device bytes to its starting value. Enqueue performs no allocation
or free.

The error bound uses `gamma(n) = n*u/(1-n*u)`, with `u = 2^-24` and
`n = 12 + K/16`, multiplied by the absolute sum of plane contributions
before sign cancellation. This covers four local additions, five shuffle
additions and up to two roundings per plane/group accumulation, with margin.
FP16-times-u8 amplitudes are exact in FP32. A small underflow term is added.
The maximum observed fraction of this bound is 0.008254662 in debug and
0.007359363 in release. These are correctness bounds, not quality thresholds
against the original source model.

The stream test checks another 12,528 values through two chained projections
in both stream directions. Two workspaces remain live, input addresses are
overwritten across three generations, and each invocation verifies that the
other workspace's intermediate and final output stayed unchanged. The
argument test rejects 11 malformed dimension/pointer/overlap cases before
any launch, preserving all buffers. Both tests verify tracked cleanup.

All three tests pass with cuBLAS enabled and disabled in debug. All three
also pass in the optimized release build with cuBLAS enabled. Release
executable SHA-256:

```text
56031d0b3c72bc4216968e588579989f9abf4b2202197c39b0aaaedc690d8116
```

The release `sm_52` function uses 40 registers, zero stack bytes, zero shared
memory and zero local memory. Its retained SASS has no DP4A, Tensor Core or
local-memory load/store instructions. Static instruction counts include
80 FADD, 40 SHFL.DOWN, eight FMUL and eight FFMA instructions. These counts
describe the compiled function, not dynamic execution counts or measured
memory traffic. Repeat the binary inspection for the final timing executable.

## Validation and limitations

Source `scripts/dev-env.sh`, set this worktree's `CARGO_TARGET_DIR`, select
`CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`, and check
compute processes before serial hardware execution:

```bash
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --test shiftadd_cuda -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=0 cargo test --features cuda --test shiftadd_cuda -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --release --features cuda --test shiftadd_cuda -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo clippy --features cuda,server --all-targets -- -D warnings
RUSTFLAGS='-D warnings' cargo test --no-default-features --locked
cargo fmt --all -- --check
```

The 227-test non-CUDA suite and CUDA/server all-target Clippy pass. Existing
regressions pass 11 attention checks, four Q8 attention projection checks,
two FP32 FFN checks and one runtime-weight ownership check. The explicit
dense head128 Qwen forward/prefill parity check also passes.

An initial `forward_one_token_with_layer_smoke` invocation fails because its
head-dimension-4 fixture uses the current compressed-KV default. It does not
call ShiftAdd. That failure is retained, not counted as a pass or repaired
by changing defaults. The existing explicit dense head128 test provides
the forward regression for this change. Initial Clippy rejected two temporary
vectors in test setup; they were replaced with fixed slices without changing
device inputs or assertions.

Raw logs, release SASS and resource reports have prefix
`2026-10-31-shiftadd-cuda-` in this directory. The source/artifact SHA manifest
records the qualified files. The CUDA implementation depends on encoder
commit `a6ef091`. Next measure source-bound Qwen projection costs and fitting
error against matched Q4_0 and Q8_0 controls under the frozen research gates.
