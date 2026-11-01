# CUDA Upload-Ordering Checkpoint: 2026-10-25

## Decision

The pageable H2D audit is complete. Central host uploads, native weight upload
and KV initialization now publish device-ready storage; legacy append length
publication no longer depends on asynchronous copies from stack variables.
Production explicit-position append remains enqueue-only. These are
correctness fixes, not a new kernel optimization or a changed inference mode.

A fresh five-pair release HTTP check measures **31.937 tok/s** for m40-llm and
**43.163 tok/s** for Ollama 0.40.11. The selected exact configuration preserves
all 65 prompt IDs and 512 generated IDs from C11. Neither speed objective is
met. Resume C23's retained input-layout prerequisite assessment, independently
of the rejected C21 sparse kernel. C22 and C27 remain postponed.

## Correctness Evidence

At pre-fix `3818104`, a repeated pageable upload consumed immediately on a
nonblocking stream reads zero instead of 0.022460938 at rows=16, K=11008,
iteration=1, index=76032. This regression uses native Q8, not sparse weights.
Its exact binary fixture avoids ordinary rounding ambiguity. All 48 repeated
uploads pass after the fix, with immediate decode and prefill consumers and
no readback or allocation between upload and consumption.

The dependency-ordered corrections are:

| Commit | Change | Scope |
| --- | --- | --- |
| `ad217af` | Complete synchronous H2D before returning | Default-stream completion only; caller still owns prior nonblocking destination users |
| `cca45db` | Publish initialized native weights and dense/compressed KV | Checked fills, one setup completion fence, partial-failure cleanup |
| `3ab1377` | Publish legacy append lengths on device | Drain length producer before host read; scalar kernel arguments replace stack-owned async copies |
| `7245d26` | Isolate fast-fits graph fixtures | Explicit dense KV, compiled async cuBLAS and explicit hardware invocation |

The last item addresses an independent pre-existing test problem found during
validation. A detached pre-fix `3818104` worktree reproduces the four cuBLAS-off
graph failures, 20 forward cases passing and four failing. The raw graph case
also depended on another test having selected dense KV. The fixtures now state
those requirements explicitly. This does not make all CUDA graphs cuBLAS-only,
extend capture coverage, or fix the synchronous F16 graph fallback. Runtime
graphs remain experimental and off.

See [the transfer/ownership contract](../cuda_transfer_contract.md) and
[the pre-fix graph failure log](2026-10-25-upload-baseline-graph-off.txt).
No numeric tolerance was relaxed and no production per-kernel wait was added.
Explicit-position head64/head128 tests verify two launches with zero stream
syncs/waits, host copies, allocations or frees in the measured append scope.

## Validation

All hardware work ran serially on Tesla M40, with explicit escalation.
Builds/tests did not overlap timed inference.

| Check | Result |
| --- | --- |
| Transfer ordering, native setup and legacy append | Five explicitly included M40 regressions pass |
| Attention parity grid | 11 cases pass with cuBLAS on and off |
| CUDA allocation ownership, elementwise, KV cast/RoPE/reset, projection/MLP | All selected cases pass |
| cuBLAS-on full forward suite | 24 pass, including all four explicitly included graph fixtures |
| Isolated raw forward graph fixture | Passes without relying on earlier test state |
| cuBLAS-off supported transfer/attention/forward/KV/projection/MLP matrix | 43 pass; four cuBLAS-specific graph fixtures are not compiled |
| Prepared Q8 owners/projections/families, layer masks, teacher forcing, GPU profiling | Selected hardware cases pass |
| Non-CUDA warning-as-error suite | Full locked suite passes |
| NVCC/no-device warning-as-error suite | Full locked CUDA suite passes; ignored hardware checks are not GPU evidence |
| Explicit graph request with GPU hidden | Fails as intended: `m40llm_create_context returned null` |
| All-targets Clippy | Non-CUDA and CUDA/server with cuBLAS on/off pass |
| Formatting | `cargo fmt --all -- --check` passes |

The optional real-Qwen test inside the forward suite had no model env set;
its skip is not another real-model quality result. The release runs below
provide the explicit-model evidence. Passing GPU-less CI is not a parity claim.

Reproduction starts with `source scripts/dev-env.sh`:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --lib \
  --test cuda_transfer_order --test cuda_allocation_errors \
  --test attention_parity_cuda_grid --test cuda_elementwise \
  --test kv_f32_to_f16_append --test proj_wrappers --test mlp_wrappers \
  --test forward_with_layer_smoke --test q8_execution_weights \
  --test q8_separate_projection --test q8_tiled_projection \
  --test q8_runtime_weights --test q8_prepared_qkv --test q8_native_mlp_cache \
  --test q8_family_selection --test q8_prepared_greedy \
  --test layer_mask_cuda --test teacher_forced_cuda --test gpu_stage_profile \
  -- --include-ignored --test-threads=1

M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=0 \
  cargo test --features cuda --test cuda_transfer_order \
  --test attention_parity_cuda_grid --test forward_with_layer_smoke \
  --test kv_f32_to_f16_append --test mlp_wrappers --test proj_wrappers \
  -- --include-ignored --test-threads=1

RUSTFLAGS="-D warnings" cargo test --no-default-features --locked
CUDA_VISIBLE_DEVICES='' M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  RUSTFLAGS="-D warnings" \
  cargo test --features cuda --locked --all --no-fail-fast
```

Detailed [cuBLAS-on](2026-10-25-upload-cuda-on.txt),
[cuBLAS-off](2026-10-25-upload-cuda-off.txt) and
[expected no-device failure](2026-10-25-upload-no-device-negative.txt) logs
are retained. The no-device failure is a separate negative test, not part of
the passing CI result.

## Release HTTP Confirmation

Measured clean source is `7245d26476385109f623a029ceb383edf13ac4d7`.
Rust 1.95.0, NVCC 12.8.93 and g++ 14.3.1 build conventional Maxwell CUDA with
`-O3`, `sm_52` SASS and `compute_52` PTX, no fast-math flag. The benchmark
executable SHA-256 is
`a8ee0c55e67948d82a45a512f9475a633a99b3852acb238f048314797568d3e6`.
The manifest records full compiler, tensor, GPU and executable identity.

Both runtimes use unchanged Qwen2.5-3B-Instruct Q8_0 bytes, SHA-256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
The canonical system-message/raw-ChatML browser prompt has 65 oracle-matched
IDs. All warmup and screen requests produce 512 actual tokens with length
stops, greedy top_k=1 and context1024. Dense FP16 KV is explicit for this
comparison; the general compressed top8 default is unchanged.

Use the [controlled comparison setup](../controlled_benchmark.md) with the
pinned isolated Ollama 0.40.11 CUDA 12 runner. The selected environment is:

```bash
export M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_Q8_EXECUTION_WEIGHTS=off
export M40LLM_Q8_DECODE_STREAM=decode
export M40LLM_Q8_DECODE_SPLIT_QBLOCK=4 M40LLM_Q8_DECODE_COL_TILE=4
export M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK=4
export M40LLM_FUSED_Q8_QKV=1 M40LLM_FUSED_Q8_MLP_SWIGLU=1
export M40LLM_CUDA_GREEDY_ARGMAX=1
export M40LLM_Q8_LM_HEAD_ARGMAX=1 M40LLM_Q8_LM_HEAD_ARGMAX_SHARED_A=1
export M40LLM_PROJECTION_BACKEND=large-model
export M40LLM_MATERIALIZE_F32_WEIGHTS=0 M40LLM_DECODE_GRAPH=0
export M40LLM_ATTN_HEAD128_KERNEL=parallel
export M40LLM_ATTN_HEAD128_PARTITIONS=8 M40LLM_ATTN_HEAD128_THREADS=256
export M40LLM_STAGE_PROFILE=0 M40LLM_LAUNCH_LOG=0
export M40LLM_SYNC_LOG=0 M40LLM_ALLOC_LOG=0

cargo build --release --features cuda,server \
  --bin m40-llm --example compare_ollama
target/release/examples/compare_ollama \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ollama-url http://127.0.0.1:18435 \
  --ollama-pid OLLAMA_SERVER_PID \
  --ollama-log /tmp/m40-upload-ollama.log \
  --output /tmp/m40-upload-order-checkpoint.jsonl
```

Five screen pairs alternate runtime order after untimed warmup. Both models
can be resident; inference is never concurrent. Every accepted row verifies
zero reused KV and 65 evaluated prompt tokens, including actual Ollama runner
cache lines after zero-common-prefix replacement. HTTP client time excludes
load, replacement, oracle, boundary GPU queries and trace capture. No invalid
row is dropped. Clocks are not locked; boundary samples retain P0 and
1113/3004 MHz SM/memory clocks.

| Runtime | Median HTTP tok/s | Range tok/s | MAD/median | HTTP ms | Internal prefill ms | Internal decode ms | Internal total ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| m40-llm | 31.937 | 31.433-31.983 | 0.0614% | 16031.497 | 1687.257 | 14298.087 | 15985.384 |
| Ollama 0.40.11 | 43.163 | 42.775-43.211 | 0.0734% | 11862.091 | 229.988 | 11626.885 | 11861.552 |

These are independent medians. m40 prefill includes initialization through
first sampling; Ollama reports native prompt evaluation. Do not subtract the
phases as matched measurements or add independent medians into a false total.
m40 uses sequential dense prefill here, with the packed-prefix knob unset.

m40-to-Ollama throughput is **-26.008%**, paired-bootstrap 95% interval
**[-26.515%, -25.984%]**, 10,000 resamples, seed0. Neither variation extension
nor borderline confirmation triggers. The original **53.06 tok/s** floor and
**1.30x fresh Ollama = 56.1115 tok/s** both remain unmet.

C11's earlier 31.850 tok/s and this 31.937 result are separate checkpoints,
not a contemporaneously paired pre/post-fix experiment. Their small difference
is not evidence of a speed improvement from upload synchronization. The fresh
pairs compare runtimes, not upload implementations. C07's original promotion
evidence remains separate.

All five outputs per runtime are stable. The untimed m40 companion matches
the measured text and C11's entire actual token trace. Cross-engine text/IDs
differ as before; no cross-engine equality is claimed. m40 uploads
3,279,519,744 weight bytes with zero prepared/materialized cache, 37,748,736
dense KV bytes and a reusable 66,560-byte attention workspace. These payloads
are not total process VRAM. The isolated Ollama runner is unloaded/stopped;
after all CLI checks the device reports 0 MiB and 0% utilization.

## Short CLI Quality Checks

The first untimed user-only Auto checks reproduce existing limitations:
arithmetic returns `4`; exact-OK returns empty text; config lookup returns
`footer\nNote: The value of M40_BATCH_LIMIT is 3`. Both default compressed
and explicit dense mode behave identically. The same failures were retained
in the [earlier native/prepared checkpoint](2026-10-25-q8-prepared-http.md).
They are not new upload or compression regressions, and dense-failed rows
remain inconclusive for compression quality.

Rerunning the exact canonical Qwen system-message/raw-ChatML fixtures from
C11A returns `4`, `OK` and `37` in **both** default compressed top8 and dense
`--kv-compress-mode off`: six of six passes. No prompt-format implementation,
runtime default or quality rubric was changed. The CLI artifact retains all
twelve observations, actual commands and startup logs rather than replacing
the unsuccessful user-only rows. These are untimed text smokes, not captured
token IDs, a long-context memory claim, or an E2E performance comparison.

The default compressed startup uses FP16-K/q4-V direct attention, top-k/top8.
At this short context1024 bound it allocates 61,747,488 KV bytes versus
37,748,736 dense bytes because the recent ring itself is 1024 tokens. This is
compatibility evidence, not a compression-saving result.

## Artifacts

- [Raw manifest, trials, cache proofs and actual token traces](2026-10-25-upload-order-checkpoint.jsonl).
- [Benchmark console summary](2026-10-25-upload-checkpoint-run.txt).
- [C11 actual-ID comparison](2026-10-25-upload-trace-comparison.jsonl).
- [Twelve CLI observations and startup configurations](2026-10-25-upload-cli-smokes.jsonl).

No sparse kernel, approximate attention, pruned layer, fallback policy or new
default is selected by this checkpoint. The remaining research retains its
separate feasibility and quality gates.
