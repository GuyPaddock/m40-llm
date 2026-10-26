# Exact Decode Reduction Hypothesis

## Source And Measurement Evidence

Clean base `2958096` closes split QKV/output storage without promotion.
The [qualified packed-runtime profile](2026-10-26-maxwell-packed-stage-profile.md)
assigns middle-token GPU-stage shares 1.986% attention norm, 2.066% FFN
norm and 1.324% greedy argmax. The combined scope is about 5.375%; these
event spans are not measurements of removable kernel work or a guarantee
of E2E improvement.

`cuda/kernels.cu::rms_norm_f32_weighted_parallel` uses 256 threads per row,
one shared publication barrier and a block barrier after each of eight
reduction steps. Each thread then computes the same normalization scale.
`argmax_f32_kernel` launches one 256-thread block for the entire vocabulary.
At Qwen's 151936 entries, each thread visits 593 or 594 values. Other SMs
cannot participate in that one-block scan. No hardware counter has yet
attributed either stage's latency to these source observations.

These are independent exact reduction experiments, not revivals of closed
FFN/projection mappings. The existing consumers and qualified Q8 execution,
prepared gate/up/down/vocabulary, prefix SGEMM and dense KV stay fixed.
No fusion, approximate math, attention change or serving schedule is added.

## Fixed Weighted RMSNorm Candidate

Keep 256 threads, each thread's original input stride and FP32 square/FMA
accumulation, one block per row, 1024 dynamic shared bytes and the original
FP32 sum tree. Publish all 256 partials. The first warp reproduces strides
128/64/32 through shared reads, then the same 16/8/4/2/1 tree with shuffles.
Thread zero computes the existing rsqrt expression and publishes the scale
in the reused shared array. A second block barrier precedes parallel output.

Retain scalar input/weight loads and the original `(input * scale) * weight`
association, supporting FP16 and FP32 weights. Do not combine vector loads,
cache hints, residual fusion, a new warp count or a changed summation tree.
Require exact old-kernel bits across supported shapes and CPU tolerances.

## Fixed Partitioned Greedy Argmax Candidate

Use at most 48 blocks, 256 threads each, striding across vocabulary values.
Each block writes one value/index partial; a second block selects the final
index. Keep the existing comparison, lowest-index tie rule and initial
`-FLT_MAX`/index-zero sentinel, including observable NaN/infinity behavior.
Use safe wide scan indexing and test tails around partition boundaries.

Scratch holds 48 value/index pairs, 384 bytes, in one caller-owned buffer
reused across calls. Include both kernel launches, intermediate traffic and
the final reduction in timing. No allocation, copy or synchronization may
occur inside enqueue. Do not change sampling, token stopping or logits.
No block-count/thread-count rescue is permitted if this fixed mapping loses.

## Validation, Measurements And Gates

Add benchmark-only primitives first; do not expose runtime selectors yet.
Test weighted norm control bits, FP16/FP32 weights, zeros, non-binary inputs,
ragged dimensions, distinct rows and ordered producers on both streams.
Argmax checks raw old-kernel indices, CPU indices for ordinary finite inputs,
ties, extremes, NaNs/infinities, small/large vocabularies and boundary tails.
Verify guards, immutable inputs, scratch reuse, no timed side effects,
rejected arguments and RAII cleanup. Run cuBLAS on/off primitive checks and
affected attention/projection/MLP/forward, non-CUDA, Clippy and formatting.

Extend the existing Q8 execution benchmark, not a separate framework. Use
real model norm weights and frozen M02 input vectors, and explicitly labeled
argmax input distributions. Compare rows/requests 1 and 4 separately. Capture
compiled sm_52 instructions, registers, spills, actual operation counts,
memory and GPU isolation. Keep cache experiments off. Use bounded warmup,
five alternating pairs and ten primary iterations, with existing fresh
confirmation rules for borderline findings.

Each independent stage needs at least 5% complete-operation throughput gain
and a positive paired interval. Only passing stages may contribute to the
latest profile-weighted feasibility estimate; it must clear 3% before
runtime integration. A hypothetical 2x norm and 10x argmax would be enough
in that optimistic calculation, but neither is a prediction. If the measured
scope is insufficient, keep the implementation benchmark-only without
geometry, fusion or register-cap rescue.

Only passing gates permit immutable runtime choices and session-owned
argmax scratch, then actual 512-ID/frozen-task/logit and incremental/fresh
official Ollama HTTP qualification. Preserve all model identities and the
original **53.06 tok/s** and **1.30x fresh Ollama** requirements. No numerical
or model-changing technique substitutes for those exact-output goals.

## Atomic Delivery

1. Commit this hypothesis before code.
2. Commit weighted RMSNorm primitive and behavioral tests after verification.
3. Commit independent partitioned argmax primitive and behavioral tests.
4. Commit shared benchmark extension before release timing.
5. Commit raw results, independent audit and measured eligibility decision.
6. Only on passing gates, separate runtime, quality and HTTP commits.

Keep losing code committed before any later removal. No runtime default
changes follow from primitive correctness or isolated timing.

## Shared Benchmark Contract

`examples/q8_execution_bench --decode-reductions` requires the frozen M02
`--ffn-inputs` and `--ffn-capture-report`, release, a clean worktree and rows
`1,4`. It excludes all other experiment switches and cache experiments.
Actual layer 0/17/35 attention/FFN norm weights retain their source names,
offsets, dtype, shape, byte hashes and model epsilon. Frozen normalized FFN
vectors are **norm-input surrogates**, not captured pre-norm hidden states.

Argmax uses labeled synthetic `random`, `ascending` and `late-tie` finite
logits at Qwen's 151936-entry vocabulary. Seed, distribution, input hashes
and CPU indices are recorded. Four requests use distinct inputs and reuse
one 384-byte partial workspace serially, not concurrently. Norm rows4 is
one four-block kernel; argmax requests4 is four independent operations.
There are twelve norm cases and six argmax cases, each with five alternating
pairs by default. Complete argmax timing includes both candidate kernels.

The common warmup sampler accepts a finite caller-selected budget. Reduction
warmup uses 4096 iterations per batch, at most 256 batches, retaining the same
1000-ms control / 100-ms candidate targets. Existing projections retain their
256-iteration budget. Primary measurements remain ten iterations. Warmup
samples are excluded, and exhaustion is an error rather than a warm result.

Each interval checks actual operation counters and forbids device allocation,
free, H2D/D2H, stream sync/wait and cuBLAS calls inside enqueue. Guards and
immutable input/weight bytes are checked outside timing. Prepared model keys
and residency are revalidated, and all allocations must return to the initial
count. Timing excludes setup, upload, validation, readback and the pre-drain.

The combined optimistic scope uses only stages passing all three one-request
cases, taking the minimum observed gain per stage. It is not E2E evidence.
Even a passing scope gate does not itself permit runtime promotion.
