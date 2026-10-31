# Native Q8 graph state qualification

The current native graph path fails this experiment's exact FP16 cache gate.
This closes the bounded graph investigation with an unmet prerequisite. No
graph performance or compound-capture claim follows from these results.

## Contract and fixture

The ignored hardware checks in
[`graph_tests.rs`](../../src/decode_session/graph_tests.rs) use the retained
two-layer Qwen-style Q8 fixture. It has width 256, FFN width 512, two query
heads, one KV head, head dimension 128, split-half RoPE, QKV biases and
320 vocabulary entries. Both models use native FFN/prefill/attention
projections, split down/vocabulary execution weights and explicit dense KV.
Each model owns four physical cache slots with capacity 32 tokens.

Two sessions interleave distinct token prefixes. Each case runs two complete
session lifetimes on the same models, with cache reset between lifetimes.
Every lifetime processes 16 tokens across the two sequences. After each token,
the checks compare all initialized FP16 K/V entries and all four lengths.
They also require one callback, matching processed length, finite logits with
absolute error at most 0.002, and identical greedy token choice. The 0.002
logit tolerance follows the existing full-forward parity tests. Cache bytes
must match exactly; this stricter state gate is distinct from numerical
logit parity. It was not relaxed after the first mismatch.

Fault injection exists only in NVCC-enabled library test builds, scoped to
one session. Capture rejection occurs after recording the real forward
operations and exercises CUDA capture cancellation. Launch rejection occurs
before submitting a graph, either at the first position or after three
successful replays. Both exercise the real ordinary-forward fallback.
They do not demonstrate recovery from an asynchronous CUDA execution fault.

## Results

Each final case runs in a fresh process because a failing assertion poisons
the shared test lock. All four reach the final exact-cache assertion after
32 token comparisons and complete tracked-allocation cleanup.

| Case | Repeated cache comparisons with unequal FP16 values | Maximum logit error | Maximum cache absolute error | Result |
|---|---:|---:|---:|---|
| Ordinary graph replay | 93 | 3.7718564e-8 | 6.1035156e-5 | Exact cache gate fails |
| Capture cancellation at position 0 | 37 | 3.7718564e-8 | 6.1035156e-5 | Exact cache gate fails |
| First launch rejection | 37 | 3.7718564e-8 | 6.1035156e-5 | Exact cache gate fails |
| Launch rejection at position 3 | 63 | 3.7718564e-8 | 6.1035156e-5 | Exact cache gate fails |

Counts include comparisons of retained earlier positions after each new
token; they are not counts of unique corrupted cache cells. All 128 token
comparisons pass the numerical-logit, greedy-ID, callback, position and
sequence-length assertions. Every case returns tracked device allocation
bytes to its initial value. This accounting does not independently measure
CUDA's internal graph allocations.

The capture-cancellation and first-launch-rejection cases have no cache
differences in sequence 0, which receives the fault. Their mismatches occur
in the other sequence, which continues graph replay. The later-rejection
case retains differences created during successful graph execution. These
results establish the tested fallback state behavior, but do not turn the
four overall failed research checks into passes.

The first isolated mismatch was one FP16 key in layer 1 at sequence 0's
third token, differing by 5.9604645e-8. The full checks show small accumulated
numerical differences, not a demonstrated sequence-addressing defect or
generation-quality failure.

## Dispatch prerequisite

The ordinary path in [`kernels.cu`](../../cuda/kernels.cu) dispatches head128
attention to `attention_last_token_gqa_head128_fast_kernel` for the tested
`M40LLM_ATTN_HEAD128_KERNEL=native` setting. Graph-parameter forward calls
`attention_last_token_gqa_seq_len_dev_kernel`, which uses a scalar per-head
dot product and repeats dot products while accumulating values. Their
reduction orders differ. This source difference explains why replay is not
an execution-only transformation of the ordinary path; no isolated kernel
ablation has proved that it is the sole cause of every differing cache bit.

A future candidate must preserve the intended attention dispatch with
device-resident lengths and pass the state contract before launch-overhead
or compound-runtime qualification. Such a change is a separate CUDA kernel
candidate with its own correctness and timing gates. This investigation
does not infer slow replay from source inspection or claim a speedup from
the small fixture. Capture, first-replay, steady-replay and buffered-HTTP
performance remain unmeasured under the required parity contract.

## Reproduction and evidence

Base checkpoint is `bb20672`. Final tested source SHA-256 values:

```text
606279599a3a0edcf5f4233726876c71ebc2cb976124c14a25a56e253721fd06  src/decode_session.rs
469a3247508b06abad10b9f1b01dfcef598daa5d60d4810addc3ee2bf3d9f183  src/decode_session/graph_tests.rs
4af26fb26e9609cbaffad5921e3f645af6b507189f7a177e0094304a6de2322a  src/lib.rs
```

Source the project toolchain, select the idle M40, then invoke each ignored
case separately. These are debug correctness checks, not performance runs.

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/legacy-graphs/target
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --lib \
  decode_session::graph_tests::test_native_q8_graph_sequences \
  -- --include-ignored --nocapture --test-threads=1
```

Replace the final case name with `test_native_q8_graph_capture_cancellation`,
`test_native_q8_graph_first_launch_rejection` or
`test_native_q8_graph_later_launch_rejection`. Exit 101 is the retained failed
exact-state result, not a successful qualification.

Raw logs use the `2026-10-31-graph-state-` prefix in this directory. They retain
the initial failure and lock-poisoned run, isolated diagnosis, four final
case runs, initial/final Clippy and non-CUDA validation. The first Clippy run
found an immediately invoked closure and an indexed range loop in the new
test code; both were corrected without changing the gates.

Validation: CUDA/server all-target Clippy passes; warning-free non-CUDA tests
pass 220 checks. The four existing graph compatibility tests remain passing
and do not substitute for the four failed native-Q8 exact-state checks.
Formatting and whitespace checks pass. No production graph restriction,
attention kernel, runtime default, or quantization representation changed.
