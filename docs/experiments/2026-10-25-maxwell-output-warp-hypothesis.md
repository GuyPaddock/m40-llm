# Native Output Reduction Hypothesis

## Evidence Before Implementation

The qualified split-FFN/SGEMM profile measures native output projection at
1.773 ms / 8.25% of middle-token GPU time. Frozen column-tile=4 and split4
settings select `gemm_f32xq8_0_gguf_f32_decode_coltile4_split4_kernel` at
256 threads, four output columns and 4096 dynamic shared bytes. Source has
one publication barrier plus a reduction-loop barrier for strides 128 to 1,
reducing four partial arrays. That is nine dynamic block-wide barriers.

Current sm_52 code has publication at `1490`, loop barrier at `15d0` and
backward branch at `15f0`. Extracted control SASS SHA-256 is
`a122594fde39a84160359f5035fbd40882a1df7098a5a4c45988858ed6e9dc1e`.
Static barrier sites are not dynamic counts or measured stall attribution.

## One Independent Candidate

Keep the four-column mapping, 256 threads, split4 task assignment, original
raw 34-byte Q8_0 weights, FP32 activations, per-weight scaling/FMA and
column-tail guards. Publish the same four arrays, then finish each through
the existing `reduce_published<256>` tree in the first warp. Preserve FP32
association and reuse the original source layout; do not pack new weights,
change geometry, quantize activations or touch prepared down/vocabulary.
Keep the original control body and emitted instructions unchanged.

Reuse the shared projection probe introduced for QKV. Generalize its
internal ownership/timing code for QKV versus output rather than clone a
second benchmark. Retain explicit QKV report identities. Frozen actual
normalized FFN vectors remain disclosed projection surrogates, not newly
captured attention outputs. Real layer 0/17/35 output weights are used.
Rows 1/4 are distinct independent one-token calls, not a batched GEMM.

Require CPU agreement, control-bit equality, guards, signed extremes, zero
scales, N tails, stream ordering, actual dispatch counts and RAII cleanup.
No timed allocation, transfer or synchronization. Inspect registers/spills
without retuning. Use five alternating pairs and ten repetitions. The output
stage must gain >=5% with positive paired intervals on all three one-request
cases; a borderline result gets one fresh twenty-pair confirmation, not a
geometry rescue.

## Combined Gate And Atomic Delivery

The positive [QKV stage result](2026-10-25-maxwell-qkv-warp-screen.md) remains
unchanged and unqualified standalone. Bind its canonical raw report SHA-256
`9c1468a93abc8b4828e04d5880f1d1fbb70c36fe4790c73c27087a0098217b73`
as a fixed peer input. Compute the optimistic combined ceiling per matching
layer/request case, using measured stage throughput gains gq and go:

`1 / (1 - 0.1172 - 0.0825 + 0.1172/(1+gq) + 0.0825/(1+go)) - 1`.

Only if both independent stage gates and the combined >=3% ceiling pass
for all three one-request cases should a combined immutable opt-in runtime
be implemented. This estimate is not measured HTTP performance. Actual
512-token equality, frozen dense-valid tasks, current split-FFN/SGEMM
control, original E2E >=max(3%, twice variation) and positive paired interval,
and fresh official Ollama remain mandatory. No runtime defaults change.

1. `docs(perf): scope native output reduction`: this audit and fixed gates.
2. `feat(cuda): test native output warp finish`: one CUDA primitive, shared
   projection probe, fixed-peer evidence binding and behavioral tests.
3. `docs(perf): record native output reduction`: raw results, ownership and
   numerical evidence, instruction hashes and measured proceed/stop decision.

The last qualified runtime remains 45.075 versus Ollama 43.119 HTTP tok/s.
Both original goals remain unchanged and unmet: 53.06 and >=1.30x fresh
Ollama, currently 56.055. Approximate/model-changing methods, more FFN
variants and unrelated scheduling do not substitute for those requirements.
