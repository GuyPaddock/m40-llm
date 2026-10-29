# Exact-Dense Dataflow Audit

## Decision

**No new speed kernel is justified by this bounded audit.** No individual
unexplored boundary has demonstrated removable work above the existing complete
request threshold, let alone the remaining 938.958 ms. Combining optimistic
elimination estimates for several unrelated kernels is not a supported design.
This is a negative feasibility result, not a hardware ceiling claim.

The immutable reference remains **50.910426 HTTP tok/s**, with goals 53.06 and
56.153149. Joint-only remains preferred for the explicitly selected compound
FFN. Runtime defaults, KV policy, exact arithmetic and all kernels are unchanged.
The separately qualified four-row verifier and copy-only drafter are preserved.

## Measurement And Correctness

Source `30290eb4f405c8eee51adc4dbc7d979d81585ecb`; release executable SHA-256
`ba57368bcefbe0ea5978800ead6b92d3e426a053a4c79fdc7d4232d20565935f`.
Use the frozen reference model/environment, CUDA 12.8.93 `-O3`, sm_52/compute_52,
no fast math, compound128, packed-warp256 QKV/output, prepared down/vocabulary,
joint-only ownership, bounded SGEMM prefill, dense FP16 KV/context1024 and the
canonical 65-token prompt with 512 actual greedy output IDs and length stopping.

Five alternating profiling-off/on buffered HTTP pairs, after both warmups:

| Metric | Profiling off | Profiling on |
| --- | ---: | ---: |
| HTTP tok/s, median | 50.780194 | 50.251923 |
| HTTP total ms, median | 10082.671 | 10188.665 |
| Prefill ms, median | 492.363 | 524.922 |
| Decode ms, median | 9536.765 | 9616.067 |
| Generation total ms, median | 10037.838 | 10142.454 |
| HTTP minus generation ms, median | 44.265 | 44.748 |

These are independently calculated medians and need not add. Instrumentation
adds **1.05125% latency**, below the fixed 3% attribution limit. Throughput
change is -1.04031%, paired CI [-1.17216%,-0.96103%]. MAD/median is below 0.10%
for each arm; neither extension nor confirmation is triggered. Do not subtract
instrumentation overhead or call the new off rate a replacement qualification
of the historical reference. No fresh Ollama comparison occurred.

All twenty timed boundaries show only PID3361367 on the selected M40, P0,
1113 MHz SM/3004 MHz memory. No build, test, other inference, CPU timing or
MemPalace work overlapped primary timing. Final process inspection is empty.

Whole-model preflight retains 109 logical sources, 37 physical split owners
and 36 joint pairs. Resident split/joint bytes are 1,192,935,424/1,724,645,376;
original GGUF remains 3,279,519,744 bytes. All actual browser IDs, seven sampled
full-vocabulary vectors and twelve frozen task outputs match qualification.
Seven tasks are dense-valid; five remain inconclusive, not new quality passes.
The additional profiler off/on check preserves all 512 IDs and stop reasons,
seven full-vocabulary vectors and **10,616,832 valid KV words bit-identically**.
Both qualification and HTTP teardown return tracked allocation to zero.

All **138 compiled CUDA instruction streams** match the completed draft study.
The raw profiler adds CPU-relative stage offsets and named H2D durations only.
It adds no device synchronization, dependency or kernel. Historical records with
absent offsets remain readable as unknown, not zero.

## Sampled Token Schedule

Primary event ranges are bound to each HTTP row by byte offsets, excluding all
untimed preflight, warmup and replay rows. Fifteen samples cover steps0/255/511,
absolute positions64/319/575, all36 layers, 400 spans and one vocabulary per
token. The 592-pair event pool is reused after the first sampled token.

| All-layer stage ms | Step0 | Step255 | Step511 |
| --- | ---: | ---: | ---: |
| Joint gate/up/SwiGLU | 7.421 | 7.417 | 7.425 |
| Prepared down | 4.770 | 4.773 | 4.763 |
| Vocabulary | 1.566 | 1.564 | 1.564 |
| QKV | 1.411 | 1.394 | 1.391 |
| Output projection | 1.019 | 1.010 | 1.010 |
| Attention including merge | 0.638 | 0.779 | 0.972 |
| Attention norm | 0.406 | 0.401 | 0.391 |
| FFN norm | 0.411 | 0.404 | 0.404 |
| Attention residual | 0.153 | 0.161 | 0.161 |
| FFN residual | 0.158 | 0.165 | 0.165 |
| Q RoPE | 0.295 | 0.294 | 0.293 |
| K RoPE + KV append | 0.315 | 0.320 | 0.320 |
| Argmax | 0.259 | 0.260 | 0.259 |
| Embedding | 0.039 | 0.031 | 0.028 |
| Output norm | 0.011 | 0.011 | 0.011 |

Steady tokens have 435 launches: 36 times12 layer launches, then output norm,
vocabulary and argmax. Attention uses two launches per layer. There is no
separate SwiGLU launch during compound decode. All decode work uses the decode
stream. Two stream synchronizations are H2D embedding completion and the final
greedy drain, not per-projection waits. There are no steady allocations/frees,
cuBLAS calls or cross-stream waits. Each token copies 2,176 embedding bytes D2H,
8,192 dequantized bytes H2D and one four-byte token D2H.

Whole requests retain 237,013 launches, 252 cuBLAS calls, 324 stream waits,
1,097 synchronizations, 580 H2D and 1,088 D2H copies, and 42 allocations/frees
per request, identical between arms. These include packed prefill and its
workspace transition, not just steady decode. Step0 records16 allocations and
17 frees as the packed workspace changes to one row; steady samples record0/0.
The 90,177,536-byte prefill conversion capacity remains bounded; this audit
does not resurrect SGEMM target verification or change prefill.

| Independent interval medians, ms | Step0 | Step255 | Step511 |
| --- | ---: | ---: | ---: |
| GPU stage union | 18.882 | 19.022 | 19.182 |
| GPU timeline span | 27.315 | 19.916 | 19.988 |
| Token wall | 27.348 | 19.957 | 20.015 |
| Wall minus GPU union | 8.483 | 0.905 | 0.835 |
| CPU enqueue span sum | 1.817 | 1.804 | 1.714 |
| CPU interval union including blocking calls | 14.991 | 15.116 | 15.544 |
| Host stream-synchronize duration | 13.155 | 13.252 | 13.818 |

The long host synchronize overlaps device execution. It is not thirteen
milliseconds of additional CPU work. Host timestamps and CUDA timestamps have
different origins. Their durations, nested copy calls and enqueue spans must
not be added. Event intervals also include queue/host effects. First-token
workspace/event initialization is not steady overhead. Sparse profiling passes
the request-level perturbation gate, but does not prove each sampled interval
has negligible local perturbation. No physical bandwidth, occupancy or stall
counter was measured.

## Buffer And Dependency Ledger

Sizes below describe one valid token, not reserved packed-workspace capacity.
The model's exclusive workspace lease prevents reuse by another request. One
decode stream orders producers and consumers; layer/sequence addressing and
absolute RoPE positions stay unchanged.

| Buffer | Bytes | Producer -> consumers; required lifetime |
| --- | ---: | --- |
| Session input and final hidden, each | 8192 | embedding -> first layer; last residual -> output norm |
| `d_xn` | 8192 | attention RMSNorm -> QKV; reusable next layer |
| `dq`, `dk`, `dv` | 8192,1024,1024 | QKV with bias -> in-place Q RoPE and K-RoPE/KV append |
| Dense KV token, all layers | 36864 | append -> causal attention through subsequent tokens |
| Attention partials | 66560 | split8 softmax/max/sum/V -> exact merge; one reusable buffer |
| `datt`, `dy_attn`, each | 8192 | attention -> output projection -> attention residual |
| `d_x1` | 8192 | attention residual -> FFN norm AND final residual; cannot discard after norm |
| `d_x1n` | 8192 | FFN norm -> gate/up; one row shared across output CTAs |
| `dhid` | 44032 | fused gate/up/SwiGLU -> down; complete vector required |
| `dy_mlp` | 8192 | down -> final residual |
| `scratch_a`, `scratch_b`, each | 8192 | layer residual ping-pong -> next layer and its residual |
| Output-norm scratch | 8192 | final norm -> vocabulary |
| Logits, token | 607744,4 | vocabulary -> stable greedy argmax -> host stop/token handling |
| `dgate`, `dup`, each | 44032 | allocated workspace slots, not written/read by qualified fused decode |

Source evidence: `src/infer/{forward,workspace,embeddings,logits}.rs`,
`src/decode_session.rs`, `src/cuda/dense_attention.rs`, `cuda/q8_ffn.cuh` and
`cuda/q8_execution.cuh`. Views describe disjoint allocations except intentional
Q in-place rotation and alternating layer inputs. The retained residual input
must survive projection writes. KV persists beyond the layer workspace.

For compound gate/up, 11,008 CTAs each logically read the same 8,192-byte
activation row: 90,177,536 requested bytes/layer before caches. Prepared down
shares one activation load across four outputs, giving512 CTAs times44,032
bytes =22,544,384 logical bytes/layer. Vocabulary has37,984 four-output CTAs
times8,192 bytes =311,164,928 logical activation bytes/token. These are static
instruction-level load estimates, **not measured DRAM traffic**. The reused
8 KiB/43 KiB inputs may be cache-resident. Deleting a producer's small store
does not automatically eliminate all those reads.

## Avoided-Work Assessment

Use stage budgets as optimistic scope bounds, not predictions. At the immutable
reference, 3% throughput requires about293 ms/request saved; the two objectives
require407.426/938.958 ms. The stronger target is equivalent to1.834 ms/token.

| Boundary | Optimistic scope at middle sample | Why it does not yet justify implementation |
| --- | --- | --- |
| Embedding host round trip | 0.031 ms/token, about16 ms/request | Tiny scope; GPU dequant must preserve host FP32 results; cannot count the final target drain as an embedding saving |
| Residual epilogues or residual+norm | both residuals total0.326 ms/token, about167 ms/request | Below3% even if residual cost vanished; normalization still needs all2048 outputs and its original reduction |
| Q RoPE combined with append | Q RoPE0.294 ms/token, about150 ms/request | Rotation/stores remain; at most remove one launch, not both full kernels |
| Vocabulary+argmax | argmax0.260 ms/token, about133 ms/request | Vocabulary projection remains; cross-CTA reduction still needed, existing prepared output contract must survive |
| Output norm | 0.011 ms/token, about5 ms/request | Reduction cannot disappear; moving scale into a dot changes FP32 association |
| Gate/up -> down | FFN12.190 ms/token, but only44,032-byte intermediate written | Down needs all gate outputs from11,008 independent CTAs. Avoiding the handoff needs cross-block synchronization or repeated gate projections; no measured cheaper exact design |
| Layer residual -> next norm/projection | visible norms0.806 ms/token are not wholly removable | Projection CTAs do not collectively own a completed row. Recomputing normalization per output replicates reductions; algebraic rescaling changes rounding |
| CPU launch/dependency gaps | measured wall residual0.835-0.905 ms/token | Even eliminating all is below1.834 ms/token; it contains instrumentation and cannot be identified as removable CPU time |

The isolated small boundaries do not clear the unchanged3% request gate.
Combining all their zero-cost upper bounds would require several new kernels,
would still retain their actual arithmetic, and would not establish the full
target. We therefore recommend **no follow-up kernel from this checkpoint**.
Do not trade numerical association, duplicate weight streaming, assume cache
misses from logical load counts, or reopen rejected FFN/graph variants to force
an opportunity. A future proposal needs new evidence of substantial avoided
work, bounded resource costs and an independently qualified full-request gain.

## Reproduction And Evidence

The retained executable/source are under
`local-artifacts/qualified-references/dataflow-30290eb/`. Restore the exact
environment from `2026-10-27-compound-reference.json`. With a clean measured
revision, run `joint_only --model MODEL --phase quality --output NEW_QUALITY`,
then `--phase dataflow --quality-report NEW_QUALITY --profile-report NEW_EVENTS
--output NEW_HTTP`. Quality must match both source and executable; event files
must not exist. The HTTP phase uses one joint-only model and does not time
Ollama. Do not run concurrent GPU workloads or other measurement activity.

Raw records are `2026-10-28-dataflow-{quality,http,events}.jsonl`.
`2026-10-28-dataflow-audit.pl` reproduces the summary, checks byte-range sample
ownership, clocks/PID, hashes, counts, logits, interval unions and cleanup.
It independently enumerates all3,125 paired bootstrap resamples; the interval
agrees with the fixed10,000-draw protocol within the preexisting0.001 audit
tolerance. `--self-test` rejects seven corrupted records. Float32 CUDA offsets
are reconstructed before checking unions; no numeric gate was relaxed.

Tests pass: four profiler unit tests, three profiler hardware tests, eight
runner/protocol tests, warning-as-error non-CUDA suite, CUDA/server all-target
Clippy, and25 additional CUDA attention/transfer/embedding/forward/prepared/
joint tests with no ignored cases. Immutable reference and preceding experiment
checksum sets pass. Logs, compiled identity and source/binary hashes accompany
this checkpoint. No new performance default is authorized. Next is the
separate learned-drafter paper assessment; then stop for review.
