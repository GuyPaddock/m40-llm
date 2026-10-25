# Exact Qwen Kernel Checkpoint: 2026-10-25

## Decision

C11 closes the first bounded exact-kernel/layout round. Retain native Q8
execution with C07's split8/256 full-context head128 attention as the selected
benchmark configuration. Preparation remains off. Do not change global KV,
scheduler, graph, or experimental-policy defaults.

A fresh controlled release comparison after cleanup confirms **31.850 HTTP
E2E tok/s** for m40-llm versus **43.032** for Ollama 0.40.11. The original
objectives remain unmet. Move to C11A's offline n-gram feasibility gate rather
than extending layout tuning or presenting an unqualified prepared result as
the winner.

## Keep and Remove

| Unit | Evidence | Decision |
| --- | --- | --- |
| C07 exact head128 attention | Five-pair +33.98% HTTP gain, matching 512 IDs; context/CPU/CUDA parity | Keep split8/256 in the opt-in benchmark configuration; retain reference kernel |
| C08 typed execution owners | Lossless identity/bounds, RAII and partial-failure tests | Keep shared experimental infrastructure |
| C09 separate planes | Isolated real-shape projection gains, but no qualified C10A HTTP winner | Keep opt-in research; no default promotion |
| C10 vector-tiled access | Promising rows=64 Q/down; not a global M1 winner | Keep opt-in research for multi-row work; no model/verifier speed claim |
| C10 coalesced-lane access | Slower projections, no register reduction | Remove active kernel, launch selector, and CLI option in `f5cda0d`; preserved in `bb4d30e` |
| C10A prepared fused MLP | Full preparation -64.21% throughput; native-MLP hybrid avoids most loss | Remove active consumer/FFI/dispatch in `9044eea`; preserved in `e2dbc61`, measured in `2e2bb5c` |
| C10A remaining cache/QKV/vocabulary consumers | Model ownership/parity passes; native-MLP hybrid -6.71% HTTP throughput | Keep diagnostic/opt-in only, not selected for generation |
| C10A down/vocabulary subset | +12.53% timing, but actual IDs branch at generated step 310 | Preserve timing lead and mismatch; reject combined promotion, no further subset tuning |

Removal is separate from preservation. Native fused MLP remains one async
launch even with prepared caches; model shape/pointer checks and CPU/native
parity are retained. Gate/up preparation still serves standard/multi-row APIs.
Vector-only descriptors no longer expose compatibility aliases for the removed
lane selector. Historical commands remain tied to their recorded revisions.

Details and raw evidence are in the [baseline](2026-10-24-qwen-baseline-checkpoint.md),
[attention](2026-10-24-head128-attention.md), [split-plane](2026-10-24-q8-separate-projections.md),
[tiled](2026-10-24-q8-input-tiled.md), and [prepared HTTP](2026-10-25-q8-prepared-http.md)
reports. Isolated gains are not whole-model gains. The down/vocabulary result
does not prove a kernel bug or harmless rounding without further evidence;
this checkpoint neither retunes it nor relaxes its parity gate.

## Fresh Controlled Confirmation

Measured revision is `f5cda0d`, with a clean initial worktree. Build is release
Rust 1.95.0, NVCC 12.8.93, g++ 14.3.1, CUDA `-O3`, sm_52 SASS plus compute_52
PTX. Same Qwen2.5-3B-Instruct Q8_0 bytes are used in both engines, SHA-256:

```text
4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d
```

The unchanged browser question is rendered with the canonical Qwen system
message and raw ChatML assistant prefix, SHA-256:

```text
3005a73b43160b265f25b4545853c3fdab0ab75014d315c701ec6bef62648f9a
```

The tokenizer oracle verifies all 65 actual prompt IDs. Every warmup and
screen row generates 512 tokens with a length stop, greedy argmax, context
1024, and dense FP16 KV. Natural stopping is enabled. Warmup is separate from
five screen pairs, alternating runtime order. Both models may be resident;
requests, inference, builds, and GPU tests never overlap during timing.

Ollama's RAM prompt cache is disabled. Each measured Ollama call follows a
replacement prompt with zero common token prefix, including special tokens.
Actual native diagnostics verify **0 reused / 65 evaluated** prompt tokens
for every accepted row; nominal prompt_eval_count is not the sole evidence.
m40-llm resets prompt KV for every request. No invalid rows are dropped.

GPU is Tesla M40 24GB, UUID `GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`, sm_52,
24 SMs, driver 580.159.04. Screen boundary samples show P0, 1113/3004 MHz
SM/memory clocks, and 47-69 C temperatures. Clocks are not locked and these
are boundary samples, not continuous counters. Only the benchmark and isolated
Ollama runner hold compute allocations. Both exit after capture; final usage
is 0 MiB / 0% utilization.

The primary interval is client send through the complete buffered HTTP body,
before JSON parsing. Model loading, replacement requests, oracle checks,
boundary GPU queries, and actual-ID companion generation are outside it.

| Runtime | Median HTTP tok/s | Range | MAD/median | HTTP wall | Internal prefill | Internal decode | Internal total |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| m40-llm | 31.850 | 31.819-31.936 | 0.0981% | 16.075 s | 1.683 s | 14.348 s | 16.031 s |
| Ollama 0.40.11 | 43.032 | 42.616-43.067 | 0.0755% | 11.898 s | 0.231 s | 11.663 s | 11.898 s |

These are independent medians. m40 prefill includes initialization through
first sampling, whereas Ollama prefill is native prompt evaluation. Do not
subtract them as matched phase measurements or sum independent medians into
a false HTTP total. Dense m40 prefill is sequential in this configuration;
the packed-prefix env knob is unset.

m40 throughput relative to Ollama is **-25.985%**, paired-bootstrap 95% interval
**[-26.118%, -25.305%]**, 10,000 resamples, seed zero. Neither the >5% variation
extension nor borderline confirmation triggers. This is an Ollama comparison,
not a new promotion test against the legacy m40 attention kernel. C07's
original matched comparison remains the evidence for its +33.98% promotion.

The objectives still require both **53.06 tok/s** and **1.30x fresh Ollama**,
now **55.9416 tok/s**. The latter requires another 75.64% throughput increase
from this checkpoint. Do not lower the 53.06 floor or replace HTTP E2E with
decode-only timing to claim success.

## Actual Output and Memory

Each runtime's five screen outputs have one stable hash. Untimed companions
reproduce their measured outputs and save all 65 prompt / 512 generated IDs.
The m40 IDs match the prior qualified native/C07 trace exactly, including after
the cleanup's native MLP checks. This is behavioral evidence beyond a passing
build. Cross-engine continuations are different; no cross-engine ID equality
is claimed. Both use the same unchanged Q8 model and full attention, not
pruned layers, reduced-context attention, or lossy KV for this workload.

m40 output SHA-256 is
`0ee35fce99522bf7321d5a059493db3588ff97c5328fe0f429e16520577339de`;
Ollama output SHA-256 is
`cc8850a02e8761a180b2880f37d5d5053536f9dfe76e85052ef305e1281bcd59`.
Both full texts and actual traces remain in the JSONL. The selected m40 trace
is the formal C11A browser-study input, not the faster mismatching prepared
continuation or a re-tokenized string.

m40 uploaded weight payload is 3,279,519,744 bytes, with **zero prepared
bytes**, 37,748,736 dense KV bytes, and one reusable 66,560-byte head128 partial
workspace. No per-token split-attention scratch allocation is introduced.
Boundary nvidia-smi allocation includes contexts/granularity and idle resident
weights in the other process; it is not peak VRAM or measured DRAM traffic.

One m40 construction/upload observation is 5.720 s. The oracle reports an
Ollama load duration of 2.505 s, with different setup boundaries. Neither is a
repeated, matched cold-loading benchmark; no cold-loading speed claim is made.
Hash/preparation time is zero with preparation off. Exact environment,
executable hashes, tensor inventory, setup, clocks, and memory are in the
manifest and trial records.

## Remaining Costs and C11A

The last selected-config [stage diagnostic](2026-10-24-head128-attention.md#separate-stage-diagnostic)
is from C07, not a new C11 profile. It measured middle/late attention at
0.791/0.973 ms across all 36 layers, gate/up/SwiGLU about 11.8 ms, down about
6.3 ms, and vocabulary/argmax about 3.4 ms. FFN is about 64% of the middle
GPU interval union and 78.90% of stored parameters. The measured HTTP gain
does not justify another unbounded attention/layout tuning round.

CUPTI's legacy counter preflight still fails actual event-group creation with
NOT_SUPPORTED. DRAM traffic, cache hits, and occupancy remain unavailable,
not inferred hardware measurements. See [counter limitations](../maxwell_counters.md).

Next, implement the offline history-only n-gram study over the selected actual
browser trace and fixed regression prompts. Test suffixes 3-8 and drafts
2/4/8/16, longest match with most-recent usable tie-breaking. Score future IDs
without adding them to draft history. Report per-position opportunities,
simulated committed rounds, rejection/correction/bonus behavior, drafting CPU
time, and cost sensitivity at hypothetical verifier costs 1/1.5/2/4 target
steps. The formal gate requires at least 10% optimistic estimated browser E2E
gain before C12-C17. A negative result postpones that infrastructure and leads
through C18 to C19's separately quality-tested model research.

## Reproduce and Checks

Use the full [controlled comparison command](../controlled_benchmark.md),
including its Q8 stream/fusion/greedy/backend flags, and explicitly add:

```bash
export M40LLM_Q8_EXECUTION_WEIGHTS=off
export M40LLM_ATTN_HEAD128_KERNEL=parallel
export M40LLM_ATTN_HEAD128_PARTITIONS=8
export M40LLM_ATTN_HEAD128_THREADS=256
```

These select the checkpoint for that command; they do not replace its other
backend assignments. Start the isolated pinned Ollama 0.40.11 and
use a new report path. No new Ollama or model download is needed.

Post-cleanup checks pass: execution-weight library 17 tests, attention grid 11,
forward suite 24, retained vector CPU/GPU tails, native/prepared cache ownership,
family selection, QKV, greedy vocabulary, native MLP and projection/MLP wrappers.
Attention/vector/native-MLP/wrappers also pass with cuBLAS disabled. Non-CUDA
warning-as-error tests, CUDA/server all-targets Clippy, and formatting pass.

Raw [HTTP JSONL](2026-10-25-qwen-exact-checkpoint.jsonl),
[benchmark log](2026-10-25-qwen-exact-checkpoint.txt), and
[Ollama backend/cache log](2026-10-25-qwen-exact-ollama.txt) retain the evidence.
Only trailing horizontal whitespace is normalized in the copied Ollama log.
No observations, invalid cases, or token traces are rewritten.
