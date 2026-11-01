# Packed Q8 Prefill Attribution

## Finding

The current transposing Q8 conversion costs **88.787 ms** and its SGEMMs
cost **91.566 ms** across all 36 layers of the 64-token prefix. These are
profiled CUDA-event spans, not unprofiled kernel latency or measured DRAM
traffic. They do not justify another conversion/layout implementation by
themselves: even eliminating both gives only a **1.771% optimistic HTTP
throughput estimate**, below the unchanged 3% E2E floor.

Five alternating release profiling-off/on pairs measure **49.390303 /
48.707494 HTTP tok/s**, a **1.382475% instrumentation reduction**. Actual
512-token continuations, output, every ordinary enqueue counter and all
109 prepared identities remain unchanged. This is attribution, not a new
optimization, promoted throughput result or fresh Ollama comparison.

The qualified exact result remains **49.146 HTTP tok/s versus 43.106
official Ollama**. Both original requirements, **53.06** and **56.037**
HTTP tok/s for 1.30x that official median, remain unmet. Defaults and the
selected packed-QKV/split-FFN/prepared-down/vocabulary backend are unchanged.

## Timing Boundaries And Coverage

Seven profiled sessions contain one warmup, five primary requests and one
actual-ID replay. The five primary prefixes each have 540 spans: seven
conversion/SGEMM pairs per layer plus one existing Q/K/V bias scope per
layer. Each prefix processes 64 tokens at their original positions. Ordinary
decode profiling still samples generated steps 0/255/511 with 400 spans
each. Prefix rows are a distinct JSONL kind, not fabricated token rows.

| Profiled prefix measurement | Median ms |
| --- | ---: |
| Packed-forward wall scope | 363.252 |
| Conversion event spans | 88.787 |
| SGEMM event spans | 91.566 |
| Q/K/V bias event spans | 43.749 |
| Captured GPU-span union | 223.945 |
| Wall minus captured union, unattributed | 139.894 |

The prefix scope includes workspace setup and the existing stream drains,
but not all request setup or the final prompt token. Its GPU spans omit
embedding, most norms, RoPE, KV append, attention and residuals. The
unattributed wall interval is **not** measured CPU-only time. Host and GPU
spans may overlap and their independently calculated medians need not sum.

| Full request measurement | Profile off | Profile on |
| --- | ---: | ---: |
| Buffered HTTP ms | 10,366.407 | 10,511.729 |
| Reported prompt interval ms | 498.848 | 572.593 |
| Reported decode interval ms | 9,824.345 | 9,896.727 |

Profiling perturbs phase timing as well as total throughput. Do not subtract
its overhead or present these event spans as the unprofiled request's exact
stage breakdown. The footprint, output and counter checks establish that
the same work ran, not that instrumentation was free.

## Projection Families And Source Evidence

Family labels below follow the checked standard forward order Q/K/V,
output, gate/up, down within each layer. They are derived from span order,
not independent cuBLAS kernel-symbol observations.

| Family, all layers | Conversion ms | SGEMM ms |
| --- | ---: | ---: |
| Q | 4.750 | 5.525 |
| K | 0.833 | 2.280 |
| V | 0.808 | 2.215 |
| Attention output | 4.756 | 5.652 |
| FFN gate | 25.701 | 26.799 |
| FFN up | 25.646 | 26.779 |
| FFN down | 26.297 | 22.359 |

Current `dequantize_colmajor` transposes through a 32x33 shared tile into
N-leading FP32 weights. The wrapper calls `cublasSgemm(N,N)`. The pinned
llama.cpp F32 path at `ggml-cuda.cu:1615` instead uses source-layout weights
and `cublasSgemm(T,N)`. This source difference remains real, but its scope
is too small for a standalone E2E promotion proposal at this checkpoint.
No alternative layout or backend is implemented.

Per-sample conversion-plus-SGEMM sums have median **180.376 ms**. Applying
`T/(T-180.376)-1` to the off-profile HTTP median gives the optimistic 1.771%
figure. This combines a profiled scope with a separate ordinary median;
it is a feasibility estimate, not a causal bound, predicted speedup, or
permission to promote a microbenchmark. It cannot satisfy either original
throughput goal.

The bias scope contains **192 residual-add launches per layer**, 6912 for
the prefix, and has 41.271 ms median summed host call time. The source
loops over 64 rows and three projections. This is concrete work to record,
not a quantitative attribution of all remaining prefix time or evidence
that eliminating it alone could clear the E2E gate. Do not expand this into
unrelated scheduler work or tune a closed FFN candidate.

## Identity, Memory And Isolation

Clean source `c44ab95af680441848b755870cc967d6418766bc`, release comparison
executable SHA-256
`3efa3916d7309057f540cb7a1a60be381ed374a9f63fddb0f99b89d2755936ea`.
CUDA 12.8.93 build flags remain `-O3`, sm_52/compute_52, no fast math.
The frozen model is Qwen2.5-3B Q8_0, SHA-256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
Use the same 65 ChatML prompt IDs, greedy sampling, 512 natural generated
tokens, dense KV/context 1024, packed-warp-256 QKV/output, split-f32x4-128
FFN and dequant-SGEMM prefix as completed quality.

The environment is unchanged except for the report path, explicit
`M40LLM_STAGE_PROFILE=0` at preflight and the new requested-prefill flag.
The controller toggles stage profiling only between requests. Raw manifests
retain these differences; the audit requires their exact values.

All 109 prepared owners retain 2,917,580,800 bytes plus 3,279,519,744 source
bytes, totaling 6,197,100,544 bytes per model before KV/workspace. Dense KV
is 37,748,736 bytes. The reusable prefill temporary is 90,177,536 bytes,
released after the prefix. Every request retains 42 ordinary allocations
and frees, 252 SGEMMs, 237,013 kernel launches, 1097 synchronizations and
324 waits. Event-pool setup is about 0.5-0.6 ms and has no tracked device
payload allocation; this is not a peak-VRAM claim.

All twenty primary GPU boundaries show P0, 1113 MHz SM and 3004 MHz
memory. Only benchmark PID 2607976 and idle owned oracle runner 2608760
are present. No inference engines run concurrently, and no builds, tests
or MemPalace work overlap timing. Benchmark session 35688 exits zero after
actual-ID verification. Verified owned server 2607480/session 11738 is
stopped and the final GPU compute-process query is empty.

## Validation And Reproduction

Hardware tests pass the existing GPU profiler checks and prefill parity
for head64 and Qwen-shaped head128, including bit-identical logits/sampled
IDs, unchanged counters, explicit profiling-off behavior and event reuse.
The synthetic Qwen math fixture uses explicit valid IDs, not a fabricated
Qwen tokenizer; real-model token IDs are verified by this full run.
Attention parity, no-cuBLAS conversion/rejection tests, 211 warning-as-error
non-CUDA checks, 20 comparison-tool tests, CUDA/server all-target Clippy and
formatting pass. Negative audits reject missing layer coverage, added
synchronization, incorrect prefix length and changed actual output IDs.

Source `scripts/dev-env.sh`, export the frozen capture environment, build
release with `cuda,server`, and start the verified
[isolated official oracle](../controlled_benchmark.md#ollama-setup). Reuse
the selected configuration/evidence arguments from the
[packed-runtime reproduction](2026-10-25-maxwell-packed-projection-ollama-http.md#fixed-setup-and-reproduction),
but replace `--time-ollama` with `--profile-q8-runtime`. Set:

```bash
export M40LLM_STAGE_PROFILE=0
export M40LLM_STAGE_PROFILE_PREFILL=1
export M40LLM_STAGE_PROFILE_REPORT=/tmp/new-prefill-events.jsonl
```

Use a new HTTP output path and the actual owned oracle PID/log. Both
report paths must be new. The oracle is untimed here, and no model download
or Hugging Face cache lookup is needed.

- [Raw HTTP requests and actual IDs](2026-10-26-maxwell-prefill-attribution-http.jsonl)
- [Raw prefix/decode events](2026-10-26-maxwell-prefill-attribution-events.jsonl)
- [Independent audit](2026-10-26-maxwell-prefill-attribution-audit.jq) and [derived results](2026-10-26-maxwell-prefill-attribution-audit.json)
- [Run log](2026-10-26-maxwell-prefill-attribution-run.log.gz), [oracle log](2026-10-26-maxwell-prefill-attribution-ollama.log.gz), [validation log](2026-10-26-maxwell-prefill-attribution-validation.log.gz), [checksums](2026-10-26-maxwell-prefill-attribution.sha256)

The next optimization needs a larger demonstrated exact-workload scope.
Do not implement the source-layout variant solely because it resembles
Ollama, revive rejected FFN mappings, or relax any original gate.
