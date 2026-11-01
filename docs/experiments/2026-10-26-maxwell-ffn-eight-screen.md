# Eight-Value FP32 FFN: Improvement Below Gate

## Decision

Keep `45d0214` benchmark-only. Wider payload tasks improve complete one-row
FFN throughput **2.00-2.03%**, with positive paired intervals, but miss the
unchanged **5% complete-FFN gate** at every layer. All three one-row upper
confidence bounds are below 2.35%, so this is not a borderline result.
Four-row point gains are 1.71-1.98% and do not replace the one-row gate.
Do not vary task width, threads, unroll or register limits to rescue it.
Do not combine it with rejected mappings or layouts.

This [prespecified FP32 experiment](2026-10-26-maxwell-ffn-eight-hypothesis.md)
changes summation ownership without quantizing activations or changing
weights. Primitive CPU bounds pass, but outputs differ from the four-value
control. No actual-512/frozen-model quality or HTTP run is authorized by
this micro result. It is not a new exact-output inference configuration.

Qualified runtime stays **49.146 HTTP tok/s** versus **43.106 official
Ollama**, not remeasured here. Both original goals remain unmet: **53.06**
and **1.30x fresh official Ollama**, currently **56.037 HTTP tok/s**.
Runtime defaults, KV hierarchy and prepared down/vocabulary remain unchanged.

## Complete FFN Results

Each shape has five alternating contemporaneous pairs, ten iterations per
gate/up, isolated-down and complete stage. Milliseconds are medians; gain
is control/candidate latency minus one. Intervals use 10,000 paired bootstrap
resamples with seed zero. Timing control is qualified split-f32x4-128, not
an older raw-GGUF or failed candidate path.

| Layer | Rows | Four-value control ms | Eight-value ms | Gain | Paired 95% gain interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.351747 | 0.344765 | 2.03% | 1.99% to 2.19% |
| 17 | 1 | 0.351856 | 0.344931 | 2.01% | 1.70% to 2.34% |
| 35 | 1 | 0.352093 | 0.345181 | 2.00% | 1.79% to 2.33% |
| 0 | 4 | 1.592330 | 1.565587 | 1.71% | -1.51% to 3.76% |
| 17 | 4 | 1.533430 | 1.504822 | 1.90% | 0.69% to 7.81% |
| 35 | 4 | 1.534848 | 1.505034 | 1.98% | 1.56% to 2.71% |

One-row gate/up/SwiGLU falls from approximately 0.2218-0.2221 ms to
0.2137-0.2142 ms. Isolated prepared down remains approximately 0.131-0.132 ms.
Stage sums need not equal complete timing because cache histories differ.
Four-row down variation is not caused by a changed down kernel. None of
these measurements establishes a generated-token or HTTP improvement.

## Numerical Contract

The candidate uses the same FP32 activations, signed Q8 payloads, FP16
scale bits, per-weight FP32 scaling/FMA and precise SwiGLU expression.
Eight consecutive values per task replace four, changing which terms share
a per-thread partial. The original 128-partial reduction tree remains, but
bit equality against four-value partials is not promised. This distinction
was committed before code; existing exact-control variants' tests stay strict.

Capture and ordinary candidate SwiGLU bits match. Independent FP64 references
and four-value-control comparisons retain the existing
`1e-3 + 2e-5 * abs(reference)` bound, including all finite-value checks.
No numerical tolerance was widened after seeing results.

| Output | Maximum absolute drift vs four-value control | Maximum absolute error vs CPU |
| --- | ---: | ---: |
| Gate | 1.91e-6 | 1.45e-6 |
| Up | 3.81e-6 | 3.41e-6 |
| SwiGLU | 3.81e-5 | 3.24e-5 |
| Complete down | 6.10e-5 | 6.07e-5 |

These are layer-local checks on frozen real calibration vectors, not final
model logits or retrieval quality. Unchanged precision alone does not prove
unchanged generated IDs. The original actual-512 requirement remains intact.

## Compiled Loads And Memory

Clean source `45d0214ff5b22b924a5508d327b940f20619382f`; release executable
SHA-256 `b770a4e73610b0cb2a29cc9b471daa499c45b7ab69a4031fc9c8b5086b05821f`.
NVCC 12.8.93 uses `-O3`, sm_52 SASS plus compute_52 PTX, no fast math.
cuobjdump 12.8.90 inspects this same executable before timing.

Both kernels use 32 registers, zero reported stack/local storage, 128 threads,
one output per CTA, 1024 dynamic shared bytes and one publication barrier.
The candidate emits aligned `LDG.E.CI.64` payload loads, two `LDG.E.CI.128`
activation loads per task and ordinary FP16 scale loads/conversions. Original
control encoded words remain byte-identical, SHA-256
`31cd38f6ca0d4a0bebd03206f484bf8045834d8af592a645cdb168cfaf02e321`.
Prepared down/head words also match, SHA-256
`fafef036cf36582ad7c230dd9b2502575a53356ada1396aaeac8e9baf0872397`.

The source-level task/load reductions and sector-request tradeoff are in the
hypothesis. Compiler loop transformations make whole-function instruction
site counts unsuitable as dynamic traffic counts. No measured DRAM bandwidth,
cache hit rate, stall fraction or achieved occupancy is inferred from this
screen. Legacy CUPTI event groups remain unsupported on the installed stack.

Both arms retain the same original source, qualified 37 down/vocabulary keys
and one shared gate/up pair. The microbenchmark model uses 3,279,519,744 source
bytes plus 1,192,935,424 prepared down/vocabulary bytes. The per-case gate/up
pair adds 47,906,816 bytes to both arms, already present in the fully qualified
runtime's preparation. The candidate introduces no layout, extra weight plane
or quantized activation storage. Guarded scratch stays 201,152 bytes at rows1
and 803,264 at rows4. Tracked bytes return zero to zero after cleanup.

## Validation And Isolation

Primitive tests pass 2/2 with cuBLAS on/off, including every signed byte
position, zero scales, tiny/odd output widths, ragged legal K, ordered input
producers, capture/ordinary bits, immutable inputs/planes, guards and early
alignment/alias rejection. Eight-value payload pointers require eight-byte
alignment; older APIs retain their previous contracts. The complete-FFN
adapter test passes explicitly on/off with actual dispatch and unchanged
scratch/cache. The benchmark CPU suite passes 18 tests; 14 hardware tests
are ignored in that invocation, not counted as passes.

Affected enabled regressions pass attention 11, forward 20, projection 2 and
MLP 1, with four optional forward tests ignored. Warning-as-error non-CUDA
tests pass 211. CUDA/server all-target Clippy, formatting and diff checks pass.

The release screen exits zero, session 99466 / PID 2944575. Thirty pairs have
180 non-overlapping primary intervals. All sixty GPU pair-boundary samples
show only the benchmark, P0 and 1113 MHz SM / 3004 MHz memory. No other agent
GPU work, builds, tests or MemPalace operations overlap timing. The GPU query
is empty after process exit. Warmup reaches 1000 ms control / 100 ms candidate
with a finite 32-iteration batch budget; it is excluded from primary pairs.

Complete FFN includes gate/up/SwiGLU and unchanged down. Ten repetitions
issue 20 launches complete, 10 per isolated stage; the actual eight-value
operation is checked separately. Timed enqueue allocates/frees no device
memory, copies no data, calls no cuBLAS and synchronizes no stream. Existing
four-row dependency waits remain included. Completion is one event wait
after each repeated interval, outside enqueue counters.

The independent audit recomputes raw medians, gains, stage gates and possible
bootstrap bounds, checks recorded numerical/control-drift fields, immutable
identities, scratch, geometry, actual operation counts, clocks, serial work
and cleanup. Eight negative controls reject missing work, timed allocation,
wrong payload width, changed cache identity, excessive drift, failed capture
parity, false medians and a competing GPU process.

## Reproduction And Artifacts

Source `scripts/dev-env.sh` and restore the exact frozen environment using
the [reduction-screen setup](2026-10-26-maxwell-decode-reductions-screen.md#reproduction-and-artifacts).
Request hardware access and check competing processes, then use a new output:

```bash
cargo build --release --features cuda --example q8_execution_bench
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-f32-eight-values --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-eight-value-ffn-screen.jsonl
```

- [Raw report](2026-10-26-maxwell-ffn-eight-screen.jsonl), [audit](2026-10-26-maxwell-ffn-eight-audit.jq) and [derived results](2026-10-26-maxwell-ffn-eight-audit.json)
- [Control SASS](2026-10-26-maxwell-ffn-eight-control.sass.gz), [candidate SASS](2026-10-26-maxwell-ffn-eight-candidate.sass.gz), [resources](2026-10-26-maxwell-ffn-eight-resources.txt.gz) and [control words](2026-10-26-maxwell-ffn-eight-control.words.gz)
- [Prepared control SASS](2026-10-26-maxwell-ffn-eight-prepared-control.sass.gz) and [words](2026-10-26-maxwell-ffn-eight-prepared-control.words.gz)
- [Run log](2026-10-26-maxwell-ffn-eight-screen.log.gz), [release build log](2026-10-26-maxwell-ffn-eight-build.log.gz) and [checksums](2026-10-26-maxwell-ffn-eight.sha256)

Next seek independently measured larger-scope costs rather than extending
the task-width sweep or treating request-footprint estimates as measured
memory traffic. The exact-output goals and acceptance thresholds stay fixed.
