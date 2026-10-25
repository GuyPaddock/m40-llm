# Modified-Model Cost Comparison

## Decision

C26's individual comparison covers completed C20/C23/C24/C25 evidence without another fit or sweep.
No modified model is promoted. Ranked layer removal gives the smaller observed
quality cost, and compact export reclaims real storage. Activation sparsity
gives larger matched-step gains but costs more loss and device memory.
Neither establishes the original unchanged-Qwen throughput objective.

The small corpus is a research checkpoint, not enough evidence to recommend
a modified model for general use. There are only six tasks and 101 scored
loss tokens in held out. Passing the four dense-valid tasks is not proof of
unchanged quality; perplexity and format regressions make that distinction
concrete.

## Comparable Inputs, Different Timing Boundaries

All quality rows use the same frozen C19 corpus, canonical Qwen ChatML,
greedy top1, explicit dense KV/context 1024 and unchanged rubrics. Calibration
and held-out prompts remain disjoint; the browser benchmark never fits a
threshold or mask. Qwen2.5-3B-Instruct Q8_0 source SHA-256 is
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`;
corpus SHA-256 is
`2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38`.

The [derived comparison records](2026-10-25-modified-model-tradeoffs.jsonl)
bind each of 13 source reports by file SHA-256, clean revision and environment.
They retain both splits, task scores/output, timing/counts/stops, loss, storage
and setup. Each row is derived from a completed source report, not a new trial.
The compact row explicitly uses a masked reference; its zero PPL difference
is equivalence to that reference, not to the original full model.

The paired speed column below measures identical observed-prefix predictions:
five alternating release pairs, 43 held-out timed steps after eight warm
tokens, including remaining forward/logits, greedy readback and final stream
completion. It excludes reset, scratch setup and preparation. These are
matched-step rate changes, not free-generated or HTTP tokens per second.
Trials from different revisions are not pooled into new historical pairs.

## Held-Out Quality and Matched-Step Rate

| Variant | Task Passes | New Dense-Valid Failures | PPL | PPL Change vs Full Model | Paired Rate Change |
| --- | --- | ---: | ---: | ---: | ---: |
| Full 36-layer native Q8 reference | 4/6 | 0 | 39.612159 | reference | reference |
| C20 fixed one-layer mask `[17]` | 4/6 | 0 | 47.656541 | +20.31% | +2.564% |
| C23 activation target 25% | 4/6 | 0 | 40.025773 | +1.04% | -0.875% |
| C23 activation target 40% | 3/6 | 1 | 40.582514 | +2.45% | +2.197% |
| C23 activation target 50% | 4/6 | 0 | 42.920954 | +8.35% | +4.325% |
| C23 activation target 60% | 4/6 | 0 | 46.495763 | +17.38% | +6.684% |
| C24 ranked mask `[8]` | 5/6 | 0 | 40.984621 | +3.46% | +2.484% |
| C24 ranked mask `[8,24]` | 3/6 | 1 | 46.184633 | +16.59% | +5.116% |
| C24 ranked mask `[8,24,25]` | 4/6 | 1 | 49.373639 | +24.64% | +7.937% |
| C25 compact `[8]` export | 5/6 | 0 | 40.984621 | +3.46%, inherited | +0.060% vs masked source only |

The two dense-failed held-out tasks are inconclusive for compression/model
quality conclusions, including any extra candidate pass. This is why 5/6 is
not evidence that removal improves quality. Calibration remains in the raw
records: C24's three-removal mask also regresses a dense-valid calibration
task, unlike the first two masks.

C23 target 40% repeats workshop JSON and hits the length cap. C24's two-layer
mask returns workshop facts inside an unclosed JSON fence; the three-layer
mask regresses a different valid format. The scorer is not relaxed to retain
these variants. C20's larger fixed masks are included in the raw comparison
but are worse loss/quality candidates than the ranked alternatives.

Ranked `[8]` has a substantially smaller same-corpus PPL cost than fixed `[17]`.
Its approximately 2.5% rate change is below the exact-engine 3% screen and is
not an E2E promotion result. C25 adds neutral compaction timing, not another
layer of compute savings. Do not add +0.060% to +2.484%.

## Ordinary Generation Observations

These examples retain complete resident generation timing for the same
110-token held-out workshop prompt. Each is one observation within its source
run, not a matched HTTP speed trial. All listed rows stop naturally.

| Variant | Pass | Generated Tokens | Prefill ms | Decode ms | Generation Total ms |
| --- | --- | ---: | ---: | ---: | ---: |
| C24 full reference control | yes | 39 | 2752.938 | 1056.874 | 3809.825 |
| C23 activation 50% | yes | 39 | 2623.061 | 1007.130 | 3630.205 |
| C23 activation 60% | yes | 39 | 2558.472 | 984.249 | 3542.735 |
| C24 ranked `[8]` | yes | 35 | 2686.195 | 923.467 | 3609.674 |
| C24 ranked `[8,24]` | no | 38 | 2614.336 | 980.780 | 3595.129 |
| C25 compact `[8]` | yes | 35 | 2678.916 | 923.049 | 3601.981 |

Different token counts and styles make this unequal work. Lower total wall
time alone is not an engine speedup. The separate caller admission-to-return
boundary and full output are retained per task in the comparison records.

## Requested Storage and Preparation

| Approach | Original/Compact Upload Bytes | Native Dense KV Bytes at 1024 | Extra Prepared Weights/Scratch |
| --- | ---: | ---: | ---: |
| Native full model | 3,279,519,744 | 37,748,736 | 0 |
| C20/C24 masked source, any tested mask | 3,279,519,744 | 37,748,736 | 0 |
| C23 positive activation target | 3,279,519,744 | 37,748,736 | 862,411,332 |
| C25 compact 35-layer `[8]` | 3,197,605,888 | 36,700,160 | 0 |

The activation extra is 862,322,688 bytes of duplicate down weights, 88,068
bytes of shared selection workspace and 576 counter bytes. It is about
822.46 MiB, with no source-weight reclamation. All rows also use the same
66,560-byte head128 attention workspace after execution, plus ordinary forward
and logits scratch not itemized in this table. C25 independently queries the
successful owned upload size; historical rows retain their recorded payload
requests. Native KV is allocated separately from the Rust device ledger.
These component sizes are not standalone peak VRAM.

| Preparation | Recorded Cost | Boundary |
| --- | --- | --- |
| C23 calibration capture | 110 rows/layer; 174,366,720 B observed, 304,349,184 B reserved | Generated calibration inputs only; no held-out fitting |
| C23 threshold fit | 5701.533 ms | Fit stage, not complete capture or process time |
| C23 positive application | 10280.775-10538.725 ms | Identity checks, packing and upload; one setup per variant |
| C24 ranking | 99 trials; 279375.359 ms loss, 307.122 ms setup | Diagnostic loss includes full-logit D2H/CPU scoring; excludes other preparation |
| C24 frozen-mask application | No isolated complete preparation timer | Do not call artifact validation or model loading free |
| C25 CPU export | 38683.148 ms | Two source hashes, output hash, streaming copy, fsync and publication |
| C25 live consistency preflight | 26295.971 ms | Separate from source/compact load; not a fresh cold-I/O measurement |

Masking avoids compute but keeps all 36 physical KV slots and source weights.
Only the measured compact artifact saves 81,913,856 weight bytes and one
1,048,576-byte layer of dense KV at this context. Saving these allocations
does not remove the inherited quality loss or establish a general load-time
advantage. One-shot preparation observations do not establish amortization
for a deployment workload.

## Negative Gates and Combinations

C21's fixed 16-of-32 sparse Q8 stores 24 rather than idealized 22 bytes/block,
saving 29.41% against 34-byte native blocks. It nevertheless takes
2.29/2.31/2.66x gate/up/down latency at M=1. Multi-row partial wins do not
justify decode pruning. C22 saliency calibration and C27 sparse Q4 remain
postponed. No integrated sparse-model quality or E2E result exists.

No combined model is tested or inferred in these original 13 reports. Static-plus-activation sparsity
lacks an independently useful static decode kernel. Layer-plus-activation
removal has independently measured costs. The initial comparison required a
controlled generated-E2E win or approved quality budget before further research;
that was an extra restriction, not C26's authorized experiment gate. The next
bounded checkpoint therefore composes frozen mask `[8]` with existing targets
0/50/60% on both unchanged splits, with no fitting or promotion. It needs
original-ID routing, a zero control and measured held-out evaluation, not an
assumption that individual gains add.
The existing activation artifact is bound to the original source/hash and
must not be applied blindly to a renumbered compact model.

The C11A n-gram gate is independently negative: best optimistic browser gain
is 4.949%, below its 10% feasibility floor. C12-C17 remain postponed, not
implemented. None of these outcomes is erased to keep a research branch busy.

## Exact Objective and Handoff

The latest qualified unchanged-model HTTP checkpoint is 31.937130 tok/s
m40-llm versus 43.162711 Ollama 0.40.11, not any rate from the modified-model
table. Both original thresholds remain unmet: 53.06 tok/s and 1.30 times the
fresh Ollama reference, approximately 56.1115 tok/s. That matched five-pair
run uses the same Qwen Q8_0 model, 65 prompt tokens, 512 generated tokens,
verified fresh KV and length stops.

This records individual costs, not completion of C26's combined experiment or
the original speed objective. The bounded kernel, layout and feasibility
screens are not reopened by composition. General runtime defaults
remain preferred compressed exact-old top8; the short exact-engine benchmark
continues to request dense mode explicitly.

## Evidence

- [C19 dense reference and corpus](2026-10-25-model-quality-dense-reference.md).
- [C20 fixed-mask study](2026-10-25-layer-mask-sensitivity.md).
- [C21 sparse kernel and negative gate](2026-10-25-sparse-q8-screen.md).
- [C23 activation calibration/quality](2026-10-25-activation-quality.md).
- [C24 frozen ranking/quality](2026-10-25-ranked-layer-tradeoffs.md).
- [C25 export preparation](2026-10-25-compact-layer8-export-checkpoint.md) and
  [live equivalence/storage](2026-10-25-compact-layer8-live.md).
- [Latest qualified exact HTTP checkpoint](2026-10-25-upload-ordering.md).
- [N-gram feasibility decision](2026-10-25-ngram-feasibility.md).

This documentation unit adds no inference, timing trial, threshold, policy or
default. Source reports retain their numerical/CI validation. The derived
record identities/completeness, local links, formatting and whitespace are
checked; no extra GPU test is needed for this comparison.
