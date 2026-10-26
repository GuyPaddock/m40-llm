# Q8_1 Quality With Completed Sequential Prefill

## Decision And Scope

Both existing Q8_1 geometries remain **numerical research only**. They
retain all seven dense-valid task passes but change the corrected control's
actual browser continuation. Held-out perplexity worsens **2.67% / 2.94%**
for 128/256 threads. The small frozen sets do not establish general quality.
No kernel, selector, tolerance, prompt, rubric or default changes.

This completes the evidence repair identified by the
[plan audit](2026-10-26-maxwell-plan-evidence-audit.md). The historical
[M08 report](2026-10-25-maxwell-runtime-quality.md) predates sequential
prefix completion. Its step-42 divergence is not the repaired runtime's
result. Old JSONL remains unchanged and cannot qualify current execution.

Clean source `33839478c35e20a59bf0a8f7930db66eb8109dda`; release executable
SHA-256 `bf581bbabe5eb0f21d30f9387c0d3bf398e65381c2ea6bae336852310cdf6b32`.
NVCC 12.8.93 uses `-O3`, sm_52 SASS/compute_52 PTX, no fast math. Both
models retain original Qwen2.5-3B Q8_0 bytes, all 36 layers, head128, dense
FP16 KV/context 1024, canonical 65 ChatML IDs, GPU greedy and natural stops.
Explicit corrected `native-sequential` is the M08 control; candidates change
only the existing Q8_1 FFN geometry. QKV/output stay native and prepared
down/vocabulary stay fixed. This is not an incremental comparison against
the latest packed-QKV/split-FFN/SGEMM configuration.

Both reports carry `prefix-decode-completed-v1`. Their actual reference
IDs/output match the corrected control in completed packed128 quality.
The frozen corpus SHA remains
`2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38`.

## Quality Results

| Candidate | Actual browser tokens | First divergence, zero-based | Dense-valid regressions | Changed task outputs | Max sampled browser logit diff | Min top-10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8_1/128 | 512 | 60 | 0 / 7 | 2 / 12 | 0.297407 | 10 |
| Q8_1/256 | 512 | 10 | 0 / 7 | 2 / 12 | 0.290829 | 9 |

Controls also naturally generate 512 tokens with length stops. Five
reference-failed task rows are inconclusive, not numerical-kernel failures.
Both changed task outputs belong to that group; task scores do not change.

At actual divergence, the expected token drops from rank 1 to rank 2 in
candidate full logits. Top-10 membership does not preserve the chosen token:

| Geometry / step | Reference token ID | Candidate top ID | Reference expected logit | Candidate expected logit |
| --- | ---: | ---: | ---: | ---: |
| 128 / 60 | 3070 | 2657 | 24.297022 | 24.186565 |
| 256 / 10 | 315 | 504 | 25.919527 | 25.853235 |

Full-vocabulary probes replay identical absolute reference prefixes at
0/1/8/32/128/256/511 plus actual divergence. Maximum sampled mean differences
are 0.048885 / 0.051495. These are not every-token parity, comparisons of
already-diverged prefixes or accuracy improvements over different historical
probe positions.

| Split | Scored tokens | Reference PPL | Q8_1/128 PPL | Q8_1/256 PPL |
| --- | ---: | ---: | ---: | ---: |
| Calibration | 102 | 31.556846 | 36.478550 | 36.609940 |
| Held-out | 101 | 39.612157 | 40.670991 | 40.776474 |

Loss is token-weighted over all scored positions, not averaged passage PPL.
These losses match historical teacher forcing, which already completed
logits per token. Prefix ownership changes the free-generation reference,
not these passage results. M04's explicit activation reconstruction and
quantized-input CPU references remain separate from FP32-activation drift.

## Dispatch, Memory And Validation

Each candidate browser request executes **20,736 quantizers** and
**20,736 requested Q8_1 kernels**, including sequential prompt forwards.
Both sides retain **21,248 prepared down/vocabulary launches**. Other
operation counts and enqueue counters match. Quantization adds 20,736
launches, not request allocations or stream syncs. Each request has 1155
syncs, including 64 completed no-logits prefix boundaries.

Both models retain identical **37 prepared identities** and
**1,192,935,424 prepared bytes** plus **3,279,519,744 source bytes** per
model, totaling **4,472,455,168 bytes** before KV/workspace. Each dense KV
is **37,748,736 bytes**. Initialization logs report **2304 reusable Q8_1
activation bytes**. No new weight cache, SGEMM temporary, compression,
staging or full FP32 materialization is involved. Combined tracked fields
describe both resident models, not twice that total.

Q8_1 hardware parity/reuse and invalid-argument tests pass **2/2** with
`--include-ignored`. The initial invocation without it ran no tests; its
two ignored cases are not passes. Runner CPU tests pass **24**, with
**9 hardware tests ignored**, not called passed. No inference/CUDA code
changed; earlier broader on/off, non-CUDA and Clippy evidence is not
presented as rerun here.

Both audits independently recompute divergence, corrected reference IDs,
dispatch, task status, prefix/target coverage, NLL sums, token-weighted
loss and PPL. Twenty-four corruptions are rejected, twelve per geometry.
Sessions **66539 / 97758** terminate zero, with tracked bytes zero to zero.
GPU compute is empty before each and after both. Mid-run observations show
only the corresponding PID **3286464 / 3290717**, P0 and 1113/3004 MHz.
These are samples, not continuous counters. No other inference, build,
test or MemPalace call overlaps either evaluation.

## Reproduction And Evidence

Source `scripts/dev-env.sh`, restore frozen M02 environment from the
[task-pair reproduction](2026-10-26-maxwell-ffn-task-pair-screen.md#reproduction-and-artifacts),
request hardware access and verify idle GPU state. Build once and run serially
with new output paths. `MODEL` is the unchanged source blob with SHA-256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`:

```bash
cargo build --release --features cuda,server --example model_quality
for threads in 128 256; do
  target/release/examples/model_quality \
    --reference-model "$MODEL" --model "$MODEL" \
    --variant-id "q81-corrected-$threads" \
    --q8-runtime-candidate "q8-1-$threads" \
    --q8-runtime-control native-sequential \
    --q8-runtime-prefill sequential --split all \
    --output "/tmp/new-q81-corrected-$threads.jsonl" || break
done
```

- [128 raw report](2026-10-26-maxwell-q81-corrected-128.jsonl), [audit](2026-10-26-maxwell-q81-corrected-128-audit.json), [log](2026-10-26-maxwell-q81-corrected-128.log.gz).
- [256 raw report](2026-10-26-maxwell-q81-corrected-256.jsonl), [audit](2026-10-26-maxwell-q81-corrected-256-audit.json), [log](2026-10-26-maxwell-q81-corrected-256.log.gz).
- [Audit program](2026-10-26-maxwell-q81-corrected-audit.jq), [corruption controls](2026-10-26-maxwell-q81-corrected-negative-controls.sh), [control log](2026-10-26-maxwell-q81-corrected-negative-controls.log.gz), [checksums](2026-10-26-maxwell-q81-corrected.sha256).
- [Build log](2026-10-26-maxwell-q81-corrected-build.log.gz), [explicit hardware tests](2026-10-26-maxwell-q81-corrected-primitive-included.log.gz), [initial skipped invocation](2026-10-26-maxwell-q81-corrected-primitive.log.gz), [runner tests](2026-10-26-maxwell-q81-corrected-runner-tests.log.gz).

Resident durations are single quality observations, not paired HTTP speedups.
Qualified exact performance stays **49.146 HTTP tok/s** versus **43.106
official Ollama**, below both **53.06** and **56.037 HTTP tok/s** targets.
Next work remains exact execution optimization from independent cost evidence,
not Q8_1 promotion, closed geometry tuning or model modification.
