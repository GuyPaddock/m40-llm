# Prepared Activation Vector Loads Miss The Gate

## Decision

The fixed activation-load probe preserves control bits but **regresses
one-row complete FFN throughput 0.665-0.875%** on layers 0/17/35. All three
paired intervals are negative. Vocabulary regresses **2.608% at one row**
and **2.651% at four rows**, also with negative intervals. No case passes
its required micro gate. Keep implementation `5579ea2` benchmark-only.

Do not add a runtime selector, apply a geometry/register-cap rescue, combine
the earlier below-gate warp finish, or proceed to actual-model/HTTP promotion
for this candidate. Five single-row pairs are decisively below the 5% gate,
not a borderline positive finding needing confirmation. Four-row results
remain separate and do not qualify single-token decode.

This closes the [prespecified activation-load experiment](2026-10-26-maxwell-prepared-activation-hypothesis.md).
The qualified **49.146 HTTP tok/s** runtime and its unchanged outputs remain
the selected exact path. Both original requirements remain unmet: **53.06**
and **1.30x fresh official Ollama**, most recently 56.037 HTTP tok/s.
Runtime defaults do not change.

## Fixed Setup

Clean source `5579ea252839439c347fe6641ba9559ab6fef537`; release benchmark
SHA-256 `b49529e1c68382c0728d11d991f2a3764881a7ddf11c5cc4c8571c990477bcaa`.
Same original Qwen2.5-3B Q8_0 weights, frozen M02 fixture/capture report,
real down tensors on layers 0/17/35 and real vocabulary weights. No KV
cache is allocated. Vocabulary inputs are distinct frozen normalized FFN
layer-0 surrogates, not actual captured output-norm hidden vectors.

Only activation reads change: eight scalar bounds-checked values become
two aligned `float4` reads for complete Q8 blocks. Original packed weight
loads, signed conversion, scale/FMA association, shared reduction tree,
four-column ownership and launches remain. Down uses 256 threads at one
row and 128 at four; vocabulary uses 128. Gate/up stays split-f32x4-128 in
both complete-FFN arms. There is no extra layout, scratch or quantization.

Source `scripts/dev-env.sh`, export the frozen environment from the
[quality reproduction](2026-10-25-maxwell-packed-projection-runtime-quality.md#configuration-and-reproduction),
build the release `q8_execution_bench` with CUDA/cuBLAS enabled, then run:

```text
--model <same Qwen Q8_0 GGUF>
--ffn-inputs target/research-fixtures/m02-ffn-inputs.json
--ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl
--prepared-f32-packed-activation --rows 1,4 --trials 5 --iterations 10
--output <new JSONL path>
```

Each of eight cases has five alternating contemporaneous pairs after
warmup. Events include required dependencies. Gate/up, isolated down and
complete FFN are timed separately; vocabulary has its own complete
projection interval. Inputs/setup/CPU references/plane readbacks remain
outside timing. Four rows are distinct matrix rows, not repeated M=1 calls.

## Measured Results

| Layer | Rows | Complete FFN control/candidate ms | Complete gain | Isolated down control/candidate ms | Down gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.351568 / 0.354227 | -0.751% | 0.131811 / 0.133222 | -1.059% |
| 17 | 1 | 0.352067 / 0.354426 | -0.665% | 0.131888 / 0.133296 | -1.056% |
| 35 | 1 | 0.352022 / 0.355130 | -0.875% | 0.131766 / 0.133840 | -1.549% |
| 0 | 4 | 1.541277 / 1.539491 | +0.116% | 0.573754 / 0.558573 | +2.718% |
| 17 | 4 | 1.567280 / 1.528230 | +2.555% | 0.575978 / 0.565917 | +1.778% |
| 35 | 4 | 1.592989 / 1.596646 | -0.229% | 0.589018 / 0.587222 | +0.306% |

One-row complete-FFN paired 95% gain intervals are respectively
[-0.956%, -0.613%], [-1.027%, -0.610%], [-1.377%, -0.797%].
Unchanged one-row gate/up medians stay approximately 0.2211-0.2214 ms
in both arms. Four-row complete intervals are wider, with point estimates
below 5%; they do not establish a reproducible qualifying benefit.

| Vocabulary rows | Control / candidate ms | Throughput gain | Paired 95% gain interval |
| --- | ---: | ---: | ---: |
| 1 | 1.563338 / 1.605203 | -2.608% | [-2.690%, -2.566%] |
| 4 | 6.212829 / 6.382000 | -2.651% | [-2.671%, -2.633%] |

Bootstrap uses 10,000 paired resamples, seed zero. The audit reconstructs
medians and gain from individual arm timings rather than trusting summary
labels. Because all candidates fail, there is no combined feasibility
estimate or inferred runtime/E2E claim.

## Numerical, Compiled And Ownership Evidence

All eight cases preserve control bits, including complete/isolated down
and hidden intermediates. CPU references, guards and immutable inputs/planes
pass. The new primitive separately tests small/real widths, column tails,
both streams and dependent async producers/consumers. Invalid incomplete-K,
layout, null/aliased/misaligned requests fail without fallback effects.

The current control SASS is unchanged, SHA-256
`333013085bd1e550afc35c6bf329183b675be4679e71bea69cdf994ea6673fa1`.
Candidate SHA-256 is
`634fb6a64a5326043f85f0ed5c11767227e850d7e3cbe534a4eaa0774d77fc57`.
It emits two `LDG.E.CI.128` activation loads instead of eight scalar sites,
retains four `LDG.E.CI.64` weight sites and the original shared-tree barrier
loop. Registers rise **40 to 46**, with zero reported stack/local storage.
That tradeoff is observed, not proof that occupancy alone causes the loss.
No measured occupancy/cache-hit/DRAM-bandwidth claim follows.

Both arms retain 37 down/vocabulary layouts totaling **1,192,935,424 bytes**
plus **3,279,519,744 source bytes**. Down cases share one additional
47,906,816-byte gate/up pair, not a full production 109-tensor cache.
Guarded down scratch is 201,152 / 803,264 bytes for rows 1/4; vocabulary
scratch is 616,064 / 2,463,872 bytes. Candidate adds no storage. Timed
enqueue records zero allocations/frees, H2D/D2H, cuBLAS or synchronization.
Matrix-row dependencies still use the existing explicit stream waits.

All eighty primary GPU boundaries show P0, 1113 MHz SM / 3004 MHz memory
and only benchmark PID 2409104. All 200 timed intervals are serial. No
build, test, other inference or MemPalace work overlaps timing. Session
42678 exits zero; tracked bytes return zero to zero, and the GPU is empty
after process exit.

Validation passes new primitive tests in both cuBLAS modes, the original
warp probe with cuBLAS off, attention/projection/MLP guardrails in both
modes, forward smoke with cuBLAS off, eleven benchmark CPU tests and the
explicit full-FFN adapter test. Nine other hardware benchmark tests remain
ignored, not reported as run. The warning-as-error non-CUDA suite passes
211 tests; CUDA/server Clippy and formatting pass.

## Evidence And Next Work

- [Raw pairs, validations and complete outputs](2026-10-26-maxwell-prepared-activation-screen.jsonl)
- [Audit program](2026-10-26-maxwell-prepared-activation-audit.jq) and [derived results](2026-10-26-maxwell-prepared-activation-audit.json)
- [Control SASS](2026-10-26-maxwell-prepared-activation-control.sass.gz), [candidate SASS](2026-10-26-maxwell-prepared-activation-candidate.sass.gz), [resources](2026-10-26-maxwell-prepared-activation-resources.txt.gz)
- [Run log](2026-10-26-maxwell-prepared-activation-screen.log.gz), [validation log](2026-10-26-maxwell-prepared-activation-validation.log.gz) and [checksums](2026-10-26-maxwell-prepared-activation-screen.sha256)

Negative artifact checks reject wrong dispatch, altered measurements,
changed numerical flags or memory capacity. Preserve this failed experiment
without further geometry/layout/combination tuning. Next inspect the
remaining fused gate/up work distribution and the pinned backend's warp
output ownership before proposing a genuinely independent exact-FP32
experiment. Do not replace either original goal with sparsity, activation
quantization, approximate attention or model changes.
