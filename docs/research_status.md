# Research Status

This is the task checkpoint, not agent instructions. Update it after meaningful
progress; keep working rules in [AGENTS.md](../AGENTS.md). Verify this snapshot
against the current code, worktree, Git history, and linked raw results.

## Active Plan

The active sequence is the workspace's
[Maxwell Q8 execution plan](../../M40_LLM_MAXWELL_Q8_EXECUTION_PLAN.md), outside
this Git repository. It orders pinned Ollama dispatch verification, complete
FFN measurements, bounded FP32/Q8_1 experiments, independent prefill SGEMM,
and controlled runtime qualification. See [the current investigation](maxwell_q8_execution.md).
The [earlier research plan](../../M40_LLM_RESEARCH_PLAN.md) and
[legacy task roadmap](archive/legacy-task-roadmap.md) remain history, not the
active execution order. Their negative conditional gates remain recorded.

## Current Phase: Maxwell Q8 Execution

**Active bounded round:** [eight-value FP32 FFN tasks](experiments/2026-10-26-maxwell-ffn-eight-hypothesis.md)
tests wider payload tasks in the original one-output/128-thread CTA, reducing
source-level load/scale groups without changing weights or activation precision.
Summation ownership changes, so use unchanged CPU tolerances and explicit
control-drift reporting rather than claiming bit equality. Actual-512/frozen
quality and HTTP gates remain mandatory; no closed mapping is combined and
no task-width/geometry rescue is allowed. Defaults and prepared down/head stay
fixed. Next implement the benchmark-only entry and behavioral checks.

**Latest fixed screen:** [two physical FFN cohorts per CTA](experiments/2026-10-26-maxwell-ffn-cohorts-screen.md)
preserve control bits but regress complete one-row FFN throughput **0.49-0.52%**
with all intervals negative; four-row results also miss 5%. Registers rise
32 to 40 despite two accumulators per thread. Control and prepared projection
instruction words remain unchanged. Keep `ac505bf` benchmark-only, without
cohort/block/register rescue or runtime integration. Next inspect qualified
four-value task/load granularity for an independent hypothesis, not a
combination of rejected mappings. Both original targets remain unmet and
defaults unchanged.

**Latest fixed screen:** [exact weighted-norm and argmax reductions](experiments/2026-10-26-maxwell-decode-reductions-screen.md)
preserve control bits/indices, guards and model preparation. Complete argmax
is **15.43-16.28x faster** in ten fresh pairs, but contributes only **1.2533%
optimistic E2E scope**, below 3%. Norm point gains are **10.84-22.44%**;
four of six one-row intervals cross zero, so neither norm stage qualifies.
Initial five-pair and fresh ten-pair reports are not pooled. Keep primitives
`f8c00bf` / `125d3a3`, shared warmup `91fd4ea` and benchmark `61a48d9`
as reference without runtime wiring, geometry/fusion/register rescue or
default changes. All 540 primary intervals are serial and both runs clean
up allocations. Next seek independent evidence of enough removable cost in
the qualified runtime, rather than combining closed probes. Both exact
targets remain unmet; there is no new HTTP throughput claim.

**Latest fixed screen:** [split-plane QKV/output storage](experiments/2026-10-26-maxwell-split-projection-screen.md)
preserves qualified packed control bits, but QKV regresses **0.78-2.12%**
at one request and output gains only **2.02-2.49%**, below 5%, with intervals
spanning zero. Output's optimistic standalone scope is **0.102-0.125%**,
not an HTTP result. Keep `0b96c46` / `34a2427` benchmark-only without layout/
geometry rescue or runtime integration. `5719ffc` corrects the pre-existing
warmup budget; initial and fresh screens are separate, not pooled. Control
instruction words and 32/40 registers are unchanged. All gates and defaults
remain unchanged, both original goals unmet; the next candidate needs
independent evidence rather than recombining closed probes.

**Previous fixed screen:** [reusable FFN blocks](experiments/2026-10-26-maxwell-persistent-tiles-screen.md)
preserve control bits but regress one-row complete FFN **9.79-9.85%** and
four-row **5.61-7.58%**, with all intervals negative. The fixed 192-block
grid increases registers 32 to 48 and adds a block-local reuse barrier;
those observations are not measured occupancy attribution. Keep `2c8f753`
benchmark-only, without cap/thread/unroll/register rescue or runtime
integration. All six gates fail. Both original targets remain unmet and
defaults unchanged; next work needs independent measured/source evidence,
not recombining closed FFN candidates.

**Latest attribution:** [packed-prefix projection costs](experiments/2026-10-26-maxwell-prefill-attribution.md)
preserve actual 512 IDs/counters with **1.3825%** off/on instrumentation
overhead. Conversion is **88.787 ms**, SGEMM **91.566 ms**, and bias
**43.749 ms** across the prefix; spans omit other GPU/host work. Eliminating
conversion plus SGEMM gives only a **1.771% optimistic E2E estimate**, below
3%. The current N-leading/NN versus pinned source-layout/TN difference
does not justify implementation by itself. Keep `c44ab95` as opt-in
measurement infrastructure, not an optimized backend. The next candidate
needs a larger demonstrated scope without reopening closed probes. Both
original targets and defaults remain unchanged.

**Previous fixed screen:** [register-owned virtual FFN warps](experiments/2026-10-26-maxwell-virtual-warp-screen.md)
preserve the control tree and bits but regress one-row complete FFN
**5.36-5.40%**, with negative paired intervals; four-row results also lose.
Removing shared publication/barrier raises registers 32 to 56 with zero
reported stack/local bytes. Keep `21ca349` benchmark-only, without geometry
or register-cap rescue. The pinned generic Maxwell MMVQ mapping uses four
physical warps per output; its MoE warp mapping is not dense Qwen evidence.
The next step needs a new independent source/measurement-backed hypothesis,
not retuning a closed probe. Exact targets and defaults remain unchanged.

**Previous fixed screen:** [prepared activation vector loads](experiments/2026-10-26-maxwell-prepared-activation-screen.md)
preserve bits and the qualified shared tree, but regress one-row complete
FFN **0.665-0.875%** and vocabulary **2.608-2.651%**. Registers rise 40 to
46 without reported spills. Keep `5579ea2` benchmark-only; no runtime or
geometry/register-cap rescue. The subsequent gate/up work-ownership audit
and virtual-thread experiment are recorded above.
Both original targets and defaults remain unchanged.

**Latest selected-runtime profile:** [packed Q8 event/HTTP diagnostic](experiments/2026-10-26-maxwell-packed-stage-profile.md)
preserves actual IDs with 1.0864% perturbation. Middle-stage shares are
41.11% gate/up, 24.37% prepared down and 7.98% vocabulary. Source/current
SASS shows scalar activation loads remain in the prepared projection while
weights already use packed 64-bit reads. The [bounded activation-load-only
probe](experiments/2026-10-26-maxwell-prepared-activation-hypothesis.md)
is now implemented and measured above, retaining original reduction,
planes/math/geometry and all existing gates. No combining rejected
reduction/FFN candidates or new throughput claim follows.

**Requested KV-stack side diagnostic:** [five resident-generation rounds](experiments/2026-10-26-maxwell-packed-kv-stack.md)
measure 49.104 tok/s dense, 19.694 compressed defaults, 20.217
window128/top8 and 22.765 window128/top4 with the optimized Q8 path fixed.
All generate 512 tokens; defaults preserve dense IDs but reserve more KV
at the short 1024 context bound. Smaller windows save KV but change output
and remain slower. This is not HTTP/exact-goal qualification or a reason
to change defaults; the exact-track next action below remains unchanged.

**Latest selected-runtime gap checkpoint:** the
[fresh packed-runtime/Ollama comparison](experiments/2026-10-25-maxwell-packed-projection-ollama-http.md)
measures **49.146 versus 43.106 buffered HTTP tok/s, +14.014%**, paired
95% interval **[13.974%, 14.370%]**. Actual 512 IDs, intended packed
dispatch, fixed preparation and prompt-cache resets pass. Both original
goals remain unmet: **53.06** and **56.037 tok/s** for 1.30x this fresh
official median. Defaults remain unchanged. The profile above now completes
its follow-up; do not revive below-gate FFN candidates.

**Latest incremental qualification:** [packed QKV/output](experiments/2026-10-25-maxwell-packed-projection-incremental-http.md)
passes twenty fresh confirmation pairs at **49.082 versus 47.186 HTTP
tok/s, +4.018%**, paired interval **[3.802%, 4.144%]**. Actual 512 IDs,
quality, all 109 prepared identities and dispatch match. The five-pair
screen triggered the prescribed fresh confirmation; it is not pooled.
The fresh official comparison above completes its cross-engine gate;
both original goals and defaults remain unchanged.

**Earlier selected-runtime gap checkpoint:** the
[corrected fresh projection/Ollama comparison](experiments/2026-10-25-maxwell-projection-ollama-http.md)
measures **47.379 versus 43.097 buffered HTTP tok/s, +9.937%**, with a
positive paired interval, verified actual 512 IDs and intended warp dispatch.
Both original goals remain unmet: **53.06** and **56.026 tok/s** for 1.30x
this fresh official median. The initial wrong-dispatch primary-loader run
is rejected and preserved separately; shared typed loading and dispatch
guards are fixed. Defaults remain unchanged. The
[new selected-runtime profile](experiments/2026-10-25-maxwell-projection-stage-profile.md)
preserves actual IDs with 1.01% instrumentation overhead. Middle-stage
shares are 39.43% gate/up, 23.33% down and 3.80% attention. The subsequent
[prepared-projection warp screen](experiments/2026-10-25-maxwell-prepared-warp-screen.md)
preserves bits but gains only 2.44-2.60% complete FFN, below 5%. Vocabulary
passes its 6.54% projection screen but has only 0.474% optimistic E2E scope.
Keep `8d1c15d` / `5218f41` benchmark-only without integration or a geometry
rescue. Next inspect the qualified gate/up load/conversion dependency chain
before proposing an independent bounded exact hypothesis; do not revive
failed layouts or replace either original goal with numerical/model changes.
The subsequent [fixed unfused decomposition](experiments/2026-10-25-maxwell-unfused-ffn-screen.md)
preserves control bits but regresses complete FFN **14.14-14.27%** at one
row and 12.18-12.76% at four. Keep `286cb9a` benchmark-only with no geometry
rescue or runtime wiring. Source inspection confirms existing load overlap
and read-only PTX loads, so generic prefetch or adding `__ldg` is not an
established missing optimization. The independent
[L2-only weight-cache screen](experiments/2026-10-25-maxwell-cache-ffn-screen.md)
now confirms distinct compiled weight instructions and preserved bits, but
only **0.095-0.184% complete one-row gains**, below 5%. Keep `0cff8ab`
benchmark-only and close cache-operator tuning without runtime integration.
The compiler raises registers from 32 to 40; no cache-hit, bandwidth or
occupancy conclusion follows. The subsequent
[exact byte-permute conversion screen](experiments/2026-10-25-maxwell-byte-ffn-screen.md)
removes I2F instructions with unchanged 32-register use and control bits,
but gains only **0.138-0.295% complete one-row throughput**, with all
intervals spanning zero. Keep `d5a3350` benchmark-only; no conversion or
geometry rescue follows. The independent source/SASS finding shows scalar
U8 weight and FP32 activation loads remain in qualified QKV/output kernels.
The resulting [packed projection screens](experiments/2026-10-25-maxwell-packed-projection-screen.md)
now pass: one-request **QKV throughput +46.87-51.97%**, **output +40.02-46.78%**,
positive paired intervals, exact control bits, unchanged 32/40 registers and
no extra weight preparation. Both independent gates and the combined
optimistic **5.23-5.33%** scope passed before runtime integration, preserving
split-f32x4-128 FFN, SGEMM prefix, dense KV and all 109 prepared identities.
Runtime integration `c264c89` and qualification tooling
`43ea00a` now pass the [actual-model quality gate](experiments/2026-10-25-maxwell-packed-projection-runtime-quality.md):
512 actual IDs and twelve frozen outputs match, all seven sampled full-vocabulary
logit differences are zero, and all 109 prepared identities remain fixed.
Both incremental HTTP and fresh Ollama gates above now pass. Their claims
come from paired HTTP timing, not surrogate-input microbenchmarks or
unpaired quality timing. Do not combine rejected FFN candidates.

The independent
[combined QKV/output HTTP run](experiments/2026-10-25-maxwell-projection-incremental-http.md)
passes at **47.323 versus 44.950 tok/s, +5.279%**, with a positive paired
interval, unchanged actual IDs and all 109 existing prepared tensor identities.
Five pairs are sufficient under the original rules; no extension is needed.
The fresh comparison above completes the subsequent cross-engine check;
neither result satisfies either original goal.

The earlier
[fresh split-plane/Ollama comparison](experiments/2026-10-25-maxwell-split-ollama-http.md)
measures FP32 split-f32x4-128 plus SGEMM at **45.075 vs 43.119 HTTP tok/s**,
**4.536% faster**, with verified actual IDs, quality and cache resets.
Both original targets remain unmet: **53.06** and **56.055 tok/s** for
1.30x this fresh Ollama median. The selected exact combination is retained
experimentally; no global defaults change. The independent
[incremental comparison](experiments/2026-10-25-maxwell-split-incremental-http.md)
passes at **44.972 vs 41.205 tok/s, +9.142%** against current packed128/SGEMM.
Both use five alternating pairs with positive intervals and no extension.
The [new split-runtime profile](experiments/2026-10-25-maxwell-split-stage-profile.md)
preserves actual IDs and measures 0.986% instrumentation overhead. Middle
stage shares are 37.43% gate/up, 22.31% down and 3.62% attention. The fixed
[two-column follow-up](experiments/2026-10-25-maxwell-split-cols2-screen.md)
preserves bits but regresses complete one-row FFN throughput 5.85-6.01%.
Keep `414f2a5` benchmark-only and stop column tuning. The
[one-column instruction audit](experiments/2026-10-25-maxwell-joint-ffn-audit.md)
motivated one lossless joint gate/up layout. Its
[completed screen and fresh twenty-pair confirmation](experiments/2026-10-25-maxwell-joint-ffn-screen.md)
preserve bits and confirm three total loads per task instead of five, but
complete one-row gains are only **4.66-4.74%**, below the unchanged 5% gate.
Keep `8881763` benchmark-only, without runtime integration or more packing/
geometry tuning. Both arms retained 95,813,632 bytes of control plus joint
preparation. An independent
[fixed-width address/loop hypothesis](experiments/2026-10-25-maxwell-fixed-k-ffn-hypothesis.md)
identified 21 static prologue XMAD sites and runtime loop control despite
Qwen gate/up always using K=2048. The
[completed fixed-width screen](experiments/2026-10-25-maxwell-fixed-k-ffn-screen.md)
removes those sites and preserves bits, but complete one-row gains are only
**1.24-1.32%**, clearly below 5%. Keep `deef80b` benchmark-only without a
width/unroll/geometry rescue. The independent
[native QKV reduction screen](experiments/2026-10-25-maxwell-qkv-warp-screen.md)
now passes the stage gate at **27.47-27.76%** throughput gain with unchanged
bits, biases, raw weights and 32-register usage. Its optimistic standalone
whole-token ceiling is only **2.59-2.61%**, below 3%, so `499d2de` remains a
positive benchmark candidate, not a runtime win. The independent
[native output reduction screen](experiments/2026-10-25-maxwell-output-warp-screen.md)
now also passes at **30.42-31.57%** one-request throughput gain, preserving
bits, storage and 40-register usage. Both stage gates and the combined
optimistic **4.68-4.72%** ceiling pass. Primitive implementation `12191f5`
is followed by the combined model-local runtime `b7edf08` and
explicit-control/evidence tooling `1349613`, which pass the
[actual-model quality gate](experiments/2026-10-25-maxwell-projection-runtime-quality.md):
**512 actual IDs and all twelve task outputs match**, with zero regressions
across seven dense-valid tasks and **zero sampled full-vocabulary logit
differences**. All 109 gate/up/down/head cache identities are unchanged;
QKV/output each dispatch 18,432 candidate launches. The incremental HTTP
and fresh official Ollama gates are now passed above. No default promotion
follows from single diagnostic generation times. Do not rescue geometry or revive failed
prepared layouts. Prepared down/head and original micro/quality/HTTP gates
remain fixed. Do not
replace the exact-engine objective with numerical, pruning or scheduling results.

Earlier checkpoints below record their historical next steps, not a second
active execution order.

The [previous packed128/SGEMM stage profile](experiments/2026-10-25-maxwell-selected-stage-profile.md)
now preserves actual IDs off/on and locates middle-token GPU stage time at
**42.94% gate/up/SwiGLU, 20.32% down, 3.34% attention**. Instrumentation
lowers HTTP throughput by 3.79%, so it is diagnostic, not a new speed claim.
This supported one fixed 128-thread FP32 scale-factored gate/up experiment,
with current packed FP32/128 as timing control and all original gates intact.

That [bounded follow-up](experiments/2026-10-25-maxwell-factored-ffn-screen.md)
now fails the performance screen: complete one-row FFN throughput is
**0.07-0.28% slower**, and four-row gains are only **1.59-2.96%**. All
primitive numerical/ownership checks pass. Fewer static multiply sites did
not yield useful speed. Implementation `498699a` is retained benchmark-only;
do not wire it into generation or expand geometry to rescue it. The completed
load-distribution follow-up is below. Memory bandwidth or
occupancy is not yet measured.
The exact-engine objective and qualified packed128/SGEMM selection remain.

The [adjacent-pair load follow-up](experiments/2026-10-25-maxwell-coalesced-ffn-screen.md)
also misses the 5% screen: complete one-row FFN gains **1.84-1.98%** with
positive paired intervals; four-row gains are only **0.70-1.70%**. All
primitive checks pass. Implementation `aa767eb` remains benchmark-only.
[CUPTI group creation](experiments/2026-10-25-maxwell-counter-capability.md)
returns unsupported even as administrator; enumerated metrics are not
measurements. A static request model and sm_52 code confirm improved input/
payload coalescing, but the smaller tasks issue more scale-load groups.
The resulting split-plane screen is below; no further native two-value
geometry was added.

The [split-plane/four-value FFN screen](experiments/2026-10-25-maxwell-split-ffn-screen.md)
now passes: **15.54-15.65%** complete one-row throughput gain over qualified
packed128, positive paired intervals, unchanged numerical tolerance. Four-row
work gains **13.47-15.91%**. Primitive implementation `3420c33` now has
opt-in model-local runtime integration `05607ab`. One current-layer pair adds
**47,906,816 bytes** for both arms; median packing/upload is **30.615 / 7.950 ms**.
The microbenchmark's all-layer extra-storage estimate is now confirmed by
the runtime quality checkpoint below. Primitive tests alone do not establish
generated-token parity.

The [split-plane runtime quality checkpoint](experiments/2026-10-25-maxwell-split-runtime-quality.md)
passes against explicit current packed128/SGEMM control: **512 actual IDs and
all twelve task outputs equal, zero regressions across seven dense-valid
tasks**. Maximum sampled logit drift is 0.004531, top-10 overlap stays 10.
Observed gate/up preparation adds **1,724,645,376 bytes**; source plus all
prepared weights is **6,197,100,544 bytes**, excluding KV/workspace. Both arms
retain 252 prefix conversions/SGEMMs, identical enqueue counts, no quantized
activation storage and complete tracked cleanup. Missing preparation,
budget, foreign-pointer and ownership checks pass. New tooling `f6078fe`
binds control/candidate quality and pins down/vocabulary identities while
allowing only the gate/up delta. Incremental and fresh Ollama HTTP gates are
now complete above. Loader correction `f93bb89` ensures selected-runtime
Ollama/profile modes prepare gate/up too; the failed preflight collected no
timing samples. No default change; both original unmet objectives remain intact.

**Current checkpoint: sequential prefill ownership repaired.** The
[real-model attribution](experiments/2026-10-25-maxwell-prefill-drift.md)
shows that draining each no-logits prefix forward or running a greedy head
at every prefix changes the old sequential output to the packed first token.
Manual no-drain reproduces the old result bit-for-bit; layer-0 KV already
differs substantially. Final cached attention replay agrees with CPU within
2.51e-6. The fix in `2ca0de1` now completes each no-logits prefix forward.
The [matched corrected probe](experiments/2026-10-25-maxwell-prefill-drift-completed.md)
shows session, drained and per-prefix-head execution match every sampled
K/V and final hidden/logit bit. A queued-work fixture fails before the fix
and passes after it. Packed prompt-logit differences fall to about 0.011
max, but full-continuation equality still needs fresh validation.
Prior M08 exact-to-control gates and timings remain historical evidence,
not corrected-control qualification. M08/M09 and both original goals remain
incomplete. Runtime defaults are unchanged. Next regenerate M08 quality
reports for the completed-prefix contract before any HTTP qualification;
old reports cannot qualify the repaired runtime. The five remaining
[historical HTTP comparisons](experiments/2026-10-25-maxwell-runtime-http-screen.md)
are preserved, including FP32/256's twenty fresh below-gate pairs.

The first [fresh corrected-control quality run](experiments/2026-10-25-maxwell-completed-packed128-quality.md)
passes for FP32/128: all 512 actual browser IDs and all twelve task outputs
match, with zero regressions across seven dense-valid tasks and unchanged
held-out loss conclusions. It carries the new completion contract. Next
regenerate native-packed and dequant-SGEMM quality, then remeasure qualifying
independent/combined runtimes and the selected-runtime Ollama comparison.
The old +8.133% HTTP number is not corrected-control timing evidence.

Both independent [fresh packed-prefill quality checks](experiments/2026-10-25-maxwell-completed-prefill-quality.md)
now pass too: native packed and dequant SGEMM preserve all 512 actual IDs
and all twelve frozen task outputs against corrected sequential, with zero
regressions across seven dense-valid tasks. Their largest sampled logit
differences are 0.010689 / 0.013475, not bitwise parity. Diagnostic prompt
intervals are about 3.21 s native packed and 0.48 s SGEMM versus 1.59 s
sequential; controlled HTTP evidence is still required. Next remeasure the
independent candidates, then a qualifying combination and fresh Ollama.

The [corrected-control independent HTTP screen](experiments/2026-10-25-maxwell-completed-http.md)
now qualifies FP32/128 at **38.382 vs 35.537 tok/s, +8.003%**, and SGEMM
prefill at **38.416 vs 35.421, +8.456%**, with positive paired intervals and
all actual 512 IDs equal. Native packed loses 10.085% and is retained only
as a negative control. Each uses five alternating pairs with stable clocks;
none requires extension. Next validate the combination, then time it only
if quality passes. The existing tool now has a tested explicit
`--time-ollama` mode for the eventual fresh selected-runtime comparison.
No default changes or original target achievement follow from this screen.

The [combined FP32/128 plus SGEMM quality gate](experiments/2026-10-25-maxwell-completed-combined-quality.md)
now passes: all 512 actual IDs and all twelve task outputs match corrected
control, with zero dense-valid regressions. Maximum sampled logit drift is
0.011406; top-10 overlap stays 10. Next run combined/native HTTP pairs and
the fresh selected-runtime Ollama comparison. Single diagnostic prompt and
decode intervals are not substituted for those controlled measurements.

Combined [corrected-control HTTP qualification](experiments/2026-10-25-maxwell-completed-combined-http.md)
passes at **41.360 vs 35.467 tok/s, +16.616%**, with a paired 95% interval
of **[16.439%, 16.708%]** and all actual 512 IDs equal. Five alternating
pairs meet the unchanged timing gate without extension. This retains the
explicit typed combination, not a global default or an Ollama target claim.

M01's [source and runtime dispatch checkpoint](experiments/2026-10-25-maxwell-q8-dispatch.md)
is complete. The SDK-only observer in `723a456` traces the official binary,
not a rebuilt surrogate. Its completed M40 runner trace observes Q8_1,
unfused 128-thread Q8 MMVQ, Q8-to-FP32 dequantization and Maxwell SGEMM,
with zero dropped records, unknown symbols or launch errors. The bounded
65-prompt/4-generated request is diagnostic, not a throughput trial. sm_52
PTX is shipped; offline ptxas resource checks are not the actual driver JIT.
The source pin, compatibility scope, licenses, raw data and limitations are
recorded. No kernel, runtime default or acceptance gate changes.

M02 is complete. The [paired FFN input contract](ffn_benchmark_inputs.md)
has eight CPU tests covering unmodified paired vectors, generated positions,
source/provenance binding, bounded storage and rejection of invalid captures.
It does not enable activation sparsity. The model-local read-only CUDA observer
now preserves exact logits/KV bytes under prepared down/vocabulary, with fused
and unfused gate/up and selected-layer capture. Two M40 tests verify replay,
CPU-reference agreement, copy/sync counts, allocation cleanup and unsupported
path rejection. Clean release `f459c97` now has
[frozen real calibration inputs](experiments/2026-10-25-maxwell-ffn-inputs.md):
21 paired rows/layer, all 163 inactive/active generated IDs equal, complete
provenance and allocation cleanup. Three control tasks pass and three fail
their fixed rubrics; capture changes no output. The shared benchmark now has a
complete-FFN mode using production wrappers, genuine rows 1/4, CPU references,
guarded reusable scratch and enqueue counters. Clean release `29dee55` now has
[warm complete-FFN measurements](experiments/2026-10-25-maxwell-ffn-baseline-warm.md):
about 0.457 ms/layer at one row, versus 4.67-4.72 ms for rows4, with gate/up
dominant. All numerical/ownership checks pass. The initial 949 MHz clock-ramp
run is preserved; bounded warmup yields 1113 MHz at every primary sample.
This is a baseline, not an inference speed improvement.

M03 is complete: [clean release five-pair evidence](experiments/2026-10-25-maxwell-fp32-ffn-screen.md)
on all three layers shows packed/warp256 improving complete one-row FFN
throughput 7.75-8.27%, with native hidden and completed-output bits preserved
on these real inputs. Packed/warp128 gains 12.18-12.63% with small nonzero
arithmetic drift. The other three fixed variants fail the decode micro gate;
all implementations and measurements are retained. Four-row gains also
include fusion changes and are reported separately. All 36 validations and
900 primary-interval audits pass, with stable observed 1113/3004 MHz clocks.
Release sm_52 uses 32 registers and zero stack/local bytes throughout.
No default changed; whole-model 512-token parity and E2E qualification remain
pending. M04/M05 screening and M06 integration are recorded below. M07 prefill
SGEMM remains independent.
Do not add capture infrastructure or expand the bounded geometry search.

M04 has a [pinned Q8_1 encoding and packed-dot contract](q8_activation.md).
The encoder preserves FP32-before-FP16 rounding and original-input XOR sums;
release M40 tests match every CPU byte. Both fixed 128/256-thread consumers
pass full gate/up/SwiGLU CPU-reference tests using reconstructed activation
bytes, including two-byte weight bases and genuine distinct rows 1/4.
Optional intermediate capture matches the timed kernel's SwiGLU bits.
Release code uses 17 quantizer / 32 consumer registers, no stack/local bytes
and no DP4A/Tensor instructions. MIT provenance and the full upstream notice
are included. The [clean release real-input screen](experiments/2026-10-25-maxwell-q81-ffn-screen.md)
now completes M04: quantization-inclusive one-row FFN gains are 27.26-27.41%
at 128 threads and 21.50-21.71% at 256. All 18 numerical checks and 60 paired
trials pass. Original-FP32 drift is measurable, including up to 0.538361 in
sampled completed output; no whole-model quality or E2E claim follows.
The report closes M05's bounded micro screen with all failed implementations
preserved. Packed/warp FP32 and Q8_1 qualify conditionally for M06, not default
promotion. M06 now captures model-local selection and reusable scratch;
full-model logits/512 IDs and quality remain pending.
Test FP32 128-thread actual continuation equality rather than assuming drift
is harmless or disqualifying it without evidence. Q8_1 stays a separate
numerical track with task/loss/perplexity gates. M07 prefill remains independent.
Use layer 0/17/35 tensors and read-only frozen-calibration activation samples
for complete FFN baselines at genuine rows 1/4. Preserve
the qualified prepared down/vocabulary control. FP32 changes require exact
512-token parity; Q8_1 stays a separate opt-in numerical track. Independent
prefill work has a bounded temporary dequant buffer, not full FP32 caching.

M06 is complete: [runtime integration and regressions](experiments/2026-10-25-maxwell-ffn-runtime.md).
`M40LLM_Q8_FFN_KERNEL=native|packed-f32|q8-1` is captured at load time, with
native unchanged by default. Typed choices retain both screened geometries.
Q8_1 reserves one reusable RAII workspace buffer; packed FP32 reserves none.
Prepared down/vocabulary remain fixed, multi-row packed-prefix projections
stay native, and only one-row steps dispatch the candidate. Unsupported
models, scratch, graph/batched decode and research combinations fail clearly.
Runtime tests cover CPU references, scratch reuse, zero warm allocation,
producer streams, packed-prefix dispatch and packed256 fixture logits/KV bits.
The mixed-flag regression exposed a pre-existing Q8 logits producer-stream bug,
fixed independently in `d211726`; the qualified Q8 decode schedule is unchanged.
Release M40 affected CUDA suites pass with cuBLAS on/off, the shared benchmark
tests pass, and non-CUDA warning-as-error tests plus CUDA/server Clippy pass.
Runtime-disabled-upload all-target checks also pass; NVCC still compiles that
build. These integration tests alone make no full-model quality or speed
claim. Independent M07 and the subsequent M08 qualification are recorded below.

M07 is complete: [bounded temporary Q8-to-FP32 prefill SGEMM](q8_prefill.md).
Checked shapes cap one weight at 128 MiB and exclude the vocabulary matrix.
Release M40 tests preserve every converted weight bit for both real Qwen FFN
orientations and feed the existing async SGEMM with genuine distinct rows 4/64.
The converter enqueues one kernel with no allocation, host copy or sync.
Source/output guards, two-byte bases, capacity rejection and cleanup pass.
Model-local `native|dequant-sgemm` selection now reuses one RAII temporary
weight across all seven projections/layer, on prefill. Decode/vocabulary keep
their original paths and no full-model FP32 cache is retained. Tests cover
final logits/KV at existing tolerance, device lengths, Qwen head128 GQA with
bias/split-half RoPE, pointer reuse and unchanged warm decode allocations.
GPU-greedy now honors an explicit packed-prefix bound while keeping its final
token/head path; disabled/over-bound experimental requests fail, not silently
fall back. Actual temporary capacity is retained in generation reporting.
The real Qwen arithmetic smoke observes 252 conversions/SGEMMs and output
`4`, matching native packed. Sequential outputs `2` in both the parent
`5d7472d` and current integration: not a new sequential regression, but not
exact-output equality either. Raw evidence and limitations are linked above.
No default changes or throughput claim follow. Next is M08: full-model
512-token equality, dense-valid tasks, Q8_1 quality/loss, isolated prefill and
decode E2E pairs, then a qualifying combination and fresh Ollama comparison.

M08 now has [full-model runtime quality evidence](experiments/2026-10-25-maxwell-runtime-quality.md)
from release `3d6833a`. Explicit request-owned prefill keeps scientific
choices independent of ambient settings; both models retain prepared
down/vocabulary. Packed FP32/128 and /256 preserve all 512 browser IDs and
all twelve frozen task outputs. Seven dense-valid task rows have no
regressions; packed128 has small nonzero logit drift, while packed256 sampled
logits match exactly. Both may advance to HTTP timing, not default promotion.
Q8_1 diverges at step 42 with higher held-out PPL, so remains numerical
research. Native packed and SGEMM prefill produce identical candidate IDs but
both diverge from sequential at token 0. Their shared large prompt-logit drift
must be isolated before an exact combination can qualify; a faster SGEMM
prompt interval is not enough. FP32-versus-FP16 attention K/V consumption is
a source-supported hypothesis, not a measured explanation. All six cases
release tracked allocations. Next run bounded controlled HTTP comparisons,
retain the failed gates, and resolve packed-path semantics before combination.
Neither original E2E target is achieved.

M08's first [controlled HTTP qualification](experiments/2026-10-25-maxwell-runtime-packed128-http.md)
now passes: five alternating pairs give packed FP32/128 38.493 versus native
FFN control 35.598 HTTP tok/s, +8.133% with positive [7.938%, 8.275%] paired
95% interval. All 512 actual IDs, frozen dense-valid tasks and HTTP smokes
pass. Prepared down/vocabulary, dense/context 1024, launch counts and
allocation counts stay fixed. The result qualifies this explicit typed
candidate only; environment/default selections remain unchanged. The isolated
Ollama runner is stopped, final CUDA regressions complete with cuBLAS on/off,
and release examples are restored to cuBLAS-on builds. Remaining bounded
candidate comparisons, packed-path attribution and fresh selected-runtime
Ollama comparison are still required; M08/M09 and both original speed goals
remain incomplete.

## Previous Research Checkpoints

The entries below retain their original checkpoint sequence. Their historical
"next" steps do not override the current M01-M09 plan above.

C18's [unchanged-model audit](experiments/2026-10-25-unchanged-model-checkpoint.md)
is complete. C01-C11, conditional C10A, and C11A are verified checkpoints.
C12-C17 are **postponed**, not implemented, under the negative n-gram gate.
Individual and combined research results are recorded through C26 below.
C22/C27 remain postponed under
the negative sparse gate. Neither exact-engine speed objective is achieved.

Validation follow-up: the local NVCC/no-device CI reproduction exposed 25
hardware-dependent assertions across 14 targets, including prior kernel/cache
checks and C19/C20. They now use explicit `#[ignore]` markers, not successful
no-device skips. Use `--include-ignored --test-threads=1` for actual hardware
validation; an explicit request still fails without a device. New evaluation/
mask checks also serialize shared GPU accounting and process environment.
The normal NVCC CI lane is not evidence of GPU correctness. Local non-CUDA
and NVCC/no-device warning-as-error lanes now pass, including default parallel
test threads. All 25 gated lib/integration hardware assertions execute and pass
on M40 with explicit inclusion; the six evaluator tests also pass. An explicit
hardware request with the device hidden fails as intended. No inference or
benchmark behavior changes in this test-policy follow-up.

**C19 is complete.** Disjoint inputs/rubrics were frozen before the dense run;
teacher-forced loss and the explicit-model release runner now have a recorded
Qwen2.5 Q8_0 reference over both splits. See the [C19 evidence and limitations](experiments/2026-10-25-model-quality-dense-reference.md).
Modified-model speed must not be presented as an unchanged-engine win.

**C20 is complete.** Model-local masks preserve original IDs and allocations;
empty-mask parity and skipped work are verified. The fixed one/two/three-layer
release probe records matched-step gains about 2.6/5.2/7.9%, but held-out PPL
worsens from 39.61 to 47.66/52.72/53.78, with task regressions at two/three
layers. Keep all masks diagnostic; no source-model or runtime default changes.
See [the measured tradeoffs and raw artifacts](experiments/2026-10-25-layer-mask-sensitivity.md).

**C21 is complete with a negative decode gate.** Fixed 16-of-32 packing passes
scalar/matched-zero parity and saves 29.41% of requested weight bytes, but
the best single-row sparse kernels take 2.29/2.31/2.66x native gate/up/down
latency. The complete release matrix has 240 pairs across real FFN shapes,
rows 1/4/16/64 and four geometries. Some multi-row gate/up wins are insufficient
for a decode pruning backend. C22 and C27 are postponed under this negative
gate. See [the screen, raw data and limitations](experiments/2026-10-25-sparse-q8-screen.md).

The separate pageable H2D/default-stream audit is now complete, including
native setup and legacy metadata publication. The release checkpoint preserves
C11's exact trace and does not establish a speedup from the correctness fixes.
The C23 input-layout screen below is independent of the rejected sparse
kernel. Layer ranking remains C24, compact export
C25. Do not tune on held-out/browser inputs or reopen failed n-gram/dense-layout
screens. See [the audit and raw results](experiments/2026-10-25-upload-ordering.md).

**C23 is complete as an opt-in approximate-model experiment.** The clean
`bd76484` release screen has 100 alternating serial pairs across four fixed
geometries and five sparsity targets. At achieved 40.2/51.2/60.6% sparsity,
width16/128 reduces isolated selection-plus-down latency by 18.6/30.4/39.1%.
These synthetic inputs and real layer-0 Qwen weights do not establish model
quality, real activation sparsity, DRAM savings or E2E gains by themselves.
Zero-threshold compaction is numerically correct but slower. See
[the prerequisite screen](experiments/2026-10-25-activation-input-sparsity.md)
and [the completed calibration/quality evaluation](experiments/2026-10-25-activation-quality.md).

Selection/quantile contracts are committed in `10478a8`, reusable CUDA scratch
and selected-input consumption in `9b5148c`, and the shared benchmark mode in
`bd76484`. Two explicit M40 tests verify deterministic GPU compaction,
same-zeroed native/CPU parity, tails, nonfinite bits and scratch reuse. Enqueue
adds two launches with zero allocation/copy/sync. Existing tiled-Q8,
attention/projection/MLP checks pass with cuBLAS on/off; cuBLAS-on forward/
ownership, both warning-as-error CI lanes, Clippy and fmt pass as well.
GPU-less CI is not GPU correctness evidence. The one-iteration preflight is
not pooled with the measured screen.

The calibration artifact contract is committed in `c89b939`. Model-local
capture/application now passes two-layer M40 tests: capture leaves native
logits bit-identical; zero-threshold layers retain native execution and need
no prepared allocation; nonzero layers use immutable tiled weights, reusable
scratch and device retained/input counters. Unsupported batch/packed/graph
paths fail before launches. Artifact identity/budget failures publish no
partial cache. Model-local integration is committed in `1f2a65c`, and the shared
capture/evaluation runner in `5697e64`. See [the contract and commands](activation_sparsity.md).

The calibration artifact was frozen before variant/held-out evaluation, SHA
`a1721123485a8f1195f751538c56dc1db4dc450592df312b44cd58fc06f7fcc9`.
All 36 layers observe 110 generated-input rows, capped at 32/task from only
C19 calibration tasks. All six capture traces match C19. Five clean-source
release evaluations cover targets 0/25/40/50/60 on both frozen splits.
All references match C19, and the zero control is exact for 12 task traces,
203 teacher-forced tokens and all timed predictions. No held-out/browser
fitting or new geometry sweep occurs.

Target 25% is about 0.8-0.9% slower. Target 40% gives about 2.2% matched-step gain
but regresses held-out workshop JSON/stop behavior. Targets 50/60% preserve
dense-valid task passes and improve held-out matched-prefix rate 4.325/6.684%,
but held-out PPL rises 8.35/17.38%. Each positive target adds 862,411,332
requested device bytes for duplicate down weights/scratch/counters. These are
approximate-model tradeoffs, not generated/HTTP TPS or unchanged-engine wins.
Keep all thresholds research-only; no default promotion. Both exact speed
objectives remain unmet.

Post-matrix CUDA parity/forward/transfer checks pass with cuBLAS on, focused
parity also passes with it off, and all seven shared runner tests pass.
Both warning-as-error CI lanes, three all-targets Clippy configurations and
fmt pass. The explicit hardware test fails when requested without a device,
rather than treating missing hardware as success.

**C24 is complete.** Read-only block influence and all 34+33+32 interior
calibration-loss trials ran on clean `68f1b15`. Freeze SHA
`a9f0f2cc8c952eacaf96d6fe2576e30a35ef225a5f85f4574630545af3ab4d3c`
binds original-ID masks `[8]`, `[8,24]` and `[8,24,25]` before held-out
evaluation. No held-out/browser fitting or BI shortlist occurs. The first
loss winner ranks 22nd by dense BI, so BI alone is not a removal recipe.
See [the contracts](layer_ranking.md) and
[complete measurements and raw data](experiments/2026-10-25-ranked-layer-tradeoffs.md).

The empty control exactly matches current full logits and C19's recorded
traces/statistics. All applied masks exactly reproduce their frozen
calibration losses. Held-out PPL is 39.6122 dense versus
40.9846/46.1846/49.3736 with one/two/three removals. The one-layer mask
preserves all seven dense-valid task passes but still increases held-out
PPL 3.46%. Two/three removals introduce unclosed-JSON-fence regressions with
requested facts still visible. Dense-failed tasks remain inconclusive;
aggregate task counts do not prove preserved quality.

Five alternating pairs give held-out matched-prefix rate gains
2.484/5.116/7.937%, including greedy readback and final completion. These
model-changing measurements are not generated/HTTP throughput. Every model
still uploads 3,279,519,744 weight bytes and reserves 37,748,736 dense KV
bytes. No mask or runtime policy is promoted. Post-matrix cuBLAS-on/off
hardware validation passes; both warning-as-error CI lanes, Clippy and fmt
also pass. Implementation commits are `7eea030`, `de98d27`, `c00fca7` and
`68f1b15`.

Next is C25 compact export of the explicitly quality-evaluated `[8]`
candidate, recorded before held-out evaluation as the least structural
change. Write a separate artifact with retained-layer tensor renumbering,
updated metadata and provenance; never modify the source GGUF. Require
round-trip output/logit equivalence to the masked source and actual weight/KV
reclamation. C26 then compares measured quality, speed, allocation and
preparation costs. C22/C27 remain postponed under their negative sparse
kernel gate; do not reopen rejected n-gram or dense-layout screens.

C25's [copy plan and streaming writer](compact_gguf.md) now map retained
original IDs to contiguous tensor names and preserve unchanged metadata wire
bytes. The CPU exporter requires the frozen ranking and completed paired
quality report; it retains regressions/PPL rather than calling evaluated masks
quality-approved. SHA-256 checks and create-new publication protect the source
and existing outputs. Four plan, two evidence and four writer tests pass.
Both warning-as-error CI lanes, all-targets Clippy and fmt pass as well.
Real compact-model load, masked-source numerical/task parity and measured
weight/KV reclamation remain. No memory-saving or engine-speed result is
claimed from the copy plan or successful file export alone.

The clean `890b9c3` release CPU tool now exports the real Qwen `[8]` artifact
to ignored `target/research-exports/qwen-q8-pruned-layer8.gguf`, SHA
`7d04260214a3b57379310e963c4f6b4d39208be9b644d45c83aefa4430d03070`.
It retains 35 layers / 422 tensors and reduces actual tensor-data file bytes
to 3,197,605,888, with the source unchanged. Preparation is 38.683 s in one
serial CPU-only run. This is not a GPU allocation, inference parity or speed
result. See [the checkpoint and reproduction identities](experiments/2026-10-25-compact-layer8-export-checkpoint.md).
Next is the live masked-source/compact-model comparison, not another mask fit.

The shared quality runner now has an explicit `--compact-export-report`
comparison mode. It masks only the original source, verifies compact file and
metadata lineage, checks actual owned uploads and physical KV capacity, and
requires matching generation plus zero full-logit drift. Additional task-prefix
diagnostics stay outside generation and paired timing. All 16 runner tests,
including the real exported Qwen artifact, pass on M40 with cuBLAS on/off.
Normal reference-mask guardrails remain intact. Clean release measurements
over both frozen splits complete that checkpoint below.

**C25 is complete.** Clean release `0dfff93` compares the separate compact
35-layer Qwen artifact against the original source with only frozen layer 8
masked. All 12 task traces, 951 task-prefix full-logit comparisons, 203 passage
tokens and paired-step predictions match exactly, including the historical
C24 candidate traces/statistics. Owned uploads reclaim 81,913,856 bytes and
native dense KV reclaims 1,048,576 bytes at context 1024. These are separate
successful allocation requests, not peak driver VRAM. Compact-versus-masked
timing is neutral at +0.060% held out; do not double-count C24's 35-versus-36
layer gain. Held-out PPL still costs 3.46% versus the unchanged model.
See [the live report and raw release evidence](experiments/2026-10-25-compact-layer8-live.md).

**C26 individual comparison is recorded.** The [modified-model comparison](experiments/2026-10-25-modified-model-tradeoffs.md)
reconciles 13 completed raw reports, including held-out quality, paired-step
speed, actual generation counts/stops, allocation and preparation boundaries.
It promotes no mask, threshold or combined model. Sparse decode and n-gram
gates remain negative; no independently useful static sparse kernel exists
for static-plus-dynamic sparsity. The prior checkpoint stopped before combined
layer/activation measurements; its extra E2E/quality-budget prerequisites were
not a requirement for research in the authorized C26 plan. Do not infer additive
gains or reuse source-bound activation metadata on a compact model.

The completed bounded checkpoint composes frozen mask `[8]` with existing activation
targets 0/50/60%, on original source IDs and both unchanged C19 splits, five
paired trials each. The zero control matches recorded masked-native behavior;
positive thresholds retain all original prepared down weights and skipped-layer
counters stay zero. No new fitting or default promotion occurs.
Implementation permits application after the mask and freezes mask changes
afterward; calibration capture still requires every original layer.

**C26 now includes measured composition.** Clean `26a3a5d` runs both splits
with targets 0/50/60% and five alternating pairs. The zero control matches C24's
12 masked-native task traces and 203 recorded loss-token statistics; every full
reference matches historical C24. Positive variants retain original IDs and
all original prepared down weights, adding 862,411,332 bytes. Held-out rate
gains are 7.012/9.247% with PPL 44.293885/50.199624, increases of 11.82/26.73%.
The 60% combination regresses calibration config lookup from `6` to `19`.
Both are research-only model changes, not unchanged-engine gains. See
[the report and complete raw evidence](experiments/2026-10-25-combined-model-tradeoffs.md).

Latest qualified unchanged HTTP rate is 35.724705 tok/s with explicit split
down/vocabulary preparation after the native-order correction below. Its
contemporaneous native control is 31.962281. The latest separate timed Ollama
baseline remains 43.162711; both 53.06 and 1.30x-Ollama 56.1115 targets are
unmet. Runtime defaults and negative research gates are unchanged;
combined-model work cannot satisfy an unchanged-model objective.

Next is a bounded diagnostic of C10A's existing down/vocabulary timing lead,
not another kernel/layout search. `model_quality --q8-prefix-drift` forces the
verified recorded native prefixes into both unmodified source models. It
records actual greedy predictions through the token-310 neighborhood, plus
native/prepared heads on both hidden vectors. Generic full-logit reductions
are named separately from the actual fused native greedy head. Same-model
controls, original absolute positions, evidence identity and cleanup must
validate before the clean release probe. The old candidate remains unqualified
until the branch is explained and the existing acceptance gate is satisfied.

The clean `6ff0f63` probe reproduces the archived branch exactly. All 317 native
predictions match history; the only forced-prefix disagreement is step 310.
Both heads choose 1008 from native hidden state and 5815 from prepared hidden
state. Same-hidden head drift is at most 3.815e-6, while forward-induced
generic-logit drift reaches 0.004250050 across a native top-two margin of
0.000270844. See [the report and raw data](experiments/2026-10-25-q8-prepared-prefix-drift.md).
Next check existing 4-column/256-thread separate projection against native
decode bitwise at Qwen down dimensions: cache dispatch currently selects 128,
and the selected native coltile4/split4 path uses 256. No new kernel/layout
search or parity-gate relaxation is authorized by this observation.

The pointwise check now passes: existing 4-column/256-thread separate
projection matches every native FP32 bit at K/N 2048/33 and 11008/2048.
The 128-thread control differs in 30/33 and 1893/2048 outputs, respectively,
using non-binary activations and synthetic Q8 payloads. Both separate-projection
tests pass with cuBLAS on/off. This is numerical evidence, not real-model
timing. Next correct only the single-row prepared-down launch selection,
record it in cache telemetry, and require real same-prefix and HTTP checks.

The dispatch correction now selects existing 4-column/256-thread separate
execution only for single-row prepared down. Other families, multi-row paths,
storage and global preparation-off defaults stay unchanged. Cache telemetry
records `single_row_down_split_launch` for actual prepared-down ownership.
Cache/parity checks, the full 21-test runner, both warning-as-error CI lanes
and non-CUDA/CUDA-server Clippy pass. A clean real prefix probe and the
original full-workload gate were required before making the result below.

The clean `dd8c5f6` release probe now matches all 317 actual predictions and
all twelve sampled hidden vectors exactly. The old step-310 branch is gone.
The original five-pair HTTP screen then measures 35.724705 versus 31.962281
tok/s, +11.771%, 95% paired bootstrap interval 11.631-11.866%, with identical
512 actual generated IDs and no dense-valid smoke regressions. See
[the complete confirmation, costs and raw evidence](experiments/2026-10-25-q8-native-order-http.md).
This qualifies the explicit split down/vocabulary benchmark configuration,
not general preparation, compressed attention, model changes or either original
throughput objective. It adds 1,192,935,424 prepared device bytes.

The [qualified-config profile](experiments/2026-10-25-q8-native-order-profile.md)
is complete on clean `1a5388c`. Five ordinary/profiled HTTP pairs show 0.717%
instrumentation cost and equal 512-ID traces. Reused middle-token GPU spans
are 11.872 ms gate/up/SwiGLU, 4.774 ms down and 0.777 ms attention. FFN is
65.96% of interval union, attention 3.07%; completion waits overlap execution.
Sampled tokens have no device allocations. First-use profile wall inflation
is not ordinary CPU-cost evidence.

Next investigate the remaining native gate/up cost before another kernel
change; sequential prefill remains about 1.58 s as well. New candidates still
require the full parity and repeated HTTP gate, followed by a fresh controlled
Ollama comparison for goal completion. Keep C11 controls for frozen quality
artifacts. Do not change defaults, refit held-out data, or reopen negative
n-gram/static-sparse gates from these timings.

C21 used these dependency-ordered commits:

1. `df9de8a`, `feat(infer): define fixed sparse Q8 blocks`: checked 16-of-32 host
   packing, matched-zero reconstruction, explicit tail/alignment accounting
   and CPU tests. Reuse C08 source identity, without adding a runtime policy.
2. `ba57428`, `perf(cuda): add sparse Q8 projection probe`: owned device storage,
   conventional Maxwell projection and CUDA parity against the same pruned
   dense matrix. Depends on the format contract.
3. `f2c5063`, `test(bench): compare matched sparse Q8 work`: bounded release paired
   microbenchmarks at real FFN shapes, packing cost and memory accounting.
   Depends on numerical parity; no model calibration or default changes.
4. `6d02feb`, `fix(bench): finish sparse probe input upload`: setup-only input
   readback fence, scalar failure values and expanded large-K coverage.
5. `51fa76b`, `docs(perf): record sparse Q8 screening`: measured negative C22
   gate and preserved raw evidence before separate removal.
6. `refactor(cuda): remove failed sparse Q8 probe`: remove the losing device
   code/tool, preserving CPU reconstruction and historical reproduction.

The initial block is 24 bytes: mask at byte 0, original FP16 scale at 4,
two zero alignment bytes at 6, and sixteen signed values at 8 in ascending
mask-bit order. The ideal representation is 22 bytes. Exactly 16 physical
positions are retained even in a partial final block; out-of-range payloads
are zero, so logical-tail sparsity is reported separately. This experiment
does not select weights for a model or promise an end-to-end speedup.

The retained CPU contract has three passing format tests. The archived CUDA
probe passed all four geometries, ragged K/N, signed/zero scales, distinct
rows and invalid-call rejection; the release matrix passed full matched-zero
and scalar-sample checks. Results are preserved in `51fa76b`, after clean
implementation `6d02feb`. The active sparse CUDA kernel, FFI/device API,
hardware target and tool are removed separately. Reproduction commands in
[the format/probe guide](sparse_q8.md) refer to `6d02feb`, not the post-removal
tree. Runtime defaults are unchanged. Broader upload ordering was audited in
the subsequent checkpoint below.

Post-removal validation passes: attention grid and projection/MLP wrappers
with cuBLAS on/off, the CUDA forward suite, full warning-as-error non-CUDA
and NVCC/no-device lanes, both all-targets Clippy configurations, and fmt.
Production CUDA/build/FFI files match the pre-C21 tree; only the standalone
CPU format is retained as new code. No GPU benchmark runs alongside tests or
builds, and the benchmark process releases all GPU use on exit.

### Upload-Ordering Correctness Checkpoint

The pre-existing pageable-copy contract dates to `8c9e3b45`; nonblocking
consumer streams make default-stream completion an explicit requirement.
This is a separate correctness follow-up, not a sparse-kernel correction or
a reason to rewrite the C21 preservation commits.

1. `ad217af`, `fix(cuda): complete synchronous host uploads`: make the central upload
   return device-ready, document producer/consumer ownership, account for the
   completion wait and add immediate-consumer regressions on both streams.
2. `cca45db`, `fix(cuda): publish initialized device storage`: finish native weights
   and checked KV initialization before publishing usable pointers. Depends
   on the central upload contract, without per-kernel hot-path waits.
3. `3ab1377`, `fix(kv): publish legacy append lengths on device`: separate the legacy
   metadata correction from setup; remove stack-owned async transfers and
   establish the length producer's dependency before host reads. Production
   explicit-position scheduling stays enqueue-only.
4. `7245d26`, `test(cuda): isolate cuBLAS graph fixtures`: independent pre-existing
   cuBLAS-off/test-order issue found by broad validation; explicitly scope
   dense KV and require the fixtures' async-cuBLAS backend, not all graphs.
5. `docs(perf): record upload ordering validation`: publish native/cuBLAS-on/off
   parity, local CI lanes and the bounded release checkpoint against the pinned
   exact trace. No throughput improvement is attributed to the fixes.

This audit preceded the C23 input-layout and calibration work recorded above.
It did not authorize reopening static sparse-Q8 tuning or runtime policy changes.

The immediate-consumer regression reproduces the pre-fix failure on M40:
rows=16,K=11008, iteration=1, index=76032 reads zero instead of 0.022460938.
The central synchronous upload now drains the default stream and records
`memcpy_h2d_completion`. All 48 repeated uploads match exact input values on
decode and exact signed Q8 reference sums on prefill. The completed contract
still requires caller-side ownership of prior nonblocking destination users.
See [the explicit transfer contract](cuda_transfer_contract.md).

Native weight and KV setup now returns initialized, device-ready storage.
Immediate append/attention tests cover supported dense, summary/q8, mixed q4
head64/head128 and representative metadata layouts. No unsupported head128
diagnostic attention is enabled. All audited native H2D operations now use
the completed helper; default-stream KV fills share one checked setup fence.

Legacy dense append now waits for its decode length producer, then publishes
FP32/RoPE length inside the append kernel rather than copying a stack value
asynchronously. FP16 append finishes default-stream copies before return.
Five transfer-order tests and four cast/RoPE/reset tests pass on M40, including
queued length publication, odd widths and overflow. Explicit-position append
remains enqueue-only on head64/head128, verified by launch/copy/wait counters.
No new waits enter production explicit-position append.

Broader cuBLAS-on ownership/projection/forward/profile checks pass. The
cuBLAS-off run exposed four old fast-fits graph fixture failures; a pre-fix
`3818104` worktree reproduces all four. Their dense-reference state was also
implicit in isolated runs. A separate test-only correction gives each case
explicit dense KV, the required compiled cuBLAS backend, and explicit hardware
invocation. Runtime graphs remain experimental/off with unchanged coverage.

The corrected forward suite passes all 24 cuBLAS-on cases, including explicit
graph inclusion, and the raw graph case now passes alone. The cuBLAS-off
matrix passes 43 supported transfer/attention/KV/forward/projection/MLP cases;
its four fast-fits graph fixtures are not compiled without their backend.
Both CUDA/server Clippy configurations pass. An explicit graph request with
the GPU hidden fails with `m40llm_create_context returned null`, as required,
rather than reporting a successful no-device skip. Full non-CUDA and
NVCC/no-device warning-as-error suites, non-CUDA Clippy and fmt also pass.
GPU-less CI does not establish hardware correctness.

The clean `7245d26` release check uses five alternating serial HTTP pairs with
the pinned Ollama 0.40.11. m40 median is 31.937 tok/s versus Ollama 43.163,
with all 65 prompt and 512 generated IDs matching C11. Both speed objectives
remain unmet: 53.06 tok/s and 1.30x fresh Ollama, now 56.1115 tok/s. The prior
31.850 result is not a contemporaneous paired control; do not claim a tiny
upload-fix speedup. No inference overlapped tests/builds or other inference.

Canonical ChatML CLI arithmetic/OK/config outputs are `4`/`OK`/`37` in default
compressed top8 and dense `off`, six passes. The initial user-only Auto
exact-OK/config failures exactly reproduce archived native behavior in both
modes; retain them as dense-failed/inconclusive, not new compression failures.
Full commands, startup configs, traces, timing, memory and limitations are in
the [upload-ordering report](experiments/2026-10-25-upload-ordering.md).
The isolated Ollama runner is stopped and final GPU use is 0 MiB / 0%.

C20 was split into dependency-ordered commits:

1. `9674452`, `feat(infer): define diagnostic layer masks`: checked interior IDs, original
   chronological execution order, empty-mask behavior and CPU tests.
2. `04739c3`, `feat(infer): apply masks to full-model forward`: model-local setup before
   KV allocation, all full-model forward paths, active-index scratch routing,
   and CUDA parity/skip tests. Depends on the mask contract.
3. `261faf9`, `test(quality): evaluate masked layer variants`: candidate-only CLI masks,
   transform/memory identity, and matched forced-prefix step timing. Depends
   on forward wiring and C19's frozen evaluation runner.
4. `docs(perf): record layer-mask tradeoffs`: fixed one/two/three interior
   layer diagnostics against the unchanged reference, with quality, loss,
   logit drift and scoped timing/memory. Depends on verified implementation.
   No ranking or compact export in C20; those remain C24/C25.

The mask contract is committed in `9674452`. Forward integration covers
sequential, batched, packed-prefill and device-parameter paths with original
KV IDs; session graphs and all-slot packed-then-compress reconstruction are
explicitly rejected for nonempty masks. See [the mask contract](layer_masks.md).
The broader validation exposed an independent minimal-forward fixture using
dense head_dim=4 KV with the compressed default; `fe48028` makes that reference
explicitly dense. Forward integration is committed in `04739c3`; 45 CUDA
cuBLAS-on tests, 21 cuBLAS-off tests, the full non-CUDA warning-as-error suite
and both Clippy lanes pass. The runner now adds candidate-only masks and
matched forced-prefix timing. Before any mask result, the fixed probes are
none, `[17]`, `[17,18]`, `[17,18,19]`, both frozen splits, five alternating
step-timing pairs per split. All four reports complete with clean source
manifests; references match C19, paired predictions repeat, and GPU use returns
to zero. These forced-prefix rates are not generated/HTTP TPS or a promoted
exact-engine improvement. Source weights and physical KV capacity stay fixed.

C19 was split into dependency-ordered commits:

1. `d1294fc`, `test(quality): freeze pruning evaluation splits`: immutable authored V1
   data, split/leakage validation, deterministic task rubrics and tests.
2. `c32ec4e`, `feat(eval): score teacher-forced tokens`: stable token-weighted NLL/PPL,
   full-logit drift, strict-prefix prediction, and CUDA session integration
   tests. Depends on the frozen scoring/data contract.
3. `f974162`, `test(quality): evaluate explicit model variants`: release CLI with model/
   corpus/config hashes, serial dense-reference comparisons, output/task/loss/
   memory/timing records. Depends on the loss evaluator and existing sessions.
4. `docs(perf): record dense quality reference`: evaluate both frozen splits,
   retain dense failures as capability evidence, and publish the exact current
   C19 completion status before C20/C21. No model tuning in this unit.

Corpus/scoring is frozen in `d1294fc`, before dense results. Its SHA-256 is
`2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38`.
Teacher-forced CPU loss and CUDA session comparison are committed in `c32ec4e`,
including strict prefixes, token weighting, full-vocabulary drift, cache reset,
and shared-KV rejection. See [the evaluation contract](model_quality.md).
The explicit-model release runner now adds model/corpus/config identity,
reference-dependent scoring, full-logit loss comparisons, actual token IDs,
timing boundaries, and scoped memory accounting. It preserves both frozen
splits and rejects tokenizer mismatch, truncation, or implicit compressed KV.
The release self-reference run matches all 12 generated traces and all 203
teacher-forced tokens exactly, including zero full-logit drift. Fixed-rubric
passes are 3/6 calibration and 4/6 held-out; five dense-failed tasks remain
inconclusive for pruning quality. Token-weighted PPL is 31.556848 calibration
and 39.612159 held-out. These small synthetic passages are not a standard
benchmark claim. Single in-process timing observations are not HTTP speed
promotion evidence. Both models are released, with zero tracked/device use.

## N-Gram Feasibility Checkpoint: 2026-10-25

C11A is complete. The selected actual C11 browser trace gives at most 4.949%
optimistic estimated HTTP throughput improvement, below the fixed 10% gate.
Limits8/16 both replay in 485 visited rounds, with 28 accepted draft tokens;
33/43 eligible proposals reject immediately. No GPU verifier or measured
speedup is claimed. See the [study and raw artifacts](experiments/2026-10-25-ngram-feasibility.md).

Actual fixed regression captures are stable across two runs; the canonical
smokes return `4`, `OK`, `37`. CPU-only release reproduces acceptance/replay/cost
results from the CUDA-feature CPU path. Lookup/study tests, non-CUDA
warning-as-error tests, both Clippy lanes, and formatting pass. GPU allocation
returns to zero after capture. Runtime/KV defaults and the measured engine
configuration are unchanged.

C18 records this negative gate and the two unmet speed objectives in the
phase audit above. Do not build a verifier or reopen layout/lookup tuning to
force the screen to pass; C19 is next.

C11A was split into dependency-ordered commits:

1. `fe392cd`, `feat(decode): add bounded ngram lookup`: history-only borrowed drafts,
   longest/recency matching, stop/budget bounds, and CPU behavioral tests.
2. `739c195`, `test(bench): study ngram acceptance offline`: position/round replay,
   actual-trace validation and regression capture, hypothetical cost sensitivity,
   and the formal screening gate. Depends on the lookup and C11 artifacts.
3. `docs(perf): record ngram feasibility decision`: release CPU analysis and
   raw results, with proceed/postpone status before conditional verifier work.
   Depends on the tested study tool and selected actual traces.

## Exact Kernel Checkpoint: 2026-10-25

C01-C11 are complete, including conditional C10A evaluation and the preserved
failure removals. The selected opt-in benchmark remains native Q8 plus C07
parallel head128 attention, partitions=8/threads=256, with preparation off.
Global runtime/KV defaults are unchanged.

Fresh five-pair release HTTP confirmation at `f5cda0d` measures 31.850 tok/s
m40-llm versus 43.032 Ollama. Every row has 65 prompt IDs, 512 generated tokens,
length stopping, and verified fresh KV. The actual m40 trace matches the prior
qualified trace exactly. Both objectives remain unmet: 53.06 tok/s and 1.30x
fresh Ollama, now 55.9416 tok/s. See the [C11 report and raw records](experiments/2026-10-25-exact-kernel-checkpoint.md).

The slower prepared fused MLP is removed in `9044eea`, after preservation in
`e2dbc61` and measurements in `2e2bb5c`. Native fusion and shape/pointer checks
remain. The slower tiled coalesced-lane mapping is removed in `f5cda0d`, after
preservation in `bb4d30e`; vector-only research remains. Remaining preparation
paths stay opt-in, with no qualified HTTP win. CUDA/cuBLAS-on/off, non-CUDA
warning-as-error, Clippy, and formatting checks pass. Benchmark processes are
stopped; final GPU allocation is zero.

This selected trace is the source for the completed C11A study above. Its
negative >=10% gate postpones C12-C17; C18/C19 follow without another layout
tuning round.

## Prepared Q8 HTTP History: 2026-10-25

C01-C10 and C10A implementation/HTTP validation are complete. No prepared
candidate passes the combined timing and actual-ID parity gate. Keep the
qualified C07 native-Q8/split8/256 attention benchmark configuration and
preparation off. Runtime/KV defaults are unchanged; both overall speed goals
remain unmet.

Five serial release pairs per subset, each with 65 prompt IDs and 512 generated
IDs, give:

| Prepared families | Native / prepared HTTP tok/s | Actual IDs equal | Decision |
| --- | --- | --- | --- |
| All | 31.946 / 11.433 | Yes | Timing regression |
| QKV/output/down/vocabulary; native MLP | 31.907 / 29.765 | Yes | Timing regression |
| Down/vocabulary only | 31.847 / 35.837 | No | Timing lead, parity rejection |

The last subset first diverges at zero-based generated step 310. It is not a
new qualified best result. Canonical arithmetic/OK/config smokes pass native
and both reduced subsets; the first run's user-only OK/config reference
failures remain inconclusive. The three runs never overlap; the isolated
oracle has been stopped and GPU allocation returned to zero.

See the [full release report and raw artifacts](experiments/2026-10-25-q8-prepared-http.md)
for confidence intervals, timing boundaries, memory/setup costs, and limitations.
Implementation/test contracts and earlier correctness canaries remain in
[Q8 execution weights](q8_execution_weights.md).

C10A commits preserve cache/loading (`11f2fb6`), fused QKV (`4db3c1e`), fused
MLP (`e2dbc61`), greedy vocabulary (`b76177d`), immutable families (`3d176fe`),
and HTTP comparison (`26bc810`). `e74ed87` fixes only quality-smoke formatting,
without changing the measured browser prompt. CUDA family/ownership/forward/
attention tests, cuBLAS on/off checks, non-CUDA warning-as-error tests,
Clippy, and protocol/CLI tests pass.

This checkpoint led to C11's keep/remove decision and exact-kernel confirmation
above. It did not authorize another layout/geometry search or reinterpret
isolated gains as qualified generation speedups.

### C11 Commit Sequence

1. Completed `9044eea`, `refactor(cuda): remove failed prepared MLP`: remove
   the preserved
   `e2dbc61` fused consumer after the negative `2e2bb5c` release gate, retaining
   native fusion, shape/pointer checks, and cache coexistence tests.
2. Completed `f5cda0d`, `refactor(cuda): drop coalesced Q8 experiment`: remove
   the losing lane
   mapping preserved in `bb4d30e`, keeping useful vector-tiled projection
   research. This is independent of MLP removal.
3. Completed in this checkpoint,
   `docs(perf): record exact Qwen kernel results`: rerun the selected native
   configuration through the controlled release HTTP comparison after cleanup
   and reconcile all C07-C10A keep/remove decisions. Depends on both removals.
4. Completed `739c195`, `test(bench): study ngram acceptance offline`: deterministic history-only
   lookup and simulation over the selected configuration's actual traces, with
   cost sensitivity and the formal proceed/postpone gate. Depends on C11.

The prepared MLP consumer is now removed. CUDA native/cache fusion, family,
prepared QKV/greedy, and MLP wrapper tests pass, as do non-CUDA warning-as-error
tests and CUDA/server all-targets Clippy. The
remaining cache and split/vector consumers stay opt-in research, not a promoted
configuration. No new layout tuning or default change is authorized here.
The coalesced-lane mapping preserved in `bb4d30e` is now removed from CUDA,
Rust launch descriptors, and the microbenchmark CLI. Its original measurements
remain unchanged. Retained vector/library/cache tests, the 11-case attention
grid, 24-case forward suite, projection/MLP wrappers, cuBLAS-off checks,
non-CUDA warning-as-error tests, and CUDA/server Clippy pass. The controlled
native/Ollama release confirmation is complete in the C11 report above.

## Committed Checkpoint: 2026-10-24

The following checkpoint was moved from `AGENTS.md` at `81948df`. Paths in
backticks are repository-root relative. Earlier implementation history is in
[archived progress](archive/agent-progress-through-2026-10-24.md); authoritative
measurements and raw records remain in [performance history](perf_baselines.md)
and `docs/experiments/`.

- C01/C02 are complete: MemPalace practice and opt-in buffered HTTP metrics.
- C03 now has a release-only `compare_ollama` example with identity checks,
  verified cache resets, paired trials/bootstrap, and untimed actual token traces.
- The controlled five-pair baseline is 23.735 HTTP E2E tok/s m40-llm versus
  43.035 for prebuilt Ollama 0.40.11 on M40. Both use the same Qwen Q8_0 blob,
  65 shared ChatML IDs, dense FP16 KV, context 1024, and 512 generated IDs.
  All rows are length-limited/cache-valid. See `docs/controlled_benchmark.md`
  and the dated raw records in `docs/perf_baselines.md`.
- The initial 66-versus-65-ID attempt was rejected. `2f4d7ca` fixes extra BOS
  insertion for preformatted Qwen Raw/Auto prompts; standard automatic Qwen
  chat formatting is unchanged.
- C04 now records buffered per-layer CUDA-event spans with session-owned reused
  events and nonblocking resolution at existing logits completion. CUDA tests
  preserve results and launch/sync/copy/allocation counts, including two streams.
  Five same-workload HTTP pairs show 23.780 tok/s off versus 23.667 on, a 0.47%
  profiling reduction with identical generated IDs. Mid-token FFN spans total
  18.17 ms and attention 11.42 ms; late attention rises to 20.39 ms. First-capture
  uncovered wall time is not ordinary CPU-cost evidence. See the dated profile
  record in `docs/perf_baselines.md`.
- C05's optional standalone CUPTI preflight enumerates 154 M40 metrics, but
  actual legacy event-group creation and domain enumeration both fail with
  `CUPTI_ERROR_NOT_SUPPORTED` on driver 580.159.04/CUPTI 12.8.90, even as root.
  Traffic/occupancy measurements remain unavailable, not estimated counters.
  See `docs/maxwell_counters.md`; ordinary builds do not depend on CUPTI.
- C06 now consolidates the raw baseline, sampled stage medians, counter
  limitations, actual tensor inventory, and explicitly estimated Amdahl limits
  in `docs/experiments/2026-10-24-qwen-baseline-checkpoint.md`. FFN parameters
  are 78.90%; gate/up is the larger fixed cost, while attention grows with
  context. The report's jq program reconciles source records and calculations.
- C07 compares eight exact head128 geometries with reusable RAII split scratch
  and stable softmax merge. Split8/256 passes the paired HTTP gate: 31.852
  tok/s versus 23.774, +33.98% with 95% interval [33.868%,34.203%]. Actual
  512-token traces match. Keep it as the selected opt-in benchmark config:
  `M40LLM_ATTN_HEAD128_KERNEL=parallel`, partitions=8, threads=256; full env
  names/commands and short-context limitations are in `docs/diagnostics.md`.
  Runtime/graph/scheduler/compressed-KV defaults are unchanged. Scratch is
  65 KiB per KV cache, not per layer/token. The separate middle/late stage
  diagnostic measures attention at 0.791/0.973 ms and FFN at about 64% of the
  middle GPU interval union. See `docs/experiments/2026-10-24-head128-attention.md`.
  Canonical-system Qwen arithmetic/OK/config smokes pass both kernels; the
  earlier user-only reference EOS is retained, not called an attention bug.
- C08's allocation-error prerequisite now releases the CUDA context mutex
  before querying error properties. An impossible-allocation hardware test
  verifies contextual errors, unchanged accounting, live data, and subsequent
  allocation instead of deadlocking before partial-buffer cleanup.
- C08 now provides benchmark-only Q8 source identities/bounded views, explicit
  column/split/tiled allocation plans, and typed RAII host/device ownership in
  `infer::execution_weights`. Identity includes model SHA, tensor, offset,
  dtype/shape, and execution layout rather than device addresses. CPU failure
  fakes plus a real second-plane CUDA allocation failure verify cleanup; device
  roundtrip and tail-shaped projection tests pass. Production GGUF views,
  materialized weights, loader/cache, dispatch, and defaults are unchanged.
  See `docs/q8_execution_weights.md` for ownership contracts.
- C09 now copies original Q8 scale/value bytes into separate planes and tests
  four bounded Maxwell configurations with a benchmark-only async consumer.
  Real Qwen release projection medians improve gate 0.229->0.139 ms, down
  0.191->0.143 ms, vocabulary 2.584->1.624 ms; this is not an HTTP speedup.
  K/V needed a longer ten-pair check and confirms only about 14% improvement.
  Tail/CPU/GPU parity, invalid-call rejection, CUDA grid/wrappers, non-CUDA
  warning-as-error tests, and Clippy pass. Repacking is lossless; retaining
  original plus execution buffers doubles tensor device payload. Defaults,
  fused generation kernels, and model loader/cache remain unchanged. See
  `docs/experiments/2026-10-24-q8-separate-projections.md` for raw trials,
  memory/setup costs, clock/measurement limitations, and negative configs.
- C10 completes independent input-tiled Q8 packing and twelve CUDA cases:
  widths4/8/16, threads128/256, vector/coalesced-lane mappings. All source
  bytes/scales and tails are preserved; padded output guards and CPU/GPU parity
  pass. Best vector M1 gate/down/vocabulary are 0.141/0.161/1.661 ms; refreshed
  split planes are 0.140/0.142/1.623 ms. Vector rows=64 Q/down are promising
  at 1.244/6.480 ms, but no model/HTTP speedup is claimed. Coalesced lanes
  were slower and did not reduce registers; code/results are preserved before
  any later removal. Width4 adds 5.88% storage, and tiled preparation is slower.
  All 2,460 measured intervals are serial; raw records, compiler output,
  configuration bounds, and clock limits are in
  `docs/experiments/2026-10-24-q8-input-tiled.md`. The C10 tuning budget is closed.
  CUDA parity with cuBLAS on/off, forward/ownership, non-CUDA warning-as-error
  tests, and Clippy pass. Loader/cache, generation dispatch, and defaults do not change.
- Next: conditional C10A useful per-family integration, not another layout
  tuning round. Preserve useful fused operations and explicitly budget duplicate
  source/prepared weights, then require model parity and HTTP E2E gates. Use the selected
  C07 benchmark config for subsequent controls, require parity and repeated
  HTTP E2E gates, and bound experiments before C11/C11A. Missing counters or
  the original speed target do not block the later n-gram feasibility decision
  and separate model-changing research. No further attention tuning round yet.
- C10A preparation tests exposed pre-existing uploaded-weight accounting drift
  from `bd9b68fd`: final context drop freed native GPU weights but never removed
  their allocation records or subtracted tracked bytes. Final-owner cleanup now
  accounts only successful frees, with a clone/readback/repeated-model regression.
  This is an independent correction, not a kernel performance result.
- The unchanged-model target requires both 53.06 E2E tok/s and 1.30x the fresh
  Ollama median, currently 55.945 tok/s. Neither is reached. The short benchmark
  explicitly uses dense `off`; the project's compressed-KV defaults are unchanged.
