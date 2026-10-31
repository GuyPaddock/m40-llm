# Eight-Value FP32 FFN Tasks

## Source Evidence

The qualified `gate_up_split_f32x4` in `cuda/q8_ffn.cuh` assigns four adjacent
weights to each task. It loads one `float4` activation, one packed 32-bit
payload from each gate/up matrix, and two FP16 scales. At K=2048, each of
128 threads processes four tasks, with two FP32 accumulators. Gate/up/SwiGLU
still occupies 41.11% of the qualified middle-token GPU-stage profile.

The existing `q8_separate_accumulate` in `cuda/q8_execution.cuh` reads eight
payload bytes per task using `uint2`. That down/vocabulary consumer remains
unchanged; it only supplies evidence that aligned eight-byte payload loads
already fit the lossless execution representation.

One eight-value task could replace two four-value tasks. At source level,
the number of warp task groups per output would fall from 16 to 8, reducing
gate/up payload load instructions and scale-load/conversion groups from 32
to 16 each. Activation `float4` loads remain 16. An aligned 32-byte-sector
request model predicts input requests increasing 256 to 512, payload requests
remaining 128 and scale requests falling 32 to 16 per output. These are
issued-request footprints, not physical DRAM bytes or measured bandwidth.
Cached reuse, registers and instruction dependencies may negate the benefit.

## One Independent Candidate

Keep the qualified one-output, 128-thread CTA and existing separate planes.
Use eight consecutive FP32 activations per task, two aligned `float4` loads,
and one aligned `uint2` gate/up payload load per matrix. Load one FP16 scale
per matrix for the eight weights; retain per-weight FP32 scaling and FMA,
two accumulators, one publication barrier, the 128-partial reduction tree
and the original precise SwiGLU expression. Support rows 1/4 and K%32=0.
Reject payload pointers not aligned to eight bytes; keep input alignment 16.

Summation **ownership changes**: a thread accumulates eight values before
advancing by 128 tasks, rather than four. Do not promise control bit equality.
This is a conventional FP32 arithmetic-order experiment, not activation
quantization, reduced precision or a change to weights/model mathematics.
The original plan's unchanged actual-512-token requirement still applies.
Use the same unchanged CPU tolerance and report gate/up, SwiGLU and complete
down drift against the four-value control before timing. Capture and ordinary
candidate execution must match bits. Do not weaken old variants' bit checks.

Do not combine this with cohorts, virtual warps, per-thread extra outputs,
looping workers, joint packing, fixed-K, scale factoring, byte conversion or
cache hints. The control and prepared down/vocabulary remain untouched.
There is no task-width/block-size/unroll/register-cap rescue if it loses.
Inspect compiled loads, registers, spills and the unchanged control words.
No occupancy, bandwidth or launch-gap attribution follows from source alone.

## Atomic Delivery And Gates

1. `docs(research): bound eight-value FFN work`: commit this hypothesis and
   checkpoint before code.
2. `feat(cuda): probe eight-value FP32 FFN`: one benchmark-only CUDA/API
   entry, shared validation, behavioral tests and exclusive shared benchmark
   mode. Depends on step 1.
3. `docs(perf): record eight-value FFN screen`: raw paired release results,
   compiled evidence, numerical/ownership audit and measured decision.
   Depends on verified step 2.

Use frozen M02 actual gate/up/down tensors from layers 0/17/35 and distinct
rows 1/4. Retain the same prepared pair and guarded scratch in both arms.
Test signed extremes/every byte position, zero scales, ragged legal widths,
tails, alignment/alias rejection, ordered producers, unchanged input/weight
planes, scratch reuse, absence of enqueue alloc/copy/sync and cleanup with
cuBLAS on/off. Check independent FP64 gate/up/SwiGLU/complete references,
finite results and the existing `1e-3 + 2e-5 * abs(reference)` tolerance.

Run five alternating pairs, ten iterations per gate/up, isolated-down and
complete stage, with bounded 1000/100-ms control/candidate warmup. Require
>=5% complete one-request gain and positive paired 95% lower bound at all
three layers before model-local runtime integration; four-row work is
reported separately. Existing fresh confirmation rules handle variation or
borderline results. A primitive/micro pass does not establish quality.

Only a qualifying micro result permits unchanged actual 512 IDs, frozen
dense-valid tasks/full logits, incremental HTTP and fresh official Ollama
qualification. FP32 drift failing any original guardrail blocks promotion.
Both **53.06 HTTP tok/s** and **1.30x fresh official Ollama** remain mandatory,
currently 56.037 tok/s for the latter. Runtime defaults stay unchanged.
