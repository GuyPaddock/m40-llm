# Fixed rotated activation preparation experiment

Compare staged sign multiplication, normalized 128-coordinate Hadamard
rotation and Q8_1 encoding with one fused producer. Both paths write the same
canonical FP32 activation and the same Q8_1 bytes. Both feed the same existing
Q8_1 gate/up/SwiGLU consumer. This is an isolated rotated fixture experiment;
the selected Qwen model does not gain a rotation in its runtime graph.

## Shapes and ownership

Use K2048 and K11008 with 1, 2, 4 and 8 distinct retained activation rows at
layers 0, 17 and 35. K2048 takes normalized FFN inputs; K11008 takes the
captured SwiGLU inputs to down projection. Use one deterministic seed-zero
sign vector per width, shared by weights and inputs. Normalize after the
butterfly, identically in staged and fused code.

The current `Q8ActivationShape` supports rows 1 and 4. Preserve that runtime
contract. The staged quantizer treats the contiguous, block-aligned activation
as one row of `rows*K` values; this produces the same block order without
crossing a real row boundary. Consume one row at a time through the existing
Q8_1 API, identically in both arms. Report that consumer schedule explicitly.

Use invocation-owned signs, signed scratch, canonical FP32 and Q8_1 buffers.
The producer recomputes every call; there is no pointer-keyed cache or cross-
invocation reuse. Keep all buffers resident in both timing arms. Reject
incompatible shapes and writable aliases. Callers order stream dependencies
and serialize reuse of one owner; independent owners may use separate streams.

## Fixed implementation

Use 128 threads per rotation block. Staged execution launches a sign kernel,
a rotation kernel, and the existing Q8_1 quantizer. Fused execution applies the
sign on load and writes canonical FP32 plus Q8_1 from the rotation kernel.
Factor the existing per-warp encoder into a shared device function, preserving
its FP32 reduction, rounding and stored FP16 scale/sum. Retain its upstream
license notices. No geometry sweep, Tensor Core or DP4A path is introduced.

For the unchanged consumer, freeze 256 gate and up output channels. At K2048,
take the first 256 original gate/up rows; at K11008, use two disjoint 256-row
slices of the original down matrix as synthetic gate/up projections. Rotate
their dequantized Q8 values using the same signs and block transform, then fit
native Q8_0 blocks with absmax/127, round-to-nearest with ties away from zero,
and FP16 scale storage. Record source, rotated encoding and quantization error.
These matched synthetic consumers establish cost and byte reuse, not a model
architecture or accuracy result.

## Checks and timing gate

Validate rotation against an independent FP64 Walsh matrix and encoding against
the existing CPU Q8_1 reference applied to validated FP32 rotation outputs.
Require staged/fused canonical and encoded bytes, and consumed output bits,
to match. Include zero/tied inputs, changed inputs at reused addresses, shape
changes through separate owners, sequential calls, separate owners, both
stream directions, guards, immutable signs/weights and allocation cleanup.

Use a clean release binary and the exact retained capture environment. Record
producer-only and producer-plus-consumer latency separately, with one-second
GPU warmup, five alternating pairs and the common bounded variation and
confirmation rules. The complete operation needs at least
`max(3%, 2*variation)` gain and a positive confidence bound. Charge required
launches and dependencies. Report setup, payload/scratch/residency, GPU events,
host submission/completion, and logical traffic separately from unavailable
DRAM counters.

No runtime integration follows without an actual compatible rotated model
consumer. A successful isolated result can therefore finish with a priced
prerequisite report; an unprofitable complete operation stops the candidate.

## Atomic units

1. `feat(cuda): fuse rotated activation preparation`: owned producer API,
   shared Q8_1 encoding and numerical/lifetime tests.
2. `feat(bench): measure rotated preparation fusion`: retained activations,
   matched rotated weight fixtures and the two timing boundaries.
3. `docs(perf): record rotated preparation results`: raw evidence and the
   measured decision, including the missing-model prerequisite if applicable.
