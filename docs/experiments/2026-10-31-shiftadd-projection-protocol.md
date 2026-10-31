# Matched ShiftAdd projection screen

Freeze this protocol before measuring the candidate. Encoder `a6ef091`,
CUDA consumer `07e7fd7` and shared Q4 fixture `b6d57cd` are prerequisites.

Use the existing `q8_execution_bench` FFN capture checks and original Qwen
Q8_0 model. For layers 0, 17 and 35, fit all gate/up/down output channels
from the same original Q8 values. Gate/up consume captured normalized inputs;
down consumes captured SwiGLU inputs. Use distinct rows at batches 1/2/4/8.
Fitting is offline, deterministic and performed once per projection.

Compare four fixed arms without a tuning sweep:

- SA4_128 FP32 sign-add, four warps per block, one launch for all rows.
- Native Q4_0 decode, one launch per row.
- Native Q8_0 decode, one launch per row.
- Prepared Q8 separate planes with packed activation loads, four columns
  and 128 threads, one launch for all rows.

All weights and guarded input/output buffers stay resident across arms.
No activation quantization is included. Native arms explicitly use the decode
stream from the frozen capture environment. Q4_K remains unavailable.

Before and after timing, compare each arm against independent FP64 dots of
its encoded weights. Use operation-count FP32 error bounds, including the
sum of absolute plane contributions for ShiftAdd. Separately report weight
and projection fit errors against the original Q8 values. These errors do
not establish model quality. Check finite outputs, guards, immutable inputs
and weights, actual dispatch counts, allocation-free enqueue and cleanup.

Use release builds from a clean commit, reusable events, 100 operations per
sample and at least one second of GPU warmup per arm. Compare each control
to ShiftAdd using five alternating pairs. Extend once to ten when throughput
MAD/median exceeds 5%; unresolved variation prevents a pass. Reuse paired
bootstrap with 10,000 resamples and seed zero. The projection gate is
`max(3%, 2 * variation)` with a positive lower confidence bound. A result
within one percentage point of the gate, or a passing point estimate with
a nonpositive interval, receives one fresh twenty-pair confirmation.

Record event and host timing, source/executable/configuration hashes, fitting
and allocation/upload costs, logical weight payload, scales and amplitude
bytes, scratch, retained model/prepared weights and peak tracked residency.
Sector counts are estimates with explicit cache assumptions; hardware DRAM
and dynamic instruction counters remain unavailable. Inspect the final
release kernel's SASS and register/local-memory use.

Advance to complete FFN comparison if batch-one projections consistently
beat both Q4 and prepared Q8 under the latency gate. A separate memory route
requires at least 3% payload saving versus Q4, no greater weight or captured
projection squared error in any sampled tensor, and no more than 5% latency
regression versus Q4. This is a bounded engineering gate, not a paper claim.
Mixed results remain scoped to the passing shapes; no general FFN claim.
If neither route passes, retain the negative result and close dependent
integration/quality work. If either passes, require complete FFN timing
against existing fused controls and model-level quality before adoption.

The next atomic unit adds the benchmark and its behavioral checks with draft
message `feat(research): compare ShiftAdd projections`. A later evidence-only
commit records measured results and the gate decision. No runtime defaults
or model weights change.
