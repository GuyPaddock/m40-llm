# Bounded Fully Direct Four-Row Target Verifier

The user authorizes one feasibility prototype following the
[operator result](2026-10-28-four-row-operator.md) and
[updated cost assessment](2026-10-28-four-row-feasibility.md).

## Implementation Boundary

Use an explicit research-only `VerificationSession` constructor. Reuse its
exclusive mutable model borrow, dense-cache alias rejection, accepted-prefix
rules and drop rollback. Keep the existing rejected SGEMM constructor/path
unchanged. Do not add a generation selector or change runtime defaults.

The new session owns a four-row workspace. Execute layer-major: original
one-row attention norm, qualified direct QKV/bias, absolute-position RoPE,
KV append and causal attention for each row; original output projection,
residual and FFN norm; one measured shared-read four-row gate/up kernel;
four original one-row prepared-down/residual calls. Preserve per-row output
norm, vocabulary and greedy evaluation. No new CUDA kernel or alternative
schedule is authorized.

The bounded prototype requires the frozen Qwen2.5-3B Q8_0 configuration,
joint-only compound ownership, packed-warp QKV/output, original prepared
down/vocabulary, dense batch-one KV and fixed four-row dimensions. Reject
unsupported configurations rather than silently invoking generic multi-row
dequantization/SGEMM. Prompt preparation may retain the qualified packed
SGEMM prefix path shared with the serial control; verification may not.

## Correctness Before Timing

- Compare actual frozen 512-token continuation and all twelve frozen task
  continuations with ordinary qualified serial target decoding. Preserve
  actual IDs, counts and stop reasons; frozen candidates are perfect drafts,
  not outputs substituted for target predictions.
- Compare full-vocabulary logits within `1e-3 + 2e-5*abs(reference)` on the
  browser continuation. Preserve valid dense KV bits, causal lengths and
  absolute positions. Inspect every committed KV position, retained prefixes
  and cleared rejected suffixes; diagnostics are outside primary timing.
- Exercise every mismatch position 0/1/2/3, full acceptance, forced partial
  verified commits, dropped attempts, correction, invalid commits, capacity,
  stop-token truncation, invalid dimensions and aliased-session storage.
- Require unchanged original prepared identities/payloads, all 138 existing
  compiled CUDA entries, zero tracked teardown, warning-free non-CUDA builds,
  affected CUDA parity/ownership/forward checks, Clippy and formatting.
- Stop on a genuine numerical, output, KV-state or ownership failure. Preserve
  code and evidence; do not alter numerical gates or try a schedule rescue.

## Measurement And Decision

After a committed passing correctness checkpoint, collect five alternating
release pairs of qualified serial and direct-four-row resident generations
using the same 65 prompt tokens and 512 actual generated IDs. Run the GPU
alone with matching clocks and flags. No builds, tests, MemPalace or other
inference overlap timed work.

Record session setup, prefill, decode, teardown and total request time;
complete batch begin/commit costs, serial tails, cold/warm allocations,
copies, dependencies, synchronizations and observed residency. Include all
required work; exclude only benchmark instrumentation and debug readbacks.
Warm once per arm before primary timing. Report CUDA-event intervals and host
elapsed intervals separately. Do not label resident results as HTTP results.

Retain the existing variation-aware `max(3%,2*variation)` promotion floor and
positive 95% paired-bootstrap bound, 10,000 resamples/seed zero. Over 5%
variation extends once to ten pairs; continued instability stops. Borderline
estimates within one percentage point of the gate, or qualifying points with
an interval crossing zero, require twenty fresh confirmation pairs, separate
from the screen. No other search or repeated confirmation is allowed.

Measure five rejection/correction cycles at each accepted prefix 0/1/2/3,
including target work, rollback, cleanup and ordinary correction. Recalculate
finite-511-row break-even/3%/53.06/56.153149 budgets from the measured batches
and correction costs. Distinguish the cached first candidate from three
drafted guesses, candidate availability and conditional acceptance. Use no
real drafter. A successful perfect-drafter result only supports recommending
a separately reviewed draft-acceptance study. An unprofitable result closes
this implementation without kernel or scheduling rescue.

## Atomic Sequence

1. `docs(research): bound direct verifier prototype` records this contract and
   MemPalace checkpoint before implementation.
2. `feat(research): add direct four-row verification` adds the isolated host
   schedule, explicit session selection and behavioral transaction tests.
3. `test(research): verify direct target correctness` retains clean-source
   frozen IDs/logits/KV/ownership checks before primary measurements.
4. `docs(research): measure direct verifier feasibility` preserves paired
   release evidence, rejection costs, audits and the measured go/no-go model.
   Split measured evidence from calculation code if that is clearer.

Preserve the immutable 50.910426 HTTP tok/s reference, preferred joint-only
ownership, original model semantics, all acceptance gates and defaults.
Stop after measurements and assessment for review. No EAGLE, neural drafter,
speculative generation integration or new CUDA optimization follows.
