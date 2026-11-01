# Complete-generation FFN observer

`model_quality --capture-ffn-inputs PATH --capture-ffn-complete-generations`
enables the [fixed study](2026-10-31-selector-predictability-protocol.md).
It requires the pinned original model and corpus hashes, calibration split,
unchanged six task limits, and existing explicit native/dense capture control.
It observes layers 0/16/17/34/35 with a 160-row bound and 256 MiB FP32 budget.
Ordinary `--capture-ffn-inputs` retains its four-row, three-layer defaults.

The collector accepts at most 511 consumed inputs for its existing maximum
512 emitted tokens; callers still reserve and enforce an explicit byte budget.
The JSON parser limit is now 512 MiB for longer multi-layer observations.
Complete-generation export uses compact JSON and verifies that no task was
truncated. The prompt-terminal state is excluded and must be charged as dense
bootstrap; the last emitted token is unconsumed. Inactive/active generations
must preserve actual IDs, outputs, counts and stop reasons.

Validation passes nine collector tests, thirteen non-CUDA CLI/example tests,
five CUDA-enabled capture/argument tests and both serial M40 observer parity
tests. CUDA/server all-target Clippy, formatting and whitespace checks pass.
The new long-sequence test retains late rows beyond the old bounds, checks
task positions and paired values, enforces reservation limits and rejects a
valid prefix artifact when complete-generation evidence is requested.

The first collector run failed because an old invalid-input test still treated
33 rows as forbidden. Its boundary now checks 512, beyond the new legal limit;
the complete suite then passes. The failed log is retained alongside passing
logs in `2026-10-31-ffn-full-*.log.gz`. No full-model observation or selector
analysis result is claimed by this implementation checkpoint.
