# Clock-CSV Diagnostic Attempt

Clean source `b63512e` completed six excluded warmup checks, then stopped
before the first primary diagnostic pair. The new validator expected bare
clock strings, but the shared existing `nvidia-smi` query returns units.
The actual snapshot showed P0,1113 MHz SM and3004 MHz memory, matching the
required clocks. This is a parser defect, not a GPU-condition failure or a
performance result. Do not change the shared query or qualification gates.

Preserve the [incomplete report](2026-10-27-direct-q8-clock-attempt.jsonl)
and [original log](2026-10-27-direct-q8-clock-attempt.log). The implementation
and executable hash remain identified by the raw manifest. No pair or
rejection-cycle result exists in this attempt; no samples can be pooled.

Normalize clock presentation in the new diagnostic validator only. Verify
unit-bearing and bare-number inputs and reject changed clocks, UUID or
P-state. Rebuild with the same CUDA settings and use a new report path for
the unchanged bounded session. Keep the original rejected SGEMM verifier,
model, default settings, observer allowance and stop conditions unchanged.
