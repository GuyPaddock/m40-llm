# Exact Multi-Token Verification On Maxwell

## Scope And Order

User-requested follow-up after the single compound FFN checkpoint. Preserve
the qualified exact runtime and dense reference workload. No draft model,
EAGLE integration, approximate attention, pruning or scheduling changes.
This is a separate study, not a substitute for the original HTTP targets.

Measure candidate lengths 2, 4 and 8 against the same number of serial target
decode steps. Start with an ideal perfect drafter using frozen greedy token
traces. Include verifier setup, causal attention, all projections/logits,
KV transaction/commit or rollback, launches and dependencies in costs.
Do not count prompt echo or repeated one-row calls as batched verification.

## Correctness And Cost Contract

- Verification uses absolute positions and correct causal masks. Compare
  per-position logits and exact greedy target IDs against serial decoding.
- Every accepted prefix commits exactly once; rejected suffixes cannot
  survive in lengths, KV entries or compression metadata. Cover zero, partial
  and full acceptance, rejection at each candidate position, EOT, capacity
  boundaries, independent session ownership and cleanup.
- First establish a dense, batch-one target baseline. Any compressed-KV
  extension needs a separate transaction/state equivalence test; never silently
  substitute dense semantics or claim it has been validated.
- Report serial and verified wall/GPU time, peak/resident/scratch bytes,
  target passes and output IDs. Measure lengths independently, alternating
  at least five pairs under the controlled workload, same token counts and
  stop reasons. Account for tail lengths and warm versus setup costs.
- Report perfect-drafter E2E speedup, then break-even drafter cost and
  acceptance curves. Use measured accepted-prefix/rejection costs rather
  than assuming a full batch is accepted. Include correction-token work.
- A useful upper bound must clear the existing variation-aware HTTP gate,
  at least max(3%, twice variation), with positive paired confidence. If even
  the ideal drafter cannot, stop without EAGLE or a neural drafter.

## Dependency-Ordered Atomic Delivery

1. `docs(research): bound exact verification study`: inspect existing packed
   forward/logits and KV APIs; record actual ownership/rollback design and
   controls before implementation. Depends on compound checkpoint closure.
2. `feat(infer): add transactional verification`: implement only required
   bounded causal target verification and KV commit/rollback with behavioral
   parity/ownership tests; keep generation defaults unchanged.
3. `test(bench): measure perfect draft verification`: extend existing tools
   with real rows 2/4/8, frozen greedy traces, serial controls, allocation and
   full timing accounting. Depends on verified transaction/forward tests.
4. `docs(perf): record verification feasibility`: preserve raw trials,
   confidence/variation, correctness, memory and break-even curves; explicitly
   decide whether any real drafter is justified. No automatic EAGLE follow-up.

Keep failed implementations committed before separate removal. This is a
plan, not evidence that verification, rollback or a drafter already works.
