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

## Inspected Implementation Boundary

After compound checkpoint `0516e71`, source inspection finds that current
packed prefill starts at position zero, attends to the packed prompt's FP32
K/V and returns only the final hidden row. It cannot verify a continuation
against existing dense FP16 KV by simply changing its token slice. There is
also no sequence truncate/rollback API. Reusing either assumption would make
the experiment invalid.

Use one bounded dense-only continuation branch of the existing packed
forward routine, leaving the existing prefill entrypoints unchanged. Batch
all projections/FFN over actual candidate rows, append at absolute prefix
positions, and run each causal query through the existing dense FP16 KV
attention kernel with its own prefix-plus-row length. Return all hidden
rows. This shares projection work without inventing another attention
algorithm; per-row attention/logits overhead remains in measured costs.
Use the existing configured Q8 prefill backend, including every dequantization
and synchronization, not a cached full FP32 model.

A research-only verification session exclusively borrows a mutable model
with explicitly allocated dense KV. It owns reusable hidden/logits/norm/token
scratch, cached next-token prediction and committed length. Each attempt
holds a mutable session borrow. Commit only the verified prefix; a bounded
dense truncate operation zeroes rejected K/V and publishes per-layer lengths.
On failure or abandoned attempts, drain producers and roll back to the
original prefix. Reject compressed KV, invalid bounds and poisoned sessions;
do not touch server scheduling or create a compressed-state transaction.

Compare serial and batched target greedy IDs, per-position diagnostic logits,
prefix KV and lengths. Include wrong candidates, partial/full/zero commit,
drop rollback and capacity failure. The benchmark uses only known greedy
continuations as perfect drafts, then calculates real-drafter break-even
curves. No neural drafter or EAGLE code is authorized by a timing result.
