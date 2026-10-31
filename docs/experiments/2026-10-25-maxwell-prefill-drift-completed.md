# Completed Sequential Prefix Ownership

The conservative completion fix resolves the large real-model sequential
prefill ordering difference. This is correctness evidence, not a new speed
claim or full-continuation qualification.

## Matched Real-Model Probe

Release `2ca0de1`, executable SHA-256
`e938a2b0110d837d6a4c4494f6f9ea4fdcc0ee1e0e571f1bb55cc9a2a11465da`.
Model, frozen environment, prepared down/vocabulary, dense context 1024,
canonical 65 ChatML IDs and diagnostic procedure match the
[pre-fix probe](2026-10-25-maxwell-prefill-drift.md).

| Prompt path | Actual first token | Hidden max/mean diff vs corrected sequential | Full-logit max/mean diff | Top-10 overlap |
| --- | ---: | --- | --- | ---: |
| Corrected session sequential | 39814 | 0 / 0 | 0 / 0 | 10 |
| Manual no-drain, deliberately unsafe | 4498 | 54.50098 / 1.76412 | 9.95758 / 1.07233 | 6 |
| Sequential drained before embedding | 39814 | 0 / 0 | 0 / 0 | 10 |
| Greedy head at every prefix | 39814 | 0 / 0 | 0 / 0 | 10 |
| Native packed prefix | 39814 | 0.037247 / 0.002385 | 0.010689 / 0.001565 | 10 |
| Corrected session repeat | 39814 | 0 / 0 | 0 / 0 | 10 |
| Dequant SGEMM packed prefix | 39814 | 0.031525 / 0.002323 | 0.011316 / 0.001528 | 10 |

All 180 sampled layer/absolute-position K/V pairs match exactly between
corrected session sequential and drained sequential. Independently completed
per-prefix heads and the session repeat also match every sampled K/V value
and every final hidden/logit bit. The manual no-drain probe still reproduces
the historical unsafe result. No temperature/performance inference is made
from this untimed test.

The fix finishes decode after each prefix token that skips logits, before
the next host upload reuses its input. It does not add a head projection,
copy, allocation or kernel, and does not change normal generated-token or
packed-prefix computation. It deliberately adds a host completion boundary
to an otherwise unsafe loop. Future stream-ordered embedding work must
preserve this ownership guarantee rather than simply remove the wait.

## Remaining Packed Difference

Both packed paths now have the same first token and top-10 set as corrected
sequential. Their small nonzero logit differences remain measured, not
explained away: packed attention consumes FP32 K/V whereas token decode
reads FP16 cache, and multi-row projection/reduction arithmetic can differ.
Full 512-ID and frozen dense-valid quality checks must be rerun before any
packed candidate or combined runtime qualifies. Do not transfer an old
exact-to-unsafe-control gate to this corrected control.

## Validation and Evidence

The new queued-work head128/Qwen regression fails before the fix, returning
206 instead of 225, and passes afterward with matching hidden state and
physical K/V. cuBLAS-on release suites pass: attention grid 11, transfer
ordering 5, Q8 FFN runtime 8, Q8 prefill runtime 4. Warning-as-error full
non-CUDA tests and CUDA/server all-target Clippy pass. cuBLAS-off and
qualification-report checks also pass: 26 affected CUDA suite outcomes with
cuBLAS off, and 12 comparison-tool tests. The SGEMM-only cases are not
compiled in the off run, not claimed as off-backend coverage. Non-CUDA
example tests pass too. Qualification now rejects reports missing the
corrected prefix contract rather than accepting stale quality evidence.

- [Corrected raw comparisons and sampled K/V](2026-10-25-maxwell-prefill-drift-completed.jsonl)
- [Real-model output](2026-10-25-maxwell-prefill-drift-completed.log.gz)
- [cuBLAS-on regressions](2026-10-25-maxwell-prefix-fix-tests.log.gz)
- [cuBLAS-off regressions](2026-10-25-maxwell-prefix-fix-tests-off.log.gz)
- [Full non-CUDA regression](2026-10-25-maxwell-prefix-fix-noncuda.log.gz)
- [All-target Clippy](2026-10-25-maxwell-prefix-fix-clippy.log.gz)
- [Qualification-report tests](2026-10-25-maxwell-prefix-contract-tests.log.gz)
- [Non-CUDA example checks](2026-10-25-maxwell-prefix-contract-noncuda.log.gz)
- [Checksums](2026-10-25-maxwell-prefill-drift-completed.sha256)

Tracked device allocations return from zero to zero. GPU inference is serial;
the isolated Ollama server remains stopped. No runtime-default change or
original target achievement is claimed. M08 resumes with fresh quality
reports bound to the corrected prefix-completion contract, then controlled
HTTP comparisons and M09's fresh selected-runtime Ollama checkpoint.
