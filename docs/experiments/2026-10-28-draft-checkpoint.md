# Causal Draft Checkpoint Before Live Implementation

Fixed source/prompt revision `d2f8070` precedes held-out capture. No algorithm,
suffix length, order, threshold, class dispatch or parameter changed after
viewing development or held-out acceptance. The three concrete CPU sources
are research-only; runtime defaults and the qualified verifier are unchanged.

The development browser trace rejects broad acceleration: history lookup
projects -10.32%, prompt-copy -0.12%, online backoff -31.21% resident throughput.
This uses actual visited rejections and measured target costs, not an IID
acceptance average. The historical 2026-10-25 negative lookup gate remains
valid, not reversed by a better target verifier.

All eight fresh held-out prompts were captured twice with qualified serial
compound/joint-only/dense generation, stable actual IDs/counts/stops and zero
tracked teardown. Actual prompt lengths are 130-137 and 393-397; outputs are
52-128 tokens with natural stops retained. They fit the existing target-cost
context range. Corpus definitions were committed before target capture.

The predefined copy workload is promising for the unchanged prompt-copy
source. Both copy traces have conditional guess acceptance [1,1,1], full-batch
acceptance 1, and visited eligibility 80.0%/84.2%. Total all-in CPU costs are
at most 0.034/0.083 ms per request across five release trials, including
initialization, misses and committed-state maintenance. Modeled resident gains
are 12.43-12.51% / 6.46-6.54% across five measured target samples. These are
projections, not paired GPU results or independent final confirmation.

All three sources succeed on the simple copy cases; prompt-copy is not a
uniquely successful algorithm. Choose it for conditional live implementation
because its role was declared as prompt copying before capture, its provenance
is restricted to supplied input, and it was the least harmful development
browser candidate. Do not call this a general drafter or select a new policy
from held-out failures. No source gives a broad 3% gain on prose/code/extraction.

The user's conditional authorization now admits one explicit research-only
live prompt-copy runner using the existing direct verifier. Require actual
greedy/logit/KV checks on all disclosed/fresh traces, then five paired resident
measurements of the two unchanged copy cases. Treat those as specialized
52/68-token workloads with equal arm counts/stops, not the original 512-token
HTTP target. Apply unchanged variation/bootstrap/confirmation rules. Stop on
correctness or performance failure; no rescue or workload tuning.

Evidence: [development](2026-10-28-draft-development.jsonl),
[held-out target captures](2026-10-28-draft-heldout.jsonl),
[full CPU study](2026-10-28-draft-study.jsonl),
[independent causal audit](2026-10-28-draft-audit.pl).
