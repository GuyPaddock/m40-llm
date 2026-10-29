# Compact Frozen-Target Drafting: Paper Admission

## Decision

Admit a proposal-cost probe, not training yet. A low-rank shared vocabulary
projection has plausible storage/work scope. Acceptance remains unmeasured;
no paper result establishes the required agreement with this exact Q8 target.
The absence of a pretrained head is no longer a blocker by itself. No model
weights or dataset records were downloaded for this assessment.

Official [Medusa](https://raw.githubusercontent.com/FasterDecoding/Medusa/main/README.md)
describes frozen heads and self-distillation; its tree/typical acceptance is
not our linear exact-greedy acceptance. Official
[EAGLE](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/README.md)
describes multi-layer features and training-time testing. Its artifact list
still does not verify a compatible Qwen2.5-3B Q8 head. Both sources were
revisited on 2026-10-29. Published accelerator speedups are not M40 estimates.

## One Fixed Cost Candidate

Use the last committed target hidden vector h of width2048. Three independent
affine maps 2048->256 with SiLU produce three rows; one shared learned
256->151936 projection produces three vocabularies. Argmax gives the three
guesses after the known target seed. No target parameters, vocabulary IDs,
attention, layers or weights change. This is a proposed low-rank head, not
an existing trained Medusa checkpoint or a claimed reproduction.

At target prefix ending at t, h(t) may predict t+2/t+3/t+4; t+1 is already
the target's known seed. Feature rows for t+1 or later cannot enter a proposal
at t. Offline future labels are training targets only, never inference inputs.
Wrong guesses are rejected by the unchanged verifier. Evaluate acceptance
conditioned on preceding guesses succeeding, not independent head accuracy.

| Quantity | Shape calculation, not measurement |
| --- | ---: |
| Parameters including three hidden biases | 40,469,248 |
| FP32 inference weights | 161,876,992 bytes |
| FP32 weights, gradients, two Adam moments | 647,507,968 bytes |
| Three-guess matrix arithmetic | 236,519,424 FLOPs plus SiLU/argmax |
| Batch128 full-vocabulary training logits | 233,373,696 bytes |
| One million FP32 target feature rows | 8,192,000,000 bytes plus IDs/metadata |

The shared vocabulary matrix is 155,582,464 bytes. An ideal three-row consumer
reads it once; three independent matvec calls may read it three times. Neither
reuse nor achieved bandwidth is presumed. Measure the whole proposal using
existing cuBLAS/activation/argmax operations before writing a new CUDA kernel.
Charge feature capture, dependencies, uploads if any, state and initialization.
Current full-width three-head storage is 1,866,989,568 FP16 bytes; that is not
the cost of this rank256 proposal. No reduced-vocabulary coverage shortcut.

A causal one-block drafter could condition later guesses on earlier guesses
and use multi-layer features, but needs serial steps, draft KV and repair.
Its potentially better acceptance has no measured basis here. Do not train it
as a second candidate if the fixed low-rank pilot fails.

## Cost And Acceptance Gates

Use the existing measured verifier recurrence, including coverage, correction,
rollback, tails and setup. At full eligibility the historical HTTP scenario
needs about92.03% equal conditional acceptance at1ms; p=.95 permits3.884ms.
At5ms the requirement is96.12%; these are conditional scenarios, not forecasts.
The free perfect-draft budget is about8.999ms. A measured proposal above that
budget cannot reach the target even with perfect guesses and stops this pilot.
For a bounded training admission, require all-in median <=3.884ms and report
tail/variation plus initialization costs; freeze this threshold before probing.
Below the threshold permits an acceptance study, not a speed claim.

The 72-GPU-hour contract remains 4cost/20teacher/40training/8evaluation.
One million teacher positions at50tok/s would take5.56h; at10tok/s,27.78h
would exceed the teacher cap. These are scenarios: feature readbacks, prompts
and serialization must be measured in an initial bounded sample. Stop if the
planned dataset cannot fit the teacher/storage budget; do not silently shrink
the held-out set or substitute an FP16 teacher. Training throughput is unknown
until the selected training device/toolchain is checked. Head-only training
does not require retaining the 36-layer target or its backward graph on A4000.

## Data, Training And Licensing Before Admission

One candidate data source is the official
[Dolly15k card](https://huggingface.co/datasets/databricks/databricks-dolly-15k/raw/main/README.md),
reviewed metadata-only. It identifies CC BY-SA3.0 instructions/context and
several prose/QA/extraction categories, but is not adequate code coverage by
itself. Preserve Databricks/Wikipedia attribution and terms if used. No blanket
claim that a permissive code license covers data or teacher-derived artifacts.
The [target license](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/LICENSE)
restricts its grant to research/evaluation and specifies attribution for
distributed models trained from its outputs. No purchase or redistribution
is authorized. Pin actual revisions and terms before dataset acquisition.

Before training, commit a manifest containing exact prompt IDs, disjoint
train/calibration/new held-out splits by document/template, provenance,
de-duplication against existing evaluation, language/task/context quotas,
teacher/feature identities, seed and optimizer schedule. Known benchmark
outputs and the twelve frozen task answers must not become training data.
Use exact-target greedy labels, full-vocabulary cross entropy and one fixed
training schedule; no held-out checkpoint selection or architecture search.
This manifest is pending, so training is NOT admitted by this document.

No GPU hours have been used. Next permitted drafting work is the <=4h M40
proposal-cost probe after the independent portfolio and numerical experiments.
Any A4000 switch and return requires user confirmation. If training is later
admitted, measure three-position acceptance/cost jointly on prose, code,
retrieval and copying with short/middle/long contexts. Only a passing measured
model leads to research-verifier integration and actual exact-output HTTP tests.
