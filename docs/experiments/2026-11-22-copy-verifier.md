# Existing prompt-copy drafting with a faster target

The M40 complete-target study at 837e070 passed its strongest control
with 5.876% lower four-row latency. The controlled rejection comparison
also reduced submission-through-correction time. This admits one live
copy study with the same draft source. [C1 results](2026-11-21-small-n/target-result/README.md).

Compare three execution arms within one loaded Qwen2.5-3B Q8_0 model.

- Ordinary sequential inference with no prompt-copy drafter.
- The existing causal prompt-copy source using the exact multi-row split
  down control from C1.
- The same source using the new shared-weight down schedule.

Reuse `draft_feasibility live`, the retained prompt/held-out records,
real target predictions, the same stop IDs and a 128-token cap. Retain
all 21 correctness cases, including prose, code, extraction and natural
rejections. Require the original frozen IDs, comparable full logit rows,
exact valid KV, sound rollback and complete allocation cleanup.
Only proposals obtained from the live prompt-copy source can enter a
timed verifier. The original paragraph and configuration copy tasks
are the two timing workloads. Charge source construction, lookup misses,
actual accepted lengths, each target transaction, rollback, ordinary
correction, prompt processing and teardown.

For each timing workload, warm all arms and collect five rounds with
alternating, rotated schedule order. Report complete resident CLI
request wall, prompt time, CPU drafting and verification, makespan per
generated token, variance and 10,000 paired bootstrap resamples with
seed zero. Retain the original specialized 3% floor and positive
lower bound versus ordinary inference. Require a positive paired lower
bound and a gain exceeding twice the larger measured variation over
the existing multi-row copy arm to credit the verifier improvement.
Apply the existing single extension for high variation and a separate
twenty-round confirmation for a stable result within one percentage
point of its gate. Keep host process startup and model preparation
observations separate from resident timing. The research CLI is the
measured endpoint. This study makes no default server claim.

Stop on correctness failure, differing work or a failed complete
request gate. Preserve the outcome for both copy workloads, including
any expected decline. The general adaptive controller and non-copy
performance remain outside this experiment.
