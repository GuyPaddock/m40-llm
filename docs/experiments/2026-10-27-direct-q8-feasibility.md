# Direct-Q8 Multi-Row Verification Feasibility

Assessment completed October 28, 2026. Measurements were collected October 27.

## Decision

Recommend **one fixed four-row joint gate/up/SwiGLU operator experiment for
review**, not a speculative decoder. No new CUDA kernel was written. The
existing SGEMM verifier remains rejected, and the earlier n-gram feasibility
gate remains closed. A real drafter's acceptance and cost are still unknown.

The immutable `compound-runtime-63240c5-v1` reference remains **50.910426 HTTP
tok/s** versus official Ollama's **43.194730**. Neither 53.06 nor 56.153149
HTTP tok/s has been achieved by this assessment. Preferred joint-only compound
ownership and all runtime defaults are unchanged.

The proposal has a distinct avoided-work hypothesis: read and unpack one joint
Q8 payload for four activation rows, rather than materialize whole FP32 matrices
or independently request the payload for each row. That is enough to justify
an isolated measurement, not enough to claim a complete-request speedup.

## Provenance And Scope

- [Approved plan](2026-10-27-direct-q8-feasibility-plan.md) and
  [source/cost ledger](2026-10-27-direct-q8-ledger.md).
- Clean measured source `6fec43a1826f1e1be1e8e642925408db2e092c11`.
  Diagnostic executable SHA-256
  `bb4bac4876cef280a1204e006e7aae0440e6d50510bc113898fff632534585fd`.
- Model SHA-256
  `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
  Original Qwen2.5-3B Q8_0 model and frozen prompt/generated IDs are retained.
- Rust release build, CUDA feature, NVCC 12.8.93, `-O3`, `sm_52` plus
  `compute_52` PTX, no fast-math addition. The raw manifest binds the full
  reference environment. Compound joint-only ownership and explicit dense KV
  are selected by the diagnostic loader.
- Sole GPU process at all 140 checked boundaries; Tesla M40 UUID
  `GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`, P0, 1113/3004 MHz.
  Measurements ran serially, without another inference workload.
- One 65-token prefix, widths 2/4/8, six excluded warmup checks, five alternating
  profiler-off/on pairs per width, and five rejection cycles per distinct
  accepted prefix from `0, floor(K/2), K-1`. There are 30 full-batch observations
  and 40 rejection cycles. No broad prompt or kernel search followed.

The diagnostic measures the first verification batch after prompt construction.
Its timed interval includes target execution, commit and correction when needed.
Prefill, debug cache downloads/hashing, reporting and profiler event-pool
allocation are outside that interval. Workspace transitions inside target
execution remain charged. This is not a new 512-token HTTP run; it cannot
replace either the original full-continuation verifier rejection or the
immutable HTTP checkpoint.

An initial attempt stopped before its first measured pair because NVIDIA's CSV
reported clocks with an optional `MHz` suffix. Six excluded checks had run.
The [attempt and parser correction](2026-10-27-direct-q8-clock-attempt.md)
are retained separately; no observations were pooled or thresholds changed.

## What The Rejected Verifier Actually Costs

Medians in milliseconds. Stage columns are medians of per-batch summed CUDA
event intervals, not exclusive compute times or additive unprofiled wall costs.

| Rows | Profiler off batch | Profiler on batch | Observer change | Q8 to FP32 spans | SGEMM spans |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 172.870 | 171.024 | -1.068% | 88.659 | 53.780 |
| 4 | 176.050 | 179.203 | +1.791% | 88.759 | 54.474 |
| 8 | 200.030 | 205.536 | +2.753% | 88.737 | 62.089 |

All observer medians remain within the existing 3% allowance. The negative
two-row difference is timing variation, not an instrumentation speedup.
Do not subtract these differences to manufacture an unobserved stage latency.
Enqueue/queue gaps can be included in event intervals; wall time outside their
union is not automatically removable CPU overhead.

There are **252 conversion and 252 SGEMM calls per batch**, seven projections
per layer across 36 layers. The cost barely amortizes at four rows because
the same full matrices are converted again for each verification batch.
The original verifier's 171.687/180.576/206.486 ms batches and 19.310791 ms
serial row used the older split-FFN configuration. Those historical values
are not a current compound speed comparison.

The seven layer projections contain 2,774,532,096 weights, or 2,947,940,352
packed Q8 bytes. Conversion writes 11,098,128,384 FP32 bytes per batch;
subsequent logical FP32 consumption brings that write/read count to
22,196,256,768 bytes. **These are logical byte counts, not measured DRAM
transactions.** No bandwidth, occupancy or warp-stall counter attribution
was collected.

Other measured stage intervals still scale with rows:

| Rows | Causal attention | Vocabulary | Greedy argmax |
| ---: | ---: | ---: | ---: |
| 2 | 1.337 | 3.124 | 0.511 |
| 4 | 2.506 | 6.238 | 1.022 |
| 8 | 4.876 | 12.468 | 2.045 |

Normalization, RoPE, KV writes, residuals and bias work also remain. Weight
reuse alone does not remove them. QKV/output use their qualified packed-warp
single-row consumers; down/vocabulary use qualified prepared consumers.
A future direct verifier must retain these associations or independently
qualify replacements. Disabling SGEMM is not sufficient to guarantee that.

## Transactions And Ownership

Every full batch matched the actual frozen greedy predictions and committed
lengths. Rejection checks matched accepted-prefix predictions, correction and
next IDs. All 65 retained prefix KV positions across all 36 layers remained
bit-identical. Cleared-suffix checks in these observations cover the remaining
rejected slots **after correction**; existing transaction tests separately
check clearing before correction overwrites its first slot.

Rollback uses one kernel and three synchronizations. Its median cost is
0.0137 to 0.0147 ms. Correction costs 19.00 to 19.09 ms at widths 2/4 and
23.03 to 23.13 ms at width 8. The latter includes workspace transition costs;
it is not treated as a free correction. Begin records 17 allocations/16 frees,
252 cuBLAS calls and 360 stream waits. Correction records 16 allocations/17
frees. Counter snapshots preserve these costs rather than hiding them.

The model charges the largest sampled rollback-plus-correction median per
width, plus a 0.3274 ms early-to-late attention sensitivity allowance. This
allowance is not a measured universal bound on late-context correction.
Unmeasured rejection positions receive that same conservative sampled charge.

Ownership remains 109 logical sources, 37 physical split tensors and 36 joint
pairs. Uploaded sources occupy 3,279,519,744 bytes; prepared split owners
1,192,935,424 bytes; joint pairs 1,724,645,376 bytes. Dense KV is 37,748,736
bytes. No owner changes during the observations; tracked teardown is zero.
All 137 compiled CUDA instruction-word sequences match the immutable binary.
The diagnostic adds no target kernel.

These checks establish the bounded existing implementation's behavior, not
full future-verifier correctness. A future verifier must retain exclusive
session ownership, causal row lengths, absolute RoPE, exact committed-prefix
state and rejected-suffix cleanup under all generation paths. It must not
use compressed-cache metadata as an implicit dense backing store.

## Quantitative Break-Even Model

The first candidate is already known from cached logits. Let `p` be the
conditional probability that each of the remaining `K-1` drafted guesses is
correct. This is a scenario variable, not measured real-drafter acceptance.
Accepted prefix `a` has probability `p^(a-1)*(1-p)` for `1 <= a < K`;
full acceptance has probability `p^(K-1)`. Full acceptance advances `K` input
rows, not `K+1`; rejection followed by correction advances `a+1`.
The measured `a=0` case is an error-path diagnostic, not a normal known-seed
drafting outcome.

For each of the five immutable HTTP trials separately:

```text
S = measured decode_ms / 511
F = measured HTTP_ms - measured decode_ms
V = K*S - (1-beta)*K*7.4215 + extra
cycle = V + drafter_ms + (1-p^(K-1))*rejection_extra
HTTP = F + expected_cycles*cycle + expected_serial_tail_rows*S
```

`beta` is hypothetical gate/up time relative to `K` serial compound calls,
not a full-FFN speedup. The 7.4215 ms anchor is a sampled all-layer gate/up
stage interval. Mapping it to unprofiled wall-time savings is an assumption;
the model also credits only 90% as a sensitivity case. Other projection,
attention, vocabulary, synchronization and transfer costs are not removed.
`extra` adds 1 or 5 ms per batch for additional full-verifier costs.

A finite-horizon recurrence accounts for all 511 continuation-input rows,
rejection progress and final serial tails. It does not substitute
`511 / mean_progress` for an integer horizon. An optional per-decision draft
eligibility rate retains ordinary serial steps when no draft exists. The
main curves assume eligibility 1; lower-eligibility curves optimistically
charge zero failed-lookup time and are labeled accordingly.

The [reproducible model](2026-10-27-direct-q8-model.json) includes widths
2/4/8, `p=0,.25,.5,.75,.9,.95,.99,1`, drafter costs
`0,.25,.5,1,2,5` ms, gate ratios `.5,.75,1`, and extra costs `1,5` ms.
Budgets use the most restrictive of the five reference trials.

Necessary perfect-acceptance/free-drafter limits on whole verifier latency:

| Rows | Break-even ms | 3% improvement ms | 53.06 tok/s ms | 56.153149 tok/s ms |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 37.191 | 36.042 | 35.593 | 33.509 |
| 4 | 74.382 | 72.076 | 71.174 | 66.989 |
| 8 | 148.765 | 144.115 | 142.298 | 133.861 |

Measured SGEMM batches exceed even these necessary ceilings. The limits
do not show that a real direct reader can attain them.

For a hypothetical four-row reader halving gate/up time, with 1 ms extra
batch cost and a 0.25 ms drafter, minimum conditional guess acceptance is:

| Objective | Required p |
| --- | ---: |
| Break-even | 87.004% |
| 3% improvement | 89.200% |
| 53.06 HTTP tok/s | 90.242% |
| 56.153149 HTTP tok/s | 94.227% |

At `p=.90`, allowable drafting cost for the 56.153149 target is negative.
At `p=.95` it is about 1.028 ms/cycle; at `.99`, about 5.194 ms. Raising
extra batch cost from 1 to 5 ms raises required acceptance for that target
to 98.113% with a 0.25 ms drafter.

**Hypothetical sensitivity, not a throughput result:** `p=.95`, half gate/up
time, 0.25 ms drafting and 1 ms extra yields 56.962 projected HTTP tok/s.
Crediting only 90% of stage savings gives 55.745; adding 5 ms extra instead
gives 52.709, a minimum modeled 3.51% gain across the reference requests.
At `.90` acceptance, that latter case regresses. Eligibility `.25/.5/1`
with the optimistic 1 ms-extra scenario gives 54.140/55.597/56.962.
None of these are achieved performance, confidence intervals or drafter
measurements. Real acceptance is prompt-dependent and correlated.

The [previous n-gram study](2026-10-25-ngram-feasibility.md) remains negative:
its best optimistic 4.949% gain missed its fixed 10% gate. It used an older
trace, with only 13.672% of positions eligible and 0.578 accepted tokens per
eligible draft versus 3.978 offered. That histogram is not current known-seed
acceptance evidence. No n-gram policy or EAGLE implementation is admitted by
this assessment.

## Maxwell Resource And Numerical Budget

Maxwell supports conventional FP32, integer unpacking, vector loads, shared
memory and warp shuffles. No Tensor Cores or native DP4A are assumed.
The [NVIDIA Maxwell guide](https://docs.nvidia.com/cuda/archive/12.8.1/maxwell-tuning-guide/index.html)
documents 64K 32-bit registers per SM, 96 KiB shared memory for second-generation
Maxwell and a 48 KiB per-block shared limit.

| Rows | Independent gate/up accumulators per thread | Shared bytes | Forward workspace bytes excluding FP32 weights | Provisional KV bytes |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 2,560 | 415,760 | 73,728 |
| 4 | 8 | 4,608 | 831,520 | 147,456 |
| 8 | 16 | 8,704 | 1,663,040 | 294,912 |

The four-row shared allocation follows `1024*K+512`. Register scenarios
40/48/64 per thread imply 5,120/6,144/8,192 registers per 128-thread block,
and register-only upper bounds of 12/10/8 resident blocks. These are
uncompiled estimates, not achieved occupancy; allocation granularity and
other limits can lower them. The qualified one-row consumer uses 32 registers.

One joint gate/up payload occupies 47,906,816 bytes/layer, 1,724,645,376
across 36 layers. Ideal logical read reuse is at most `K`; actual cache reuse
by existing independent-row blocks is unknown. Do not claim `K` times less
DRAM traffic. Existing session scratch is 689,668 bytes and attention partial
scratch 66,560 bytes. Provisional KV fits the already allocated dense cache.
The 90,177,536-byte prefill weight scratch may remain resident because prompt
SGEMM is unchanged; avoiding its use during verification is not a residency
saving.

Each row must preserve the qualified thread-to-input mapping, FP32 scale
multiplication, accumulation sequence, reduction tree and SwiGLU operations.
Interleaving rows can change compiler fusion or register allocation. Bitwise
operator comparison and machine-code/resource inspection must check this;
the cost model does not waive exact target greedy output requirements.

## One Proposed Operator Probe, Pending Approval

Fixed shape only: four rows, D=2048, H=11008, 128 threads, one CTA per output
column. Use the existing joint-packed payload and shared scales. Load/unpack
each payload once and reuse it across four independent FP32 row calculations.
No alternate geometry, activation quantization, repacking or new persistent
weight representation is included.

Use layers 0/17/35 and the first four distinct frozen normalized activation
rows per layer from the [existing input capture](2026-10-25-maxwell-ffn-inputs.md).
Compare three contemporaneous arms:

1. Four qualified one-row compound calls.
2. Existing research four-row compound dispatch with independent row blocks.
3. Proposed fixed shared-read four-row consumer.

All three arms must finish with **four qualified one-row prepared-down calls**.
Do not use the generic four-row down path with a different association.
The existing four-row compound is not already a shared-read optimization;
its earlier layer-17 screen took 0.892 ms versus 0.203 ms for one row.
That historical screen is context, not a contemporaneous promotion result.

Proposed consumer ceiling is 64 registers/thread, 4,608 shared bytes, no
spills/local stack. No hot-path allocation, upload, copy or extra synchronization.
Require exact operator outputs, canaries, unchanged ownership and unchanged
qualified compiled entries. Stop on a correctness or resource failure rather
than adding rescue configurations.

Retain the existing **at least 5% complete-FFN improvement** and positive
paired bounds on all three layers against both controls. Keep the existing
5/10/fresh-20 trial and variation rules. A passing operator result would
only justify refreshing this model and separately reviewing a full causal
verifier prototype. It would not qualify runtime integration or a drafter.
Any eventual HTTP promotion still needs the existing greater-of-3%-or-twice-
variation threshold, paired confidence rules, all 512 actual greedy IDs,
twelve frozen task outputs and a fresh official Ollama comparison.

## Reproduction, Checks And Artifacts

The [raw observations](2026-10-27-direct-q8-cost.jsonl) bind measured source,
executable, model, environment, ownership, GPU boundaries, counters and spans.
The [independent audit](2026-10-27-direct-q8-audit.json) passes all 70 cases.
[Seven arithmetic checks](2026-10-27-direct-q8-math-tests.json) and
[18 rejected corruption controls](2026-10-27-direct-q8-controls.jsonl)
check the model and audit. The control log was initially copied before its
process finished; follow-up commit `838ff35` retains all 18 completed records.
The raw measurements were unaffected.

```bash
perl docs/experiments/2026-10-27-direct-q8-model.pl --self-test
perl docs/experiments/2026-10-27-direct-q8-model.pl \
  docs/experiments/2026-10-27-direct-q8-cost.jsonl --audit-only
perl docs/experiments/2026-10-27-direct-q8-model.pl \
  docs/experiments/2026-10-27-direct-q8-cost.jsonl > /tmp/direct-q8-model.json
bash docs/experiments/2026-10-27-direct-q8-controls.sh \
  docs/experiments/2026-10-27-direct-q8-cost.jsonl
```

Validation before measurement passed the affected biased-Qwen head128 and
head64 observer fixtures, transaction and profiler tests, CLI isolation and
clock-parser tests, non-CUDA warning-as-error suite with 219 passing tests,
CUDA/server Clippy with warnings denied, and formatting. The
[instruction-word audit](2026-10-27-direct-q8-sass-audit.json) reports all
137 prior entries unchanged and no added CUDA entries. Logs and audit tooling
are in [the diagnostic archive](2026-10-27-direct-q8-validation.tar.gz).

The read-only local executable and source archive are retained under
`local-artifacts/qualified-references/direct-q8-cost-6fec43a/`.
Source archive SHA-256 is
`adde84158d3611b6d2be37a6e68de4fcfbb5f4d59e556d97acf350fd0fe65a05`.
The [separate checksum list](2026-10-27-direct-q8.sha256) covers this new
evidence. It does not modify the immutable reference or ownership audit.

**Stop here.** The assessment is complete. Review the fixed operator proposal
before writing CUDA. No further kernel, verifier, drafter, compatibility,
selection-policy or runtime-default work follows automatically.
