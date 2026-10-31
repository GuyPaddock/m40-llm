# Full-Verifier Assessment After The Fixed Operator Probe

## Decision

The [fixed operator](2026-10-28-four-row-operator.md) passes its gate.
That does **not** make the rejected SGEMM verifier viable. Replacing only
gate/up in that implementation still models roughly **111.9 ms per batch**.
The remaining conversions and dispatch overhead must not disappear from the
accounting because one operator improved.

A different layer-major schedule using the measured four-row gate/up and
existing qualified direct **single-row** consumers for QKV/output/down remains
architecturally plausible. With the earlier stage-to-wall mapping and 1 ms
extra batch overhead, it models **60.17-60.43 ms per four-row target pass**.
This is not a measured assembled verifier or an implementation approval.

The necessary perfect-acceptance/free-drafter threshold remains 74.382 ms for
break-even, 72.076 ms for a 3% HTTP gain, and 66.989 ms for 56.153149 tok/s.
Non-perfect acceptance makes those thresholds stricter. Plausible target
execution alone does not establish a profitable speculative decoder; no real
cheap drafter meeting the required acceptance has been identified here.

Stop for review. No full verifier, neural drafter, EAGLE path, generation
selector, runtime-default change or additional kernel search was implemented.
The immutable reference remains **50.910426 HTTP tok/s** versus official
Ollama's **43.194730**. The goals remain **53.06** and **56.153149 HTTP tok/s**.
No modeled number below is progress toward those measured HTTP goals.

## Measured Inputs And Extrapolation

The [new raw screen](2026-10-28-four-row-screen.jsonl) uses the same three
frozen layers and first four distinct real inputs. Gate/up medians are
0.395238/0.395821/0.395571 ms versus serial 0.812483/0.812102/0.812163.
The model takes the largest sampled-layer median ratio, **0.487403**, instead
of the earlier assumed ratio 0.5. This is a measured operator ratio, not
measured all-layer target performance or a universal latency bound.

The [reproducible updated model](2026-10-28-four-row-model.json) retains the
five separate immutable HTTP trial inputs, 511 continuation-input rows,
7.4215 ms sampled all-layer gate/up anchor, and 19.426459 ms rollback/correction
charge from the [earlier diagnostic](2026-10-27-direct-q8-feasibility.md).
It preserves the 0.3274 ms late-attention allowance inside that correction
charge. Neither the sampled correction median nor that allowance is a bound
on every future context or workspace implementation.

Two independent accounting cases follow. They must not be mixed by taking
cheap stages from one and pretending the other's conversion costs vanished.

## Case A: Gate Replacement In The Rejected SGEMM Verifier

Use the five prior profiled width-four batches. Within each layer, the audited
call order is Q/K/V/output/gate/up/down. Attribution uses that source order;
the CUDA event labels themselves do not identify individual tensors.
Remove only the two gate/up conversion and SGEMM spans plus SwiGLU, then add
36 times the slowest sampled-layer candidate gate/up median, 14.249549 ms.

Median remaining conversion and SGEMM spans are about 37.345 and 24.584 ms.
The counterfactual batch median is **111.867 ms**. This subtraction is an
indicative model, not measured new execution: event intervals can contain
enqueue gaps, their mapping to wall time is imperfect, and profiler overhead
has not been subtracted to invent a better number.

There are still **180 conversions and 180 SGEMMs per batch** for the other
five projections per layer. Remaining logical Q8 sources total 1,223,294,976
bytes; conversion still writes 4,605,345,792 FP32 bytes per group, followed by
logical matrix consumption. These are representation sizes and logical work,
not measured DRAM traffic. Causal attention, vocabulary/argmax, bias, norms,
copies, waits, allocation transitions and transaction work also remain.

This case does not approach even the necessary perfect-drafter break-even
threshold. The old implementation remains rejected. Do not integrate this
operator by simply patching the losing SGEMM verifier.

## Case B: Direct One-Row Consumers Around The Four-Row Gate

This is a prospective schedule, not code implemented in this task:

- Process a group layer by layer, keeping four independent activation rows.
- Use the already qualified direct one-row QKV/bias and output consumers.
  Do not silently route generic multi-row calls through SGEMM.
- Preserve causal per-row attention over `prefix + row + 1`, absolute RoPE
  positions and exact KV layout. Never expose future candidate keys to an
  earlier row. There is no attention approximation or credited attention reuse.
- Use the new fixed gate/up consumer once per layer, followed by four
  original qualified single-row prepared-down calls, as measured here.
- Retain per-row output normalization, vocabulary and greedy argmax. No head
  reuse or free bonus token is credited.
- Retain the generation lease, provisional cache ownership and rollback
  protocol. A rejected suffix must not survive commit or leak into correction.

All non-gate costs remain charged through the measured serial baseline:

```text
S_i = immutable trial i decode_ms / 511
F_i = immutable trial i HTTP_ms - decode_ms
beta = largest sampled-layer candidate_gate_ms / serial_four_gate_ms
V_i = 4*S_i - credit*(1-beta)*4*7.4215 + extra_batch_ms
```

`credit` is 1 or 0.9. Mapping a profiled stage interval to unprofiled request
savings is still an assumption; reducing its credit tests that assumption.
`extra_batch_ms` is 1 or 5 ms for new scheduling/transfer/dependency/workspace
overhead. These are sensitivity inputs, not measured costs or proven bounds.

The serial residual `4*(S_i-7.4215)` is **44.696-44.958 ms** before the new
gate and extra allowance. It includes the rest of the target, not just FFN.
For orientation, the earlier sampled single-row stage anchors are 4.7614 ms
down, 1.5638 vocabulary, 1.3918 QKV, 1.0164 output and 0.7802 attention.
Those stage anchors are not added again to the residual or treated as exact
unprofiled wall components. Norms, RoPE, residuals, argmax, copies and launch
overhead are not zeroed out.

This case avoids repeated whole-matrix FP32 materialization during verification
only by explicitly using the qualified direct consumers for **all remaining
projections**. They still perform their ordinary Q8 reads and FP32 arithmetic
once per row. No new shared-read gain for QKV, output, down or vocabulary is
assumed. Prompt SGEMM is unchanged; its 90,177,536-byte temporary weight
capacity may remain resident and cannot be claimed as a memory saving.

The earlier provisional four-row KV count is 147,456 bytes within the existing
dense cache; its old forward workspace is 831,520 bytes. The operator itself
uses 594,240 guarded benchmark scratch bytes and 4608 shared bytes per block.
These scopes overlap different implementations and must not be summed as a
verified future memory budget. An actual schedule still needs persistent
row storage, stable lifetimes, producer/consumer ordering and measured setup
and rejection behavior. New hot-path allocations or synchronizations cannot
be hidden in this cost model.

## Acceptance, Drafting And The Full Request

The first candidate is known from cached logits. `p` is conditional accuracy
for the remaining three guesses, not whole-batch acceptance. Full acceptance
has probability `p^3` and advances four input rows. Rejection at accepted
prefix `a` advances `a+1` after correction; it pays full verification plus
the existing rollback/correction charge. No `K+1` bonus is assumed.

An exact finite-horizon recurrence counts expected verified cycles and serial
tail work over 511 input rows. The per-request equation is:

```text
HTTP_i = F_i + cycles*(V_i + D + (1-p^3)*19.426459) + serial_rows*S_i
```

`D` is drafter cost per eligible cycle. Main curves assume draft eligibility
1. Eligibility sensitivities .25/.5/1 retain ordinary serial execution on
ineligible decisions and optimistically charge zero unsuccessful lookup cost.
Actual availability, correlations, acceptance and lookup costs remain unknown.

Using credit 1, 1 ms extra and a 0.25 ms drafter, minimum conditional guess
accuracy across all five reference requests is:

| Objective | Required p |
| --- | ---: |
| Break-even | 86.610% |
| 3% HTTP throughput gain | 88.813% |
| 53.06 HTTP tok/s | 89.857% |
| 56.153149 HTTP tok/s | 93.853% |

At `p=.95`, `D=.25 ms`, the necessary maximum four-row verifier latency is
68.619 ms for break-even, 66.404 ms for 3%, and **61.518 ms** for 56.153149.
The nominal 60.17-60.43 ms model fits the latter with little unaccounted room.
At `.90`, the 56.153149 limit falls to 56.637 ms and the nominal model does
not fit. Allowable drafting time for that target is negative at `.90`,
1.402 ms/cycle at `.95`, and 5.568 ms at `.99`, under the nominal assumptions.

Sensitivity at `p=.95`, `D=.25 ms`:

| Credited stage savings | Extra ms/batch | Modeled verifier ms | Hypothetical HTTP tok/s | Minimum modeled gain |
| ---: | ---: | ---: | ---: | ---: |
| 100% | 1 | 60.165-60.427 | 57.277 | +12.454% |
| 90% | 1 | 61.687-61.949 | 56.016 | +9.986% |
| 100% | 5 | 64.165-64.427 | 54.077 | +6.189% |
| 90% | 5 | 65.687-65.949 | 52.952 | +3.985% |

**None of these are measured verifier or HTTP results.** They do not satisfy
the original goal or the real paired performance gate. The final scenario
misses even the fixed 53.06 target despite its modeled 3% gain. Under 90%
credit and 5 ms extra, the 56.153149 target needs about 99.191% conditional
guess accuracy. That is a demanding and unproven drafter requirement.

The [previous n-gram study](2026-10-25-ngram-feasibility.md) remains rejected
under its own fixed gate. The measured operator does not supply better drafts
or reverse that decision. No EAGLE integration follows. Before any future
runtime claim, actual greedy target IDs, counts and stop reasons, cache state,
all frozen outputs and original numerical/paired HTTP gates still apply,
including a fresh official Ollama comparison.

## Reproduction And Stop

The [model implementation](2026-10-28-four-row-model.pl) has
[six arithmetic checks](2026-10-28-four-row-model-tests.json). It records both
accounting cases, per-layer measured medians, per-trial reference calibration,
conditional acceptance curves, drafter budgets, finite tails and eligibility
sensitivity. The earlier measured evidence/model is unchanged.
All 34 immutable reference, 24 joint-only and 17 earlier feasibility artifact
checksums pass; the [preservation log](2026-10-28-four-row-preservation.log.gz)
retains those checks. New artifacts and locally retained operator binaries
have their own [checksum list](2026-10-28-four-row.sha256).

```bash
perl docs/experiments/2026-10-28-four-row-model.pl --self-test
perl docs/experiments/2026-10-28-four-row-model.pl > /tmp/r4-model.json
cmp /tmp/r4-model.json docs/experiments/2026-10-28-four-row-model.json
```

The [operator gate](2026-10-28-four-row-operator.md) is a real result. The
fully direct assembly is a conditional feasibility result requiring separate
review, not a qualified complete implementation. Stop here as requested.
Do not start a full verifier, neural drafter, additional speed kernel or
runtime promotion automatically.
