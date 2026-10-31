# Measured Direct-Verifier Feasibility Decision

**Go for a separately reviewed draft-acceptance feasibility study. Stop here.**
The fully direct four-row target clears correctness and complete resident
performance gates. It does not authorize a neural drafter, EAGLE, speculative
generation integration or a new default. The prior temporary-dequant/SGEMM
verifier and negative n-gram study remain rejected for their implementations
and workloads. No failed kernel or scheduling configuration was rescued.

## What Is Now Measured

The [correctness report](2026-10-28-direct-verifier-correctness.md) establishes
actual greedy IDs, all 511 full-vocabulary vectors, bit-identical valid dense
KV, all rejection positions, partial/full commits, suffix cleanup and session
ownership. All twelve frozen task continuations match. The numerical contract
remains `1e-3 + 2e-5*abs(reference)`; bitwise logits happen to match here.

The [complete release experiment](2026-10-28-direct-verifier-measurement.md)
measures **57.820 ms** per four-row batch, median of five request means.
All 635 batches range from 57.064 to 58.852 ms. Ordinary serial means range
from 18.585 to 18.630 ms per row. Complete resident requests, including setup,
prompt preparation, commit, tails and teardown, improve from **51.767457 to
65.799057 tok/s**, +27.1051%, paired interval [26.8683%,27.2526%]. These are
ideal perfect drafts using frozen continuations, not a real proposal source.

The previous 60.17-60.43 ms estimate is superseded by complete measurements.
All other Q8 projections, causal attention, absolute positions, output norm,
vocabulary, greedy readback, copies, dependencies and commit costs now execute
inside the measured target. There is no repeated FP32 target materialization
or target SGEMM to subtract hypothetically. Shared prompt SGEMM remains charged.
The immutable **50.910426 HTTP tok/s** reference is not replaced by a resident
measurement, and neither original HTTP goal has been newly qualified.

## Acceptance Model

The [calculation](2026-10-28-direct-verifier-model.pl) and
[JSON result](2026-10-28-direct-verifier-model.json) use each measured pair
independently. A finite 511-input-row recurrence includes the final serial
tail. Perfect acceptance reproduces each actual candidate decode interval
within 1e-6 ms; it does not substitute an FFN-only estimate.

The first candidate is a known target prediction. There are **three guesses**,
not four independent draft successes. With equal conditional acceptance `p`,
the retained-prefix probabilities are `1-p`, `p*(1-p)`, `p^2*(1-p)`, `p^3`
for prefixes 1,2,3,4. A rejected attempt retains its verified inputs and runs
one ordinary target correction. Useful progress is therefore 2,3,4,4 rows;
the infinite-stream mean is `2+p+p^2`. Prefix zero is a transaction diagnostic,
not a natural cached-seed branch. Full-batch acceptance is `p^3`.

The recurrence uses measured context-dependent batch costs, interpolated
between actual four-row starts. It charges observed rejection target overhead,
rollback/clearing and the larger of measured early correction or the ordinary
serial cost at that absolute position. The direct session's actual fixed
costs and loop bookkeeping remain charged. An all-in drafting cost is charged
on every decision, including attempts that find no eligible draft. Real
proposal lookup, computation, transfers and any additional synchronization
must fit that budget. There is no assumption that a real drafter is perfect.

## Drafter Budgets

These budgets take the minimum across all five measured requests. Negative
values mean even a free drafter cannot reach that objective at the stated
acceptance. They are conditional cost-model results, not statistical
qualifications for an implemented drafter.

| Conditional guess acceptance | Resident break-even D ms | Resident >=3% gain D ms | 53.06 HTTP scenario D ms | 56.153149 HTTP scenario D ms |
| --- | ---: | ---: | ---: | ---: |
| 0.90 | 6.056 | 3.954 | 2.968 | -0.903 |
| 0.95 | 11.109 | 8.927 | 7.903 | 3.884 |
| 1.00, oracle only | 16.518 | 14.250 | 13.184 | 8.999 |

The two HTTP columns borrow the **largest historical fixed HTTP overhead**,
554.538 ms, and add the largest measured extra session setup/teardown cost,
0.370 ms. They reserve this overhead before testing the measured decode model
against `512000/target`. They do not claim contemporary HTTP throughput or
reuse historical trials as new paired evidence. Prefill timing noise is not
credited as a speedup in these HTTP scenarios. A fresh controlled HTTP/Ollama
comparison remains necessary before claiming either absolute/relative goal.

| All-in drafting cost per decision | Resident break-even minimum p | Resident >=3% minimum p | 53.06 HTTP scenario minimum p | 56.153149 HTTP scenario minimum p |
| --- | ---: | ---: | ---: | ---: |
| 0.25 ms | 83.78% | 86.03% | 87.09% | 91.24% |
| 1.00 ms | 84.62% | 86.85% | 87.90% | 92.03% |
| 2.00 ms | 85.71% | 87.93% | 88.98% | 93.08% |

At `p=0.95`, all three guesses survive only 85.74% of attempts. That distinction
matters when comparing a reported drafter acceptance statistic with this table.
Minimum acceptance is modeled, not observed. Original variation and paired
confidence gates still apply to any later complete implementation.

## Sensitivity And Remaining Unknowns

- Rejection/correction was measured five times per position at the 65-token
  prefix. Later perfect-batch costs are measured through the continuation,
  but wrong-draft late-context costs are not. Per-position serial correction
  and early overhead are an explicit sensitivity model, not an upper bound.
- Unknown additional per-attempt overhead is tested at 1 and 5 ms without
  changing the acceptance gate. With D=0.25 ms, the 56.153149 HTTP scenario
  needs p=92.29% and 96.36%, respectively, rather than the nominal 91.24%.
  Do not assume such overhead vanishes in a deployable proposal pipeline.
- Eligibility matters. At p=0.95 and D=0.25 ms charged even on ineligible
  decisions, modeled minimum resident gains are 8.19%/12.65%/16.95% for
  eligibility 25%/50%/100%. These are scenarios, not measured draft coverage.
- The equal-p, constant-eligibility recurrence does not model correlations
  between prompt content, context length, acceptance and draft cost. A real
  acceptance study must measure those jointly, not just overall match rate.
- Model loading and prepared-weight construction precede the warm resident
  comparison for both arms. Every request's session construction, prompt,
  transactions and teardown is charged. This is not cold-start throughput.
- No DRAM counters, achieved occupancy or warp-stall attribution is available
  from this evidence. Static resources and unchanged compiled controls do not
  establish a Maxwell bandwidth ceiling.

## Recommendation And Stop

There is enough measured target-side margin to justify **one separate,
proposal-only acceptance/cost study**, if authorized. A cheap drafter around
1 ms per decision would need roughly 86.9% conditional acceptance for the
resident 3% screen, and about 92.0% for the conservative historical HTTP target
scenario. Those are demanding but now quantitatively defined requirements;
they are not evidence that a particular drafter can meet them.

Start that future study offline against frozen actual target traces, using
only prompt and committed past tokens. Measure eligible coverage, acceptance
at each guess position, rejection locations, proposal cost and prompt-shape
variation. Reuse the existing negative n-gram findings rather than assuming
the new verifier repaired draft eligibility. An improved or different draft
source needs its own evidence and authorization before GPU integration.

Do not build a neural drafter, EAGLE or speculative runtime in this task.
Do not retune kernels, change model semantics or promote the prototype.
Preserve preferred joint-only ownership and all defaults. The immutable
50.910426 HTTP checkpoint and the 53.06 / 56.153149 HTTP goals are unchanged.
The approved experiment is complete; stop for review.

```bash
perl docs/experiments/2026-10-28-direct-verifier-model.pl --self-test
perl docs/experiments/2026-10-28-direct-verifier-model.pl > /tmp/recomputed-model.json
cmp /tmp/recomputed-model.json docs/experiments/2026-10-28-direct-verifier-model.json
```

The [five model tests](2026-10-28-direct-verifier-model-tests.json) cover finite
tails, one known seed, rejection progress/correction, ineligible serial work
and charged extra per-attempt cost. They are separate from the GPU correctness
and performance gates.
