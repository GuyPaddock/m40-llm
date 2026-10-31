# Fixed ILP4 Admission: Stop Before Performance

## Decision

The fixed synthetic numerical admission is not satisfied. Stop this experiment
without changing the fixture, tolerance, geometry, accumulator count or kernel.
No complete-FFN timing, model quality evaluation, HTTP trial or runtime selector
was run or enabled. This is a numerical-admission stop, not a performance loss.

Importantly, the failure is in the unchanged qualified control on a new
synthetic dynamic-range case. ILP4 passes that case. This does not show that
ILP4 is less accurate, nor invalidate the qualified control's frozen actual
workload results. It shows that this stress fixture is outside the control's
existing FP64 tolerance. Neither tolerance nor inputs were adjusted to rescue
the experiment. Actual-input layers 0/17/35 remain untested for ILP4.

## Compiler And Numerical Evidence

Kernel preservation commit `e7ade95`; host API, guarded tests and unused fixed
two-arm complete-FFN runner preservation commit `8b2ba39`. The compiler audit
preserves all 138 original instruction streams. The candidate has 40 registers,
zero stack/local memory, 1536 dynamic shared bytes and eight independent FMA
accumulator registers. The qualified compound uses 32 registers. These are
compiler resources, not measurements of occupancy, traffic or stalls.

The single-row synthetic shape is D2048/H11008, with signed Q8 extremes,
positive/negative/zero scales and four deterministic activation cases. Both
arms pass signed, zero and cancellation cases. With repeating activation
`[8, -8, 1e-5, -1e-5]`, the original control fails 101 of 11008 SwiGLU values;
ILP4 fails none. Both gate and up projections pass for both arms.

| Dynamic-range error vs FP64 | Original control | ILP4 |
| --- | ---: | ---: |
| SwiGLU maximum absolute | 0.00440928905 | 0.00581220751 |
| SwiGLU mean absolute | 0.00044633613 | 0.00038375371 |
| Gate maximum absolute | 0.00002494394 | 0.00001982691 |
| Up maximum absolute | 0.00002043871 | 0.00002001463 |
| Values outside unchanged relative/absolute gate | 101 | 0 |

The gate remains `error < 1e-3 + 2e-5 * abs(FP64 reference)`. A larger maximum
absolute error does not imply a failed relative gate. The first control failure
is SwiGLU column 124: FP64 14.8182938515, FP32 14.8196821213, absolute error
0.00138826976 versus tolerance 0.00129636588.

The initial test stopped at that value and poisoned its local test mutex. The
second diagnostic records both arms before returning the same failed gate;
it does not waive failures. All output guards, immutable input and packed
owners, prefill-to-decode dependency, ordinary/capture bit equality and tracked
teardown checks pass. The independent invalid-span/shape test passes without
launches. Final committed-source run: one passing test, one rejected numerical
test, tracked bytes 0 before and after. Changed-target Clippy and the exclusive
benchmark CLI unit test pass. There is no claim that the numerical suite passes.

## Reproduction And Preservation

Source `scripts/dev-env.sh`, then run the preserved commit with:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --release --features cuda --test q8_ilp4 \
  -- --include-ignored --nocapture --test-threads=1
```

The command intentionally exits 101 under the unchanged gate. No GPU compute
process was present before admission. Only the M40 was used. Numerical checks
are not timed performance evidence, so no throughput or GPU-clock conclusion
is drawn from them.

- [Raw numeric records](2026-10-29-ilp4-synthetic.jsonl)
- [Compiled-resource and control audit](2026-10-29-ilp4-compiled-audit.json)
- [Audit program](2026-10-29-ilp4-compiled-audit.pl)
- [Artifact checksums](2026-10-29-ilp4.sha256)

Full logs, compiled test, compiler-control executable, SASS and resource output
are retained under `local-artifacts/2026-10-29-ilp4/`. The failed probe is
preserved before removal in separate commits. The immutable 50.910426 HTTP
reference, joint-only preference, exact portfolio and runtime defaults remain
unchanged. Proceed only to the independently approved compact-drafter cost
probe; do not rescue this numerical experiment.
