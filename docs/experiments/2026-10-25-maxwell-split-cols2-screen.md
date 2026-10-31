# Two-Column Split FFN Screen

## Decision

The fixed two-column/128-thread experiment **loses** against the qualified
one-column split-plane FFN. Complete one-row throughput is **5.85-6.01%
slower**; genuine four-row work is **4.38-5.45% slower**. Every paired interval
is negative. Preserve implementation `414f2a5` benchmark-only. Do not add
generation selection or more column counts to rescue this result.

The qualified runtime remains one-column split-f32x4-128 plus SGEMM, last
measured at 45.075 HTTP tok/s versus fresh official Ollama 43.119. Both
original goals remain unmet: 53.06 and 56.055 tok/s for 1.30x that median.
No full-model quality or HTTP trial of the losing candidate is warranted.

## Fixed Experiment And Results

Clean release source `414f2a521662b7ea2840eb47df98d6ba90f34a78`, benchmark
SHA-256 `a8cfe71ed2653d05c1adec506db26b7a7fbd510f03697067d9bfdcdb82e605a6`.
The new [split-runtime profile](2026-10-25-maxwell-split-stage-profile.md)
motivated activation/block-setup reuse without changing task assignment,
per-weight FP32 scaling, reduction tree, weight layout or prepared down.

Use the frozen M02 environment and inputs with the existing shared runner:

```bash
source scripts/dev-env.sh
target/release/examples/q8_execution_bench \
  --model "$MODEL" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-f32-split-two-columns --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-split-cols2-screen.jsonl
```

Each latency is the median of five alternating complete-FFN timing pairs,
ten repetitions/stage. Gain is control latency / candidate latency minus one.
Bootstrap intervals resample paired observations 10,000 times with seed zero.
The control is current **one-column split**, not the older packed/warp128.

| Layer | Rows | One-column ms | Two-column ms | Throughput gain | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.351936 | 0.373821 | -5.85% | -5.96% to -5.71% |
| 17 | 1 | 0.352304 | 0.374848 | -6.01% | -6.35% to -5.61% |
| 35 | 1 | 0.351994 | 0.374336 | -5.97% | -6.25% to -5.69% |
| 0 | 4 | 1.586515 | 1.659184 | -4.38% | -4.97% to -3.85% |
| 17 | 4 | 1.580659 | 1.658899 | -4.72% | -5.92% to -3.71% |
| 35 | 4 | 1.579165 | 1.670269 | -5.45% | -5.53% to -4.74% |

One-row gate/up/SwiGLU increases from 0.2218-0.2223 ms to
0.2425-0.2429 ms; isolated prepared down remains about 0.131 ms. The loss
is inside the changed stage, not a different down path. Stage sums need not
equal complete timing because cache histories differ. No trial extension
is justified for this clear regression, and no geometry search follows.

## Correctness, Resources And Ownership

All six real shapes preserve **one-column hidden and completed down bits**.
Captured gate/up, ordinary/captured SwiGLU, output guards and immutable
prepared planes pass the existing CPU tolerance. Largest absolute CPU
errors are 2.0644e-6 gate, 3.4088e-6 up, 1.7127e-5 SwiGLU and 5.0080e-5
completed down. Synthetic tests additionally cover signed extremes, zero
scales, odd hidden tails, alignment rejection, rows 1/4 and queued producers.

Both arms retain the same 47,906,816-byte prepared gate/up pair, comprising
2,818,048 scale bytes and 45,088,768 value bytes. Original plus prepared
down/vocabulary payload remains 4,472,455,168 bytes. Scratch is 201,152
bytes at one row and 803,264 at four rows; observed tracked totals are
4,520,563,136 / 4,521,165,248 bytes. No KV/attention or Q8_1 scratch is used.
Median temporary packing/upload time is 28.563 / 8.191 ms, outside timing.
Both arms reuse that allocation set and final tracked storage returns to zero.

The candidate uses **40 registers**, versus **32** for control, with no
reported stack/local spill storage. Dynamic shared memory is 2048 versus
1024 bytes; one block-wide barrier remains. Halving static blocks and
activation requests did not improve latency. Higher register pressure is a
measured compiler property, not proof that occupancy caused the regression.
CUPTI counters remain unavailable; there is no measured bandwidth or
occupancy attribution.

The qualified control's extracted sm_52 SASS is byte-identical before/after,
SHA-256 `84a8cfe7d64c3ac38fc52666e7f509a79893422e50f683db591f7e4c4a1c69fd`.
Its GPU body was not retuned to favor the candidate. All 30 primary pairs /
180 stage intervals are nonoverlapping; 60 boundaries observe P0 and
1113/3004 MHz clocks with only process 1300420. No build, tests, MemPalace
embedding or Ollama runs during timing; the GPU ends empty.

Before timing, both cuBLAS-enabled/disabled primitive and benchmark suites
pass 2+12 tests; enabled affected runtime/attention/forward/wrapper tests
pass, with any external-model fixture skip not counted as real-model proof.
The cuBLAS-off grid/wrappers pass 14 tests. Non-CUDA warning-as-error tests
pass 208 cases; CUDA/server all-target Clippy and formatting pass.

## Evidence And Next Direction

- [Raw pairs, validation, preparation and cleanup](2026-10-25-maxwell-split-cols2-screen.jsonl)
- [Paired/bitwise audit](2026-10-25-maxwell-split-cols2-audit.json) and [program](2026-10-25-maxwell-split-cols2-audit.jq)
- [Memory audit](2026-10-25-maxwell-split-cols2-memory-audit.json), using the unchanged [ownership audit](2026-10-25-maxwell-split-ffn-memory-audit.jq)
- [Run log](2026-10-25-maxwell-split-cols2-screen.log.gz), [compiler resources](2026-10-25-maxwell-split-cols2-resources.txt), [SASS](2026-10-25-maxwell-split-cols2-sass.txt.gz), [checksums](2026-10-25-maxwell-split-cols2-screen.sha256)

Negative controls with an obsolete packed timing control, timed allocation,
or fabricated positive gain are rejected by the audit. Earlier audits and
hash-protected results are not rewritten for the new control.

Stop output-column tuning. Next audit the retained one-column kernel's
issued weight/scale loads and address/conversion work. A possible orthogonal
experiment is losslessly pairing gate/up payloads and scale bits so one
vector load replaces separate matrix loads, while retaining one column,
four-value tasks and the FP32 tree. Establish removable instruction work and
bounded memory ownership before implementing that layout. It is a hypothesis,
not a measured bandwidth claim or permission to relax any micro/quality/HTTP
gate. Prepared down/vocabulary and runtime defaults remain fixed.
