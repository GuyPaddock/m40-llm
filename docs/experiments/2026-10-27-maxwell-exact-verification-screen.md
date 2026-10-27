# Exact Verification Works But Loses With A Perfect Drafter

## Decision

Stop performance expansion of this verifier. **Do not pursue EAGLE or a
neural drafter on this path.** Even a zero-cost perfect drafter with 100%
acceptance is slower than ordinary target decoding at lengths2/4/8. All
paired throughput intervals are negative, well away from the unchanged
3%/variation gate. No extension or borderline confirmation is required.

This tests the existing temporary-FP32/SGEMM continuation implementation,
not every possible verifier or Maxwell's hardware ceiling. A different
target-verification primitive would need a separate avoided-work case and
qualification before a drafter is justified. No kernel search, draft
model, EAGLE integration, generation default or server scheduling was added.

The [compound FFN](2026-10-27-maxwell-compound-ffn-screen.md) remains separately
eligible for runtime qualification. Preserve the qualified **49.146 HTTP
tok/s** configuration. Both original targets, **53.06** and **56.037 HTTP
tok/s**, remain unmet. The resident serial result below is not a replacement
HTTP baseline or a fresh comparison with Ollama.

## Correctness And Scope

Foundation `0b7992b` and benchmark `885cc98` implement the
[bounded design](2026-10-27-maxwell-exact-verification-plan.md): exclusive
mutable model/session ownership; explicitly dense FP16 KV; real causal
continuation rows at absolute positions; reusable hidden/logits/norm/token
scratch; verified-prefix commit; and suffix clearing/length restoration on
partial commit, rejection, failed commit or abandoned attempt.

All four arms reproduce the exact frozen **65 prompt / 512 generated IDs**
from the qualified Qwen2.5-3B Q8_0 workload. There are four standalone
preflight generations, four repeated preflights in the screen, and twenty
primary generations. No EOS/count differences are hidden. The prompt
produces the cached first token; **511 target input rows**, not 512 extra
forwards, produce the remaining 511 IDs. Widths2/4/8 execute 255/127/63
genuine multi-row batches plus 1/3/7 serial tail rows respectively.

The control is the qualified split-f32x4-128 / packed-warp-256 /
dequant-sgemm configuration with all 109 prepared identities unchanged.
Every arm uses the same packed-prefix/final-prompt-token preparation.
Projection/FFN work is batched, but each query uses the existing dense
attention kernel with its own causal length, and logits/greedy remain
per-position. Include those costs and every temporary dequantization,
workspace transition, stream dependency, completion and transaction step.
This is not four repetitions mislabeled as a batched forward.

## Five Paired Resident-Generation Trials

All measurements are release builds with CUDA/NVCC flags unchanged. Timing
starts before fresh prompt preparation and ends after all 512 IDs exist;
model loading, verifier-session construction and HTTP are outside this
boundary. This is a resident-generation feasibility screen, **not HTTP E2E
qualification**. Arm order rotates/reverses within each trial. Medians of
phases need not add to median total.

| Target rows per batch | Prefill ms | Decode ms | Total ms | Resident tok/s | Gain vs serial | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Serial 1 | 388.88 | 9867.81 | 10257.22 | 49.92 | reference | reference |
| 2 | 389.49 | 43800.73 | 44193.18 | 11.59 | -76.79% | -76.84% to -76.64% |
| 4 | 386.08 | 22992.39 | 23377.70 | 21.90 | -56.12% | -56.21% to -55.69% |
| 8 | 389.15 | 13149.89 | 13541.39 | 37.81 | -24.25% | -24.49% to -23.82% |

Paired bootstrap uses the existing protocol, 10,000 resamples, seed zero.
MAD/median throughput is 0.122% serial and 0.143% / 0.095% / 0.282% for
widths2/4/8. No case requires the ten-pair variation extension or twenty
fresh borderline pairs. Rejected performance is preserved rather than
rescued with another geometry, precision, layout or batch-length search.

The source and counters identify costs present in this implementation,
without assigning exclusive stage milliseconds. The bounded SGEMM path
dequantizes source weights again for each short verification batch; it does
not retain a full FP32 model. Whole-request cuBLAS calls are **252 serial**,
**64,512 width2**, **32,256 width4**, **16,128 width8**. The common prompt
accounts for 252. Attention/logits remain per-position. These facts do not
prove which single cost explains the full regression.

## Drafter Cost And Acceptance

Serial decode averages **19.311 ms per target input row**. Full-acceptance
verification averages **171.687 / 180.576 / 206.486 ms per batch** at
widths2/4/8. Break-even drafter budgets are therefore already negative:

| Width | Serial cost for K rows ms | Verifier batch ms | Maximum drafter cost to break even ms |
| --- | ---: | ---: | ---: |
| 2 | 38.622 | 171.687 | -133.065 |
| 4 | 77.243 | 180.576 | -103.333 |
| 8 | 154.486 | 206.486 | -52.000 |

There is no nonnegative drafting cost or feasible acceptance rate that
rescues this measured perfect-acceptance result. A 3% gain requires still
lower budgets: -134.190 / -105.583 / -56.500 ms per batch.

Measure **70 rejection/correction cycles**, five trials at every accepted
prefix 0..K-1 for K=2/4/8. Each cycle includes the complete speculative
target pass, clearing rejected K/V, publishing shortened layer lengths,
and one ordinary correction step. Every corrected next ID and layer length
matches reference. Prompt preparation is outside correction-cycle timing.
Median target-cycle ranges are 191.69-192.81 ms at K2, 198.71-200.49 at K4,
and 226.49-229.36 at K8. Full acceptance has no correction step.

JSONL includes cost curves for per-token acceptance probabilities
0/.25/.5/.75/.9/.95/1 and a break-even drafter budget at each point.
The curves use an **iid acceptance sensitivity model**, not observed n-gram
or neural-drafter acceptance. Rejection costs are measured at the 65-token
prefix; full acceptance spans the complete 511-row continuation. They are
not a universal later-context rejection-cost model. All reported budgets
are negative, including perfect acceptance.

## Memory And Ownership

Source plus prepared weight payload remains **6,197,100,544 bytes**. Dense
KV capacity is **37,748,736 bytes**, separate from Rust allocation tracking.
Session scratch is **689,668 bytes** in every arm, with capacity for eight
hidden rows and one reusable logits/norm/token slot. No draft model exists.

Observed Rust-tracked boundary high-water is **6,198,064,652 bytes serial**,
**6,288,450,068 width2**, **6,288,865,828 width4** and **6,289,697,348 width8**.
Add native KV separately. These are sampled allocation boundaries, not an
allocator/driver peak or process VRAM measurement. Temporary prefill and
verification buffers are not resident FP32 materialization. The benchmark
includes first-batch and serial-tail workspace transitions rather than
hiding their allocation/free cost. Same-width batches reuse workspace.

The first preflight retains 17 initial model/workspace allocations until
model teardown; every primary arm balances tracked allocations/frees, and
final tracked bytes return zero to zero. The independent final GPU query
is empty. Reject compressed, aliased and multi-sequence caches rather than
silently changing semantics. No compressed-KV transaction is claimed.

## Validation And Evidence

- New CUDA tests pass 2/2 with cuBLAS on and off. They cover head64 and
  Qwen-shaped head128 with Q/K/V biases and split-half RoPE, serial logits
  under the existing tolerance, exact greedy IDs, sampled K/V equivalence,
  every rejection position, prefix bytes, cleared tails, partial/full/zero
  acceptance, abandoned attempts, invalid commit, stop and capacity bounds.
- Existing attention 11, forward 24 and joint-FFN 2 tests return pass.
  Optional real-model diagnostics without supplied models are not evidence
  of additional model coverage. The actual Qwen trace checks above supply
  the real-model greedy checkpoint.
- Non-CUDA warning-as-error tests pass 217. CUDA/server Clippy and formatting
  pass. Benchmark CPU/CLI tests pass 51, with 18 hardware tests ignored in
  that invocation; the verifier CUDA tests were explicitly run separately.
- The first benchmark compilation caught an allocation-method naming error
  and missing serialization on reused pair records. Preserve that log; both
  are corrected before any release timing. No kernel or threshold rescue.
- Independent audit checks raw trace/cache identity, row counts, lengths,
  memory, actual cuBLAS work, medians, variation, cost curves, serial primary
  intervals and all **180 primary/correction GPU boundaries**. Twelve
  deliberately corrupted reports fail. Sampled boundaries contain only the
  owned benchmark at P0, 1113 MHz SM / 3004 MHz memory.
- All 136 prior CUDA entry points retain identical encoded words. The only
  added entry is the bounded dense truncate kernel. No occupancy, stall or
  physical-DRAM measurement or hardware-ceiling conclusion follows.

Clean benchmark source `885cc98acdede4bee913d94ee4592331e78c5bd5`; release
executable SHA-256
`6ef35663ed85ae60d621d1c43be7493343065b9a15ad1fda2d4d91f48ad0376d`.
NVCC12.8.93, `-O3`, sm_52 SASS / compute_52 PTX, no fast math. GPU work is
serial, with no build/test/MemPalace overlap during measurements.

Source dev-env, restore the qualified quality manifest's `.environment`,
then build release `--features cuda --example q8_execution_bench`. Run with
the Qwen Q8_0 model, `--verification-trace
docs/experiments/2026-10-25-maxwell-packed-projection-runtime-quality.jsonl`,
`--rows 2,4,8 --trials 5 --output NEW_PATH`. Optional
`--verification-only-preflight` checks actual512 IDs without primary trials.
No generation CLI or scheduler integration is exposed.

Artifacts: [raw screen](2026-10-27-maxwell-exact-verification-screen.jsonl),
[standalone preflight](2026-10-27-maxwell-exact-verification-preflight.jsonl),
[audit](2026-10-27-maxwell-exact-verification-audit.json),
[audit program](2026-10-27-maxwell-exact-verification-audit.jq),
[corruption controls](2026-10-27-maxwell-exact-verification-negative-controls.sh),
[compiled control audit](2026-10-27-maxwell-exact-verification-sass-audit.json),
[resources](2026-10-27-maxwell-exact-verification-resources.txt.gz),
[orchestration](2026-10-27-maxwell-exact-verification-orchestration.tar.gz),
[test log](2026-10-27-maxwell-exact-verification-tests-on.log.gz), and
[checksums](2026-10-27-maxwell-exact-verification.sha256).

## Next

Do not add a real drafter to this losing path. The next eligible exact-engine
step is separate compound-FFN runtime/actual512/frozen-quality/HTTP
qualification, not another verifier search or EAGLE integration.
