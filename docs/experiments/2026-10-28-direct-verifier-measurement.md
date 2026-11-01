# Fully Direct Four-Row Target Measurement

The [approved prototype](2026-10-28-direct-verifier-plan.md) passes its
complete resident-generation feasibility gate after the
[correctness checkpoint](2026-10-28-direct-verifier-correctness.md).
No new kernel, drafter, speculative generation selector or runtime default
is introduced. This is not an HTTP comparison or an Ollama rerun.

## Fixed Configuration

Measured source `50d4d17f3ca59e7a8999fea7c385ff01f288c276` is clean.
The release executable remains
`7e922a3e4e95613254b6520d6bbf38c33880ee12d03b33c7fad6b3d83dae7de4`,
the exact executable admitted by the correctness run at `83c59a8`.
Only documentation/audits changed between those commits.

Qwen2.5-3B Q8_0 model SHA256 remains
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
The manifest records every frozen environment value, NVCC version,
`-O3`, `sm_52`, `compute_52` and no fast math. Both arms use joint-only
compound ownership, original prepared down/head, qualified packed-warp
QKV/output, split8/256 exact dense head128 attention and the same packed
SGEMM prompt preparation. The direct target uses the fixed R4 gate/up reader
and original single-row consumers for everything else. It never materializes
FP32 target weights or enters target SGEMM.

One process occupies the M40 throughout. All 60 primary/rejection before/after
boundaries record P0, 1113/3004 MHz and the same sole benchmark PID. No builds,
tests, MemPalace operations or other inference overlap timed GPU work.

## Complete Request Results

One full warmup per arm precedes five alternating pairs. Each measured
request actually generates the same 512 frozen greedy IDs from the same
65-token prompt, with length stop and 576 committed KV positions per layer.
The first answer comes from prefill; the remaining 511 target input rows are
127 four-row attempts plus three ordinary serial tails for the candidate.

| Median host interval | Qualified serial | Direct four-row |
| --- | ---: | ---: |
| Session setup ms | 0.045 | 0.329 |
| Packed prompt preparation ms | 387.693 | 382.874 |
| Decode, commit and tails ms | 9508.347 | 7399.527 |
| Session teardown ms | 0.043 | 0.103 |
| Complete resident request ms | 9890.383 | 7781.267 |
| Complete resident tok/s | 51.767457 | 65.799057 |

Phase medians are descriptive and need not sum to the median total. Every
raw request retains its own complete elapsed interval and bookkeeping cost.
The direct mean batch cost has a five-request median of **57.820 ms**.
All 635 complete batch observations range from 57.064 to 58.852 ms, with
median 57.835 ms. These intervals include `begin` and verified `commit`.
They replace the earlier optimistic 60.17-60.43 ms assembly estimate.

Complete resident throughput gain is **27.1051%**, paired 95% bootstrap
interval **[26.8683%, 27.2526%]**, 10,000 paired resamples, seed zero.
Control/candidate MAD-over-median is 0.1160%/0.0481%. The unchanged floor is
3%, the interval is positive, and neither variation nor borderline
confirmation condition is met. There is no extension or pooled confirmation.
Independent exhaustive enumeration of all 3,125 paired empirical resamples
gives the same interval endpoints; it does not replace the seeded protocol.

This qualifies an idealized perfect-draft resident result only. The serial
51.767 value is not a revision of the immutable 50.910426 HTTP reference.
HTTP buffering and request boundaries differ, and no real drafting cost or
acceptance rate has been measured.

## Work And Ownership Accounting

Every target arm makes zero cuBLAS calls and zero device allocations/frees.
Both perform 511 H2D copies, 1,022 D2H copies, 1,022 stream synchronizations
and no target stream waits. Whole-request accounting includes the shared
prompt's 252 cuBLAS calls and 324 stream waits. The serial/direct request
allocates and frees 47/63 buffers respectively. Those setup transitions,
copies and dependencies are charged, not hidden behind the decode interval.

Target launches are 222,285 serial versus 208,569 direct. The candidate has
4,572 R4 gate/up launches and 108 ordinary compound tail launches. Both keep
18,396 QKV, 18,396 output and 18,907 prepared down/vocabulary launches.

Both retain 3,279,519,744 uploaded source bytes, 1,192,935,424 prepared split
bytes, 1,724,645,376 joint bytes and 37,748,736 dense KV bytes. Warm model
tracked residency is 6,197,374,984 bytes. Session scratch is 689,668 serial
versus 1,521,188 direct; candidate peak at the recorded phase boundaries is
6,198,896,172 bytes. These are tracked boundary measurements, not a claim
about every untracked driver allocation or a sampled memory high-water mark.
Each session returns to its starting residency; model teardown returns to 0.
All 36 joint and 37 prepared payload hashes are unchanged after timing.

## Rejection And Correction Costs

Five observations per rejection position follow the primary pairs. They
include the complete four-row target attempt, truncation/clearing of the
rejected suffix and an ordinary serial correction. No rejected suffix is
reused. All corrections produce the actual expected next token.

| Retained prefix | Target mean ms | Rollback mean ms | Correction mean ms | Total mean ms |
| --- | ---: | ---: | ---: | ---: |
| 0 | 57.409 | 0.023 | 18.495 | 75.928 |
| 1 | 57.363 | 0.015 | 18.467 | 75.845 |
| 2 | 57.437 | 0.018 | 18.501 | 75.956 |
| 3 | 57.378 | 0.015 | 18.494 | 75.887 |

These rejection cycles begin at the 65-token prefix, not every later context.
Prefix zero is an invalid cached-seed diagnostic, not a natural real-drafter
branch. For a known-correct first candidate, only the next three tokens are
guesses. Rejection setup/prefill/teardown is excluded from these component
intervals and charged once by the complete-request measurements above.
The subsequent feasibility model must retain this limitation and charge
ordinary correction and all fixed/transaction work; perfect acceptance alone
does not establish a practical drafter.

## Evidence And Reproduction

- [Raw measurements](2026-10-28-direct-verifier-measurement.jsonl) include
  every batch, serial step, phase, actual token ID, counter, GPU boundary,
  rejection observation, identity and before/after payload hash.
- [Record audit](2026-10-28-direct-verifier-audit.pl) and
  [result](2026-10-28-direct-verifier-measurement-audit.json) reconcile actual
  work, alternating/nonoverlapping pairs, clocks, outputs, accounting and
  summary statistics. Ten altered measurement reports are rejected. The
  synthetic extra-process case exposed and fixed a Perl list-context regex
  checker bug before audit completion; no actual GPU contamination occurred.
- [Logs and compiled resources](2026-10-28-direct-verifier-logs.tar.gz)
  retain builds, tests, both preflights, timing and final SASS/resources.
  [All 138 compiled entries](2026-10-28-direct-verifier-compiled-audit.json)
  remain unchanged from the preceding R4 operator checkpoint.
- Read-only local executable and measured-source archive are retained in
  `local-artifacts/qualified-references/direct-verifier-50d4d17/`.
  The source archive SHA256 is
  `a23a4a7f9d6768f90a822585559ffcdcce03b622a5862dcce094ae0939717900`.

```bash
source scripts/dev-env.sh
mapfile -t frozen_env < <(jq -r \
  '.configuration.environment | to_entries[] | "\(.key)=\(.value)"' \
  docs/experiments/2026-10-27-compound-reference.json)
env "${frozen_env[@]}" target/release/examples/q8_execution_bench \
  --model /mnt/array-fastest/home/guyep/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d \
  --direct-four-row-verifier \
  --direct-verifier-correctness-report docs/experiments/2026-10-28-direct-verifier-correctness.jsonl \
  --rows 4 --trials 5 --iterations 10 --output /tmp/fresh-direct-verifier.jsonl
perl docs/experiments/2026-10-28-direct-verifier-audit.pl \
  docs/experiments/2026-10-28-direct-verifier-measurement.jsonl \
  --measurement --self-test
```

Use clean committed source, the exact admitted executable and exclusive GPU
access. The output file must not already exist. The next and final authorized
step is the measured acceptance/drafter-budget assessment, then stop for review.
