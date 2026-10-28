# Live Causal Prompt-Copy Measurement

The conditional live drafter passes both specialized copy-workload gates.
It is research-only and uses the unchanged prompt-copy source, actual target
predictions and the existing fully direct four-row verifier. **No oracle or
perfect drafts enter this measurement.** No generation/server default changes.

Measured clean source `89ecb54` differs from implementation `8936b71` only by
the [correctness evidence](2026-10-28-draft-live-correctness.md). Release
executable SHA256 is
`1acb8772b58a4a808e428dc2912db8517493c93904eac41a6d316c242fcd352e`.
The Qwen2.5-3B Q8_0 model, exact dense attention, compound/joint-only ownership,
prepared down/head, QKV/output kernels and qualified environment remain fixed.
Target decode never enters temporary-dequant/SGEMM verification. Shared packed
prompt preparation remains charged.

## Whole Resident Requests

| Workload | Prompt / actual output tokens | Serial ms | Live copy ms | Serial / live tok/s | Gain | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Paragraph copy | 132 / 52 | 1789.149 | 1586.189 | 29.064 / 32.783 | +12.7955% | [12.6511%,12.8623%] |
| Config copy | 393 / 68 | 4351.158 | 4078.994 | 15.628 / 16.671 | +6.6723% | [6.6499%,6.8529%] |

Each case warms once per arm, then uses five alternating pairs. Both arms
preserve identical actual greedy IDs/counts and natural stop reasons. The
configured output cap is 128, not the reference's eventual EOS length. There
are no manufactured 128/512-token outputs or cached oracle continuations.
Both sources start fresh from the supplied prompt on every request.

| Median phase, ms | Paragraph serial | Paragraph live | Config serial | Config live |
| --- | ---: | ---: | ---: | ---: |
| Session/source setup | 0.037 | 0.302 | 0.043 | 0.340 |
| Prompt preparation | 844.451 | 844.778 | 3097.252 | 3093.139 |
| Decode including proposals/commit | 944.270 | 741.427 | 1254.761 | 985.151 |

Complete intervals also include source/session teardown, allocations, source
construction, miss handling, committed-history maintenance, copies and stream
dependencies. Phase medians are not summed to infer total gains. No debug
logit/KV readbacks or correctness oracle enters primary timing. Real proposal
counts are 12 and 16 four-row attempts, respectively, with no rejection on
these two cases. Misses and ordinary serial rows remain charged.

The original `max(3%,2*variation)` floor and positive paired-bootstrap bounds
pass. Maximum MAD-over-median is 0.1280% for paragraph and 0.0396% for config.
Both use 10,000 paired resamples, seed zero. Neither variation extension nor
fresh borderline confirmation is required. Independent exhaustive paired
resampling reproduces both interval endpoints. No threshold was lowered.

These are **52/68-token resident copy workloads**, not the browser512 HTTP
benchmark. They do not qualify the 53.06 / 56.153149 HTTP goals or update the
immutable 50.910426 HTTP reference. No HTTP or Ollama inference was measured.

## Correctness, Resources And Isolation

All 21 live preflight traces pass 1,544 full-vocabulary comparisons with zero
difference, 85,948,416 valid KV words bit-identically, actual IDs/counts/stops,
and 516,096 rejected suffix words cleared before correction across 13 natural
rejections. This includes the adverse prose/code/extraction paths as well as
copying. Identity preservation is not a semantic-accuracy claim for model
answers. The original transaction guards and numerical tolerances are intact.

Every timed target has zero cuBLAS calls and zero device allocations/frees.
Serial/live H2D copies are 51/51 for paragraph and 67/67 for config; D2H copies
are 102/102 and 134/134, with the same stream synchronization counts. Target
launches reduce from 22,185 to 20,889 and from 29,145 to 27,417. Existing R4
gate/up sharing supplies this reduction; no new CUDA kernel or geometry exists.
All 138 compiled CUDA entries remain instruction-identical to the prior
operator checkpoint. Prepared owner statistics are unchanged; teardown is 0.

The GPU is empty before invocation. All 40 timed before/after boundaries record
the sole benchmark PID on the same M40, P0, 1113/3004 MHz. No builds, tests,
CPU draft studies, MemPalace calls or other inference overlap primary timing.

Nine CPU causal/cost tests, sixteen CUDA-build CPU/protocol tests, all 219
warning-as-error non-CUDA tests, and 23 affected hardware tests pass. The CUDA
grid is 11/11, transactions 3/3, forward fixtures 2/2, compound 3/3, prepared
weights 2/2 and R4 operator 2/2. CUDA/server all-target Clippy and formatting
pass. All 113 prior reference/ownership/verifier/operator checksums pass.

## Evidence And Use

- [Raw measurements](2026-10-28-draft-live-measurement.jsonl),
  [independent audit](2026-10-28-draft-live-audit.pl) and
  [audit result](2026-10-28-draft-live-audit.json) retain/reconcile outputs,
  coverage, work, ownership, clocks, pairs and statistical decisions. Eight
  corrupted reports are rejected.
- [Compiled audit](2026-10-28-draft-compiled-audit.json) and
  [logs/resources](2026-10-28-draft-logs.tar.gz) preserve tests, capture, CPU
  studies, live runs, checksum checks and compiler evidence.
- The measured binary/source are retained read-only under
  `local-artifacts/qualified-references/draft-live-89ecb54/`. Original CPU-study
  and held-out capture executables/source remain separately retained under
  `local-artifacts/qualified-references/draft-study-d2f8070/`.

```bash
source scripts/dev-env.sh
mapfile -t frozen_env < <(jq -r \
  '.configuration.environment | to_entries[] | "\(.key)=\(.value)"' \
  docs/experiments/2026-10-27-compound-reference.json)
env "${frozen_env[@]}" target/release/examples/draft_feasibility live \
  --model /mnt/array-fastest/home/guyep/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d \
  --heldout docs/experiments/2026-10-28-draft-heldout.jsonl \
  --correctness-report docs/experiments/2026-10-28-draft-live-correctness.jsonl \
  --output /tmp/fresh-live-copy.jsonl
```

Use exclusive GPU access and the exact admitted executable. For a new
executable, omit `--correctness-report` to run all preflight cases first.
The output must be new. This command selects a bounded research runner, not
the production `generate` CLI. Complete assessment and ranking are separate
from these measured results; no further source/kernel tuning follows.
