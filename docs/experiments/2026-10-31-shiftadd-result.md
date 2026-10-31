# ShiftAdd projection result

The bounded SA4_128 FP32-input experiment does not advance to FFN or model
integration. It passes the native Q4 comparison in all 36 cases, but loses
to native and prepared Q8 in every case. Its 4.1667% payload saving versus
Q4 comes with higher weight and captured-projection error in every sampled
matrix. Neither advancement route in the
[frozen protocol](2026-10-31-shiftadd-projection-protocol.md) passes.

This result applies to the retained weight-only fit and FP32 sign-add
consumer. It does not evaluate upstream Q8_1 activation arithmetic, calibrated
released checkpoints, other ShiftAdd formats or their model quality. The
positive native Q4 comparisons remain useful evidence despite the stop.

## Controlled measurements

Source `2c8e985` was clean for the release run. The nine full Qwen matrices
cover gate/up/down at layers 0, 17 and 35, with distinct captured rows at
batches 1/2/4/8. All four arms remain resident. Each arm receives at least
one second of GPU warmup per case. There are 108 comparisons and 540
alternating pairs, each sample containing 100 complete projections.

All comparisons are stable. Maximum throughput MAD/median is 0.07373%,
so no ten-pair extension or twenty-pair confirmation is triggered. Every
Q4 comparison clears the 3% floor and positive paired confidence bound;
the smallest Q4 lower confidence bound is +27.077%. All Q8 comparisons fail.
The bootstrap uses 10,000 paired resamples and seed zero.

Batch-one medians across the three layers, in milliseconds per projection:

| Projection | ShiftAdd | Native Q4 | Native Q8 | Prepared Q8 |
|---|---:|---:|---:|---:|
| Gate | 0.416 to 0.419 | 0.529 to 0.534 | 0.205 to 0.207 | 0.127 to 0.128 |
| Up | 0.417 to 0.419 | 0.530 to 0.535 | 0.205 to 0.207 | 0.127 to 0.128 |
| Down | 0.455 to 0.457 | 0.989 to 0.991 | 0.177 to 0.178 | 0.134 to 0.135 |

Across all batch sizes, ShiftAdd throughput is 27.11 to 125.85% higher than
native Q4, 48.80 to 61.08% lower than native Q8 and 69.42 to 73.82% lower than
prepared Q8. Native Q4/Q8 issue one decode launch per row; ShiftAdd and the
selected packed-input prepared Q8 control issue one batch launch. These are
complete projection comparisons, not comparisons against fused FFNs.

Samples retain event latency, submission time and host wall latency. Host
latency spans 0.1273 to 7.5913 ms; submission spans 0.00203 to 0.03049 ms per
operation. Setup, fitting, uploads, references and warmup are excluded from
primary intervals. GPU snapshots show 1113 MHz SM and 3004 MHz memory clocks
throughout comparisons, with temperatures 38 to 86 C. Clocks were observed,
not locked. Snapshot checks found no other compute process.

## Numerical evidence and representation cost

All 72 before/after validations pass, covering 8,663,040 output values across
the four arms. Maximum absolute arithmetic error versus the encoded FP64
reference is 0.00008638; maximum fraction of the operation-count error bound
is 0.006102. Guards, immutable input/weights, actual dispatch, allocation-free
enqueue, nine projection-weight cleanups and final model cleanup pass.
These checks establish arithmetic correctness for each representation.

Fit error is a separate result. Relative to original Q8 values, ShiftAdd's
weight mean squared error is 1.705 to 1.870 times Q4's across all nine tensors.
Captured-projection mean squared error is 1.797 to 38.020 times Q4's across all
36 row cases. These are local weight/projection errors, not perplexity,
logit divergence, task quality or repetition measurements.

Every projection contains 22,544,384 weights:

| Representation or allocation | Bytes |
|---|---:|
| SA4_128 signs | 11,272,192 |
| SA4_128 amplitude codes | 704,512 |
| SA4_128 FP16 scales | 176,128 |
| SA4_128 total | 12,152,832 |
| Q4_0 total | 12,681,216 |
| Q8_0 or its prepared planes | 23,953,408 |
| All four experimental weight owners | 72,740,864 |

The experiment retains the 3,279,519,744-byte original model and
1,192,935,424-byte prepared cache equally for every arm. Peak tracked device
residency is 4,545,613,952 bytes, including a maximum 417,920 bytes of guarded
input/output storage. The kernel has no additional scratch. Final tracked
device bytes return to zero; after process exit the M40 is idle at 0 MiB.
The in-process final snapshot still includes context overhead, which is
distinct from tracked tensor ownership.

Offline ShiftAdd fitting takes 3.829 to 4.479 seconds per matrix, versus
0.351 to 0.427 seconds for Q4. Combined allocation/upload of all experimental
weights takes 11.68 to 13.21 ms. Retained CPU decoded reference arrays occupy
541,065,216 bytes per matrix, plus 72,740,864 encoded bytes. This reference
storage is experimental overhead, not an inference requirement or measured
whole-process peak RSS.

The final release SA4 kernel uses 40 registers and no stack, shared or local
memory. Its SASS contains no DP4A or Tensor Core operations; static counts
include 80 FADD, 40 SHFL.DOWN, eight FMUL and eight FFMA instructions. Dynamic
instructions and DRAM transactions remain unavailable. Reported sector
estimates describe unique backing sectors only; they do not count repeated
warp requests or cache misses.

## Decision and reproduction

The latency route requires a consistent batch-one win over Q4 and prepared
Q8. Only the Q4 part passes. The memory route requires at least 3% payload
saving, no worse sampled weight/projection error and at most a 5% Q4 latency
regression. Storage and latency pass; both error conditions fail. Preserve
the implementation and close its conditional complete-FFN, runtime and
model-quality stages. No generated tokens or whole-model speedup are claimed.
Inference defaults remain unchanged.

The existing [implementation report](2026-10-31-shiftadd-benchmark.md) records
65 passing CPU example tests, 230 passing non-CUDA tests, CUDA/server Clippy
and formatting. Candidate hardware qualification is in the
[CUDA report](2026-10-31-shiftadd-cuda.md). This result adds full-source M40
validation and controlled timing, without rerunning unrelated GPU suites.

Release executable SHA-256:

```text
cf8dba7786991f862e7cf0c707196b8774c06fd9cd2b6636080eb8136352a291
```

Run from the ShiftAdd worktree after sourcing `scripts/dev-env.sh`, selecting
its disk-backed target directory, checking the GPU is idle and building
`cargo build --release --features cuda --example q8_execution_bench` with
cuBLAS enabled. Restore the exact captured environment without `eval`:

```bash
while IFS= read -r research_env_name; do unset "$research_env_name"; done \
  < <(compgen -e | rg '^(M40LLM_|CUDA_|OMP_NUM_THREADS$)')
mapfile -d '' research_capture_env < <(
  jq -j 'select(.kind == "ffn_capture_manifest") |
    .execution_configuration.environment | to_entries[] |
    .key + "=" + .value + "\u0000"' \
    docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl)
env "${research_capture_env[@]}" target/release/examples/q8_execution_bench \
  --ffn-shiftadd \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --output target/shiftadd-projection-02.jsonl \
  --ffn-inputs /mnt/array-fastest/home/guyep/playground/2025-12-20-m40-llm/m40-llm/target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --rows 1,2,4,8 --trials 5 --iterations 100
```

Raw records, build/run logs, SASS, resource counts and the post-exit snapshot
use prefix `2026-10-31-shiftadd-projection`. Reproduce the retained summary:

```bash
gzip -dc docs/experiments/2026-10-31-shiftadd-projection.jsonl.gz |
  jq -s -f docs/experiments/2026-10-31-shiftadd-summary.jq
```

The source/model/capture identities are in the raw manifest. Artifact hashes
are in `2026-10-31-shiftadd-result.sha256`. The next independent topic is the
lossless FP16 KV investigation in `research/1030-lossless-kv`.
