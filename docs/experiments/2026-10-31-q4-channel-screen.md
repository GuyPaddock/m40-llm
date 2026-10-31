# Supplied-index Q4 execution passes the operator screen

The repacked selected-channel implementation passes the complete-FFN gate
in all 216 cases. This supports further channel-selection research. It does
not establish faster generation or preserved model quality. Selection and
recovery costs are absent, and the comparison uses the existing native Q4
consumer rather than the faster qualified Q8 runtime.

Source commit `e46bf13e4728cab22602dbd4a3ae2943ca9dd856` was clean for both
release runs. They completed on the M40 with driver 580.159.04 and no competing
compute process. The frozen Qwen Q8 checkpoint and retained activation capture
were unchanged. Matched Q4 weights were derived using
[the fixed encoder](2026-10-30-q4-fixture-encoding.md).

## Measurements

The grid covers layers 0, 17 and 35; distinct batches 1, 2, 4 and 8; contiguous,
seeded scattered and exact-hidden-magnitude selections; and retention of
10%, 20%, 40%, 60%, 80% and 100%. Each selected variant is compared against
both applicable existing dense controls. At batch one only the fused serial
control applies. The multi-row conclusions below require both comparisons
to pass, rather than choosing the slower control.

Batch-one ranges across all three layers and all three selection patterns:

| Channels retained | Repacked FFN ms | Speedup over matched dense Q4 | Native-storage selected FFN ms |
| ---: | ---: | ---: | ---: |
| 10% | 0.1503 to 0.1540 | 9.89 to 10.14x | 0.4034 to 0.4317 |
| 20% | 0.2518 to 0.2580 | 5.92 to 6.08x | 0.7570 to 0.7958 |
| 40% | 0.4818 to 0.4878 | 3.09 to 3.13x | 1.4888 to 1.5362 |
| 60% | 0.7080 to 0.7195 | 2.08 to 2.11x | 2.2496 to 2.3157 |
| 80% | 0.9010 to 0.9147 | 1.67 to 1.69x | 2.9380 to 3.0005 |
| 100% | 1.1419 to 1.1462 | 1.330 to 1.334x | 3.6616 to 3.7100 |

At 20%, contemporaneous dense Q4 medians are 1.5282 to 1.5334 ms. Every repacked
batch-one comparison has a positive lower confidence bound. At full retention,
the smallest lower gain bound is 32.77%. Native-storage selection passes the
batch-one gate only at 10% and 20% retention. It has no useful margin at 40%
and becomes slower at larger budgets.

Repacked execution also passes all 54 cases at each of batches 2, 4 and 8.
At 20% retention, speedups against the stronger dense control are respectively
6.53 to 6.73x, 7.17 to 7.31x and 7.26 to 7.42x. Native-storage selection is inconsistent
at batch eight: only three of nine cases pass at each of 20% and 40%.

There are 756 comparisons and 3,815 alternating timing pairs. Seven screens
extended from five to ten pairs for variation; none required borderline
confirmation. Two native-storage full-retention comparisons at layer 0,
batch eight remained unstable and cannot support a speed conclusion.
They compare contiguous selection against dense batched, and importance
selection against dense serial. Their candidate throughput MAD/median is
6.33% and 6.14%. These failures remain in the raw report.

All numbers above measure complete FFN execution with supplied indices.
There is no selector, low-rank recovery, token generation, attention or HTTP
request in the timed interval. The 100% result shows that the overall gains
include a kernel/layout improvement independent of sparsity. Comparisons
across retention levels in this grid are not fresh paired measurements
against that improved full-retention implementation.

## Correctness and memory

The separate correctness run completed all 216 cases and 810 independent
FP64-reference checks. The timing run repeated those checks successfully.
Maximum absolute error was 0.000635173, within the fixed elementwise tolerance
`1e-3 + 2e-5*abs(reference)`. Native/repacked selected outputs were bit-identical.
Guards, immutable weights and indices, input values, and per-case allocation
cleanup passed. The GPU had no compute process and zero allocated MiB when
checked after the run.

Each native Q4 matrix occupies 12,681,216 bytes. The repacked down matrix has
the same payload size. Every timing arm retains all three native matrices
and the additional repacked down representation. Maximum tracked residency
was 4,527,169,856 bytes, including the original model and its prepared Q8
weights. This is benchmark residency, not a production sparse-model footprint.

At layer 0, batch one, scattered 20% retention touches an estimated
12,681,216 unique native down-sector bytes but only 3,655,680 packed down-sector
bytes. Gate/up selected logical payload is another 5,073,408 bytes, with
8,808 bytes of indices. Contiguous 20% retention instead estimates 2,588,672
native versus 2,537,472 packed down-sector bytes. These estimates account for
whole 32-byte sectors and shared scales; they are cold unique-sector lower
bounds. They do not measure warp requests, repeated transactions, cache hits
or DRAM traffic. Dynamic counters remain unavailable.

## Next gate

Before implementing the paper's predictor, pair the selected packed path
against packed 100% execution and the qualified Q8 compound/packet FFN on the
same captured inputs. Freeze batch-one 10% and 20% importance selections for
this follow-up. Retain all comparison owners in every arm. This separates
the incremental sparsity benefit and establishes the budget available for
screening and recovery against the relevant current runtime.

The [earlier Q8 operator report](2026-10-29-portfolio-operator-checkpoint.md)
measured approximately 0.325 ms for the qualified packet FFN. That is historical
context, not a contemporaneous cross-representation speed claim. The Q4 path
adds quantization and pruning error, so equal-input timing alone cannot
justify adoption. Sequence quality, logits, repetition, total selector and
recovery cost, and whole-model throughput remain unmeasured. Dense-Qwen
results do not validate routed-expert claims.

## Reproduction and artifacts

Follow the [benchmark protocol and command](2026-10-30-q4-channel-benchmark.md).
The completed runs used `--rows 1,2,4,8 --iterations 10 --trials 5`, with
`--ffn-q4-channels-correctness-only` for the first run. Restore the exact
environment from the retained capture manifest using data arguments, not
shell evaluation:

```bash
source scripts/dev-env.sh
while IFS= read -r research_env_name; do
  unset "$research_env_name"
done < <(compgen -e | rg '^(M40LLM_|CUDA_|OMP_NUM_THREADS$)')
mapfile -d '' research_capture_env < <(
  jq -j 'select(.kind == "ffn_capture_manifest") | .execution_configuration.environment | to_entries[] | .key + "=" + .value + "\u0000"' \
    docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl
)
# Invoke the benchmark command through env "${research_capture_env[@]}".
```

- [Correctness records](2026-10-31-q4-channel-correctness.jsonl.gz)
- [Timing records](2026-10-31-q4-channel-timing.jsonl.gz)
- [Summary](2026-10-31-q4-channel-summary.json) and [summary program](2026-10-31-q4-channel-summary.jq)
- [Release build log](2026-10-31-q4-channel-release-build.log.gz)
- [Correctness log](2026-10-31-q4-channel-correctness.log.gz) and [timing log](2026-10-31-q4-channel-timing.log.gz)

Regenerate the summary with `gzip -dc` on the timing artifact piped to
`jq -s -f docs/experiments/2026-10-31-q4-channel-summary.jq`. The program rejects
an incomplete grid or failed numerical checks. The benchmark's 64 CPU tests,
CUDA/server all-target Clippy and formatting passed before the clean build.
Existing hardware tests ignored in that CPU invocation are not counted as passes.
