# Live copy-only speculation with shared Q8 down weights

**PASS for both frozen copy workloads on the M40.** The same
causal prompt-copy drafter improved the complete resident CLI
request with both target schedules. Shared down weights improved
it further. Five fresh alternating rounds per workload exceeded
the frozen 3% improvement over ordinary decoding. The incremental
gain over the existing multi-row verifier exceeded twice the
observed variation and its paired interval stayed above zero.

| Workload | Ordinary tok/s | Existing copy tok/s | Shared copy tok/s | Gain vs ordinary, 95% interval | Gain vs existing copy, 95% interval |
|---|---:|---:|---:|---:|---:|
| Paragraph, 128 prompt / 52 output IDs | 28.813 | 32.990 | 33.832 | 17.416% [16.678%, 18.520%] | 2.551% [1.759%, 3.243%] |
| Configuration, 384 prompt / 68 output IDs | 15.525 | 16.694 | 16.912 | 8.934% [8.644%, 8.986%] | 1.308% [1.156%, 1.572%] |

The paragraph ran 12 four-row verification attempts with 15 lookups.
The configuration ran 16 attempts with 19 lookups. Every prompted
draft was accepted in these two timing cases. Median shared target
verification plus commit cost was 626.56 ms and 846.50 ms, compared
with 666.58 ms and 903.29 ms under the existing multi-row schedule.
Median CPU lookup and committed-source maintenance took 0.042 ms and
0.117 ms in the shared arm. Those times exclude source construction,
which the full timed request includes along with setup, packed prefill,
ordinary steps and teardown. There were no allocations, frees or
cuBLAS calls during the measured decode interval.

The executable-bound correctness run preceded timing. Both copy
schedules matched all 21 fixed generations, 1,544 full 151,936-value
logit rows and 85,948,416 valid FP16 KV words bit for bit. Thirteen
natural rejection transactions per candidate cleared 516,096 suffix
words and reproduced all ordinary corrections. Both processes ended
at zero tracked device bytes. The 250-ms sampler observed up to
6,228 MiB across the correctness and timing processes. Both schedules
used the same prepared weight owners. Neither creates a server
throughput or general-drafting claim.

Retain the result as an opt-in copy workload candidate after the
clean-base Q8 and verifier dependencies qualify. The successful
scope is greedy prompt copying on this Qwen2.5-3B Q8_0 checkpoint.
The general adaptive controller remains closed. Any future
serving version needs an independent complete HTTP workload.

The [raw archive](measurements.tar.gz), [checksum](archive.sha256)
and [independent replay](replay.json) retain both stages, targets,
source costs and GPU preflight. Use the
[frozen protocol](../../2026-11-22-copy-verifier.md) and
[M40 runner](../run.sh) on source
`8c490d1ad81df139a941afb48ba3f23dd789de7a`. Replay with
`bash docs/experiments/2026-11-22-copy-verifier-audit/replay.sh EXTRACTED RESEARCH_HTTP_PAIRS`
from the consolidation worktree. The audit reproduced both
paired decisions before and after extraction.
