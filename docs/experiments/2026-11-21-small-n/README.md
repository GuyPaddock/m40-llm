# Matching the small-row measurement to verification cost

The direct target in `src/infer/direct_verification.rs` rejects a down
owner unless its single-row launch is Q8SeparateLaunch(4, 256). Each
of 36 layers invokes that prepared-down projection four times and adds
the four residuals. Its direct four-row gate/up stage already reuses
weights. The November serial arm matches those four separate down calls.
Existing multirow and shared weights are additional same-arithmetic
controls. The native, warm SGEMM and dequant-plus-SGEMM arms have distinct
resource and arithmetic costs. A cached FP32 weight for one measured
11008-by-2048 down projection occupies 90,177,536 bytes.

On the historical M40 run, each of five direct requests verified 127
complete batches, used three ordinary tails and returned the same 512
greedy IDs as its serial control. All 36 layers ended at 576 KV positions.
The median of the five per-request mean four-row intervals was 57.8203 ms.
This measures begin through verified commit on a perfect draft. A 5%
reduction requires 2.8910 ms per complete target batch, equivalent to
80.31 microseconds per layer if the entire gain comes from down projection.
A 5% increase in four-token throughput has a slightly lower requirement
of 2.7533 ms per batch. Compare actual complete costs after an admitted
prototype. Resident request, rollback/correction and live drafting have
different denominators. A 5% reduction in the historical 7,781.27 ms
resident request would require 389.06 ms across the request.

Reconstruct [the budget](target-budget.json) with
`jq -es -f docs/experiments/2026-11-21-small-n/target-budget.jq docs/experiments/2026-10-28-direct-verifier-measurement.jsonl`.
The retained historical JSONL SHA-256 is
555646bab7f3200452c0dce9b30b3317f19d42bdecdc517016a62ec10a87b69e.
The checks cover the model identity, committed positions, all direct trials,
batch widths and agreement of all ten generated ID vectors.

## Compiled resource cost

The [sm_52 ELF](compiled-resources.txt) reports 40, 56, 72 and 93
registers per thread for one through four input rows in a block. The
original four-output-column kernel uses 40. Each launch has 256 threads
and 4,096 bytes of dynamic shared memory; the ELF reports zero additional
static shared memory, local memory and stack for these kernels. Inputs of
eight or sixteen use two or four groups of four. For width four, the
split-plane source contains 23,953,408 bytes per weight traversal;
the shared kernel can use the loaded scale and codes in four input
accumulations. Actual DRAM transactions depend on cache behavior.
Resource counts and source-level reuse alone do not establish speed.

Read the five alternating wall and CUDA-event spans for each of the three
actual down weights separately. Measure serial minus shared per pair
and compare against existing multirow and both SGEMM controls. Scaling
the three sampled-layer differences to 36 estimates headroom for the
gate. It is not an observed full-verifier latency. An ambiguous estimate
needs a matched target measurement; a successful rank at one, three or
sixteen rows alone cannot reopen live speculation.

The build checks at 6d58b4e passed formatting, the locked non-CUDA test
suite with warnings denied (223 reported passes, zero failures), and
CUDA/server all-target Clippy with NVCC and cuBLAS enabled. The [build
records](build/status.tsv) include exits, exact commands and compressed
logs. The CPU-only run contributes no GPU parity or timing evidence.
