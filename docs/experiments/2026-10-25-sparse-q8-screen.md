# C21 Sparse-Q8 Kernel Screen

## Decision

**Negative decode gate.** Fixed 16-of-32 storage is correct and saves 29.41%
of requested weight-buffer bytes, but all four tested single-row kernels are
slower than the matched-zero native Q8 control. Even the fastest per-family
geometry takes 2.29x gate, 2.31x up, and 2.66x down latency. Do not calibrate
C22 model pruning or proceed to C27 sparse-Q4 on this backend. There is no
end-to-end inference result, model-quality claim, or runtime default change.

Some gate/up multi-row microbenchmarks improve, but the down projection still
regresses and decode is the research objective. These partial wins do not
justify a calibrated sparse runtime. The 64-row gate/up gains are also below
the existing 3% promotion floor. No extra geometry/density search was run.

## Provenance and Method

- Clean preservation revision `6d02feb6efa89856146d103269abda3590b2860a`.
  Host format `df9de8a`, CUDA `ba57428`, measurement tool `f2c5063`,
  setup-order correction `6d02feb`. Commit the negative result before
  separately removing the active losing CUDA probe/tool.
- Qwen2.5-3B Q8_0 model SHA-256
  `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
  Layer-0 gate/up are K=2048,N=11008; down is K=11008,N=2048.
- Release Rust, NVCC 12.8.93, CUDA `-O3`, `sm_52` SASS and `compute_52`
  PTX, cuBLAS enabled, no fast-math/Tensor-Core requirement. GPU Tesla M40
  24GB, driver 580.159.04, UUID recorded in the raw manifest.
- Supplied masks are uniform deterministic sixteen-position subsets, seed
  203494962, independent of source values and evaluation prompts. Original
  FP16 scale bits are retained. This is not saliency/pruning calibration.
- Dense control is the *same zeroed matrix*, not the original nonzero tensor.
  Single-row native control uses C11 coltile4/split4 on decode; multi-row
  control uses existing shared-activation/block-loop dispatch on prefill.
  Sparse uses decode, no per-call allocation or synchronization.
- Three warmups per side/configuration. Five alternating serial pairs for
  rows 1/4/16/64, columns 1/4, threads 128/256. Repetitions per measurement are
  16/4/2/2 respectively. CUDA events drain the actual consumer stream;
  setup/input validation is outside timing. Competing GPU processes cause
  failure. No other GPU workload or build overlaps timed work.
- Complete artifact has one manifest, three preparations, 240 timing pairs,
  and one completion record. JSON includes wall/event times, GPU state,
  storage/setup costs and source/mask/packed/matched-dense hashes. It is not
  240 independent trials of a selected runtime configuration.

## Per-Shape Results

GPU milliseconds per complete projection, medians of five pairs. The fastest
sparse configuration is shown for each shape, chosen only for this screening
summary. Speedup is median dense divided by median sparse. Pair ranges are
descriptive ratios, not confidence intervals or E2E promotion evidence.
All 48 configurations and wall times are in the [derived summary](2026-10-25-sparse-q8-summary.json).

| Projection | Rows | Columns/Threads | Dense ms | Sparse ms | Speedup | Pair Range |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gate | 1 | 1/128 | 0.227844 | 0.522788 | 0.436x | 0.435-0.436x |
| gate | 4 | 1/128 | 2.342072 | 2.057208 | 1.138x | 1.135-1.139x |
| gate | 16 | 1/128 | 8.289008 | 7.518752 | 1.102x | 1.102-1.106x |
| gate | 64 | 1/128 | 29.808176 | 29.186943 | 1.021x | 1.019-1.022x |
| up | 1 | 1/128 | 0.201090 | 0.465030 | 0.432x | 0.431-0.435x |
| up | 4 | 1/128 | 2.342552 | 2.058632 | 1.138x | 1.091-1.138x |
| up | 16 | 1/128 | 7.901936 | 7.304864 | 1.082x | 1.073-1.092x |
| up | 64 | 1/128 | 29.853840 | 29.194208 | 1.023x | 1.021-1.024x |
| down | 1 | 1/256 | 0.173776 | 0.461760 | 0.376x | 0.375-0.380x |
| down | 4 | 1/256 | 1.654560 | 1.799424 | 0.919x | 0.916-0.922x |
| down | 16 | 1/256 | 4.692064 | 7.136832 | 0.657x | 0.657-0.658x |
| down | 64 | 1/256 | 20.320065 | 28.493776 | 0.713x | 0.713-0.714x |

All complete-run numerical checks pass. The largest full-output absolute
sparse/dense difference is 7.272e-6; the largest sampled FP64 scalar difference
is below 1.20e-7. Both signed/zero scales and synthetic ragged K/N are covered
by separate format/CUDA tests. Different FP32 reduction order is expected;
this is tolerance parity, not bitwise generated-model equivalence.

The measured single-row gap is much larger than paired timing variation, so
additional borderline-gain confirmation is not warranted. Small event/wall
differences do not change that conclusion. Kernel profiling counters were not
collected; this run does not establish which instruction, occupancy or memory
transaction mechanism causes the regression.

## Storage and Preparation

Each tensor has 704,512 blocks, 22,544,384 logical weights, and 11,272,192
retained logical positions, exactly 50%. All three shapes have K divisible by
32; the separate CPU/GPU tests cover padded tails.

| Requested Bytes Per Tensor | Bytes | MiB |
| --- | ---: | ---: |
| Matched native Q8_0 | 23,953,408 | 22.84375 |
| Ideal sparse, 22 bytes/block | 15,499,264 | 14.78125 |
| Stored sparse, 24 bytes/block | 16,908,288 | 16.12500 |
| Retained i8 payload | 11,272,192 | 10.75000 |
| Mask | 2,818,048 | 2.68750 |
| Original FP16 scale | 1,409,024 | 1.34375 |
| Alignment padding | 1,409,024 | 1.34375 |

Host packing plus validation takes 186.83-193.99 ms/tensor, separate from
186.55-196.37 ms synthetic-mask generation. Sparse upload observations are
6.09-6.45 ms; dense upload 3.01-3.22 ms. Pageable upload calls can return before
final DMA, so these setup call times are not isolated DMA-completion metrics.
The readback fence precedes checks/timing. Benchmark weights occupy both
representations concurrently, 40,861,696 requested bytes per tensor; that is
not an integrated model memory saving. Driver/context overhead is separate.
No DRAM traffic counter is available; `measured_dram_bytes` remains null.

## Aborted Setup-Ordering Evidence

The [first unfenced attempt](2026-10-25-sparse-q8-unfenced-aborted.jsonl)
stopped before down/M16 timing on a 0.0456742 parity difference.
An [isolated M16 replay](2026-10-25-sparse-q8-unfenced-isolated16.jsonl)
passed, but the [M1/M4/M16 diagnostic](2026-10-25-sparse-q8-unfenced-repro.jsonl)
reproduced row5/column0: sparse 0.5484369, scalar 0.5484367777, dense 0.5027627.
The detailed replay used an uncommitted diagnostic-only checker; its manifest
records that fact. These files are failure evidence, not pooled timing data.

The final setup copy used pageable H2D on the default stream without a fence
before nonblocking prefill consumed it. CUDA permits that copy to return
before DMA completion. The correction performs a same-default-stream D2H
readback and validates all input values before either numerical consumer.
The full matrix then passes without changing either kernel or loosening
numeric tolerances. See the [format/copy contract](../sparse_q8.md).
This does not audit or fix every pre-existing runtime H2D site; inspect those
producer/consumer contracts before attributing any similar mismatch to
quantization or model behavior.

## Active-Code Removal

Results are committed in `51fa76b` after the complete tested code in `6d02feb`.
A separate follow-up removes the failed CUDA kernel, FFI/device interface,
hardware target and benchmark. The CPU format and all raw artifacts remain.
Production CUDA/build/FFI files match the pre-C21 tree after that cleanup.
Post-removal attention/projection/MLP checks pass with cuBLAS on/off, along
with the forward suite, warning-as-error non-CUDA and NVCC/no-device CI lanes,
both all-targets Clippy configurations and formatting. No performance rerun
is claimed after removal; there is no sparse execution path to promote.

## Validation and Reproduction

At the preservation revision, use the commands in [the probe guide](../sparse_q8.md).
The qualified run uses the guide's defaults with the model blob above,
`CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852` and launch,
sync and allocation logging disabled. Collect into a new file outside the
worktree so the manifest records a clean source revision.

CPU format tests, expanded sparse M40 parity, attention grid, projection/MLP
wrappers, non-CUDA warning-as-error tests, CUDA/server all-targets Clippy and
formatting pass. Sparse/projection/MLP wrappers also pass with cuBLAS disabled
before the expanded setup regression. Hardware tests require explicit
`--include-ignored --test-threads=1`; a no-device skip is not GPU validation.
After process exit, nvidia-smi reports no compute processes, 0 MiB, 0%.

Qualified raw SHA-256:
`c032d70b1fbb3ecfb8cc10bd1b51b1995ccd5e0eed2eeba0f177d27a73dadeaf`.
Raw [JSONL](2026-10-25-sparse-q8.jsonl) is the authority; the summary uses
per-configuration medians and pair ranges only. No quality inputs, native
engine configuration, or runtime/KV defaults changed.
