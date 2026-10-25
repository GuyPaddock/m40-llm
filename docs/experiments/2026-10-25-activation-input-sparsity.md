# Sparse SwiGLU Input Screen: 2026-10-25

## Decision

C23's input-layout prerequisite passes a bounded isolated speed screen.
With real Qwen2.5-3B Q8_0 down weights and synthetic completed-input vectors,
width16/128 threads reduces selection-plus-projection latency about
18.6%, 30.4% and 39.1% at achieved sparsity 40.2%, 51.2% and 60.6%.
Keep the standalone consumer and proceed to calibration/quality evaluation.
This is not a generated-token speedup, a calibrated model or completed C23.

Activation thresholding changes the model. No runtime dispatch, model weight,
KV policy, sampler or default is changed. C21's negative sparse-weight gate
does not apply to this different dense-weight/sparse-input representation;
C22/C27 remain postponed. Both unchanged-engine speed objectives are unmet.

## Implementation and Method

`10478a8` defines finite magnitude thresholds, quantile fitting and stable
retained original indices. `9b5148c` adds a reusable `8*K+4`-byte workspace,
deterministic CUDA compaction and selected-input projection. It shares tiled
vector loads and FP32 reduction with C10, rather than copying another dense
kernel. `bd76484` extends the existing execution-weight runner with explicit
`--sparse-inputs`; source hashing, tensor loading, ownership, JSONL and GPU
isolation checks are reused.

Measured clean source is `bd76484732b5289b26e1815c2c650e2efbefe9c9`.
Release Rust 1.95.0, NVCC 12.8.93, g++ 14.3.1, CUDA `-O3`, sm_52 SASS and
compute_52 PTX, cuBLAS enabled. Executable SHA-256 is
`4bb18c6a520a638f46b9b5b02032aae2252d3a8955139cd980d04884d8f4c524`.
Model SHA-256 is
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
Only `blk.0.ffn_down.weight`, GGUF `[11008,2048]`, is loaded onto the GPU.
This is not a full-model residency experiment.

Input element i is `((i*13%127)-63)/128`, a deterministic synthetic input
proxy, not captured model SwiGLU. Requested sparsities are 0/25/40/50/60%.
Ties make actual sparsity differ. Magnitude fitting on this vector tests
geometry only; it does not use the browser, C19 calibration, or held-out set.
At requested zero, existing zeros alone give 0.79% omission.

Native Q8 consumes the same CPU-thresholded dense input used by the selected
input reference. The candidate always includes GPU thresholding, compaction,
device count/index reads and projection. Gate/up is unchanged and unmeasured.
Before timing, every output matches native Q8 within existing tolerance,
sixteen FP64 samples agree, and retained indices/values exactly match CPU.
The earlier one-iteration preflight is not pooled with this screen.

Four fixed geometries, widths 8/16 and threads 128/256, cover twenty cells.
Three warmup pairs precede five alternating serial native/candidate pairs per
cell. Each CUDA-event interval repeats 100 operations. Separate selection-only
and projection-only timings diagnose costs; they are not additive predictions
of full-token or HTTP time. Setup, CPU checks, readback, events and boundary GPU
queries are outside the event span. Host submission gaps may affect events.

All 100 pairs / 400 measured operation intervals have zero overlaps. No other
GPU compute process is present. No tests or builds overlap the timed screen.
The M40 starts idle in P8; recorded screen boundaries are P0, SM 949/1063/1113
MHz and memory 3004 MHz, 27-32 C. Clocks are not locked and samples are not
continuous counters. Compare paired ratios within cells; the first cell's
slower native median should not be used as a shared global control.

## Results

CUDA-event milliseconds per operation, width16/128 threads throughout this
table. Percentages use the median paired candidate/native latency ratio.
Stage columns and pipeline columns are independent medians.

| Requested Sparsity | Achieved | Retained Inputs | Native ms | Selection + Projection ms | Selection-Only ms | Projection-Only ms | Pipeline Latency Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.79% | 10921 | 0.172750 | 0.210915 | 0.045072 | 0.165249 | +22.16% |
| 25% | 25.99% | 8147 | 0.172651 | 0.167173 | 0.035166 | 0.131423 | -3.20% |
| 40% | 40.16% | 6587 | 0.172722 | 0.140674 | 0.029371 | 0.110718 | -18.55% |
| 50% | 51.19% | 5373 | 0.172701 | 0.120183 | 0.025089 | 0.094587 | -30.43% |
| 60% | 60.64% | 4333 | 0.172767 | 0.105122 | 0.021700 | 0.082841 | -39.14% |

The zero-threshold consumer is numerically correct but slower. Do not enable
the research path globally or claim that zeros alone improve performance.
Selection itself costs 21-45 us at the steady-clock cells and cannot be
excluded from the result. The marginal 25% row is not a promotion claim;
collect more paired trials if a later decision hinges on such a small gain.

All four configurations at each target are retained in the summary artifact.
The best paired latency ratios across those four are approximately
1.094 / 0.922 / 0.790 / 0.696 / 0.609 at requested 0/25/40/50/60%. Width16/256
leads at 25/40%; width16/128 leads at 50/60%. These are per-cell screen choices,
not measured full-model dispatch policies. No additional geometry tuning was
added. At 40-60%, width16/128 is already sufficiently useful to justify the
next quality stage without promoting a marginal low-sparsity result.

The summary includes baseline/candidate MAD and descriptive paired-bootstrap
95% latency-ratio intervals, 10,000 resamples from a shared xorshift32 stream
seeded 2654435769 in row order. Width16/128 ratios at 40/50/60% are
0.81447 / 0.69567 / 0.60863, intervals
[0.81349,0.81599] / [0.69537,0.69655] / [0.60789,0.60964]. Five observations
do not characterize systematic clock or workload effects. This is an isolated
feasibility screen, not C03's end-to-end promotion gate.

Worst full-output max/mean native drift is `5.07e-7 / 1.08e-7`;
worst sampled FP64 error is `3.21e-7`. GPU selected indices and values match
CPU in every cell. These errors compare the same thresholded vector, not
the unthresholded model. Model logit/quality drift is still unmeasured.

## Memory and Load Accounting

Each original and tiled down tensor occupies **23,953,408 bytes**. They are
both resident, **47,906,816 bytes** combined, excluding input/output/scratch.
The preparation record's old `extra_resident_bytes=0` field means layout-size
growth versus the source, not zero duplicate memory. The sparse timing record
explicitly reports **23,953,408 extra tiled weight bytes**.

Workspace capacity is **88,068 bytes**, allocated once and reused. Active
selected index/value/count payload at requested40/50/60% is
52,700 / 42,988 / 34,668 bytes. Capacity does not shrink with active count.
Repacking width8/16 takes 53.93/55.37 ms and upload 3.84/3.51 ms, single setup
observations. Host peak payload estimate is 71,860,224 bytes; it excludes
allocator/context overhead. If all 36 same-shaped down tensors were prepared,
their extra payload would be 862,322,688 bytes, **an estimate**, not a measured
model-cache allocation. Original GGUF/device weights are not reclaimed.

At requested 40/50/60%, width16 requests 13,490,176 / 11,003,904 / 8,873,984
Q8 value bytes. Every input qblock is still touched, so unique scale payload
remains 1,409,024 bytes. Source-level repeated scale loads and index/value
loads are reported separately; cache broadcasts, reuse, transaction granularity
and actual DRAM bytes are not measured. Reduced requested values are not proof
of proportional DRAM savings. The latency measurement is the positive gate.

The terminal record has zero tracked device payloads. The primary CUDA
context still accounts for about 102 MiB while the process queries its final
state; after process exit, nvidia-smi reports **0 MiB / 0% utilization**.
No model server or benchmark process is left running.

## Validation and Next Work

Four CPU selection/calibration tests and two explicitly included M40 tests
pass. Coverage includes zero/full omission, nonfinite payload preservation,
strict cutoff ties, K/N tails, real down dimensions, scratch reuse, guarded
outputs, rejected calls and owner release. Warm enqueue records two launches
with zero synchronization, copies or allocation/free. The retained dense-tiled
test passes after sharing its internal load/reduction functions.

Existing attention grid, projection/MLP and new sparse/tiled checks pass with
cuBLAS on/off. The cuBLAS-on full forward/ownership suite also passes, with
all four graph fixtures explicitly included. The optional real-Qwen forward
test has no supplied model and is not counted as quality evidence.
Full warning-as-error CI, Clippy and formatting results are recorded with this
checkpoint; GPU-less CI is not numerical GPU evidence.

The non-CUDA locked suite and NVCC/no-device locked `--all --no-fail-fast`
suite both pass with `RUSTFLAGS=-D warnings`. The new hardware tests are
explicitly ignored in that no-device lane. Invoking one with `--ignored
--exact` and the GPU hidden fails with `m40llm_create_context returned null`,
as intended; the [negative-test log](2026-10-25-activation-no-device-negative.txt)
is retained. CUDA/server with cuBLAS on/off and non-CUDA all-targets Clippy pass; formatting and
local report links pass. Hardware parity uses:

```bash
source scripts/dev-env.sh
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test sparse_activation_cuda \
  --test q8_tiled_projection -- --include-ignored --test-threads=1
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=0 \
  cargo test --features cuda --test sparse_activation_cuda \
  --test q8_tiled_projection --test attention_parity_cuda_grid \
  --test proj_wrappers --test mlp_wrappers \
  -- --include-ignored --test-threads=1
```

The next C23 units must collect completed SwiGLU observations from C19's
calibration-only generated prefixes, freeze per-layer/model-identified
threshold artifacts near 25/40/50/60%, and add immutable opt-in application
with reusable scratch and explicit unsupported-path errors. Preserve zero-
threshold/reference parity and count duplicate prepared weights, selection
and setup. Then evaluate both frozen splits with loss/PPL, task scores,
matched-prefix logit drift, actual outputs/counts/stops and complete latency.
No held-out/browser threshold fitting or runtime promotion is authorized.

## Reproduction and Artifacts

Use [the release command and API contracts](../activation_sparsity.md).
The screen is `--sparse-inputs --layout input-tiled --rows 1`, explicit
layer-0 down tensor, `--trials 5 --iterations 100`. The manifest retains all
source, executable, model, compiler, environment and device identity. Logs
are opt-out for timing: stage/launch/sync/allocation logging is disabled.

- [Raw manifest, preparation and 100 timing pairs](2026-10-25-activation-input-screen.jsonl).
- [All twenty cell summaries](2026-10-25-activation-input-summary.jsonl).
- [Runner output and parity diagnostics](2026-10-25-activation-input-screen.txt).

No full-model generation was run under activation sparsity in this checkpoint.
