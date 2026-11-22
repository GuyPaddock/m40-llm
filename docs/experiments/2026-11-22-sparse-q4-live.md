# Real SwiGLU vectors on the Q4 checkpoint

## Admission and baseline

The [fixed-mask M40 study](2026-11-21-sparse-q4/result/README.md) cleared
the operator gate for the existing lossless channel-major layout and the
regrouped zigzag layout. Admit one real-activation experiment on the
same Qwen2.5-3B Q4_0 GGUF, SHA-256
`670d82d0fbee6289b661eb323612c65bd2e97ec8c65cdbd9131aa44d30e9816a`.
Use only layers 0, 17 and 35 with 2,048 outputs and 11,008 inputs.
A dense comparison uses this checkpoint's actual projections and
dedicated vocabulary head. Require full forward, explicit dense KV,
greedy sampling, identical tokenizer IDs, no graph and no Q8 preparation.

**Hypothesis.** A fixed per-layer magnitude threshold fitted on completed
post-SwiGLU vectors can reduce charged down-projection time on held-out
Q4 inputs while retaining useful model outputs. The threshold depends
only on that layer's current, already computed vector. It neither predicts
channels from earlier tokens nor skips gate/up computation.

## Calibration and charged timing

1. Reuse the normal one-row forward observer at the down-projection input.
   Bind captured rows to the actual model, tensors, token IDs, layer, and
   position. Check a trace-off dense replay. Capture synchronizations,
   host copies and weight conversion belong outside operator timing.
   Preserve existing Q8-only capture/apply contracts for their callers.
2. Calibrate once per layer on the six calibration tasks in
   [v1.json](../../data/research_quality/v1.json), SHA-256
   `2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38`.
   Keep the first eight generated-input rows per task or all available
   before natural stop. Use the empirical median and 75th percentile of
   absolute magnitudes to target 50% and 25% active. Specify the
   device selector's equality rule, record achieved density and save
   the cutoffs before using held-out data. Keep a zero-cutoff control.
3. Use separate rows from the six held-out tasks, up to eight positions
   each per layer. Compare full native GGUF, full lossless packed,
   thresholded native, thresholded packed, dense zigzag and thresholded
   zigzag on identical inputs. Reuse the existing weights, selector,
   reduction, alternating timing and numerical gates. Include device-side
   selection cost. Compare masked outputs against their own FP64 reference
   and the original unmasked projection. Record per-vector normalized
   RMS error, cosine, live count and actual retained activation mass.
   An all-zero vector must produce finite zero output without dividing
   by its norm.
4. Require at least 10% lower charged held-out latency on at least two
   of the three layers compared with both native dense and the matched
   full-layout control, above twice the larger MAD/median and with a
   positive paired lower bound. Freeze the 50%-active case first for
   the lossless layout. For zigzag, the 25%-active case is the only
   fixed-mask primary density that cleared 10% against dense zigzag.
   Keep the other density as a diagnostic. No kernel geometry search.

## Quality, resource and stopping rules

For each admitted layout evaluate the identical real Q4 model with the
approximation restricted to those three layers. Compare original dense,
unmasked repacked, and thresholded repacked to isolate both sources of
error. Start with the existing held-out tasks and likelihood passages.
Record dense failures separately. Stop an arm on any lost dense pass,
nonfinite value, new degeneration, or a held-out perplexity increase
over 3% relative to original dense. Do not refit thresholds or replace
failed tasks. Require expanded retrieval, code, reasoning, language and
long-continuation evidence before a runtime recommendation; use the
already-frozen November suite and prepare its IDs with the Q4 tokenizer.

Retain per-arm live/peak model VRAM, duplicate down weights, workspace
and load cost. A model-level performance recommendation still needs the
5% ordinary complete-request gate and its independently checked quality
evidence. Insufficient dense coverage or an unsupported normal Q4 path
makes model qualification inconclusive. An operator advantage or a Q8
model's previous calibration cannot qualify Q4 inference. Keep all
branch work explicit and experimental.

## Captured inputs

**PASS for the frozen dense capture.** A release M40 build at
`a971f9011a09096ad49095c541363ad883b1fd40` captured 35 calibration
and 36 held-out SwiGLU rows per layer at layers 0, 17 and 35. All
12 captured generations matched the uninstrumented dense IDs and
both processes ended at zero tracked device bytes. The 50th- and
75th-percentile calibration cutoffs retained exactly 192,640 and
96,320 of 385,280 values per layer. The separate CPU replay
reproduced the complete cutoff document byte for byte. The held-out
vectors and report remained separate from calibration. Capture adds
no performance or approximate-inference quality result.

The [capture checksum list](2026-11-22-sparse-q4-live/capture.sha256)
names the original records in `research-1121/raw/sparse-q4-live-01`,
including both full vector files and their 57,480,279 bytes of JSON.
Verify from that durable directory with `sha256sum -c` and the
checksum list's absolute path. The
[frozen cutoff document](2026-11-22-sparse-q4-live/cutoffs.json)
can be replayed with the release CPU example
`sparse_q4_calibrate calibration.json calibration.jsonl NEW_CUTOFFS.json`.

## Capture and threshold replay

From this worktree, source `scripts/dev-env.sh` and build the tools.

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo build --locked --release --features cuda,server --target-dir target/capture --example model_quality
cargo build --locked --release --no-default-features --target-dir target/capture --example sparse_q4_calibrate
flock ../raw/m40-benchmark.lock bash docs/experiments/2026-11-22-sparse-q4-live/capture.sh ../raw/sparse-q4-live-01
```

The calibration tool checks the checkpoint and corpus identities, the six
task histories, and the capture artifact digest. For each layer it pools
absolute FP32 magnitudes from the eligible generated rows and takes the
nearest-rank 50th and 75th percentiles. The selector retains values whose
magnitude strictly exceeds the cutoff. Ties are discarded. The runner
writes the threshold digest before opening the held-out split and checks it
again after the run. Both splits use the original dense GGUF and explicitly
disabled graph and prepared-weight paths. These observations are untimed.

## Held-out operator replay

The explicit full packed control uses all indices in order and the original
uncompressed input. It pays no selection or compaction cost. Each charged
path includes selection, the device count, projection and reduction. A
diagnostic all-zero row precedes the actual held-out rows. For each vector,
the numerical check compares every output with the matched FP64 sum and
reports error against the original full native and full matched layouts.
The run measures all three cutoffs on all six arms in rotating order with
one warm-up and five measured rounds of twenty calls.

For promotion, use both the synchronized wall interval and the CUDA-event
span. Compute paired fractional reductions within each row and weigh each
task equally by its median row. Use 4,096 deterministic task-level bootstrap
resamples (seed 1121) for the paired interval. Require the median gain to
exceed twice the median within-row relative MAD of the noisier arm. Fewer
than three represented tasks gives an inconclusive interval. The primary
cutoffs, two control comparisons, 10% threshold and two-layer rule above
were fixed before capture. Each arm receives a separate decision. No
kernel parameter search or timing extension is part of this follow-up.

```bash
source scripts/dev-env.sh
cargo build --release --locked --no-default-features --target-dir target/capture --example sparse_q4_rows
bash docs/experiments/2026-11-21-sparse-q4/build.sh sparse_q4_live
flock ../raw/m40-benchmark.lock bash docs/experiments/2026-11-22-sparse-q4-live/operator.sh ../raw/sparse-q4-live-01 ../raw/sparse-q4-live-operator-01
```
