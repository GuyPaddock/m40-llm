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
