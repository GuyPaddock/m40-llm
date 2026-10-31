# Frozen Maxwell FFN Inputs

## Result

Clean release `f459c9777d49ffaace6d0fd27a31d5f340919106` captures paired,
unmodified FFN inputs from original Qwen2.5-3B Q8_0 layers 0, 17 and 35.
Each layer has 21 generated-input rows from the six frozen calibration tasks.
All 163 actual generated IDs, complete output and stop reasons match between
inactive and active capture. Each layer's first four normalized and SwiGLU
rows are distinct, suitable for genuine one/four-row projection benchmarks.

This completes the real-input prerequisite of M02, not its complete-FFN timing
or the broader execution plan. No new CUDA kernel, selection policy, model
transform, prepared-weight default or KV default is enabled. This run is not
performance evidence; active capture includes diagnostic readbacks/drains.

## Identity and Control

- Original GGUF SHA-256
  `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
  All 36 layers execute, with full dense FP16 KV capacity 1024 and original
  Qwen2 tokenizer/ChatML, QKV biases, split-half RoPE and head dimension 128.
- Frozen corpus `data/research_quality/v1.json`, SHA-256
  `2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38`.
  Calibration tasks only; no browser or held-out observations.
- Qualified split down/vocabulary preparation, 6144 MiB extra-payload budget,
  single-row down four columns/256 threads, native fused QKV/MLP, Q8
  decode-stream coltile4/split4 and parallel head128 split8/256. No full FP32
  materialization, sparsity, skipped layers or graph replay.
- Greedy top1, seed zero, sequential prefill, actual natural stopping within
  each frozen task's original output bound. The configured observer is inactive
  for baseline/warmup; KV resets for every baseline/capture generation.
- Rust release 1.95.0, NVCC 12.8.93, g++ 14.3.1; CUDA `-O3`, sm_52 SASS and
  compute_52 PTX, no fast math. Executable SHA-256
  `940c2bede231d4d4560b91e15a45fbcbba28a49de44bc8c8198bbf462b938a48`.
- Execution-configuration SHA-256
  `6b66d08282476c88b39a416817978dc61f387abae50f50762c4d1e9c09035662`.
  The [companion manifest](2026-10-25-maxwell-ffn-capture.jsonl) stores its
  complete configuration, actual cache telemetry, device/build identity and
  clean worktree. Its hash recomputes from canonical compact configuration JSON.

The process was the only inference workload. GPU compute-process queries are
empty before and after it. No tests, builds, other inference or MemPalace work
ran during capture. These conditions do not turn the diagnostic into timing.

## Observations and Quality Scope

| Calibration Task | Actual Generated IDs | Captured Inputs per Layer | Task Score | Inactive/Active IDs |
| --- | ---: | ---: | ---: | --- |
| `cal-chat-reminder` | 21 | 4 | 1.00, pass | Equal |
| `cal-fact-depot` | 11 | 4 | 1.00, pass | Equal |
| `cal-multi-delivery` | 34 | 4 | 0.67, fail | Equal |
| `cal-config-retry` | 2 | 1 | 1.00, pass | Equal |
| `cal-arithmetic-invoice` | 16 | 4 | 0.00, fail | Equal |
| `cal-long-photo` | 79 | 4 | 0.75, fail | Equal |

Three original-control task rows pass and three fail their fixed rubrics.
Capture does not change any answer. Failed baseline rows are not evidence of
a candidate quality regression or improvement. Do not retune prompts, rubrics
or sampling to improve this input-capture checkpoint. The raw report retains
all outputs, token IDs, scoring details and stop reasons.

All six baseline traces and outputs also exactly match the
[frozen C19 dense reference](2026-10-25-model-quality-dense-reference.jsonl).
Its same three tasks fail the same rubrics. This cross-check rules out a new
task-level regression from the prepared control or observer in this checkpoint;
it does not establish broader model quality or candidate acceptance.

Capture starts at the first processed generated token, absolute position equal
to prompt length, and stops after four eligible rows/task. The final emitted
token has not been processed, so the two-token config answer supplies one row,
not two. The first four benchmark rows belong to `cal-chat-reminder`, absolute
positions 86 through 89, and differ at each layer. Four rows are independent
matrix inputs, not four sequential one-row calls or an autoregressive verifier.

## Memory and Frozen Files

| Requested Payload | Bytes |
| --- | ---: |
| Original uploaded weights | 3,279,519,744 |
| Prepared down/vocabulary | 1,192,935,424 |
| Combined weight payload | 4,472,455,168 |
| Dense FP16 KV | 37,748,736 |
| Used paired host observations | 3,290,112 |
| Reserved paired host observations | 3,760,128 |
| Reusable host read scratch | 52,224 |
| Host observation/read-scratch budget | 33,554,432 |

Observation/read-scratch counts exclude metadata, serialization and allocator
overhead, not process RSS. GPU payloads exclude driver/library allocator
overhead; native KV is accounted separately from Rust allocation tracking.
Tracked device bytes return from zero to zero after model release.

- [Compressed frozen inputs](2026-10-25-maxwell-ffn-inputs.json.gz), gzip SHA-256
  `b2b186100c3e4cb3e91a73c25802af5f027c1f541f346b598d29ff41f69f7b5c`.
- Expanded JSON is 17,252,371 bytes, SHA-256
  `eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687`.
  The local working copy is `target/research-fixtures/m02-ffn-inputs.json`.
- Uploaded weight-region SHA-256
  `c4a944caa8c27c326a7c0a9a1b5da4d3192d905436dc4e614eeed6279361be4d`.
  This is the GGUF tensor data region, not the full-file identity.
- [Capture JSONL](2026-10-25-maxwell-ffn-capture.jsonl), SHA-256
  `d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801`.
- [Raw stderr](2026-10-25-maxwell-ffn-capture.stderr.log) and
  [artifact audit](2026-10-25-maxwell-ffn-inputs-audit.json).

All audit checks pass. Nine gate/up/down source hashes independently match
byte ranges read from the original GGUF. Its tensor-data offset is 5,956,512,
the full file size minus the recorded uploaded data-region length. Source
relative offsets, shapes, Q8_0 types, identities and paired positions remain
inside the frozen artifact. `gzip -t` and expanded/archive hashes pass.

## Reproduction and Remaining Work

Use [the capture command](../ffn_benchmark_inputs.md#real-model-capture-command)
with new artifact/report paths and the recorded environment. This run used
`target/research-fixtures/m02-ffn-inputs.json` and
`target/research-fixtures/m02-ffn-capture.jsonl`. Both command implementations
and the read-only model observer were committed before the clean release run.

The eight CPU contract tests, two explicit M40 observer tests and affected
projection/MLP/attention/forward/ownership regressions pass. cuBLAS-on/off
parity, warning-as-error non-CUDA tests, CUDA/server all-targets Clippy and fmt
pass. The shared quality-runner lane passes 23 tests, including existing GPU
evaluator checks and new capture arguments/preflight; the unrelated C25 real
compact-model test is explicitly excluded because it requires separate
source/export inputs. An initial unconfigured invocation correctly failed on
that missing input; this is not a passing C25 rerun.

Next extend the existing `q8_execution_bench` to consume the frozen artifact
and companion, verify identities and measure complete gate/up/SwiGLU, isolated
down and complete FFN for real layers 0/17/35 at rows 1 and 4. Keep prepared
down arithmetic and the vocabulary control fixed. Only after this baseline
should bounded FP32 load/reduction and independent Q8_1 candidates be screened.
Both original exact-output throughput goals remain unmet.
