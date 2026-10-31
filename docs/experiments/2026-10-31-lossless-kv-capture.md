# Real FP16 KV capture implementation

`model_quality --capture-lossless-kv <new-binary-path>` extends the existing
model loader, tokenizer, dense session and report writer. It requires the
original Qwen Q8 model, context 8192, calibration mode and the frozen FFN
control environment. The separate quality corpus is not an input to this
capture. Other transforms, timing studies and capture modes are excluded.

Input text is `0eee578:docs/perf_baselines.md`, SHA-256
`a918722b18973bb2621a0e2f0fd9ccde0d01ff610456904773c1a975a9b10ade`.
The first eight disjoint 8192-token spans are frozen in the manifest before
any compression observation. Insufficient or duplicated spans fail. These
are technical-document compression fixtures, not a task-quality corpus.

The capture uses explicit dense FP16 KV and graphs off. Each sequence runs
native packed prefix prefill and processes its final input through ordinary
decode. The predicted next token is recorded but not fed back; generated
token count is zero. Sessions reuse physical sequence zero after full cache
reset. Capture reads layers 0/17/35 using physical layer/sequence mapping
after both streams drain and checks every selected cache's valid length.

Raw words are preserved, including exceptional patterns. The binary stores
sequence, layer, KV head, then K and V streams, each ordered by token/channel.
Per-stream offsets, sizes and hashes accompany CPU codec measurements for
1K/2K/4K/8K prefixes. Every encoded prefix must reconstruct exactly. Header,
directory, tail and raw-block bytes all count. CPU encoding/validation and
decoding costs are separately labeled; no attention timing follows from them.

The expected raw input fixture is 201,326,592 bytes. Keep this large binary
in the ignored, disk-backed `target` fixture directory and retain its hash
and source inputs in the report. Commit raw JSONL measurements and summaries
under `docs/experiments/`; do not commit the large binary. The capture refuses
existing or aliased artifact/report destinations and requires a clean release
source tree. The manifest identifies the executable, model, tokenizer and
environment. Memory snapshots are observations, not process-wide peak RSS.

Five capture tests pass, including one explicitly invoked M40 test. CPU
tests cover disjoint token windows, bit-preserving head order, exact binary
contents/prefix hashes and CLI exclusions. The GPU test checks two layers,
repeated snapshots, two cache lifetimes with different valid lengths,
independent direct FP16 readback, codec round trips and tracked cleanup.
The final empty-head rejection also passes its focused test. Twelve
non-CUDA model-quality tests and CUDA/server all-target Clippy pass. The
[codec unit](2026-10-31-lossless-kv-codec.md) previously passed the full
225-test non-CUDA suite; its source is unchanged in this unit.

Raw validation logs use prefix `2026-10-31-lossless-kv-capture-`. No complete
real-model capture or compression result exists at this implementation
checkpoint. Next build the committed source in release, inspect GPU process
state, restore the frozen environment and run serially:

```bash
target/release/examples/model_quality \
  --reference-model "$research_model" --model "$research_model" \
  --variant-id lossless-fp16-512-v1 --context-tokens 8192 --split calibration \
  --capture-lossless-kv target/lossless-kv-inputs-01.bin \
  --output target/lossless-kv-capture-01.jsonl
```

`research_model` is the original blob ending in
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
The frozen environment is recorded in
`2026-10-25-maxwell-ffn-capture.jsonl`; `run::environment` and an exact map/hash
check enforce it. CUDA compression and staged/fused attention remain
conditional on observed net storage savings under the
[fixed protocol](2026-10-31-lossless-kv-audit.md).
