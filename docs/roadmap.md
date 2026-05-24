# M40 Performance Roadmap

This project prioritizes correctness and measurement before deeper scheduling
work. The main fast path is currently TinyLlama-class "fast-fits" inference:
materialized FP32 projection weights plus cuBLAS on Tesla M40.

## Performance Strategy

- FP16 storage, FP32 compute.
- cuBLAS `Sgemm` for materialized projection weights when the model fits.
- Explicit CUDA streams and async enqueue wrappers to avoid unnecessary
  per-kernel synchronization.
- CUDA Graph experiments for warm steady decode once graph replay is proven
  beneficial.
- Packed variable-length decode before packed prefill in the server scheduler.
- Read-only cache and texture experiments only when profiling identifies a
  specific bottleneck.

## Current Execution Model

CLI and non-streaming server generation share `DecodeSession`, which owns
per-request scratch such as token activations, output buffers, logits, and
optional normalized hidden state. Server generation remains serialized at the
workspace level while batching and per-session workspace ownership mature.

Model-level KV addressing is explicit:

```text
KV[layer][sequence][position][kv_head][head_dim]
```

Physical KV slots are mapped internally from logical layer and sequence IDs.

## Server Batching Direction

`M40LLM_SERVER_BATCH_DECODE=1` enables the experimental buffered decode
scheduler path. Requests lease logical KV sequence slots and can use packed
batched GQA decode attention for compatible `head_dim=64` and `head_dim=128`
models while the shared workspace lock remains in place.
`M40LLM_SERVER_BATCH_DECODE_SLOTS=N` overrides the default logical sequence slot
count.

`M40LLM_SERVER_BATCH_PREFILL=1` opts into packed variable-length prompt prefill
for compatible head64 dense-KV server cases. Head128/Qwen server requests fall
back to sequential prompt prefill until real-model multi-request output parity
is validated. Unsupported or single-request cases fall back to the normal path.

The intended order is:

1. Batched decode with safe request/session ownership.
2. Packed prefill once decode batching is correct.
3. Mixed prefill/decode overlap after scheduler behavior is stable.

Benchmark the buffered batch-decode path with:

```bash
source scripts/dev-env.sh
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  TRIALS=3 MAX_TOKENS=2 scripts/bench_server_batch_decode.sh
```

The script compares dense batch-decode modes and writes detailed logs and
`results.tsv` under `/tmp` by default. It passes `--kv-compress-mode off`
because the current server scheduler batching path is dense-KV-only. Set
`BATCH_DECODE_MODES=1 PREFILL_MODES="0 1"` to compare batched decode with
packed prefill disabled versus enabled. Set `CARGO_RUN_ARGS="--release"` for
optimized Rust timing checks. Set `CASES="batch2_same batch4_mixed"` or another
space-separated subset to keep longer decode-focused checks bounded.

## CUDA Graph Direction

Graph infrastructure exists and can capture warmed decode-shaped work, including
one-layer production smoke coverage. Full-token graph coverage should expand
only after graph replay timing is clearly beneficial and stream dependencies are
explicit.

Normal async decode remains the default.

## Backend Direction

Fast-fits backend:

- Materialized FP32 projection weights.
- cuBLAS `Sgemm`.
- Intended for models that fit comfortably in 24 GB with workspace and KV.

Large-model backend:

- Compact GGUF/quantized weights.
- No full FP32 materialization.
- Future fused dequant plus projection kernels.

The large-model backend should not be mixed into the fast path until the
fast-fits path is stable and measured.

## Variable-Length Batching Citation

Variable-length batching is inspired by Zhang and Lu's SC25 research poster on
GEMM acceleration for variable-length LLM inference. The M40 implementation
adapts the high-level ideas to conventional CUDA FP32 accumulation on Maxwell
sm_52, without Tensor Core or WMMA assumptions:

```bibtex
@inproceedings{zhang2025efficientgemm,
  title        = {An Efficient GEMM Acceleration Method for LLM Inference with Variable-Length Sequences},
  author       = {Zhang, Yu and Lu, Lu},
  booktitle    = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC25), Research Posters},
  year         = {2025},
  organization = {ACM/IEEE},
  url          = {https://sc25.supercomputing.org/proceedings/posters/poster_files/post167s2-file2.pdf},
  note         = {Research poster}
}
```
