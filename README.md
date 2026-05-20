# m40-llm

Tesla M40-optimized Rust + CUDA LLM runtime/server. The project targets
Maxwell `sm_52` directly: GGUF models, FP16 storage, FP32 compute, CUDA/cuBLAS
hot paths, and an optional HTTP `/generate` server.

## What It Is

- Single-GPU runtime focused on NVIDIA Tesla M40.
- GGUF model loading with CUDA forward/decode paths.
- C FFI symbols under the `m40llm_*` prefix.
- FP16 weights/KV storage with FP32 compute on Maxwell.
- CLI generation and optional HTTP server.
- Experimental long-context KV compression and batching work. The current
  preferred experimental KV path is direct FP16-K/q4-V exact-old retrieval with
  plain top-k block selection; see [`docs/kv_compression.md`](docs/kv_compression.md).

This is for M40 owners and researchers who want an M40-first runtime rather
than a broad portability layer. It aims to be faster than generic runtimes on
this specific card by exploiting stable sm_52 constraints.

## How It Compares

- vs ollama: m40-llm competes directly on Tesla M40 by specializing kernels,
  layouts, and decode-path optimizations for Maxwell.
- vs vLLM: vLLM is excellent on modern GPUs, but its main acceleration paths are
  not practical on M40. m40-llm is designed to set up and run on sm_52.
- vs llama.cpp: llama.cpp is much more portable, but many of its fastest GPU
  paths target newer hardware. m40-llm trades portability for M40-specific
  optimization.

## Quick Start

Use the project dev environment when possible:

```bash
source scripts/dev-env.sh
```

Build without CUDA:

```bash
cargo build --no-default-features
```

Build with CUDA:

```bash
M40LLM_ENABLE_CUBLAS=1 cargo build --features cuda
```

Run one CLI generation:

```bash
M40LLM_ENABLE_CUBLAS=1 cargo run \
  --features cuda \
  -- generate path/to/model.gguf "Hello" \
  --max-tokens 16 \
  --top-k 1 \
  --require-sm52
```

Run the HTTP server:

```bash
M40LLM_ENABLE_CUBLAS=1 cargo run \
  --features cuda,server \
  -- run path/to/model.gguf \
  --addr 0.0.0.0:58439
```

Send a request:

```bash
curl -sS -X POST http://127.0.0.1:58439/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":8,"temperature":1.0,"top_k":1}'
```

CLI and server responses contain generated text only; they do not echo the
prompt. `max_tokens` counts generated token IDs, not decoded characters.

## CUDA Toolchain

CUDA builds require `nvcc`. The build compiles kernels for `sm_52` and embeds
`compute_52` PTX. cuBLAS is enabled only when `M40LLM_ENABLE_CUBLAS=1` is set
and the headers/libraries are found.

If micromamba is available, this creates the recommended CUDA toolchain without
requiring a system CUDA install:

```bash
micromamba create -y -n m40-llm -c conda-forge -c nvidia/label/cuda-12.4.1 \
  cuda-nvcc=12.4.99 cuda-cudart=12.4.99 cuda-cudart-dev=12.4.99 \
  libcublas=12.4.5.8 libcublas-dev=12.4.5.8

micromamba run -n m40-llm env M40LLM_ENABLE_CUBLAS=1 \
  cargo build --features cuda
```

`scripts/dev-env.sh` is the preferred local entrypoint once configured.

## Cargo Features

- `cuda`: builds CUDA kernels and enables GPU runtime paths.
- `server`: includes the HTTP server entrypoint and `/generate` route.

Common checks:

```bash
cargo test --no-default-features
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda,server
```

Some CUDA tests require an actual sm_52 device and will skip when unavailable.

## Generation Notes

Prompt formatting defaults to `--prompt-format auto`. For Llama 3 GGUF files
using `tokenizer.ggml.model=gpt2` and `tokenizer.ggml.pre=llama-bpe`, auto mode
wraps unformatted prompts in the Llama 3 chat template. Use
`--prompt-format raw` for already-formatted prompts or
`--prompt-format llama3-chat` to force that wrapper.

Qwen2/Qwen2.5 GGUF files are recognized through `qwen2.*` metadata and Qwen
tokenizer markers. Auto prompt formatting wraps unformatted Qwen prompts in the
ChatML-style `<|im_start|>` / `<|im_end|>` template; use
`--prompt-format qwen-chat` to force that wrapper.
`Qwen2.5-3B-Instruct-f16.gguf` has a basic one-token CUDA smoke on Tesla M40;
long-context KV quality validation now has the required head_dim=128 kernel
support for dense packed prefill and direct FP16-K/q4-V exact-old retrieval.

Host sampling supports greedy, top-k, top-p, temperature, and deterministic RNG
paths. The normal CUDA path keeps full-layer decode on device and copies logits
back for host sampling.

## Experimental Features

These areas are active research and should be treated as opt-in:

- Compressed KV-cache modes and long-context quality harness:
  [`docs/kv_compression.md`](docs/kv_compression.md)
- Runtime diagnostics, profiling flags, and graph debug knobs:
  [`docs/diagnostics.md`](docs/diagnostics.md)
- M40 performance roadmap, batching, CUDA Graphs, and backend direction:
  [`docs/roadmap.md`](docs/roadmap.md)
- Current benchmark history:
  [`docs/perf_baselines.md`](docs/perf_baselines.md)

## Useful Docs

- CUDA device selection and M40 guardrails:
  [`docs/device_selection.md`](docs/device_selection.md)
- CUDA parity grid and KV layout:
  [`docs/cuda_parity_and_kv_layout.md`](docs/cuda_parity_and_kv_layout.md)
- Minimal forward path notes:
  [`docs/minimal_forward.md`](docs/minimal_forward.md)

## Citations

Variable-length batching work is inspired by:

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

Compressed KV-cache experiments are inspired by:

```bibtex
@misc{deepseekai2026deepseekv4,
  title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
  author={DeepSeek-AI},
  year={2026},
}
```

The implementation does not attempt to reproduce either work exactly; both are
used as design inspiration for M40-specific experiments.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Use Conventional Commits with a
descriptive body, and keep changes focused.
