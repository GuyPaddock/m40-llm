# llm-context index (detailed)

This folder contains design notes and conversations used to design and build m40-llm, a Rust + CUDA LLM runtime optimized for Tesla M40 (sm_52). This index summarizes each file so you can quickly decide relevance when resuming work with a limited context buffer.

Conventions:
- Relevance: High = actively useful now; Medium = background or future work; Low = mostly historical.
- Focus tags: [CUDA], [Rust], [Design], [Tests], [Docs], [Roadmap]

Files:

- [01-machine_context_summary.md](./01-machine_context_summary.md)
  - Summary: Single-file snapshot of project identity, dtype policy (FP16 storage, FP32 compute), pipeline, implemented pieces, and roadmap. Originally authored earlier; we will keep this file up-to-date and brief.
  - Why useful: Primary on-ramp; should be read first when resuming.
  - Relevance: High [Design] [Roadmap]

- [02-scope-and-hardware-hacks.md](./02-scope-and-hardware-hacks.md)
  - Summary: Initial high-level plan to build a vLLM/Ollama-like server optimized for M40. Covers stack (Rust + CUDA + cuBLAS), FP16 storage/FP32 compute, pinned host memory, prefill/decode phases, batching, CUDA Graphs, and extensive M40-specific guidance (Hyper‑Q, streams, __ldg, cache behavior, concurrency, KV compression ideas).
  - Why useful: Grounding document for M40 constraints/advantages and server shape; still matches our current direction.
  - Relevance: High [Design]

- [02b-cuda-rust-architecture.md](./02b-cuda-rust-architecture.md)
  - Summary: Rust/CUDA split with ASCII diagram; what lives in CUDA vs Rust; FFI surface examples; build.rs integration flow; where to place kernels; guidance on when to use cuBLAS/CUTLASS vs handwritten kernels.
  - Why useful: Clarifies module boundaries and ABI expectations; helpful when adding kernels or FFI.
  - Relevance: High [Design] [CUDA] [Rust]

- [03-project-skeleton.md](./03-project-skeleton.md)
  - Summary: Early multi-file scaffold for a new repo (Cargo.toml, src layout, minimal gguf, server CLI). Predates current repository structure and is superseded by real code.
  - Why useful: Historical only; not needed for current codebase.
  - Relevance: Low (removed) [Historical]

- [04-ring-buffer.md](./04-ring-buffer.md)
  - Summary: Persistent decode kernel + host/GPU ring buffer design using pinned memory; job/result queues; motivation (avoid per-token launches); scheduling and concurrency notes; fits Hyper‑Q; outlines GPU resident decode loop.
  - Why useful: Directly relevant to future tasks t32 (persistent kernel) and t33 (stream separation). Good blueprint when we implement the decode engine.
  - Relevance: Medium‑High [CUDA] [Design] [Roadmap]

- [05-attention-and-kv-cache.md](./05-attention-and-kv-cache.md)
  - Summary: KV cache layout and attention planning on M40; FP16 K/V storage with FP32 compute; indexing/stride considerations; host append path; compatibility with last‑token attention.
  - Why useful: Background when touching KV cache and attention; supports current KV layout API and parity tests.
  - Relevance: High [CUDA] [Design]

- [06-parallelization.md](./06-parallelization.md)
  - Summary: Guidance to parallelize attention across lanes/warps for Maxwell; occupancy/register tradeoffs; shared memory tiling ideas.
  - Why useful: Tuning reference when optimizing attention beyond correctness.
  - Relevance: Medium [CUDA]

- [07-rust-interfaces.md](./07-rust-interfaces.md)
  - Summary: Proposed FFI and Rust wrapper shapes for KV cache and kernels; how Rust orchestrates model pieces; interface-focused.
  - Why useful: Still maps to our src/cuda.rs/src/infer.rs approach; helps keep ABI tidy.
  - Relevance: Medium [Rust] [Design]

- [08-minimal-decode.md](./08-minimal-decode.md)
  - Summary: Minimal forward/decode path plan (GGUF → one layer → attention → MLP → logits); sequencing and test strategy.
  - Why useful: Aligns with t26 minimal forward. Good checklist when wiring full layer.
  - Relevance: High [Design] [Roadmap]

- [09-recap.md](./09-recap.md)
  - Summary: Project layout recap and naming confirmation; reiterates directory structure and rationale.
  - Why useful: Redundant with code and updated summary, but harmless quick reference.
  - Relevance: Low‑Medium [Historical]

- [10-rename-api.md](./10-rename-api.md)
  - Summary: Discussion about renaming API from fastllm to m40llm.
  - Why useful: Historical only; decision settled and code reflects m40llm.
  - Relevance: Low (removed) [Historical]

- [11-device-allocator.md](./11-device-allocator.md)
  - Summary: Proposal for a centralized DeviceAllocator in Rust with typed DevicePtr<T>, RAII frees, and future backends (cudaMallocAsync pools). API and integration notes.
  - Why useful: Solid future direction; useful when refactoring memory management.
  - Relevance: Medium [Rust] [Design]

- [12-new-allocator.md](./12-new-allocator.md)
  - Summary: Steps to convert model loader to use DeviceAllocator; typed buffers for Q/K/V/context/logits; integration example.
  - Why useful: Companion to 11-… when we adopt allocator; can be adapted.
  - Relevance: Medium [Rust]

- [13-real-gguf.md](./13-real-gguf.md)
  - Summary: Hand-rolled GGUF parser and mapping to LLaMA/Mistral hparams; tensor descriptors and offsets; shows how to derive shapes from metadata.
  - Why useful: Although we plan to use gguf-rs-lib + gguf-llms, this doc helps cross-check shapes/metadata and understand GGUF fields.
  - Relevance: Medium [Design]

- [14-cuda-kernels.md](./14-cuda-kernels.md)
  - Summary: RMSNorm and MLP kernel designs (CUDA) + Rust FFI integration stubs; outlines interfaces and simple kernels.
  - Why useful: Reference for kernel signatures and integration patterns.
  - Relevance: Medium [CUDA]

- [15-dtype-consistency.md](./15-dtype-consistency.md)
  - Summary: Canonical dtype policy for M40: FP16 storage, FP32 compute; detailed policy across embeddings, RMSNorm, QKV, KV cache, attention, MLP, output proj; required code changes.
  - Why useful: Core principle document; aligns with current implementation and tests.
  - Relevance: High [Design]

- [16-tests.md](./16-tests.md)
  - Summary: Strategy and examples for CUDA tests with CPU references (RMSNorm focus, extendable to MLP/attention); utilities expected on CudaContext.
  - Why useful: Template for future kernels/tests.
  - Relevance: Medium [Tests]

- [17-convert-attention-kernel.md](./17-convert-attention-kernel.md)
  - Summary: Conversion of attention to FP32-Q / FP16-KV with FP32 output; C API, kernel, and Rust side changes; aligns with our current parity grid.
  - Why useful: Reinforces correct interfaces for attention; good when revisiting kernel.
  - Relevance: High [CUDA] [Design]
