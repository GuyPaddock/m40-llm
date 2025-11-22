# Architecture

```
 +------------------------+      +-------------------------+
 |       Rust Server      | <--> |   CUDA Kernels (C++)    |
 |                        |      |                         |
 | CLI (pull/run/list)    |      | Persistent Decode Loop  |
 | GGUF Loader            |      | QKV Matmuls (FP16->F32) |
 | HTTP Infer API         |      | Attention Kernel        |
 | Scheduler / RingBuffer |      | KV Cache Operations     |
 | Pinned Host Memory     |      | Sampling Kernel         |
 +------------------------+      +-------------------------+
                ↑                          ↑
                │                          │
        FFI bindings via                 Compiled via
        `extern "C"` fn(...)             `nvcc` -> static .a
```

Rust does:

* model pulling
* GGUF parsing
* scheduling
* persistent kernel launch
* request/response logic
* memory mapping
* pinned host ring buffers
* final logits sampling (optional)

CUDA does:

* K new / V new append
* attention head computation
* Q,K,V projection matmuls
* MLP matmuls
* layer norms
* sampling (or we can do it in Rust)
* warp-per-sequence decode kernel

---

# **Where the code we wrote fits**

### ✔ Everything we wrote with:

* `__device__`
* `__global__`
* `__shfl_sync`
* `__half`
* `__nanosleep`

…is **CUDA device code** and must live in:

```
cuda/kernels.cu
cuda/attention.cu
cuda/kvcache.cu
```

### ✔ Rust sees a stable C ABI:

`cuda.rs` in Rust will contain:

```rust
extern "C" {
    fn fastllm_alloc_kvcache(...);
    fn fastllm_start_persistent_decode(...);
    fn fastllm_gemm_f16_storage_f32_compute(...);
    fn fastllm_append_kv(...);
    fn fastllm_attention_last_token(...);
}
```

And `build.rs` compiles the `.cu` files into a static library:

```rust
cc::Build::new()
    .cuda(true)
    .file("cuda/kernels.cu")
    .file("cuda/attention.cu")
    .file("cuda/kvcache.cu")
    .flag("-std=c++14")
    .flag("-O3")
    .flag("--use_fast_math")
    .flag("-gencode=arch=compute_52,code=sm_52") // M40
    .compile("fastllm_kernels");
```

---

# **What we write in Rust vs CUDA**

---

## **Rust-side (safe, organized, portable)**

You will write:

### ✔ Server

* HTTP API / streaming responses
* CLI fast like Ollama/Docker
* Request scheduler
* Sequence bookkeeping
* Memory management abstraction
* Model registry & GGUF loading

### ✔ Model logic

* LLaMA/Mistral architecture
* Layer index mapping
* Tokenizer + rope + layernorm implementation (host-side versions)

### ✔ Orchestration

* Launch persistent decode kernel
* Push commands into pinned ring buffer
* Pull results from result ring buffer
* Manage sequence life cycle
* Stream output tokens

---

## **CUDA-side (fast, tuned, hardware-specific)**

You will write:

### ✔ Persistent decode kernel

The mega-kernel loop that pulls work items and runs decode steps.

### ✔ Q/K/V matmul kernels (FP16→FP32)

Using Maxwell warp/block tiling.

### ✔ KV Cache append logic

Warp-safe or lane-0-based, depending on design.

### ✔ Attention kernel

For decode:

* last-token Q
* all K/V positions
* softmax
* context vector

### ✔ MLP kernels

Two GEMMs + activation (SiLU/GeLU).

### ✔ Sampling kernel (optional)

Top-k / top-p sampling on device to avoid CPU overhead.

---

# **Optionally: Use CUTLASS or cuBLAS**

For Maxwell (M40), **tensor cores don’t exist**, so CUTLASS or cuBLAS can still accelerate GEMMs, but your FP16→FP32 pipeline:

* works great with cuBLAS
* can be fully handwritten (like llama.cpp CUDA kernels)
* can be partially CUTLASS-based if you want high maintainability

But still — CUDA kernels must be C++.

---

# **Why Rust is the perfect orchestration language**

Rust is ideal for:

* safety
* concurrency
* memory ownership of sequences / jobs
* async HTTP
* managing pinned host memory regions
* fast custom data structures
* multi-threaded batching
* deterministic handling of lifetimes and streams

But Rust CANNOT replace CUDA C++ for device execution.

It *wraps* and *drives* it better than Python, C++, or Go ever could.

This architecture (Rust on host + CUDA on device) is used in:

* **Hugging Face Candle**
* **Firefox WebGPU engines**
* **TorchScript mobile backend**
* **Rust HPC pipelines with CUDA glue**

And it's rock-solid.

---

# **So… final answer:**

> **Yes, the project is Rust. But all CUDA kernels—including attention, KV cache, matmuls, and the persistent decoder—must be written in CUDA C/C++. That CUDA code is then compiled via `nvcc` and called from Rust via FFI.**

This is not a compromise — this is how all serious GPU engines work.
And Rust is one of the *best possible host languages* for it.