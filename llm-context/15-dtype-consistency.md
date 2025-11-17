User:
Clean up dtype consistency (f16/f32) in Q/K/V and attention
---
Agent:
Thank you — and I want to be equally clear:

### **This is *our* project.**

You’re steering the ship, and I’m helping engineer it exactly the way you want.
Think of me as your co-author and GPU-kernel gremlin who lives inside the ISA manuals.
You’re not taking credit for me — I’m working *for you*.

Alright… let’s do **Step A: Clean up dtype consistency for Q/K/V and attention**.
This is the single most important architectural step before we can test correctness.

---

# 🅰️ Step A: Clean, consistent datatype policy for Q/K/V and attention

## 🧭 Current Situation (diagnosis)

Right now, we have:

* Embedding output: **f16**
* RMSNorm output: **f16**
* Q/K/V projections: **planned for f16 storage but sometimes outputting f32**
* Attention kernel: expects **f16 Q**, **f16 K**, **f16 V**, upgrades internally to f32
* KV cache: stores **f16** K/V
* MLP input: **f32** (`context` from attention)
* Everything else: a bit mixed

This is typical for early scaffolding, but:

## ❗ Inconsistent types → incorrect attention results, mismatched KV writes, broken logits.

So we need a consistent, M40-optimal policy.

---

# 🧠 The *right* policy for Maxwell (M40)

Maxwell is best when:

* We store weights and KV in **FP16**, because:

    * cuts VRAM usage
    * halves bandwidth
* We compute everything with **FP32**, because:

    * Maxwell does FP32 at full speed
    * FP16 arithmetic is not accelerated
    * f16→f32 conversion is cheap + fast

So the policy should be:

```
Storage: FP16
Intermediate: FP32
Compute: FP32
Input/Output of each kernel: FP32 where possible
Only use FP16 at the edges (weights, embeddings, KV).
```

This is how frameworks like Triton kernels are structured for pre-Volta GPUs, and it is ideal for Maxwell.

So we’ll adopt:

---

# 🧩 **Official M40-Optimized Datatype Policy**

### **1. Embeddings:**

* Stored FP16 (`tok_embeddings.weight`)
* Loaded → converted to FP32 immediately
* RMSNorm input = FP32
* We extend our RMSNorm kernel to accept FP32 input

### **2. RMSNorm:**

* Input: FP32
* Gamma: FP32
* Output: FP32

### **3. Linear projections (Q, K, V):**

* W_q / W_k / W_v: stored FP16
* Input activations: FP32
* Output: FP32
* (Before writing K/V to KV-cache, convert to FP16)

### **4. KV cache:**

* Store FP16 K and FP16 V
* Attention kernel loads FP16 K/V → converts to FP32 on-the-fly

### **5. Q input to attention:**

* Q stored FP32 (no reason to store it in FP16; not reused)
* Attention kernel converts Q to FP32 internally anyway

### **6. Attention output:**

* Output: FP32 (context vector)

### **7. MLP input:**

* FP32
* W_gate / W_up / W_down stored FP16
* Output FP32

### **8. Output projection:**

* W_out: FP16
* x: FP32
* logits: FP32

---

# 🧱 What needs to change in our code?

Here’s the surgical plan:

---

## **A.1 – RMSNorm should be FP32→FP32, not FP16→FP16**

This is a small modification:

### Current signature:

```cpp
m40llm_rmsnorm_f16_f32gamma(x_f16, gamma_f32, y_f16)
```

### New signature:

```cpp
m40llm_rmsnorm_f32(
    const float* x_f32,
    const float* gamma_f32,
    float*       y_f32,
    int n,
    float eps
)
```

### Why?

* RMSNorm is normalization → should not lose precision
* FP32 compute is much more stable for deep layers
* Storing RMSNorm output in FP32 means better Q/K/V accuracy

I will provide the updated kernel below.

---

## **A.2 – QKV GEMMs should output FP32**

Currently we had pseudo-calls like:

```rust
self.cuda.gemm_f16_f32(...)
```

We will standardize GEMMs as:

```
gemm_fp16W_fp32in_fp32out()
```

Meaning:

* W: FP16 storage
* x: FP32
* y: FP32

This matches the pattern: FP16 memory → FP32 compute → FP32 output.

---

## **A.3 – Convert K and V to FP16 before storing in KV cache**

At the end of each decode step:

```rust
convert_fp32_to_fp16(self.d_k_f32, self.d_k_f16);
convert_fp32_to_fp16(self.d_v_f32, self.d_v_f16);
self.kv.append_token(… d_k_f16, d_v_f16 …);
```

We will add a small CUDA kernel:

```cpp
__global__ void cast_fp32_to_fp16(const float*, __half*, int n)
```

---

## **A.4 – Attention kernel input should be FP32 Q + FP16 K/V**

The kernel already assumes FP16 inputs; we’ll update the signature to:

```
Q: FP32
K: FP16
V: FP16
output context: FP32
```

Then inside we convert:

```cpp
float q = Q_f32[i];
float k = __half2float(K_f16[i]);
```

(This improves numerical precision and aligns with storage policy.)

---

## **A.5 – Logits matmul should be FP16W × FP32 → FP32**

Matches the same pattern as Q/K/V projections.

---

# ✨ **Below is the fully updated RMSNorm kernel (FP32→FP32)**

Put this in `cuda/kernels.cu`:

```cpp
__global__ void rmsnorm_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float*       __restrict__ y,
    int n,
    float eps)
{
    extern __shared__ float buf[];

    int tid = threadIdx.x;
    float sum = 0.0f;

    // compute sum(x^2)
    for (int i = tid; i < n; i += blockDim.x) {
        float v = x[i];
        sum += v * v;
    }

    buf[tid] = sum;
    __syncthreads();

    // reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(buf[0] / n + eps);

    // normalize
    for (int i = tid; i < n; i += blockDim.x) {
        y[i] = x[i] * inv_rms * gamma[i];
    }
}

extern "C" int m40llm_rmsnorm_f32(
    M40llmCudaContext* ctx,
    const void* x_f32,
    const void* gamma_f32,
    void*       y_f32,
    int n,
    float eps)
{
    int block = 128;
    int grid  = 1;
    size_t shmem = block * sizeof(float);

    rmsnorm_f32_kernel<<<grid, block, shmem>>>(
        (const float*)x_f32,
        (const float*)gamma_f32,
        (float*)y_f32,
        n,
        eps
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}
```

I can also generate Rust FFI for this right away — just say the word.

---

# 🧬 **Recap of what Step A accomplishes**

After Step A, you will have:

* 🥇 Fully consistent Maxwell-optimized dtype layout
* 🥇 FP32 compute everywhere that matters
* 🥇 FP16 weight storage everywhere that helps
* 🥇 FP16 KV cache storage
* 🥇 FP32 activations → best for accuracy with long sequences
* 🥇 Simpler integration of GGUF tensors
* 🧹 Clean code with fewer type mismatches
* 🧪 A path to real unit testing

This is the **correct** foundation for M40 inference.

---

Next steps:
A. Write Rust/CUDA unit tests to validate our kernels
B. Convert the attention kernel to FP32 Q + FP16 K/V
C. Create GEMM wrapper functions (FP16W × FP32 → FP32)
D. Clean up the block0 forward path to match new types