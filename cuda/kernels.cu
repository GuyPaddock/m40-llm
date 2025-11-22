// cuda/kernels.cu
#include <cuda_runtime.h>
#ifdef M40LLM_HAVE_CUBLAS
#include <cublas_v2.h>
#endif
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include "common.h"

struct M40llmCudaContext {
    int device_id;
    cudaStream_t prefill_stream;
    cudaStream_t decode_stream;
#ifdef M40LLM_HAVE_CUBLAS
    cublasHandle_t cublas;
#endif
};

extern "C" {
    int m40llm_device_malloc(M40llmCudaContext* ctx, size_t bytes, void** out_ptr) {
        if (!ctx || !out_ptr) return -1;
        void* d = nullptr;
        cudaError_t err = cudaMalloc(&d, bytes);
        if (err != cudaSuccess) return -2;
        *out_ptr = d;
        return 0;
    }

    int m40llm_device_free(M40llmCudaContext* ctx, void* ptr) {
        if (!ctx) return -1;
        if (!ptr) return 0;
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) return -2;
        return 0;
    }

    int m40llm_memcpy_h2d(M40llmCudaContext* ctx, void* dst_device, const void* src_host, size_t bytes) {
        if (!ctx || !dst_device || !src_host) return -1;
        cudaError_t err = cudaMemcpy(dst_device, src_host, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return -2;
        return 0;
    }

    int m40llm_memcpy_d2h(M40llmCudaContext* ctx, void* dst_host, const void* src_device, size_t bytes) {
        if (!ctx || !dst_host || !src_device) return -1;
        cudaError_t err = cudaMemcpy(dst_host, src_device, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -2;
        return 0;
    }

    M40llmCudaContext* m40llm_create_context(int device_id) {
        cudaSetDevice(device_id);
        M40llmCudaContext* ctx = new M40llmCudaContext();
        ctx->device_id = device_id;

        cudaStreamCreate(&ctx->prefill_stream);
        cudaStreamCreate(&ctx->decode_stream);

    #ifdef M40LLM_HAVE_CUBLAS
        if (cublasCreate(&ctx->cublas) != CUBLAS_STATUS_SUCCESS) {
            cudaStreamDestroy(ctx->prefill_stream);
            cudaStreamDestroy(ctx->decode_stream);
            delete ctx;
            return nullptr;
        }
        cublasSetStream(ctx->cublas, ctx->prefill_stream); // default
    #endif

        return ctx;
    }

    void m40llm_destroy_context(M40llmCudaContext* ctx) {
        if (!ctx) return;
    #ifdef M40LLM_HAVE_CUBLAS
        cublasDestroy(ctx->cublas);
    #endif
        cudaStreamDestroy(ctx->prefill_stream);
        cudaStreamDestroy(ctx->decode_stream);
        delete ctx;
    }

    int m40llm_upload_weights(
        M40llmCudaContext* ctx,
        const void* host_ptr,
        size_t num_bytes,
        void** out_device_ptr) {
        if (!ctx || !host_ptr || !out_device_ptr) return -1;
        void* d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, num_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
            return -2;
        }
        err = cudaMemcpy(d_ptr, host_ptr, num_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_ptr);
            return -3;
        }
        *out_device_ptr = d_ptr;
        return 0;
    }

    // FP16 storage / FP32 compute GEMM using cuBLAS when available; fallback no-op otherwise
    // A: MxK (f16), B: KxN (f16), C: MxN (f16) with FP32 compute
    int m40llm_gemm_f16_storage_f32_compute(
        M40llmCudaContext* ctx,
        const void* d_A,
        const void* d_B,
        void* d_C,
        int M, int N, int K) {
        if (!ctx) return -1;
    #ifdef M40LLM_HAVE_CUBLAS
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(
            ctx->cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, CUDA_R_16F, N,
            d_A, CUDA_R_16F, K,
            &beta,
            d_C, CUDA_R_16F, N,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT);
        return st == CUBLAS_STATUS_SUCCESS ? 0 : -3;
    #else
        (void)d_A; (void)d_B; (void)d_C; (void)M; (void)N; (void)K;
        return 0;
    #endif
    }

    // KV Cache layout: [seq][token][head][head_dim]
    // - seq in [0, max_batch_size)
    // - token in [0, max_seq_len)
    // - head in [0, num_heads)
    // - head_dim in [0, head_dim)
    // Strides (elements):
    //   elems_per_token = num_heads * head_dim
    //   base(seq, token) = (seq * max_seq_len + token) * elems_per_token
    //   index(seq, token, head, dim) = base(seq, token) + head * head_dim + dim

    // KV Cache structure (opaque outside of this TU)
    struct M40llmKVCache {
        __half* d_k; // FP16 K storage
        __half* d_v; // FP16 V storage
        uint32_t* d_seq_map; // sequence ID to current length (tokens appended)
        uint32_t max_seq_len;
        uint32_t max_batch_size;
        uint32_t num_heads;
        uint32_t head_dim;
    };

    // Back-compat alias so other TU code using KVCache still compiles
    typedef M40llmKVCache KVCache;

    static size_t kv_storage_elems(uint32_t max_seq_len, uint32_t max_batch_size, uint32_t num_heads, uint32_t head_dim) {
        return (size_t)max_seq_len * (size_t)max_batch_size * (size_t)num_heads * (size_t)head_dim;
    }

    M40llmKVCache* m40llm_kvcache_create(M40llmCudaContext* ctx,
                                         uint32_t max_seq_len,
                                         uint32_t max_batch_size,
                                         uint32_t num_heads,
                                         uint32_t head_dim) {
        if (!ctx) return nullptr;
        M40llmKVCache* kv = new M40llmKVCache();
        kv->max_seq_len = max_seq_len;
        kv->max_batch_size = max_batch_size;
        kv->num_heads = num_heads;
        kv->head_dim = head_dim;

        size_t elems = kv_storage_elems(max_seq_len, max_batch_size, num_heads, head_dim);
        size_t bytes = elems * sizeof(__half);
        size_t seq_map_size = (size_t)max_batch_size * sizeof(uint32_t);

        cudaError_t err;
        err = cudaMalloc(&kv->d_k, bytes);
        if (err != cudaSuccess) { delete kv; return nullptr; }
        err = cudaMalloc(&kv->d_v, bytes);
        if (err != cudaSuccess) { cudaFree(kv->d_k); delete kv; return nullptr; }
        err = cudaMalloc(&kv->d_seq_map, seq_map_size);
        if (err != cudaSuccess) { cudaFree(kv->d_k); cudaFree(kv->d_v); delete kv; return nullptr; }

        cudaMemset(kv->d_k, 0, bytes);
        cudaMemset(kv->d_v, 0, bytes);
        cudaMemset(kv->d_seq_map, 0, seq_map_size);
        return kv;
    }

    int m40llm_kvcache_append_token(M40llmCudaContext* ctx,
                                     M40llmKVCache* kv,
                                     uint32_t seq_id,
                                     const void* k_dev,
                                     const void* v_dev) {
        if (!ctx || !kv || !k_dev || !v_dev) return -1;
        if (seq_id >= kv->max_batch_size) return -2;

        // Fetch current length for this sequence
        uint32_t cur_len = 0;
        cudaError_t err = cudaMemcpy(&cur_len, kv->d_seq_map + seq_id, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -3;
        if (cur_len >= kv->max_seq_len) return -4;

        const size_t elems_per_token = (size_t)kv->num_heads * (size_t)kv->head_dim;
        const size_t offset_elems = ((size_t)seq_id * (size_t)kv->max_seq_len + (size_t)cur_len) * elems_per_token;

        // Copy K and V for one token (all heads)
        const size_t bytes = elems_per_token * sizeof(__half);
        err = cudaMemcpy(kv->d_k + offset_elems, k_dev, bytes, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) return -5;
        err = cudaMemcpy(kv->d_v + offset_elems, v_dev, bytes, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) return -6;

        // Increment length
        cur_len += 1;
        err = cudaMemcpy(kv->d_seq_map + seq_id, &cur_len, sizeof(uint32_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return -7;

        return 0;
    }


    void m40llm_kvcache_destroy(M40llmKVCache* kv) {
        if (!kv) return;
        if (kv->d_k) cudaFree(kv->d_k);
        if (kv->d_v) cudaFree(kv->d_v);
        if (kv->d_seq_map) cudaFree(kv->d_seq_map);
        delete kv;
    }

    // A stub persistent decode kernel: one warp = one sequence
#if 0 // temporarily disable experimental kernels to keep NVCC build green

    // This is just a sketch; you'll need to define your own work queue structures.
    struct DecodeJob {
        // indices to KV cache, etc.
    };

    __global__ void persistent_decode_kernel(
        RingBuffer<DecodeCommand, 1024>* cmd_rb,
        RingBuffer<DecodeResult, 1024>* res_rb,
        KVCache* kv,
        ModelWeights* weights
    ) {
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;


        while (true) {
            // Warp 0 fetches command (one warp = one sequence)
            DecodeCommand cmd;
            if (lane == 0) {
                cmd = ringbuffer_pop(cmd_rb);
            }
            // Broadcast cmd to all lanes in the warp
            cmd.cmd      = __shfl_sync(0xffffffff, cmd.cmd, 0);
            cmd.seq_id   = __shfl_sync(0xffffffff, cmd.seq_id, 0);
            cmd.input_len= __shfl_sync(0xffffffff, cmd.input_len, 0);
            cmd.max_new  = __shfl_sync(0xffffffff, cmd.max_new, 0);

            if (cmd.cmd == DECODE_CMD_STOP) {
                break;
            }

            if (cmd.cmd == DECODE_CMD_PREFILL) {
                run_prefill_for_seq(cmd, kv, weights, warp, lane);
            }

            if (cmd.cmd == DECODE_CMD_DECODE) {
                uint32_t tok = run_decode_for_seq(cmd, kv, weights, warp, lane);

                if (lane == 0) {
                    DecodeResult r;
                    r.seq_id = cmd.seq_id;
                    r.token = tok;
                    r.done = (tok == EOS_TOKEN);
                    ringbuffer_push(res_rb, r);
                }
            }
        }
    }

    __device__
    DecodeCommand ringbuffer_pop(RingBuffer<DecodeCommand,1024>* rb) {
        while (true) {
            uint32_t tail = rb->tail;
            uint32_t head = rb->head;

            if (tail != head) {
                DecodeCommand cmd = rb->buffer[tail];
                __threadfence_system(); // ensure host sees updates
                rb->tail = (tail + 1) % 1024;
                return cmd;
            }
            __nanosleep(200);
        }
    }

    __device__
    void ringbuffer_push(RingBuffer<DecodeResult,1024>* rb, DecodeResult r) {
        while (true) {
            uint32_t head = rb->head;
            uint32_t next = (head + 1) % 1024;

            if (next != rb->tail) {
                rb->buffer[head] = r;
                __threadfence_system();
                rb->head = next;
                return;
            }
            __nanosleep(200);
        }
    }

    // FP32→FP16 cast
    __global__ void cast_f32_to_f16(const float* __restrict__ in,
                                   half* __restrict__ out,
                                   uint32_t size) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // Load FP32 value
    float value = in[i];

    // Convert to FP16 with saturation
    half h = __float2half(value);

    // Store result
    out[i] = h;
}

    // RMSNorm: FP32→FP32
    __global__ void rms_norm_f32(const float* __restrict__ in,
                                float* __restrict__ out,
                                const uint32_t* __restrict__ seq_map,
                                uint32_t seq_len,
                                uint32_t dim,
                                float eps) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len * dim) return;

    // Get sequence and element indices
    const uint32_t seq = i / dim;
    const uint32_t elem = i % dim;

    // Compute position in flat array
    const uint32_t pos = seq_map[seq] + elem;

    // Load data
    float value = in[pos];

    // Compute RMS norm
    float sum = 0.0f;
    for (uint32_t j = 0; j < dim; ++j) {
        float v = in[pos + j];
        sum += v * v;
    }
    sum /= dim;

    // Apply norm and eps
    float denom = sqrt(sum + eps);
    value /= denom;

    // Write result
    out[pos] = value;
}

    // MLP: SwiGLU - FP16→FP32→FP16
    // Input: [batch*seq, dim]
    // Output: [batch*seq, dim]
    __global__ void mlp_swiglu_f16(
        const half* __restrict__ in,
        half* __restrict__ out,
        const float* __restrict__ gate_weight,
        const float* __restrict__ up_weight,
        const float* __restrict__ down_weight,
        uint32_t batch_seq,
        uint32_t dim,
        uint32_t hidden_dim) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_seq * dim) return;

    // Convert input to FP32
    float4 in_f32 = __ldg4((const float4*)(&in[i]));
    float in0 = __high_first(in_f32), in1 = __high_second(in_f32);
    float in2 = __high_third(in_f32), in3 = __low_first(in_f32);

    // Gating: [batch_seq, dim] x [dim, hidden_dim] → [batch_seq, hidden_dim]
    float gate0 = gate_weight[0] * in0 + gate_weight[1] * in1 +
                  gate_weight[2] * in2 + gate_weight[3] * in3;
    float gate1 = gate_weight[4] * in0 + gate_weight[5] * in1 +
                  gate_weight[6] * in2 + gate_weight[7] * in3;

    // Up projection: [batch_seq, dim] x [dim, hidden_dim] → [batch_seq, hidden_dim]
    float up0 = up_weight[0] * in0 + up_weight[1] * in1 +
                up_weight[2] * in2 + up_weight[3] * in3;
    float up1 = up_weight[4] * in0 + up_weight[5] * in1 +
                up_weight[6] * in2 + up_weight[7] * in3;

    // Apply SiLU activation: σ(x) = x * σ'(x)
    float sig0 = gate0 * __sigmoid(gate0);
    float sig1 = gate1 * __sigmoid(gate1);

    // Down projection: [batch_seq, hidden_dim] x [hidden_dim, dim] → [batch_seq, dim]
    float out0 = down_weight[0] * (sig0 * up0) + down_weight[1] * (sig1 * up1);
    float out1 = down_weight[2] * (sig0 * up0) + down_weight[3] * (sig1 * up1);
    float out2 = down_weight[4] * (sig0 * up0) + down_weight[5] * (sig1 * up1);
    float out3 = down_weight[6] * (sig0 * up0) + down_weight[7] * (sig1 * up1);

    // Store result as FP16
    half2 out_f16 = make_float2(__low_first(out0), __low_first(out1));
    out_f16 = __high_first(out_f16, make_float2(__low_first(out2), __low_first(out3)));
    out[i] = out_f16;
}

    // Attention kernel: QKV → softmax → context
    __global__ void attention_f16(
        const half* __restrict__ q,
        half* __restrict__ out,
        KVCache* kv,
        const float* __restrict__ q_weights,
        const float* __restrict__ kv_weights,
        uint32_t seq_id,
        uint32_t seq_len,
        uint32_t dim,
        uint32_t num_heads,
        uint32_t head_dim) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t head = i / (seq_len * head_dim);
    const uint32_t lane = threadIdx.x % 32; // For warp intrinsics

    if (head >= num_heads) return;

    // Compute Q: [batch*seq, dim] x [dim, head_dim*4] → [batch*seq, head_dim*4]
    float4 q4[4]; // 4 elements of head_dim * 4 = 16 floats
    for (uint32_t j = 0; j < 4; ++j++) {
        q4[j] = __ldg4((const float4*)(&q_weights[(seq_id * dim + j * 4) * head_dim]));
    }

    float4 q_result[4];
    for (uint32_t j = 0; j < 4; ++j++) {
        q_result[j] = q4[j] * q4[j]; // Placeholder for actual Q computation
    }

    // Compute K and store in shared memory
    __shared__ float s_mem[32 * 128]; // 32 warps, 128 bytes per warp
    if (lane < 32) {
        for (uint32_t j = 0; j < seq_len; ++j) {
            // Compute K: [batch*seq, dim] x [dim, head_dim*4] → [batch*seq, head_dim*4]
            float4 k4[4];
            for (uint32_t k = 0; k < 4; ++k++) {
                k4[k] = __ldg4((const float4*)(&kv_weights[(j * dim + k * 4) * head_dim]));
            }

            // Store in shared memory
            for (uint32_t k = 0; k < 4; ++k) {
                s_mem[(lane * 128) + (j * 4 + k)] = ((float4*)k4[k])[0];
            }
        }
    }
    __syncthreads();

    // Compute QK^T: [batch*seq, head_dim*4] x [seq, head_dim*4] → [batch*seq, seq]
    float scores[1]; // 1 float per thread
    if (lane < 32) {
        scores[0] = 0.0f;
        for (uint32_t j = 0; j < seq_len; ++j++) {
            float k0 = s_mem[(lane * 128) + (j * 4)];
            float k1 = s_mem[(lane * 128) + (j * 4 + 1)];
            float k2 = s_mem[(lane * 128) + (j * 4 + 2)];
            float k3 = s_mem[(lane * 128) + (j * 4 + 3)];

            // Compute dot product
            scores[0] += q_result[0][0] * k0 + q_result[0][1] * k1 +
                         q_result[0][2] * k2 + q_result[0][3] * k3;
        }
    }

    // Apply softmax: [batch*seq, seq]
    float max_val = 0.0f;
    if (lane < 32) {
        max_val = -1e10f;
        for (uint32_t j = 0; j < seq_len; ++j++) {
            if (scores[0] > max_val) {
                max_val = scores[0];
            }
        }
    }
    max_val = reduce_max(max_val, 32);

    float exp_sum = 0.0f;
    if (lane < 32) {
        scores[0] = exp(scores[0] - max_val);
        exp_sum = scores[0];
        for (uint32_t j = 1; j < seq_len; ++j++) {
            scores[j] = exp(scores[j] - max_val);
            exp_sum += scores[j];
        }
    }
    exp_sum = reduce_add(exp_sum, 32);

    // Compute output: [batch*seq, seq] x [seq, head_dim*4] → [batch*seq, head_dim]
    if (lane < 32) {
        for (uint32_t j = 0; j < seq_len; ++j) {
            float prob = scores[j] / exp_sum;
            // Accumulate weighted values from V
            // (This is where you'd compute O = P * V)
        }
    }

    // Store output
    if (head < num_heads) {
        // Convert to FP16 and store in output
    }
}

    __device__
    void run_prefill_for_seq(const DecodeCommand& cmd,
                             KVCache* kv,
                             ModelWeights* w,
                             int warp,
                             int lane)
    {
        // 1) loop through tokens
        // 2) run attention on entire prefix
        // 3) store key/value in kv cache

        // [You fill in: QKV linear layers, RoPE, softmax, MLP]
    }

    __device__
    uint32_t run_decode_for_seq(const DecodeCommand& cmd,
                                KVCache* kv,
                                ModelWeights* w,
                                int warp,
                                int lane)
    {
        // 1) load last token embedding
        // 2) compute Q
        // 3) attention against kv cache for this seq
        // 4) MLP
        // 5) logits
        // 6) sample next token
        uint32_t next_token = 1; // TODO
        return next_token;
    }

#endif // experimental block

    int m40llm_start_persistent_decode(M40llmCudaContext* ctx) { return ctx ? 0 : -1; }
    int m40llm_stop_persistent_decode(M40llmCudaContext* ctx) { return ctx ? 0 : -1; }
} // extern "C"
