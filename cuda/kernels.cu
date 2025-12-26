// cuda/kernels.cu
#include <cuda_runtime.h>
#ifdef M40LLM_HAVE_CUBLAS
#include <cublas_v2.h>
#endif
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include "common.h"

struct M40llmCudaContext {
    int device_id;
    cudaStream_t prefill_stream;
    cudaStream_t decode_stream;
#ifdef M40LLM_HAVE_CUBLAS
    cublasHandle_t cublas;
#endif
};
  struct M40llmDeviceProps {
      int device_id;
      int major;
      int minor;
      char name[128];
  };


extern "C" {
    // KV Cache structure (visible in this TU)
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

    static int ensure_device(M40llmCudaContext* ctx) {
        if (!ctx) return -1;
        int current_device;
        cudaError_t get_err = cudaGetDevice(&current_device);
        if (get_err != cudaSuccess) {
            fprintf(stderr, "DEBUG: cudaGetDevice failed: %s\\n", cudaGetErrorString(get_err));
            return -3;
        }
        if (current_device != ctx->device_id) {
            fprintf(stderr, "DEBUG: Device mismatch: current=%d, ctx->device_id=%d\\n", current_device, ctx->device_id);
            cudaError_t set_err = cudaSetDevice(ctx->device_id);
            if (set_err != cudaSuccess) {
                fprintf(stderr, "DEBUG: cudaSetDevice failed: %s\\n", cudaGetErrorString(set_err));
                return -4;
            }
            fprintf(stderr, "DEBUG: Switched device from %d to %d\\n", current_device, ctx->device_id);
        }
        return 0;
    }

    int m40llm_device_malloc(M40llmCudaContext* ctx, size_t bytes, void** out_ptr) {
        if (!ctx || !out_ptr) return -1;
        if (ensure_device(ctx) != 0) return -2;
        void* d = nullptr;
        cudaError_t err = cudaMalloc(&d, bytes);
        if (err != cudaSuccess) return -3;
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
        if (ensure_device(ctx) != 0) return -3;
        cudaError_t err = cudaMemcpy(dst_device, src_host, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return -2;
        return 0;
    }

    int m40llm_memcpy_d2h(M40llmCudaContext* ctx, void* dst_host, const void* src_device, size_t bytes) {
        if (!ctx || !dst_host || !src_device) return -1;
        if (ensure_device(ctx) != 0) return -3;
        cudaError_t err = cudaMemcpy(dst_host, src_device, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -2;
        return 0;
    }

    int m40llm_validate_device_ptr(const void* ptr) {
        if (!ptr) return -1;
        cudaPointerAttributes attr;
        cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
        if (err != cudaSuccess) return -2;
        if (attr.type != cudaMemoryTypeDevice) return -3;
        return 0;
    }

    M40llmCudaContext* m40llm_create_context(int device_id) {
        fprintf(stderr, "DEBUG: m40llm_create_context called with device_id=%d\n", device_id);
        // Allow runtime auto-selection of Tesla M40 (sm_52) when:
        // - device_id < 0, or
        // - environment variable M40LLM_FORCE_M40=1 is set.
        int selected = device_id;
        const char* force_env = getenv("M40LLM_FORCE_M40");
        bool force_m40 = (force_env && force_env[0] == '1');
        if (device_id < 0 || force_m40) {
            int count = 0;
            if (cudaGetDeviceCount(&count) == cudaSuccess) {
                for (int i = 0; i < count; ++i) {
                    cudaDeviceProp prop;
                    if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                        if (prop.major == 5 && prop.minor == 2) {
                            fprintf(stderr, "DEBUG: Found Tesla M40 (sm_52) at device %d\n", i);
                            selected = i;
                            break;
                        }
                    }
                }
                if (selected < 0) {
                    // Fallback to device 0 if no M40 found
                    selected = 0;
                }
            } else {
                selected = 0;
            }
        }

        if (cudaSetDevice(selected) != cudaSuccess) {
            fprintf(stderr, "DEBUG: cudaSetDevice(%d) failed during context creation\n", selected);
            return nullptr;
        }
        fprintf(stderr, "DEBUG: cudaSetDevice(%d) succeeded\n", selected);

        M40llmCudaContext* ctx = new M40llmCudaContext();
        ctx->device_id = selected;

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

        fprintf(stderr, "DEBUG: Created context %p for device %d\n", ctx, selected);
        return ctx;
    }

}



    // Root Mean Square Normalization (FP32 compute)
    extern "C" __global__ void rms_norm_f32(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t n_rows,
        size_t row_stride,
        float eps) {
        const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= n_rows) return;

        float ss = eps; 
        for (size_t i = 0; i < row_stride; i++) {
            float x = input[row * row_stride + i];
            ss += x * x;
        }
        float rms = sqrtf(ss / row_stride);
        const float scale = 1.0f / rms;

        for (size_t i = 0; i < row_stride; i++) {
            output[row * row_stride + i] = input[row * row_stride + i] * scale;
        }
    }

    __global__ void kernel_f16_to_f32(const __half* in, float* out, size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            out[i] = __half2float(in[i]);
        }
    }

    // Q8_0 block layout per ggml: struct { float d; int8_t qs[32]; }
    __global__ void kernel_q80_to_f32(const int8_t* in, float* out, size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            size_t blk = i / 32;
            size_t idx = i % 32;
            const char* base = reinterpret_cast<const char*>(in) + blk * 36;
            const float d = *reinterpret_cast<const float*>(base + 0);
            int8_t q = *(reinterpret_cast<const int8_t*>(base + 4 + idx));
            out[i] = d * static_cast<float>(q);
        }
    }
extern "C" {

    // RMS Normalization (FP32)
    // Rotary Position Embedding kernel (FP32)
    __global__ void rope_kernel(
        float* x,
        float* out,
        const uint32_t* positions,
        uint32_t dim,
        uint32_t head_size,
        uint32_t n_ctx,
        uint32_t n_heads,
        uint32_t n_batch) {
        const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_batch * n_heads * head_size) return;

        const uint32_t b = i / (n_heads * head_size);
        const uint32_t h = (i % (n_heads * head_size)) / head_size;
        const uint32_t d = i % head_size;
        const uint32_t pos = positions[b];
        const uint32_t half_d = dim / 2;

        const float theta = 1.0f / powf(10000.0f, float(d % half_d) / float(half_d));
        const float theta_pos = theta * float(pos);
        const float cos_theta_pos = cosf(theta_pos);
        const float sin_theta_pos = sinf(theta_pos);

        const uint32_t idx = b * n_heads * head_size + h * head_size + d;
        const float x0 = x[idx];
        const float x1 = idx + half_d < n_batch * n_heads * head_size ? x[idx + half_d] : 0.0f;
        
        out[idx] = x0 * cos_theta_pos - x1 * sin_theta_pos;
        if (d < half_d) {
            out[idx + half_d] = x0 * sin_theta_pos + x1 * cos_theta_pos;
        }
    }

    // RMS Normalization (FP32)
    // Rotary Position Embedding (FP32)
    int m40llm_rope_f32(
        M40llmCudaContext* ctx,
        void* d_x,
        void* d_out,
        const uint32_t* d_positions,
        uint32_t dim,
        uint32_t head_size,
        uint32_t n_ctx,
        uint32_t n_heads,
        uint32_t n_batch) {
        if (!ctx || !d_x || !d_out || !d_positions) return -1;
        if (ensure_device(ctx) != 0) return -3;
        const int threads = 256;
        const int elements = n_batch * n_heads * head_size;
        const int blocks = (elements + threads - 1) / threads;
        rope_kernel<<<blocks, threads, 0, ctx->prefill_stream>>>(
            reinterpret_cast<float*>(d_x),
            reinterpret_cast<float*>(d_out),
            d_positions,
            dim,
            head_size,
            n_ctx,
            n_heads,
            n_batch);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -2;
        err = cudaStreamSynchronize(ctx->prefill_stream);
        if (err != cudaSuccess) return -3;
        return 0;
    }

    // Forward declaration
// Declaration removed - using extern "C" version below

// RMS Normalization (FP32)
extern "C" int m40llm_rms_norm_f32(
        M40llmCudaContext* ctx,
        const void* d_input,
        void* d_output,
        uint32_t rows,
        uint32_t dim,
        float eps) {
        if (!ctx || !d_input || !d_output || dim == 0) return -1;
        const int threads_per_block = 256;
        size_t shared = threads_per_block * sizeof(float);
        rms_norm_f32<<<rows, threads_per_block, shared, ctx->decode_stream>>>(
            reinterpret_cast<const float*>(d_input),
            reinterpret_cast<float*>(d_output),
            rows,
            dim,
            eps);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -2;
        err = cudaStreamSynchronize(ctx->decode_stream);
        return err == cudaSuccess ? 0 : -3;
    }

    int m40llm_f16_to_f32(M40llmCudaContext* ctx, const void* d_in_f16, void* d_out_f32, size_t n) {
        if (!ctx || !d_in_f16 || !d_out_f32) return -1;

        // Debug: Validate pointers and alignment
        cudaError_t err;
        cudaPointerAttributes attributes;
        err = cudaPointerGetAttributes(&attributes, d_in_f16);
        if (err != cudaSuccess) return -2;
        if ((uintptr_t)d_in_f16 % alignof(__half) != 0) return -3;

        const int threads = 256;
        const int blocks = (int)((n + threads - 1) / threads);

        // Sync any previous operations
        cudaStreamSynchronize(ctx->prefill_stream);

        kernel_f16_to_f32<<<blocks, threads, 0, ctx->prefill_stream>>>(
            reinterpret_cast<const __half*>(d_in_f16),
            reinterpret_cast<float*>(d_out_f32),
            n);
        err = cudaGetLastError();
        if (err != cudaSuccess) return -2;
        err = cudaStreamSynchronize(ctx->prefill_stream);
        if (err != cudaSuccess) return -3;
        return 0;
    }

    int m40llm_q80_to_f32(M40llmCudaContext* ctx, const void* d_in_q80, void* d_out_f32, size_t n) {
        if (!ctx || !d_in_q80 || !d_out_f32) return -1;
        const int threads = 256;
        const int blocks = (int)((n + threads - 1) / threads);
        kernel_q80_to_f32<<<blocks, threads, 0, ctx->prefill_stream>>>(
            reinterpret_cast<const int8_t*>(d_in_q80),
            reinterpret_cast<float*>(d_out_f32),
            n);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -2;
        err = cudaStreamSynchronize(ctx->prefill_stream);
        if (err != cudaSuccess) return -3;
        return 0;
    }

    // Query the current CUDA device properties for test/CI guardrails.
    // Returns 0 on success, negative on error.
    int m40llm_current_device_props(char* name_buf, size_t buf_len, int* major, int* minor, int* device_id) {
        if (!name_buf || buf_len == 0 || !major || !minor || !device_id) return -1;
        int dev = 0;
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess) return -2;
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) return -3;
        size_t nlen = strnlen(prop.name, sizeof(prop.name));
        if (buf_len <= nlen) {
            nlen = buf_len - 1;
        }
        memcpy(name_buf, prop.name, nlen);
        name_buf[nlen] = '\0';
        *major = prop.major;
        *minor = prop.minor;
        *device_id = dev;
        return 0;
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

        // Register memory if coming from mmap
        cudaHostRegister((void*)host_ptr, num_bytes, cudaHostRegisterDefault);
        
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

    // Last-token attention: FP32 compute, FP16 K/V storage
    // Q: [num_heads * head_dim] (f32) for the last token of a sequence
    // Output: [num_heads * head_dim] (f32)
    __global__ void attention_last_token_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        uint32_t max_seq_len,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t seq_id,
        const float* __restrict__ Q,
        uint32_t seq_len,
        float* __restrict__ Out) {
        const uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
        if (h >= num_heads) return;

        const size_t elems_per_token = (size_t)num_heads * (size_t)head_dim;
        const float inv_sqrt = 1.0f / sqrtf((float)head_dim);
        const float* qh = Q + (size_t)h * (size_t)head_dim;
        
        // KV cache always stores data as FP16, so we always use FP16 access

        // Pass 1: find max score for numerical stability
        float max_score = -1e30f;
        for (uint32_t t = 0; t < seq_len; ++t) {
            const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                               + (size_t)h * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                float kf = __half2float(K[base + d]);
                dot += qh[d] * kf;
            }
            float score = dot * inv_sqrt;
            if (score > max_score) max_score = score;
        }

        // Pass 2: sum of exp(scores - max)
        float denom = 0.0f;
        for (uint32_t t = 0; t < seq_len; ++t) {
            const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                               + (size_t)h * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                float kf = __half2float(K[base + d]);
                dot += qh[d] * kf;
            }
            float score = dot * inv_sqrt;
            denom += expf(score - max_score);
        }
        denom = denom > 0.f ? denom : 1.f;

        // Pass 3: compute output = sum_t softmax(score)*V
        float* out_h = Out + (size_t)h * (size_t)head_dim;
        for (uint32_t d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (uint32_t t = 0; t < seq_len; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)h * (size_t)head_dim;
                // Recompute score (same as passes 1 and 2)
                float dot = 0.0f;
                for (uint32_t dd = 0; dd < head_dim; ++dd) {
                    float kf = __half2float(K[base + dd]);
                    dot += qh[dd] * kf;
                }
                float score = dot * inv_sqrt;
                float p = expf(score - max_score) / denom;
                float vf = __half2float(V[base + d]);
                acc += p * vf;
            }
            out_h[d] = acc;
        }
    }

    int m40llm_attention_last_token_f32(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t seq_len,
        void* out_dev_f32) {
        if (!ctx || !kv || !q_dev_f32 || !out_dev_f32) return -1;
        if (seq_id >= kv->max_batch_size) return -2;
        if (seq_len == 0 || seq_len > kv->max_seq_len) return -3;
        const int blocks = (int)kv->num_heads;
        const int threads = 1;
        attention_last_token_kernel<<<blocks, threads, 0, ctx->decode_stream>>>(
            kv->d_k, kv->d_v, kv->max_seq_len, kv->num_heads, kv->head_dim, seq_id,
            reinterpret_cast<const float*>(q_dev_f32), seq_len,
            reinterpret_cast<float*>(out_dev_f32)
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -4;
        err = cudaStreamSynchronize(ctx->decode_stream);
        if (err != cudaSuccess) return -5;
        return 0;
    }

    // FP16 storage / FP32 compute GEMM using cuBLAS when available; fallback naive CUDA kernel otherwise
    // A: MxK (f16 row-major), B: KxN (f16 row-major), C: MxN (f16 row-major) with FP32 accumulation → FP16 output
    __global__ void gemm_f16_f32_kernel(
        const __half* __restrict__ A, // MxK
        const __half* __restrict__ B, // KxN
        __half* __restrict__ C,       // MxN
        int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y; // 0..M-1
        int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..N-1
        if (row >= M || col >= N) return;
        float acc = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
            float a = __half2float(A[row * K + kk]);
            float b = __half2float(B[kk * N + col]);
            acc += a * b;
        }
        C[row * N + col] = __float2half_rn(acc);
    }

    // Mixed GEMM: A f16 (MxK row-major) × B f16 (KxN row-major) → C f32 (MxN row-major)
    __global__ void gemm_f16xf16_f32_kernel(
        const __half* __restrict__ A,  // MxK
        const __half* __restrict__ B,  // KxN
        float* __restrict__ C,         // MxN
        int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= M || col >= N) return;
        float acc = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
            float a = __half2float(A[row * K + kk]);
            float b = __half2float(B[kk * N + col]);
            acc += a * b;
        }
        C[row * N + col] = acc;
    }

    // Mixed GEMM: A f32 (MxK row-major) × B f16 (KxN row-major) → C f32 (MxN row-major)
    __global__ void gemm_f32xf16_f32_kernel(
        const float* __restrict__ A,  // MxK
        const __half* __restrict__ B, // KxN
        float* __restrict__ C,        // MxN
        int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= M || col >= N) return;
        float acc = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
            float a = A[row * K + kk];
            float b = __half2float(B[kk * N + col]);
            acc += a * b;
        }
        C[row * N + col] = acc;
    }

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
        const __half* A = reinterpret_cast<const __half*>(d_A);
        const __half* B = reinterpret_cast<const __half*>(d_B);
        __half* C = reinterpret_cast<__half*>(d_C);
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        gemm_f16_f32_kernel<<<grid, block, 0, ctx->prefill_stream>>>(A, B, C, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -2;
        err = cudaStreamSynchronize(ctx->prefill_stream);
        if (err != cudaSuccess) return -3;
        return 0;
    #endif
    }

    // Expose mixed-dtype GEMM (f32 × f16 → f32)
    int m40llm_gemm_f32xf16_f32(
        M40llmCudaContext* ctx,
        const void* d_A_f32,
        const void* d_B_f16,
        void* d_C_f32,
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
            d_B_f16, CUDA_R_16F, N,
            d_A_f32, CUDA_R_32F, K,
            &beta,
            d_C_f32, CUDA_R_32F, N,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT);
        if (st == CUBLAS_STATUS_SUCCESS) {
            return 0;
        }
        // Fallback to simple kernel when cuBLAS does not support this mixed-type combination on this device
        const float* A = reinterpret_cast<const float*>(d_A_f32);
        const __half* B = reinterpret_cast<const __half*>(d_B_f16);
        float* C = reinterpret_cast<float*>(d_C_f32);
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        gemm_f32xf16_f32_kernel<<<grid, block, 0, ctx->prefill_stream>>>(A, B, C, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -2;
        err = cudaStreamSynchronize(ctx->prefill_stream);
        if (err != cudaSuccess) return -3;
        return 0;
    #else
        const float* A = reinterpret_cast<const float*>(d_A_f32);
        const __half* B = reinterpret_cast<const __half*>(d_B_f16);
        float* C = reinterpret_cast<float*>(d_C_f32);
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        gemm_f32xf16_f32_kernel<<<grid, block, 0, ctx->prefill_stream>>>(A, B, C, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -2;
        err = cudaStreamSynchronize(ctx->prefill_stream);
        if (err != cudaSuccess) return -3;
        return 0;
    #endif
    }


    // Expose f16×f16 → f32 GEMM (row-major)
    int m40llm_gemm_f16xf16_f32(
        M40llmCudaContext* ctx,
        const void* d_A_f16,
        const void* d_B_f16,
        void* d_C_f32,
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
            d_B_f16, CUDA_R_16F, N,
            d_A_f16, CUDA_R_16F, K,
            &beta,
            d_C_f32, CUDA_R_32F, N,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT);
        return st == CUBLAS_STATUS_SUCCESS ? 0 : -3;
    #else
        const __half* A = reinterpret_cast<const __half*>(d_A_f16);
        const __half* B = reinterpret_cast<const __half*>(d_B_f16);
        float* C = reinterpret_cast<float*>(d_C_f32);
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        gemm_f16xf16_f32_kernel<<<grid, block, 0, ctx->prefill_stream>>>(A, B, C, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -2;
        err = cudaStreamSynchronize(ctx->prefill_stream);
        if (err != cudaSuccess) return -3;
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

    // Duplicate definition removed; defined at top of extern "C" block.
    // (typedef KVCache alias also defined there.)

    static size_t kv_storage_elems(uint32_t max_seq_len, uint32_t max_batch_size, uint32_t num_heads, uint32_t head_dim) {
        return (size_t)max_seq_len * (size_t)max_batch_size * (size_t)num_heads * (size_t)head_dim;
    }

    M40llmKVCache* m40llm_kvcache_create(M40llmCudaContext* ctx,
                                         uint32_t max_seq_len,
                                         uint32_t max_batch_size,
                                         uint32_t num_heads,
                                         uint32_t head_dim) {
        if (!ctx) return nullptr;
        if (ensure_device(ctx) != 0) return nullptr;
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

    __global__ void cast_store_f32_to_f16_kernel(const float* __restrict__ in,
                                                 __half* __restrict__ out,
                                                 size_t n) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            out[i] = __float2half_rn(in[i]);
        }
    }

    __global__ void cast_store_f32_to_f16_kernel_h2(const float* __restrict__ in,
                                                    __half* __restrict__ out,
                                                    size_t n_pairs) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n_pairs) {
            const size_t idx = i * 2;
            const float a = in[idx];
            const float b = in[idx + 1];
            __half2 h2 = __halves2half2(__float2half_rn(a), __float2half_rn(b));
            reinterpret_cast<__half2*>(out)[i] = h2;
        }
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

    int m40llm_kvcache_append_token_f32(M40llmCudaContext* ctx,
                                         M40llmKVCache* kv,
                                         uint32_t seq_id,
                                         const void* k_dev_f32,
                                         const void* v_dev_f32) {
        if (!ctx || !kv || !k_dev_f32 || !v_dev_f32) return -1;
        if (seq_id >= kv->max_batch_size) return -2;

        uint32_t cur_len = 0;
        cudaError_t err = cudaMemcpy(&cur_len, kv->d_seq_map + seq_id, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -3;
        if (cur_len >= kv->max_seq_len) return -4;

        const size_t elems_per_token = (size_t)kv->num_heads * (size_t)kv->head_dim;
        const size_t offset_elems = ((size_t)seq_id * (size_t)kv->max_seq_len + (size_t)cur_len) * elems_per_token;

        __half* k_out = kv->d_k + offset_elems;
        __half* v_out = kv->d_v + offset_elems;
        const float* k_in = reinterpret_cast<const float*>(k_dev_f32);
        const float* v_in = reinterpret_cast<const float*>(v_dev_f32);

        const int threads = 256;

        // Handle potential 4-byte alignment for half2 stores. If the output pointer
        // is not 4-byte aligned, do one scalar element first to align it.
        size_t elems_remaining = elems_per_token;
        size_t skip = 0;
        if (((uintptr_t)k_out & 0x3) != 0) {
            // do first element scalar for both K and V
            cast_store_f32_to_f16_kernel<<<1, 1, 0, ctx->decode_stream>>>(k_in, k_out, 1);
            cast_store_f32_to_f16_kernel<<<1, 1, 0, ctx->decode_stream>>>(v_in, v_out, 1);
            skip = 1;
            elems_remaining -= 1;
        }

        const size_t pairs = elems_remaining / 2;
        const size_t tail = elems_remaining & 1u;

        const float* k_in_h2 = k_in + skip;
        const float* v_in_h2 = v_in + skip;
        __half* k_out_h2 = k_out + skip;
        __half* v_out_h2 = v_out + skip;

        if (pairs > 0) {
            const int blocks_h2 = (int)((pairs + threads - 1) / threads);
            cast_store_f32_to_f16_kernel_h2<<<blocks_h2, threads, 0, ctx->decode_stream>>>(k_in_h2, k_out_h2, pairs);
            cast_store_f32_to_f16_kernel_h2<<<blocks_h2, threads, 0, ctx->decode_stream>>>(v_in_h2, v_out_h2, pairs);
        }
        if (tail) {
            const float* k_tail_in = k_in_h2 + pairs * 2;
            const float* v_tail_in = v_in_h2 + pairs * 2;
            __half* k_tail_out = k_out_h2 + pairs * 2;
            __half* v_tail_out = v_out_h2 + pairs * 2;
            cast_store_f32_to_f16_kernel<<<1, 1, 0, ctx->decode_stream>>>(k_tail_in, k_tail_out, 1);
            cast_store_f32_to_f16_kernel<<<1, 1, 0, ctx->decode_stream>>>(v_tail_in, v_tail_out, 1);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "append_token_f32 kernel launch error: %s\n", cudaGetErrorString(err));
            return -5;
        }

        // Increment length
        cur_len += 1;
        err = cudaMemcpyAsync(kv->d_seq_map + seq_id, &cur_len, sizeof(uint32_t), cudaMemcpyHostToDevice, ctx->decode_stream);
        if (err != cudaSuccess) return -6;
        err = cudaStreamSynchronize(ctx->decode_stream);
        if (err != cudaSuccess) return -7;

        return 0;
    }


    int m40llm_kvcache_debug_read_token(M40llmCudaContext* ctx,
                                         M40llmKVCache* kv,
                                         uint32_t seq_id,
                                         uint32_t token,
                                         void* out_k_host,
                                         void* out_v_host) {
        (void)ctx;
        if (!kv || !out_k_host || !out_v_host) return -1;
        if (seq_id >= kv->max_batch_size) return -2;
        if (token >= kv->max_seq_len) return -3;
        const size_t elems_per_token = (size_t)kv->num_heads * (size_t)kv->head_dim;
        const size_t offset_elems = ((size_t)seq_id * (size_t)kv->max_seq_len + (size_t)token) * elems_per_token;
        const size_t bytes = elems_per_token * sizeof(__half);
        cudaError_t err;
        err = cudaMemcpy(out_k_host, kv->d_k + offset_elems, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -4;
        err = cudaMemcpy(out_v_host, kv->d_v + offset_elems, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -5;
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





// Residual add kernel
__global__ void residual_add_f32(const float* a, const float* b, float* out, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// C wrapper for residual add kernel
extern "C" void m40llm_residual_add_f32(M40llmCudaContext* ctx, const float* a, const float* b, float* out, uint32_t size) {
    if (ensure_device(ctx) != 0) return;
    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    residual_add_f32<<<blocks, threads_per_block>>>(a, b, out, size);
}

    __global__ void rope_f32(
        float* __restrict__ q,
        float* __restrict__ k,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t past_len,
        float freq_base,
        float freq_scale) {

    const uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pairs_per_row = (num_heads * head_dim) / 2;
    const uint32_t total_pairs = rows * pairs_per_row;
    if (pair_idx >= total_pairs) return;

    const uint32_t row = pair_idx / pairs_per_row;
    const uint32_t pair_in_row = pair_idx % pairs_per_row;
    const uint32_t head = pair_in_row / (head_dim / 2);
    const uint32_t offset_in_head = pair_in_row % (head_dim / 2);

    const uint32_t base = row * num_heads * head_dim + head * head_dim + 2 * offset_in_head;
    const float pos = static_cast<float>(past_len + row) * freq_scale;
    const float theta = pos * powf(freq_base, -2.0f * static_cast<float>(offset_in_head) / static_cast<float>(head_dim));
    const float c = cosf(theta);
    const float s = sinf(theta);

    const float q0 = q[base];
    const float q1 = q[base + 1];
    q[base] = q0 * c - q1 * s;
    q[base + 1] = q0 * s + q1 * c;

    const float k0 = k[base];
    const float k1 = k[base + 1];
    k[base] = k0 * c - k1 * s;
    k[base + 1] = k0 * s + k1 * c;
}

extern "C" int m40llm_rope_f32(
    M40llmCudaContext* ctx,
    float* q,
    float* k,
    uint32_t rows,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t past_len,
    float freq_base,
    float freq_scale) {
    if (!ctx || !q || !k || head_dim == 0 || num_heads == 0) return -1;
    if (head_dim % 2 != 0) return -2;
    const uint32_t pairs_per_row = (num_heads * head_dim) / 2;
    const uint32_t total_pairs = rows * pairs_per_row;
    const int threads_per_block = 256;
    const int blocks = (total_pairs + threads_per_block - 1) / threads_per_block;
    rope_f32<<<blocks, threads_per_block, 0, ctx->decode_stream>>>(
        q, k, rows, num_heads, head_dim, past_len, freq_base, freq_scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -3;
    err = cudaStreamSynchronize(ctx->decode_stream);
    return err == cudaSuccess ? 0 : -4;
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
