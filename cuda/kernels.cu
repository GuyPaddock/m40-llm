// cuda/kernels.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdint>

struct FastllmCudaContext {
    int device_id;
    cudaStream_t prefill_stream;
    cudaStream_t decode_stream;
    cublasHandle_t cublas;
};

extern "C" {
    FastllmCudaContext* fastllm_create_context(int device_id) {
        cudaSetDevice(device_id);
        FastllmCudaContext* ctx = new FastllmCudaContext();
        ctx->device_id = device_id;

        cudaStreamCreate(&ctx->prefill_stream);
        cudaStreamCreate(&ctx->decode_stream);

        cublasCreate(&ctx->cublas);
        cublasSetStream(ctx->cublas, ctx->prefill_stream); // default

        return ctx;
    }

    void fastllm_destroy_context(FastllmCudaContext* ctx) {
        if (!ctx) return;
        cublasDestroy(ctx->cublas);
        cudaStreamDestroy(ctx->prefill_stream);
        cudaStreamDestroy(ctx->decode_stream);
        delete ctx;
    }

    int fastllm_upload_weights(
        FastllmCudaContext* ctx,
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

    // FP16 storage / FP32 compute GEMM
    // A: MxK (f16), B: KxN (f16), C: MxN (f16) but computed with FP32 accum
    int fastllm_gemm_f16_storage_f32_compute(
        FastllmCudaContext* ctx,
        const void* d_A,
        const void* d_B,
        void* d_C,
        int M, int N, int K) {
        if (!ctx) return -1;

        float alpha = 1.0f;
        float beta = 0.0f;

        // All row-major vs column-major issues are up to you; for now, assume column-major
        cublasStatus_t st = cublasGemmEx(
            ctx->cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, CUDA_R_16F, M,
            d_B, CUDA_R_16F, K,
            &beta,
            d_C, CUDA_R_16F, M,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT);

        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasGemmEx error: %d\n", (int)st);
            return -2;
        }
        return 0;
    }

    // A stub persistent decode kernel: one warp = one sequence
    // This is just a sketch; you'll need to define your own work queue structures.
    struct DecodeJob {
        // indices to KV cache, etc.
    };

    __global__ void persistent_decode_kernel(DecodeJob* jobs, int max_jobs) {
        int warp_id = threadIdx.x / warpSize;
        int lane_id = threadIdx.x % warpSize;
        int block_warps = blockDim.x / warpSize;

        // Warp-per-sequence loop
        while (true) {
            // TODO: fetch job index from some global queue (atomic)
            // int job_idx = ...
            // if (job_idx < 0) break;

            // decode one token for this sequence:
            // - load KV tile from global into shared
            // - attention
            // - MLP
            // - write logits / chosen token somewhere
            // For now, just spin.
            __nanosleep(100);
        }
    }

    int fastllm_start_persistent_decode(FastllmCudaContext* ctx) {
        if (!ctx) return -1;
        // In real code: allocate DecodeJob queue in pinned host memory, map to device.
        // Launch kernel with cooperative groups or large grid.
        // For now, do nothing.
        return 0;
    }

    int fastllm_stop_persistent_decode(FastllmCudaContext* ctx) {
        if (!ctx) return -1;
        // Signal kernel via some flag or destroying context
        return 0;
    }
} // extern "C"
