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
