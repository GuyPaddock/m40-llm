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
#include <float.h>
#include <algorithm>
#include <vector>
#include "common.h"

struct M40llmPersistentDecode;
struct M40llmCudaGraphExec {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
};

struct M40llmCudaEvent {
    cudaEvent_t event;
};

struct M40llmCudaContext {
    int device_id;
    cudaStream_t prefill_stream;
    cudaStream_t decode_stream;
    cudaEvent_t prefill_waits_decode_event;
    cudaEvent_t decode_waits_prefill_event;
    std::vector<cudaEvent_t> retained_wait_events;
    int prefill_priority;
    int decode_priority;
    M40llmPersistentDecode* persistent_decode;
#ifdef M40LLM_HAVE_CUBLAS
    cublasHandle_t cublas;
    uint32_t cublas_stream_kind;
#endif
};

static void CUDART_CB destroy_wait_event_host_func(void* user_data) {
    cudaEvent_t event = reinterpret_cast<cudaEvent_t>(user_data);
    if (event) {
        cudaEventDestroy(event);
    }
}
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
        __half* d_recent_k;
        __half* d_recent_v;
        float* d_summary_k_acc;
        float* d_summary_v_acc;
        __half* d_summary_k;
        __half* d_summary_v;
        uint32_t* d_block_counts;
        __half* d_rep_k;
        __half* d_rep_v;
        uint32_t* d_rep_positions;
        int8_t* d_q8_old_k;
        int8_t* d_q8_old_v;
        float* d_q8_old_k_scale;
        float* d_q8_old_v_scale;
        __half* d_fp16_old_k;
        uint8_t* d_q4_old_v;
        float* d_q4_old_v_scale;
        uint32_t max_seq_len;
        uint32_t max_batch_size;
        uint32_t num_heads;
        uint32_t head_dim;
        uint32_t compressed;
        uint32_t recent_window;
        uint32_t block_size;
        uint32_t max_blocks;
        uint32_t top_blocks;
        uint32_t representatives;
        uint32_t representative_policy;
        uint32_t q8_old_backing;
        uint32_t fp16_k_q4_v_old_backing;
        uint32_t q4_old_v_backing;
    };
    struct M40llmAttentionGroupStats {
        float prob_mass;
        float logit_max;
        float logit_mean;
        uint32_t count;
    };
    struct M40llmAttentionTopEntry {
        uint32_t group;
        uint32_t block_index;
        uint32_t token_position;
        float score;
        float probability;
    };
    struct M40llmAttentionBlockMass {
        uint32_t block_index;
        float prob_mass;
        float logit_max;
        float logit_mean;
        uint32_t count;
    };
    struct M40llmAttentionTelemetry {
        M40llmAttentionGroupStats recent;
        M40llmAttentionGroupStats selected_old_exact;
        M40llmAttentionGroupStats summary;
        M40llmAttentionGroupStats representatives;
        M40llmAttentionGroupStats other;
        float needle_block_mass;
        M40llmAttentionBlockMass selected_block_masses[64];
        uint32_t selected_block_mass_count;
        M40llmAttentionTopEntry top_entries[8];
        uint32_t top_entry_count;
    };

    static uint32_t selected_block_order_from_env() {
        const char* value = std::getenv("M40LLM_KV_SELECTED_BLOCK_ORDER");
        if (!value || value[0] == '\0') return 0;
        if (std::strcmp(value, "chronological") == 0) return 1u;
        if (std::strcmp(value, "descending") == 0) return 2u;
        return 0u;
    }

    static uint32_t exact_block_selected_token_capacity(
        uint32_t recent_count,
        uint32_t old_len,
        uint32_t selected_old_blocks,
        uint32_t block_size) {
        const uint32_t selected_old_token_capacity =
            std::min(old_len, selected_old_blocks * block_size);
        return recent_count + selected_old_token_capacity;
    }

    static bool q8_dense_shadow_from_env() {
        const char* split = std::getenv("M40LLM_KV_Q8_PRECISION_SPLIT_DIAG");
        if (split && std::strcmp(split, "1") == 0) return true;
        const char* value = std::getenv("M40LLM_KV_Q8_DENSE_SHADOW");
        return value && std::strcmp(value, "1") == 0;
    }

    static cudaStream_t f16_decode_projection_stream(M40llmCudaContext* ctx) {
        const char* value = std::getenv("M40LLM_F16_DECODE_STREAM");
        if (value && (std::strcmp(value, "decode") == 0 || std::strcmp(value, "DECODE") == 0)) {
            return ctx->decode_stream;
        }
        return ctx->prefill_stream;
    }

    static uint32_t q8_old_k_source_from_env() {
        const char* value = std::getenv("M40LLM_KV_EXACT_OLD_PRECISION");
        if (!value || value[0] == '\0') return 0u;
        return std::strcmp(value, "fp16-k-q8-v") == 0 || std::strcmp(value, "fp16-k-fp16-v") == 0
            ? 1u
            : 0u;
    }

    static uint32_t q8_old_v_source_from_env() {
        const char* value = std::getenv("M40LLM_KV_EXACT_OLD_PRECISION");
        if (!value || value[0] == '\0') return 0u;
        if (std::strcmp(value, "fp16-k-q4-v") == 0) return 2u;
        return std::strcmp(value, "q8-k-fp16-v") == 0 || std::strcmp(value, "fp16-k-fp16-v") == 0
            ? 1u
            : 0u;
    }

    static bool q4_old_v_diag_from_env() {
        const char* value = std::getenv("M40LLM_KV_Q4_V_DIAG");
        return value && std::strcmp(value, "1") == 0;
    }

    static uint32_t block_select_policy_from_env() {
        const char* value = std::getenv("M40LLM_KV_BLOCK_SELECT_POLICY");
        if (!value || value[0] == '\0' || std::strcmp(value, "topk") == 0) return 0u;
        if (std::strcmp(value, "neighbors") == 0) return 1u;
        if (std::strcmp(value, "threshold") == 0) return 2u;
        if (std::strcmp(value, "anchor") == 0) return 3u;
        if (std::strcmp(value, "anchor-neighbors") == 0) return 4u;
        if (std::strcmp(value, "explicit") == 0) return 5u;
        if (std::strcmp(value, "score-cluster") == 0) return 6u;
        if (std::strcmp(value, "score-cluster-adaptive") == 0) return 7u;
        if (std::strcmp(value, "explicit-score-order") == 0) return 8u;
        return 0u;
    }

    static float block_score_delta_from_env() {
        const char* value = std::getenv("M40LLM_KV_BLOCK_SCORE_DELTA");
        return value && value[0] != '\0' ? (float)std::atof(value) : 0.0f;
    }

    static uint32_t block_min_blocks_from_env() {
        const char* value = std::getenv("M40LLM_KV_BLOCK_MIN_BLOCKS");
        const int parsed = value && value[0] != '\0' ? std::atoi(value) : 0;
        return parsed > 0 ? (uint32_t)parsed : 0u;
    }

    static uint32_t block_max_blocks_from_env(uint32_t fallback) {
        const char* value = std::getenv("M40LLM_KV_BLOCK_MAX_BLOCKS");
        const int parsed = value && value[0] != '\0' ? std::atoi(value) : 0;
        return parsed > 0 ? (uint32_t)parsed : fallback;
    }

    static uint64_t anchor_blocks_from_env() {
        const char* value = std::getenv("M40LLM_KV_ANCHOR_BLOCKS");
        if (!value || value[0] == '\0') return 1ull;
        uint64_t mask = 0ull;
        const char* ptr = value;
        while (*ptr) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(ptr, &end, 10);
            if (end != ptr && parsed < 64ul) {
                mask |= (1ull << parsed);
            }
            ptr = end && *end ? end + 1 : end;
            if (!ptr) break;
        }
        return mask == 0ull ? 1ull : mask;
    }

    static void block_masks_from_env(const char* name, uint64_t* low, uint64_t* high) {
        *low = 0ull;
        *high = 0ull;
        const char* value = std::getenv(name);
        if (!value || value[0] == '\0') return;
        const char* ptr = value;
        while (*ptr) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(ptr, &end, 10);
            if (end != ptr && parsed < 128ul) {
                if (parsed < 64ul) {
                    *low |= (1ull << parsed);
                } else {
                    *high |= (1ull << (parsed - 64ul));
                }
            }
            ptr = end && *end ? end + 1 : end;
            if (!ptr) break;
        }
    }

    __host__ __device__ bool block_mask_contains(uint64_t low, uint64_t high, uint32_t block) {
        if (block < 64u) return (low & (1ull << block)) != 0ull;
        if (block < 128u) return (high & (1ull << (block - 64u))) != 0ull;
        return false;
    }

    __device__ void sort_selected_blocks_chronological(uint32_t* blocks, uint32_t count) {
        for (uint32_t i = 1; i < count; ++i) {
            const uint32_t key = blocks[i];
            uint32_t j = i;
            while (j > 0 && blocks[j - 1] > key) {
                blocks[j] = blocks[j - 1];
                --j;
            }
            blocks[j] = key;
        }
    }

    __device__ void sort_selected_blocks_descending(uint32_t* blocks, uint32_t count) {
        for (uint32_t i = 1; i < count; ++i) {
            const uint32_t key = blocks[i];
            uint32_t j = i;
            while (j > 0 && blocks[j - 1] < key) {
                blocks[j] = blocks[j - 1];
                --j;
            }
            blocks[j] = key;
        }
    }

    __device__ bool append_unique_block(uint32_t* blocks, uint32_t* count, uint32_t capacity, uint32_t block) {
        if (block == 0xffffffffu) return false;
        for (uint32_t i = 0; i < *count; ++i) {
            if (blocks[i] == block) return false;
        }
        if (*count >= capacity) return false;
        blocks[(*count)++] = block;
        return true;
    }

    // Back-compat alias so other TU code using KVCache still compiles
    typedef M40llmKVCache KVCache;

    static int ensure_device(M40llmCudaContext* ctx) {
        if (!ctx) return -1;
        cudaError_t set_err = cudaSetDevice(ctx->device_id);
        return set_err == cudaSuccess ? 0 : -2;
    }

    static bool stream_log_enabled() {
        const char* env = getenv("M40LLM_STREAM_LOG");
        return env && strcmp(env, "1") == 0;
    }

    static cudaStream_t select_stream(M40llmCudaContext* ctx, uint32_t stream_kind) {
        if (!ctx) return nullptr;
        if (stream_kind == 0) return ctx->prefill_stream;
        if (stream_kind == 1) return ctx->decode_stream;
        return nullptr;
    }

    __global__ void argmax_f32_kernel(
        const float* __restrict__ values,
        uint32_t len,
        uint32_t* __restrict__ out_idx) {
        constexpr int threads = 256;
        __shared__ float best_values[threads];
        __shared__ uint32_t best_indices[threads];

        const uint32_t tid = threadIdx.x;
        float best = -FLT_MAX;
        uint32_t best_idx = 0;
        for (uint32_t i = tid; i < len; i += blockDim.x) {
            const float v = values[i];
            if (v > best || (v == best && i < best_idx)) {
                best = v;
                best_idx = i;
            }
        }
        best_values[tid] = best;
        best_indices[tid] = best_idx;
        __syncthreads();

        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other = best_values[tid + stride];
                const uint32_t other_idx = best_indices[tid + stride];
                if (other > best_values[tid] ||
                    (other == best_values[tid] && other_idx < best_indices[tid])) {
                    best_values[tid] = other;
                    best_indices[tid] = other_idx;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            *out_idx = best_indices[0];
        }
    }

    static size_t m40llm_strnlen_host(const char* s, size_t max_len) {
        size_t n = 0;
        while (n < max_len && s[n] != '\0') {
            ++n;
        }
        return n;
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
        if (ensure_device(ctx) != 0) return -3;
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

    int m40llm_memcpy_d2d_async(
        M40llmCudaContext* ctx,
        void* dst_device,
        const void* src_device,
        size_t bytes,
        uint32_t stream_kind) {
        if (!ctx || !dst_device || !src_device) return -1;
        if (ensure_device(ctx) != 0) return -3;
        cudaStream_t stream = select_stream(ctx, stream_kind);
        if (!stream) return -4;
        cudaError_t err = cudaMemcpyAsync(
            dst_device,
            src_device,
            bytes,
            cudaMemcpyDeviceToDevice,
            stream);
        if (err != cudaSuccess) return -2;
        return 0;
    }

    int m40llm_argmax_f32_async(
        M40llmCudaContext* ctx,
        const void* d_values_f32,
        uint32_t len,
        void* d_out_u32,
        uint32_t stream_kind) {
        if (!ctx || !d_values_f32 || !d_out_u32 || len == 0) return -1;
        if (ensure_device(ctx) != 0) return -3;
        cudaStream_t stream = select_stream(ctx, stream_kind);
        if (!stream) return -4;
        argmax_f32_kernel<<<1, 256, 0, stream>>>(
            reinterpret_cast<const float*>(d_values_f32),
            len,
            reinterpret_cast<uint32_t*>(d_out_u32));
        cudaError_t err = cudaGetLastError();
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
            return nullptr;
        }

        M40llmCudaContext* ctx = new M40llmCudaContext();
        ctx->device_id = selected;
        ctx->prefill_priority = 0;
        ctx->decode_priority = 0;
        ctx->persistent_decode = nullptr;
        ctx->prefill_waits_decode_event = nullptr;
        ctx->decode_waits_prefill_event = nullptr;

        int least_priority = 0;
        int greatest_priority = 0;
        cudaError_t priority_err = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        if (priority_err == cudaSuccess) {
            ctx->prefill_priority = least_priority;
            ctx->decode_priority = greatest_priority;
        }

        cudaError_t stream_err = cudaStreamCreateWithPriority(
            &ctx->prefill_stream,
            cudaStreamNonBlocking,
            ctx->prefill_priority);
        if (stream_err != cudaSuccess) {
            delete ctx;
            return nullptr;
        }
        stream_err = cudaStreamCreateWithPriority(
            &ctx->decode_stream,
            cudaStreamNonBlocking,
            ctx->decode_priority);
        if (stream_err != cudaSuccess) {
            cudaStreamDestroy(ctx->prefill_stream);
            delete ctx;
            return nullptr;
        }

        cudaError_t event_err = cudaEventCreateWithFlags(
            &ctx->prefill_waits_decode_event,
            cudaEventDisableTiming);
        if (event_err != cudaSuccess) {
            cudaStreamDestroy(ctx->prefill_stream);
            cudaStreamDestroy(ctx->decode_stream);
            delete ctx;
            return nullptr;
        }
        event_err = cudaEventCreateWithFlags(
            &ctx->decode_waits_prefill_event,
            cudaEventDisableTiming);
        if (event_err != cudaSuccess) {
            cudaEventDestroy(ctx->prefill_waits_decode_event);
            cudaStreamDestroy(ctx->prefill_stream);
            cudaStreamDestroy(ctx->decode_stream);
            delete ctx;
            return nullptr;
        }

        if (stream_log_enabled()) {
            fprintf(stderr,
                "[cuda] streams: prefill=nonblocking priority=%d decode=nonblocking priority=%d range=[%d,%d]\n",
                ctx->prefill_priority,
                ctx->decode_priority,
                least_priority,
                greatest_priority);
        }

    #ifdef M40LLM_HAVE_CUBLAS
        if (cublasCreate(&ctx->cublas) != CUBLAS_STATUS_SUCCESS) {
            cudaEventDestroy(ctx->prefill_waits_decode_event);
            cudaEventDestroy(ctx->decode_waits_prefill_event);
            cudaStreamDestroy(ctx->prefill_stream);
            cudaStreamDestroy(ctx->decode_stream);
            delete ctx;
            return nullptr;
        }
        cublasSetStream(ctx->cublas, ctx->prefill_stream); // default
        ctx->cublas_stream_kind = 0;
    #endif

        return ctx;
    }

    int m40llm_stream_synchronize(M40llmCudaContext* ctx, uint32_t stream_kind) {
        if (!ctx) return -1;
        if (ensure_device(ctx) != 0) return -2;
        cudaStream_t stream = select_stream(ctx, stream_kind);
        if (!stream) return -3;
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) return -4;
        return 0;
    }

    int m40llm_stream_wait_for_stream(
        M40llmCudaContext* ctx,
        uint32_t waiting_stream_kind,
        uint32_t signal_stream_kind) {
        if (!ctx) return -1;
        if (ensure_device(ctx) != 0) return -2;
        cudaStream_t waiting_stream = select_stream(ctx, waiting_stream_kind);
        cudaStream_t signal_stream = select_stream(ctx, signal_stream_kind);
        if (!waiting_stream || !signal_stream) return -3;
        if (waiting_stream_kind == signal_stream_kind) {
            return 0;
        }
        cudaEvent_t event = nullptr;
        cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            return -4;
        }
        err = cudaEventRecord(event, signal_stream);
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return -5;
        }
        err = cudaStreamWaitEvent(waiting_stream, event, 0);
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return -6;
        }
        err = cudaLaunchHostFunc(waiting_stream, destroy_wait_event_host_func, event);
        if (err != cudaSuccess) {
            // Host callbacks may be unsupported in some capture contexts. Keep
            // the event alive until context teardown rather than destroying it
            // before the asynchronously queued wait has consumed it.
            ctx->retained_wait_events.push_back(event);
        }
        return 0;
    }

    int m40llm_cuda_graph_begin_capture(M40llmCudaContext* ctx, uint32_t stream_kind) {
        if (!ctx) return -1;
        if (ensure_device(ctx) != 0) return -2;
        cudaStream_t stream = select_stream(ctx, stream_kind);
        if (!stream) return -3;
        cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
        return err == cudaSuccess ? 0 : -4;
    }

    int m40llm_cuda_graph_end_capture(
        M40llmCudaContext* ctx,
        uint32_t stream_kind,
        M40llmCudaGraphExec** out_graph) {
        if (!ctx || !out_graph) return -1;
        *out_graph = nullptr;
        if (ensure_device(ctx) != 0) return -2;
        cudaStream_t stream = select_stream(ctx, stream_kind);
        if (!stream) return -3;

        cudaGraph_t graph = nullptr;
        cudaError_t err = cudaStreamEndCapture(stream, &graph);
        if (err != cudaSuccess) return -4;
        if (!graph) return -5;

        cudaGraphExec_t exec = nullptr;
        err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            cudaGraphDestroy(graph);
            return -6;
        }

        M40llmCudaGraphExec* wrapped = new M40llmCudaGraphExec();
        wrapped->graph = graph;
        wrapped->exec = exec;
        *out_graph = wrapped;
        return 0;
    }

    int m40llm_cuda_graph_cancel_capture(M40llmCudaContext* ctx, uint32_t stream_kind) {
        if (!ctx) return -1;
        if (ensure_device(ctx) != 0) return -2;
        cudaStream_t stream = select_stream(ctx, stream_kind);
        if (!stream) return -3;
        cudaGraph_t graph = nullptr;
        cudaError_t err = cudaStreamEndCapture(stream, &graph);
        if (graph) cudaGraphDestroy(graph);
        return err == cudaSuccess ? 0 : -4;
    }

    int m40llm_cuda_graph_launch(
        M40llmCudaContext* ctx,
        M40llmCudaGraphExec* graph,
        uint32_t stream_kind) {
        if (!ctx || !graph || !graph->exec) return -1;
        if (ensure_device(ctx) != 0) return -2;
        cudaStream_t stream = select_stream(ctx, stream_kind);
        if (!stream) return -3;
        cudaError_t err = cudaGraphLaunch(graph->exec, stream);
        return err == cudaSuccess ? 0 : -4;
    }

    int m40llm_cuda_graph_launch_timed_sync(
        M40llmCudaContext* ctx,
        M40llmCudaGraphExec* graph,
        uint32_t stream_kind,
        float* elapsed_ms) {
        if (!ctx || !graph || !graph->exec || !elapsed_ms) return -1;
        *elapsed_ms = 0.0f;
        if (ensure_device(ctx) != 0) return -2;
        cudaStream_t stream = select_stream(ctx, stream_kind);
        if (!stream) return -3;

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        cudaError_t err = cudaEventCreate(&start);
        if (err != cudaSuccess) return -4;
        err = cudaEventCreate(&stop);
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            return -5;
        }

        err = cudaEventRecord(start, stream);
        if (err != cudaSuccess) {
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            return -6;
        }
        err = cudaGraphLaunch(graph->exec, stream);
        if (err != cudaSuccess) {
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            return -7;
        }
        err = cudaEventRecord(stop, stream);
        if (err != cudaSuccess) {
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            return -8;
        }
        err = cudaEventSynchronize(stop);
        if (err != cudaSuccess) {
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            return -9;
        }
        err = cudaEventElapsedTime(elapsed_ms, start, stop);
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return err == cudaSuccess ? 0 : -10;
    }

    int m40llm_cuda_event_create(M40llmCudaContext* ctx, M40llmCudaEvent** out_event) {
        if (!ctx || !out_event) return -1;
        *out_event = nullptr;
        if (ensure_device(ctx) != 0) return -2;
        M40llmCudaEvent* wrapped = new M40llmCudaEvent();
        wrapped->event = nullptr;
        cudaError_t err = cudaEventCreate(&wrapped->event);
        if (err != cudaSuccess) {
            delete wrapped;
            return -3;
        }
        *out_event = wrapped;
        return 0;
    }

    int m40llm_cuda_event_record(
        M40llmCudaContext* ctx,
        M40llmCudaEvent* event,
        uint32_t stream_kind) {
        if (!ctx || !event || !event->event) return -1;
        if (ensure_device(ctx) != 0) return -2;
        cudaStream_t stream = select_stream(ctx, stream_kind);
        if (!stream) return -3;
        cudaError_t err = cudaEventRecord(event->event, stream);
        return err == cudaSuccess ? 0 : -4;
    }

    int m40llm_cuda_event_elapsed_sync(
        M40llmCudaContext* ctx,
        M40llmCudaEvent* start,
        M40llmCudaEvent* stop,
        float* elapsed_ms) {
        if (!ctx || !start || !stop || !start->event || !stop->event || !elapsed_ms) return -1;
        *elapsed_ms = 0.0f;
        if (ensure_device(ctx) != 0) return -2;
        cudaError_t err = cudaEventSynchronize(stop->event);
        if (err != cudaSuccess) return -3;
        err = cudaEventElapsedTime(elapsed_ms, start->event, stop->event);
        return err == cudaSuccess ? 0 : -4;
    }

    void m40llm_cuda_event_destroy(M40llmCudaEvent* event) {
        if (!event) return;
        if (event->event) cudaEventDestroy(event->event);
        delete event;
    }

    void m40llm_cuda_graph_destroy(M40llmCudaGraphExec* graph) {
        if (!graph) return;
        if (graph->exec) cudaGraphExecDestroy(graph->exec);
        if (graph->graph) cudaGraphDestroy(graph->graph);
        delete graph;
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

        float ss = 0.0f;
        for (size_t i = 0; i < row_stride; i++) {
            float x = input[row * row_stride + i];
            ss += x * x;
        }
        float rms = sqrtf(ss / row_stride + eps);
        const float scale = 1.0f / rms;

        for (size_t i = 0; i < row_stride; i++) {
            output[row * row_stride + i] = input[row * row_stride + i] * scale;
        }
    }

    extern "C" __global__ void rms_norm_f32_weighted(
        const float* __restrict__ input,
        const void* __restrict__ weight,
        float* __restrict__ output,
        size_t n_rows,
        size_t row_stride,
        float eps,
        uint32_t weight_dtype) {
        const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= n_rows) return;

        float ss = 0.0f;
        for (size_t i = 0; i < row_stride; i++) {
            float x = input[row * row_stride + i];
            ss += x * x;
        }
        float rms = sqrtf(ss / row_stride + eps);
        const float scale = 1.0f / rms;

        const __half* w_f16 = reinterpret_cast<const __half*>(weight);
        const float* w_f32 = reinterpret_cast<const float*>(weight);
        for (size_t i = 0; i < row_stride; i++) {
            const float w = weight_dtype == 0 ? __half2float(w_f16[i]) : w_f32[i];
            output[row * row_stride + i] = input[row * row_stride + i] * scale * w;
        }
    }

    __global__ void rms_norm_f32_parallel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t n_rows,
        size_t row_stride,
        float eps) {
        const size_t row = blockIdx.x;
        if (row >= n_rows) return;

        extern __shared__ float reduce[];
        const uint32_t tid = threadIdx.x;
        const size_t base = row * row_stride;

        float ss = 0.0f;
        for (size_t i = tid; i < row_stride; i += blockDim.x) {
            const float x = input[base + i];
            ss += x * x;
        }
        reduce[tid] = ss;
        __syncthreads();

        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }

        const float scale = rsqrtf(reduce[0] / static_cast<float>(row_stride) + eps);
        for (size_t i = tid; i < row_stride; i += blockDim.x) {
            output[base + i] = input[base + i] * scale;
        }
    }

    __global__ void rms_norm_f32_weighted_parallel(
        const float* __restrict__ input,
        const void* __restrict__ weight,
        float* __restrict__ output,
        size_t n_rows,
        size_t row_stride,
        float eps,
        uint32_t weight_dtype) {
        const size_t row = blockIdx.x;
        if (row >= n_rows) return;

        extern __shared__ float reduce[];
        const uint32_t tid = threadIdx.x;
        const size_t base = row * row_stride;

        float ss = 0.0f;
        for (size_t i = tid; i < row_stride; i += blockDim.x) {
            const float x = input[base + i];
            ss += x * x;
        }
        reduce[tid] = ss;
        __syncthreads();

        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }

        const float scale = rsqrtf(reduce[0] / static_cast<float>(row_stride) + eps);
        const __half* w_f16 = reinterpret_cast<const __half*>(weight);
        const float* w_f32 = reinterpret_cast<const float*>(weight);
        for (size_t i = tid; i < row_stride; i += blockDim.x) {
            const float w = weight_dtype == 0 ? __half2float(w_f16[i]) : w_f32[i];
            output[base + i] = input[base + i] * scale * w;
        }
    }

    __global__ void rms_norm_f32_weighted_parallel_ldg(
        const float* __restrict__ input,
        const void* __restrict__ weight,
        float* __restrict__ output,
        size_t n_rows,
        size_t row_stride,
        float eps,
        uint32_t weight_dtype) {
        const size_t row = blockIdx.x;
        if (row >= n_rows) return;

        extern __shared__ float reduce[];
        const uint32_t tid = threadIdx.x;
        const size_t base = row * row_stride;

        float ss = 0.0f;
        for (size_t i = tid; i < row_stride; i += blockDim.x) {
            const float x = __ldg(input + base + i);
            ss += x * x;
        }
        reduce[tid] = ss;
        __syncthreads();

        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }

        const float scale = rsqrtf(reduce[0] / static_cast<float>(row_stride) + eps);
        const __half* w_f16 = reinterpret_cast<const __half*>(weight);
        const float* w_f32 = reinterpret_cast<const float*>(weight);
        for (size_t i = tid; i < row_stride; i += blockDim.x) {
            const float x = __ldg(input + base + i);
            const float w = weight_dtype == 0 ? __half2float(__ldg(w_f16 + i)) : __ldg(w_f32 + i);
            output[base + i] = x * scale * w;
        }
    }

    __global__ void kernel_f16_to_f32(const __half* in, float* out, size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            out[i] = __half2float(in[i]);
        }
    }

    // Q8_0 block layout per ggml: struct { ggml_half d; int8_t qs[32]; }
    __global__ void kernel_q80_to_f32(const int8_t* in, float* out, size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            size_t blk = i / 32;
            size_t idx = i % 32;
            const char* base = reinterpret_cast<const char*>(in) + blk * 34;
            const float d = __half2float(*reinterpret_cast<const __half*>(base + 0));
            int8_t q = *(reinterpret_cast<const int8_t*>(base + 2 + idx));
            out[i] = d * static_cast<float>(q);
        }
    }

    __global__ void residual_add_f32_kernel(
        const float* __restrict__ a,
        const float* __restrict__ b,
        float* __restrict__ out,
        size_t n) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            out[i] = a[i] + b[i];
        }
    }

    __global__ void residual_add_rms_norm_f32_weighted_parallel(
        const float* __restrict__ a,
        const float* __restrict__ b,
        float* __restrict__ residual_out,
        const void* __restrict__ weight,
        float* __restrict__ norm_out,
        size_t n_rows,
        size_t row_stride,
        float eps,
        uint32_t weight_dtype) {
        const size_t row = blockIdx.x;
        if (row >= n_rows) return;

        extern __shared__ float reduce[];
        const uint32_t tid = threadIdx.x;
        const size_t base = row * row_stride;

        float ss = 0.0f;
        for (size_t i = tid; i < row_stride; i += blockDim.x) {
            const size_t idx = base + i;
            const float value = a[idx] + b[idx];
            residual_out[idx] = value;
            ss += value * value;
        }
        reduce[tid] = ss;
        __syncthreads();

        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }

        const float scale = rsqrtf(reduce[0] / static_cast<float>(row_stride) + eps);
        const __half* w_f16 = reinterpret_cast<const __half*>(weight);
        const float* w_f32 = reinterpret_cast<const float*>(weight);
        for (size_t i = tid; i < row_stride; i += blockDim.x) {
            const float w = weight_dtype == 0 ? __half2float(w_f16[i]) : w_f32[i];
            norm_out[base + i] = residual_out[base + i] * scale * w;
        }
    }

    __global__ void swiglu_f32_kernel(
        const float* __restrict__ gate,
        const float* __restrict__ up,
        float* __restrict__ out,
        size_t n) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            const float g = gate[i];
            const float silu = g / (1.0f + expf(-g));
            out[i] = silu * up[i];
        }
    }
extern "C" {

    // Forward declaration
// Declaration removed - using extern "C" version below
extern "C" int m40llm_rms_norm_f32_async(
        M40llmCudaContext* ctx,
        const void* d_input,
        void* d_output,
        uint32_t rows,
        uint32_t dim,
        float eps);
extern "C" int m40llm_rms_norm_f32_weighted_async(
        M40llmCudaContext* ctx,
        const void* d_input,
        const void* d_weight,
        void* d_output,
        uint32_t rows,
        uint32_t dim,
        float eps,
        uint32_t weight_dtype);
int m40llm_residual_add_f32_async(
        M40llmCudaContext* ctx,
        const void* d_a_f32,
        const void* d_b_f32,
        void* d_out_f32,
        size_t n);
int m40llm_residual_add_rms_norm_f32_weighted_async(
        M40llmCudaContext* ctx,
        const void* d_a_f32,
        const void* d_b_f32,
        void* d_residual_out_f32,
        const void* d_weight,
        void* d_norm_out_f32,
        uint32_t rows,
        uint32_t dim,
        float eps,
        uint32_t weight_dtype);
int m40llm_swiglu_f32_async(
        M40llmCudaContext* ctx,
        const void* d_gate_f32,
        const void* d_up_f32,
        void* d_out_f32,
        size_t n);
int m40llm_attention_last_token_f32_gqa_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        void* out_dev_f32);
int m40llm_attention_last_token_f32_gqa_seq_len_dev_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        const uint32_t* seq_len_dev,
        void* out_dev_f32);
int m40llm_attention_last_token_f32_gqa_block_select_exact_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        void* out_dev_f32);
int m40llm_attention_last_token_f32_gqa_block_summary_lossy_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        void* out_dev_f32);
int m40llm_kvcache_append_token_f32_async(
        M40llmCudaContext* ctx,
        M40llmKVCache* kv,
        uint32_t seq_id,
        const void* k_dev_f32,
        const void* v_dev_f32);
int m40llm_kvcache_append_token_f32_rope_k_async(
        M40llmCudaContext* ctx,
        M40llmKVCache* kv,
        uint32_t seq_id,
        const void* k_dev_f32,
        const void* v_dev_f32,
        uint32_t past_len,
        float freq_base,
        float freq_scale);
int m40llm_kvcache_append_token_f32_rope_k_at_async(
        M40llmCudaContext* ctx,
        M40llmKVCache* kv,
        uint32_t seq_id,
        const void* k_dev_f32,
        const void* v_dev_f32,
        uint32_t position,
        uint32_t past_len,
        float freq_base,
        float freq_scale);
int m40llm_kvcache_append_token_f32_rope_k_position_dev_async(
        M40llmCudaContext* ctx,
        M40llmKVCache* kv,
        uint32_t seq_id,
        const void* k_dev_f32,
        const void* v_dev_f32,
        const uint32_t* position_dev,
        float freq_base,
        float freq_scale);
int m40llm_kvcache_append_token_f32_rope_k_compressed_at_async(
        M40llmCudaContext* ctx,
        M40llmKVCache* kv,
        uint32_t seq_id,
        const void* k_dev_f32,
        const void* v_dev_f32,
        uint32_t position,
        float freq_base,
        float freq_scale);
void m40llm_kvcache_destroy(M40llmKVCache* kv);
int m40llm_rope_f32_async(
        M40llmCudaContext* ctx,
        float* q,
        float* k,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t past_len,
        float freq_base,
        float freq_scale);
int m40llm_rope_f32_inplace_async(
        M40llmCudaContext* ctx,
        float* x,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t past_len,
        float freq_base,
        float freq_scale);
int m40llm_rope_f32_inplace_position_dev_async(
        M40llmCudaContext* ctx,
        float* x,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        const uint32_t* past_len_dev,
        float freq_base,
        float freq_scale);

// RMS Normalization (FP32)
extern "C" int m40llm_rms_norm_f32(
        M40llmCudaContext* ctx,
        const void* d_input,
        void* d_output,
        uint32_t rows,
        uint32_t dim,
        float eps) {
        int rc = m40llm_rms_norm_f32_async(ctx, d_input, d_output, rows, dim, eps);
        if (rc != 0) return rc;
        cudaError_t err = cudaStreamSynchronize(ctx->decode_stream);
        return err == cudaSuccess ? 0 : -3;
    }

extern "C" int m40llm_rms_norm_f32_async(
        M40llmCudaContext* ctx,
        const void* d_input,
        void* d_output,
        uint32_t rows,
        uint32_t dim,
        float eps) {
        if (!ctx || !d_input || !d_output || dim == 0) return -1;
        cudaGetLastError();
        const int threads = 256;
        rms_norm_f32_parallel<<<rows, threads, threads * sizeof(float), ctx->decode_stream>>>(
            reinterpret_cast<const float*>(d_input),
            reinterpret_cast<float*>(d_output),
            rows,
            dim,
            eps);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -2;
        return 0;
    }

extern "C" int m40llm_rms_norm_f32_weighted(
        M40llmCudaContext* ctx,
        const void* d_input,
        const void* d_weight,
        void* d_output,
        uint32_t rows,
        uint32_t dim,
        float eps,
        uint32_t weight_dtype) {
        int rc = m40llm_rms_norm_f32_weighted_async(
            ctx, d_input, d_weight, d_output, rows, dim, eps, weight_dtype);
        if (rc != 0) return rc;
        cudaError_t err = cudaStreamSynchronize(ctx->decode_stream);
        return err == cudaSuccess ? 0 : -3;
    }

extern "C" int m40llm_rms_norm_f32_weighted_async(
        M40llmCudaContext* ctx,
        const void* d_input,
        const void* d_weight,
        void* d_output,
        uint32_t rows,
        uint32_t dim,
        float eps,
        uint32_t weight_dtype) {
        if (!ctx || !d_input || !d_weight || !d_output || dim == 0) return -1;
        if (weight_dtype > 1) return -4;
        cudaGetLastError();
        const int threads = 256;
        const char* cache_experiment = getenv("M40LLM_CACHE_EXPERIMENT");
        if (cache_experiment && strcmp(cache_experiment, "ldg") == 0) {
            static int logged_ldg = 0;
            if (!logged_ldg) {
                fprintf(stderr, "[cuda] rms_norm_weighted cache experiment: __ldg\n");
                logged_ldg = 1;
            }
            rms_norm_f32_weighted_parallel_ldg<<<rows, threads, threads * sizeof(float), ctx->decode_stream>>>(
                reinterpret_cast<const float*>(d_input),
                d_weight,
                reinterpret_cast<float*>(d_output),
                rows,
                dim,
                eps,
                weight_dtype);
        } else {
            rms_norm_f32_weighted_parallel<<<rows, threads, threads * sizeof(float), ctx->decode_stream>>>(
                reinterpret_cast<const float*>(d_input),
                d_weight,
                reinterpret_cast<float*>(d_output),
                rows,
                dim,
                eps,
                weight_dtype);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -2;
        return 0;
    }

    int m40llm_f16_to_f32(M40llmCudaContext* ctx, const void* d_in_f16, void* d_out_f32, size_t n) {
        if (!ctx || !d_in_f16 || !d_out_f32) return -1;
        if (ensure_device(ctx) != 0) return -4;
        // Pointer validation should report the pointer state, not a stale
        // asynchronous error from a prior CUDA operation in this process.
        cudaGetLastError();

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

    int m40llm_residual_add_f32(
        M40llmCudaContext* ctx,
        const void* d_a_f32,
        const void* d_b_f32,
        void* d_out_f32,
        size_t n) {
        int rc = m40llm_residual_add_f32_async(ctx, d_a_f32, d_b_f32, d_out_f32, n);
        if (rc != 0) return rc;
        cudaError_t err = cudaStreamSynchronize(ctx->decode_stream);
        if (err != cudaSuccess) return -4;
        return 0;
    }

    int m40llm_residual_add_f32_async(
        M40llmCudaContext* ctx,
        const void* d_a_f32,
        const void* d_b_f32,
        void* d_out_f32,
        size_t n) {
        if (!ctx || !d_a_f32 || !d_b_f32 || !d_out_f32) return -1;
        if (ensure_device(ctx) != 0) return -2;
        if (n == 0) return 0;
        const int threads = 256;
        const int blocks = (int)((n + threads - 1) / threads);
        residual_add_f32_kernel<<<blocks, threads, 0, ctx->decode_stream>>>(
            reinterpret_cast<const float*>(d_a_f32),
            reinterpret_cast<const float*>(d_b_f32),
            reinterpret_cast<float*>(d_out_f32),
            n);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -3;
        return 0;
    }

    int m40llm_residual_add_rms_norm_f32_weighted_async(
        M40llmCudaContext* ctx,
        const void* d_a_f32,
        const void* d_b_f32,
        void* d_residual_out_f32,
        const void* d_weight,
        void* d_norm_out_f32,
        uint32_t rows,
        uint32_t dim,
        float eps,
        uint32_t weight_dtype) {
        if (!ctx || !d_a_f32 || !d_b_f32 || !d_residual_out_f32 || !d_weight || !d_norm_out_f32) return -1;
        if (ensure_device(ctx) != 0) return -2;
        if (rows == 0 || dim == 0) return -4;
        if (weight_dtype > 1) return -5;
        const int threads = 256;
        residual_add_rms_norm_f32_weighted_parallel<<<
            rows,
            threads,
            threads * sizeof(float),
            ctx->decode_stream>>>(
            reinterpret_cast<const float*>(d_a_f32),
            reinterpret_cast<const float*>(d_b_f32),
            reinterpret_cast<float*>(d_residual_out_f32),
            d_weight,
            reinterpret_cast<float*>(d_norm_out_f32),
            rows,
            dim,
            eps,
            weight_dtype);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -3;
        return 0;
    }

    int m40llm_swiglu_f32(
        M40llmCudaContext* ctx,
        const void* d_gate_f32,
        const void* d_up_f32,
        void* d_out_f32,
        size_t n) {
        int rc = m40llm_swiglu_f32_async(ctx, d_gate_f32, d_up_f32, d_out_f32, n);
        if (rc != 0) return rc;
        cudaError_t err = cudaStreamSynchronize(ctx->decode_stream);
        if (err != cudaSuccess) return -4;
        return 0;
    }

    int m40llm_swiglu_f32_async(
        M40llmCudaContext* ctx,
        const void* d_gate_f32,
        const void* d_up_f32,
        void* d_out_f32,
        size_t n) {
        if (!ctx || !d_gate_f32 || !d_up_f32 || !d_out_f32) return -1;
        if (ensure_device(ctx) != 0) return -2;
        if (n == 0) return 0;
        const int threads = 256;
        const int blocks = (int)((n + threads - 1) / threads);
        swiglu_f32_kernel<<<blocks, threads, 0, ctx->decode_stream>>>(
            reinterpret_cast<const float*>(d_gate_f32),
            reinterpret_cast<const float*>(d_up_f32),
            reinterpret_cast<float*>(d_out_f32),
            n);
        cudaError_t err = cudaGetLastError();
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
        size_t nlen = m40llm_strnlen_host(prop.name, sizeof(prop.name));
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


    void m40llm_destroy_context(M40llmCudaContext* ctx);

    int m40llm_upload_weights(
        M40llmCudaContext* ctx,
        const void* host_ptr,
        size_t num_bytes,
        void** out_device_ptr) {
        if (!ctx || !host_ptr || !out_device_ptr) return -1;

        // Register memory if coming from mmap. This is a best-effort fast path:
        // test fixtures and Vec-backed buffers may not be registerable, and the
        // ignored registration error must not poison the following cudaMemcpy.
        cudaError_t reg_err = cudaHostRegister((void*)host_ptr, num_bytes, cudaHostRegisterDefault);
        bool registered_for_copy = (reg_err == cudaSuccess);
        if (reg_err != cudaSuccess) {
            cudaGetLastError();
        }
        
        void* d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, num_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
            return -2;
        }
        err = cudaMemcpy(d_ptr, host_ptr, num_bytes, cudaMemcpyHostToDevice);
        if (registered_for_copy) {
            cudaHostUnregister((void*)host_ptr);
            cudaGetLastError();
        }
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
        
        // KV cache storage is FP16; compute remains FP32.

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
                // Recompute score.
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

    __global__ void attention_last_token_gqa_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim,
        uint32_t seq_id,
        const float* __restrict__ Q,
        uint32_t seq_len,
        float* __restrict__ Out) {
        const uint32_t qh_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (qh_idx >= q_heads) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 1.0f / sqrtf((float)head_dim);
        const float* qh = Q + (size_t)qh_idx * (size_t)head_dim;

        float max_score = -1e30f;
        for (uint32_t t = 0; t < seq_len; ++t) {
            const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += qh[d] * __half2float(K[base + d]);
            }
            const float score = dot * inv_sqrt;
            if (score > max_score) max_score = score;
        }

        float denom = 0.0f;
        for (uint32_t t = 0; t < seq_len; ++t) {
            const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += qh[d] * __half2float(K[base + d]);
            }
            denom += expf(dot * inv_sqrt - max_score);
        }
        denom = denom > 0.f ? denom : 1.f;

        float* out_h = Out + (size_t)qh_idx * (size_t)head_dim;
        for (uint32_t d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (uint32_t t = 0; t < seq_len; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                float dot = 0.0f;
                for (uint32_t dd = 0; dd < head_dim; ++dd) {
                    dot += qh[dd] * __half2float(K[base + dd]);
                }
                const float p = expf(dot * inv_sqrt - max_score) / denom;
                acc += p * __half2float(V[base + d]);
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_last_token_gqa_seq_len_dev_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim,
        uint32_t seq_id,
        const float* __restrict__ Q,
        const uint32_t* __restrict__ seq_len_dev,
        float* __restrict__ Out) {
        if (!seq_len_dev) return;
        const uint32_t seq_len = *seq_len_dev;
        if (seq_len == 0 || seq_len > max_seq_len) return;

        const uint32_t qh_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (qh_idx >= q_heads) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 1.0f / sqrtf((float)head_dim);
        const float* qh = Q + (size_t)qh_idx * (size_t)head_dim;

        float max_score = -1e30f;
        for (uint32_t t = 0; t < seq_len; ++t) {
            const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += qh[d] * __half2float(K[base + d]);
            }
            const float score = dot * inv_sqrt;
            if (score > max_score) max_score = score;
        }

        float denom = 0.0f;
        for (uint32_t t = 0; t < seq_len; ++t) {
            const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += qh[d] * __half2float(K[base + d]);
            }
            denom += expf(dot * inv_sqrt - max_score);
        }
        denom = denom > 0.f ? denom : 1.f;

        float* out_h = Out + (size_t)qh_idx * (size_t)head_dim;
        for (uint32_t d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (uint32_t t = 0; t < seq_len; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                float dot = 0.0f;
                for (uint32_t dd = 0; dd < head_dim; ++dd) {
                    dot += qh[dd] * __half2float(K[base + dd]);
                }
                const float p = expf(dot * inv_sqrt - max_score) / denom;
                acc += p * __half2float(V[base + d]);
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_last_token_gqa_head64_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t seq_id,
        const float* __restrict__ Q,
        uint32_t seq_len,
        float* __restrict__ Out) {
        extern __shared__ float shmem[];
        float* scores = shmem;
        float* reduce = scores + seq_len;

        const uint32_t qh_idx = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const uint32_t head_dim = 64;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 0.125f;
        const float* qh = Q + (size_t)qh_idx * (size_t)head_dim;

        float max_score_local = -1e30f;
        for (uint32_t t = 0; t < seq_len; ++t) {
            const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += qh[d] * __half2float(K[base + d]);
            }
            reduce[tid] = dot;
            __syncthreads();

            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[t] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];

        float denom_part = 0.0f;
        for (uint32_t t = tid; t < seq_len; t += blockDim.x) {
            denom_part += expf(scores[t] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

        float* out_h = Out + (size_t)qh_idx * (size_t)head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t t = 0; t < seq_len; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                const float p = expf(scores[t] - max_score) / denom;
                acc += p * __half2float(V[base + d]);
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_last_token_gqa_head128_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t seq_id,
        const float* __restrict__ Q,
        uint32_t seq_len,
        float* __restrict__ Out) {
        extern __shared__ float shmem[];
        float* scores = shmem;
        float* reduce = scores + seq_len;

        const uint32_t qh_idx = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const uint32_t head_dim = 128;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 0.08838834764831845f;
        const float* qh = Q + (size_t)qh_idx * (size_t)head_dim;

        float max_score_local = -1e30f;
        for (uint32_t t = 0; t < seq_len; ++t) {
            const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += qh[d] * __half2float(K[base + d]);
            }
            reduce[tid] = dot;
            __syncthreads();

            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[t] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];

        float denom_part = 0.0f;
        for (uint32_t t = tid; t < seq_len; t += blockDim.x) {
            denom_part += expf(scores[t] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

        float* out_h = Out + (size_t)qh_idx * (size_t)head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t t = 0; t < seq_len; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                const float p = expf(scores[t] - max_score) / denom;
                acc += p * __half2float(V[base + d]);
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_last_token_gqa_dense_recent_head64_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t seq_id,
        const float* __restrict__ Q,
        uint32_t seq_len,
        uint32_t recent_window,
        float* __restrict__ Out) {
        const uint32_t window_len = seq_len < recent_window ? seq_len : recent_window;
        const uint32_t window_start = seq_len - window_len;
        extern __shared__ float shmem[];
        float* scores = shmem;
        float* reduce = scores + window_len;

        const uint32_t qh_idx = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads || window_len == 0) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const uint32_t head_dim = 64;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 0.125f;
        const float* qh = Q + (size_t)qh_idx * (size_t)head_dim;

        float max_score_local = -1e30f;
        for (uint32_t i = 0; i < window_len; ++i) {
            const uint32_t t = window_start + i;
            const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += qh[d] * __half2float(K[base + d]);
            }
            reduce[tid] = dot;
            __syncthreads();

            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }

            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[i] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];

        float denom_part = 0.0f;
        for (uint32_t i = tid; i < window_len; i += blockDim.x) {
            denom_part += expf(scores[i] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

        float* out_h = Out + (size_t)qh_idx * (size_t)head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t i = 0; i < window_len; ++i) {
                const uint32_t t = window_start + i;
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                const float p = expf(scores[i] - max_score) / denom;
                acc += p * __half2float(V[base + d]);
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_last_token_gqa_batched_head64_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        const uint32_t* __restrict__ seq_ids,
        const uint32_t* __restrict__ seq_lens,
        uint32_t batch_size,
        const float* __restrict__ Q,
        float* __restrict__ Out) {
        extern __shared__ float shmem[];
        const uint32_t qh_idx = blockIdx.x;
        const uint32_t batch_idx = blockIdx.y;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads || batch_idx >= batch_size) return;

        const uint32_t seq_id = seq_ids[batch_idx];
        const uint32_t seq_len = seq_lens[batch_idx];
        if (seq_len == 0) return;

        float* scores = shmem;
        float* reduce = scores + seq_len;
        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const uint32_t head_dim = 64;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 0.125f;
        const float* qh = Q + ((size_t)batch_idx * (size_t)q_heads + (size_t)qh_idx) * head_dim;

        float max_score_local = -1e30f;
        for (uint32_t t = 0; t < seq_len; ++t) {
            const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += qh[d] * __half2float(K[base + d]);
            }
            reduce[tid] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[t] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];

        float denom_part = 0.0f;
        for (uint32_t t = tid; t < seq_len; t += blockDim.x) {
            denom_part += expf(scores[t] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

        float* out_h = Out + ((size_t)batch_idx * (size_t)q_heads + (size_t)qh_idx) * head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t t = 0; t < seq_len; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                const float p = expf(scores[t] - max_score) / denom;
                acc += p * __half2float(V[base + d]);
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_last_token_gqa_batched_head128_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        const uint32_t* __restrict__ seq_ids,
        const uint32_t* __restrict__ seq_lens,
        uint32_t batch_size,
        const float* __restrict__ Q,
        float* __restrict__ Out) {
        extern __shared__ float shmem[];
        const uint32_t qh_idx = blockIdx.x;
        const uint32_t batch_idx = blockIdx.y;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads || batch_idx >= batch_size) return;

        const uint32_t seq_id = seq_ids[batch_idx];
        const uint32_t seq_len = seq_lens[batch_idx];
        if (seq_len == 0) return;

        float* scores = shmem;
        float* reduce = scores + seq_len;
        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const uint32_t head_dim = 128;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 0.08838834764831845f;
        const float* qh = Q + ((size_t)batch_idx * (size_t)q_heads + (size_t)qh_idx) * head_dim;

        float max_score_local = -1e30f;
        for (uint32_t t = 0; t < seq_len; ++t) {
            const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += qh[d] * __half2float(K[base + d]);
            }
            reduce[tid] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[t] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];

        float denom_part = 0.0f;
        for (uint32_t t = tid; t < seq_len; t += blockDim.x) {
            denom_part += expf(scores[t] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

        float* out_h = Out + ((size_t)batch_idx * (size_t)q_heads + (size_t)qh_idx) * head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t t = 0; t < seq_len; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                const float p = expf(scores[t] - max_score) / denom;
                acc += p * __half2float(V[base + d]);
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_last_token_gqa_block_select_exact_head64_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t seq_id,
        const float* __restrict__ Q,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        uint32_t selected_block_order,
        float* __restrict__ Out) {
        extern __shared__ float shmem[];
        const uint32_t qh_idx = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const uint32_t head_dim = 64;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 0.125f;
        const float* qh = Q + (size_t)qh_idx * (size_t)head_dim;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t recent_start = old_len;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t top_n = top_blocks < old_blocks ? top_blocks : old_blocks;

        const uint32_t recent_count = seq_len < recent_window ? seq_len : recent_window;
        const uint32_t selected_capacity = recent_count + top_blocks * block_size;
        float* scores = shmem;
        float* token_slots = scores + selected_capacity;
        float* reduce = token_slots + selected_capacity;

        float selected_scores[64];
        uint32_t selected_blocks[64];
        for (uint32_t i = 0; i < 64; ++i) {
            selected_scores[i] = -1e30f;
            selected_blocks[i] = 0xffffffffu;
        }

        for (uint32_t b = 0; b < old_blocks; ++b) {
            const uint32_t block_start = b * block_size;
            const uint32_t block_end = block_start + block_size < old_len ? block_start + block_size : old_len;
            float dot_part = 0.0f;
            for (uint32_t t = block_start; t < block_end; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                    dot_part += qh[d] * __half2float(K[base + d]);
                }
            }
            reduce[tid] = dot_part;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float block_len = (float)(block_end - block_start);
                const float score = (reduce[0] / block_len) * inv_sqrt;
                uint32_t insert = top_n;
                for (uint32_t i = 0; i < top_n; ++i) {
                    if (score > selected_scores[i]) {
                        insert = i;
                        break;
                    }
                }
                if (insert < top_n) {
                    for (uint32_t j = top_n - 1; j > insert; --j) {
                        selected_scores[j] = selected_scores[j - 1];
                        selected_blocks[j] = selected_blocks[j - 1];
                    }
                    selected_scores[insert] = score;
                    selected_blocks[insert] = b;
                }
            }
            __syncthreads();
        }

        uint32_t selected_count = 0;
        if (tid == 0) {
            if (selected_block_order == 1u) {
                sort_selected_blocks_chronological(selected_blocks, top_n);
            } else if (selected_block_order == 2u) {
                sort_selected_blocks_descending(selected_blocks, top_n);
            }
            for (uint32_t i = 0; i < top_n; ++i) {
                const uint32_t b = selected_blocks[i];
                if (b == 0xffffffffu) continue;
                const uint32_t block_start = b * block_size;
                const uint32_t block_end = block_start + block_size < old_len ? block_start + block_size : old_len;
                for (uint32_t t = block_start; t < block_end; ++t) {
                    token_slots[selected_count++] = __uint_as_float(t);
                }
            }
            for (uint32_t t = recent_start; t < seq_len; ++t) {
                token_slots[selected_count++] = __uint_as_float(t);
            }
            reduce[0] = __uint_as_float(selected_count);
        }
        __syncthreads();
        selected_count = __float_as_uint(reduce[0]);
        if (selected_count == 0) return;

        float max_score_local = -1e30f;
        for (uint32_t idx = 0; idx < selected_count; ++idx) {
            const uint32_t t = __float_as_uint(token_slots[idx]);
            const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += qh[d] * __half2float(K[base + d]);
            }
            reduce[tid] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[idx] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];

        float denom_part = 0.0f;
        for (uint32_t idx = tid; idx < selected_count; idx += blockDim.x) {
            denom_part += expf(scores[idx] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

        float* out_h = Out + (size_t)qh_idx * (size_t)head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t idx = 0; idx < selected_count; ++idx) {
                const uint32_t t = __float_as_uint(token_slots[idx]);
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                const float p = expf(scores[idx] - max_score) / denom;
                acc += p * __half2float(V[base + d]);
            }
            out_h[d] = acc;
        }
    }

    __global__ void gather_block_select_exact_head64_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t seq_id,
        const float* __restrict__ Q,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        uint32_t selected_capacity,
        uint32_t selected_block_order,
        __half* __restrict__ staged_k,
        __half* __restrict__ staged_v,
        uint32_t* __restrict__ staged_positions,
        uint32_t* __restrict__ staged_counts) {
        extern __shared__ float shmem[];
        const uint32_t qh_idx = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const uint32_t head_dim = 64;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 0.125f;
        const float* qh = Q + (size_t)qh_idx * (size_t)head_dim;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t recent_start = old_len;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t top_n = top_blocks < old_blocks ? top_blocks : old_blocks;

        float* token_slots = shmem;
        float* reduce = token_slots + selected_capacity;

        float selected_scores[64];
        uint32_t selected_blocks[64];
        for (uint32_t i = 0; i < 64; ++i) {
            selected_scores[i] = -1e30f;
            selected_blocks[i] = 0xffffffffu;
        }

        for (uint32_t b = 0; b < old_blocks; ++b) {
            const uint32_t block_start = b * block_size;
            const uint32_t block_end = block_start + block_size < old_len ? block_start + block_size : old_len;
            float dot_part = 0.0f;
            for (uint32_t t = block_start; t < block_end; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                    dot_part += qh[d] * __half2float(K[base + d]);
                }
            }
            reduce[tid] = dot_part;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float block_len = (float)(block_end - block_start);
                const float score = (reduce[0] / block_len) * inv_sqrt;
                uint32_t insert = top_n;
                for (uint32_t i = 0; i < top_n; ++i) {
                    if (score > selected_scores[i]) {
                        insert = i;
                        break;
                    }
                }
                if (insert < top_n) {
                    for (uint32_t j = top_n - 1; j > insert; --j) {
                        selected_scores[j] = selected_scores[j - 1];
                        selected_blocks[j] = selected_blocks[j - 1];
                    }
                    selected_scores[insert] = score;
                    selected_blocks[insert] = b;
                }
            }
            __syncthreads();
        }

        uint32_t selected_count = 0;
        if (tid == 0) {
            if (selected_block_order == 1u) {
                sort_selected_blocks_chronological(selected_blocks, top_n);
            } else if (selected_block_order == 2u) {
                sort_selected_blocks_descending(selected_blocks, top_n);
            }
            for (uint32_t i = 0; i < top_n; ++i) {
                const uint32_t b = selected_blocks[i];
                if (b == 0xffffffffu) continue;
                const uint32_t block_start = b * block_size;
                const uint32_t block_end = block_start + block_size < old_len ? block_start + block_size : old_len;
                for (uint32_t t = block_start; t < block_end; ++t) {
                    token_slots[selected_count++] = __uint_as_float(t);
                }
            }
            for (uint32_t t = recent_start; t < seq_len; ++t) {
                token_slots[selected_count++] = __uint_as_float(t);
            }
            reduce[0] = __uint_as_float(selected_count);
            staged_counts[qh_idx] = selected_count;
        }
        __syncthreads();
        selected_count = __float_as_uint(reduce[0]);

        for (uint32_t idx = 0; idx < selected_count; ++idx) {
            const uint32_t t = __float_as_uint(token_slots[idx]);
            if (tid == 0) {
                staged_positions[(size_t)qh_idx * selected_capacity + idx] = t;
            }
            const size_t src = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                             + (size_t)kvh_idx * (size_t)head_dim;
            const size_t dst = ((size_t)qh_idx * selected_capacity + idx) * head_dim;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                staged_k[dst + d] = K[src + d];
                staged_v[dst + d] = V[src + d];
            }
        }
    }

    __global__ void build_q8_old_from_dense_head64_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        int8_t* __restrict__ q8_k,
        int8_t* __restrict__ q8_v,
        float* __restrict__ k_scale,
        float* __restrict__ v_scale,
        uint32_t max_seq_len,
        uint32_t num_heads,
        uint32_t seq_id,
        uint32_t seq_len,
        uint32_t recent_window) {
        __shared__ float reduce[128];
        const uint32_t vec = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        const uint32_t head_dim = 64;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        if (old_len == 0) return;
        const uint32_t head = vec % num_heads;
        const uint32_t token = (vec / num_heads) % old_len;
        const size_t elems_per_token = (size_t)num_heads * head_dim;
        const size_t src_base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)token) * elems_per_token
                              + (size_t)head * head_dim;
        float local_max_k = 0.0f;
        float local_max_v = 0.0f;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            const float k = fabsf(__half2float(K[src_base + d]));
            const float v = fabsf(__half2float(V[src_base + d]));
            local_max_k = fmaxf(local_max_k, k);
            local_max_v = fmaxf(local_max_v, v);
        }
        reduce[tid] = local_max_k;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
            __syncthreads();
        }
        const float max_k = reduce[0];
        reduce[tid] = local_max_v;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
            __syncthreads();
        }
        const float max_v = reduce[0];
        const float scale_k = max_k > 0.0f ? max_k / 127.0f : 1.0f / 127.0f;
        const float scale_v = max_v > 0.0f ? max_v / 127.0f : 1.0f / 127.0f;
        const size_t scale_idx = ((size_t)seq_id * (size_t)max_seq_len + (size_t)token) * (size_t)num_heads + head;
        if (tid == 0) {
            k_scale[scale_idx] = scale_k;
            v_scale[scale_idx] = scale_v;
        }
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            const float k = __half2float(K[src_base + d]) / scale_k;
            const float v = __half2float(V[src_base + d]) / scale_v;
            const int kq = max(-127, min(127, (int)lrintf(k)));
            const int vq = max(-127, min(127, (int)lrintf(v)));
            q8_k[src_base + d] = (int8_t)kq;
            q8_v[src_base + d] = (int8_t)vq;
        }
    }

    __global__ void quantize_evicted_recent_to_q8_old_head64_kernel(
        const __half* __restrict__ recent_k,
        const __half* __restrict__ recent_v,
        int8_t* __restrict__ q8_k,
        int8_t* __restrict__ q8_v,
        float* __restrict__ k_scale,
        float* __restrict__ v_scale,
        uint32_t max_seq_len,
        uint32_t recent_window,
        uint32_t num_heads,
        uint32_t seq_id,
        uint32_t position) {
        __shared__ float reduce[128];
        const uint32_t head = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        const uint32_t head_dim = 64;
        if (head >= num_heads || position < recent_window) return;
        const uint32_t old_pos = position - recent_window;
        const uint32_t ring = position % recent_window;
        const size_t elems_per_token = (size_t)num_heads * head_dim;
        const size_t recent_base = ((size_t)seq_id * (size_t)recent_window + (size_t)ring)
                                 * elems_per_token + (size_t)head * head_dim;
        const size_t q8_base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)old_pos)
                             * elems_per_token + (size_t)head * head_dim;
        float local_max_k = 0.0f;
        float local_max_v = 0.0f;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            local_max_k = fmaxf(local_max_k, fabsf(__half2float(recent_k[recent_base + d])));
            local_max_v = fmaxf(local_max_v, fabsf(__half2float(recent_v[recent_base + d])));
        }
        reduce[tid] = local_max_k;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
            __syncthreads();
        }
        const float max_k = reduce[0];
        reduce[tid] = local_max_v;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
            __syncthreads();
        }
        const float max_v = reduce[0];
        const float scale_k = max_k > 0.0f ? max_k / 127.0f : 1.0f / 127.0f;
        const float scale_v = max_v > 0.0f ? max_v / 127.0f : 1.0f / 127.0f;
        const size_t scale_idx = ((size_t)seq_id * (size_t)max_seq_len + (size_t)old_pos)
                               * (size_t)num_heads + head;
        if (tid == 0) {
            k_scale[scale_idx] = scale_k;
            v_scale[scale_idx] = scale_v;
        }
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            const int kq = max(-127, min(127, (int)lrintf(__half2float(recent_k[recent_base + d]) / scale_k)));
            const int vq = max(-127, min(127, (int)lrintf(__half2float(recent_v[recent_base + d]) / scale_v)));
            q8_k[q8_base + d] = (int8_t)kq;
            q8_v[q8_base + d] = (int8_t)vq;
        }
    }

    __global__ void copy_evicted_recent_k_to_fp16_old_kernel(
        const __half* __restrict__ recent_k,
        __half* __restrict__ old_k,
        uint32_t max_seq_len,
        uint32_t recent_window,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t seq_id,
        uint32_t position) {
        const uint32_t head = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        if (head >= num_heads || position < recent_window) return;
        const uint32_t old_pos = position - recent_window;
        const uint32_t ring = position % recent_window;
        const size_t elems_per_token = (size_t)num_heads * head_dim;
        const size_t recent_base = ((size_t)seq_id * (size_t)recent_window + (size_t)ring)
                                 * elems_per_token + (size_t)head * head_dim;
        const size_t old_base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)old_pos)
                              * elems_per_token + (size_t)head * head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            old_k[old_base + d] = recent_k[recent_base + d];
        }
    }

    __device__ __forceinline__ uint8_t pack_q4_pair(float a, float b, float scale) {
        const int qa = max(-7, min(7, (int)lrintf(a / scale)));
        const int qb = max(-7, min(7, (int)lrintf(b / scale)));
        return (uint8_t)((qa & 0x0f) | ((qb & 0x0f) << 4));
    }

    __device__ __forceinline__ float unpack_q4(uint8_t packed, uint32_t lane, float scale) {
        const uint8_t nibble = lane == 0u ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
        const int signed_value = nibble >= 8u ? (int)nibble - 16 : (int)nibble;
        return (float)signed_value * scale;
    }

    __global__ void quantize_evicted_recent_v_to_q4_old_kernel(
        const __half* __restrict__ recent_v,
        uint8_t* __restrict__ q4_v,
        float* __restrict__ v_scale,
        uint32_t max_seq_len,
        uint32_t recent_window,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t seq_id,
        uint32_t position) {
        __shared__ float reduce[128];
        const uint32_t head = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        if (head >= num_heads || position < recent_window) return;
        const uint32_t old_pos = position - recent_window;
        const uint32_t ring = position % recent_window;
        const size_t elems_per_token = (size_t)num_heads * head_dim;
        const size_t recent_base = ((size_t)seq_id * (size_t)recent_window + (size_t)ring)
                                 * elems_per_token + (size_t)head * head_dim;
        const size_t packed_per_token = (size_t)num_heads * (head_dim / 2u);
        const size_t q4_base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)old_pos)
                             * packed_per_token + (size_t)head * (head_dim / 2u);
        float local_max_v = 0.0f;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            local_max_v = fmaxf(local_max_v, fabsf(__half2float(recent_v[recent_base + d])));
        }
        reduce[tid] = local_max_v;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
            __syncthreads();
        }
        const float max_v = reduce[0];
        const float scale_v = max_v > 0.0f ? max_v / 7.0f : 1.0f / 7.0f;
        const size_t scale_idx = ((size_t)seq_id * (size_t)max_seq_len + (size_t)old_pos)
                               * (size_t)num_heads + head;
        if (tid == 0) {
            v_scale[scale_idx] = scale_v;
        }
        for (uint32_t pair = tid; pair < head_dim / 2u; pair += blockDim.x) {
            const uint32_t d = pair * 2u;
            const float v0 = __half2float(recent_v[recent_base + d]);
            const float v1 = __half2float(recent_v[recent_base + d + 1u]);
            q4_v[q4_base + pair] = pack_q4_pair(v0, v1, scale_v);
        }
    }

    __global__ void gather_block_select_exact_q8_old_head64_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        const __half* __restrict__ recent_k,
        const __half* __restrict__ recent_v,
        const int8_t* __restrict__ q8_k,
        const int8_t* __restrict__ q8_v,
        const float* __restrict__ k_scale,
        const float* __restrict__ v_scale,
        const uint8_t* __restrict__ q4_v,
        const float* __restrict__ q4_v_scale,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t seq_id,
        const float* __restrict__ Q,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        uint32_t selected_capacity,
        uint32_t selected_block_order,
        uint32_t old_k_source,
        uint32_t old_v_source,
        __half* __restrict__ staged_k,
        __half* __restrict__ staged_v,
        uint32_t* __restrict__ staged_positions,
        uint32_t* __restrict__ staged_counts) {
        extern __shared__ float shmem[];
        const uint32_t qh_idx = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const uint32_t head_dim = 64;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 0.125f;
        const float* qh = Q + (size_t)qh_idx * (size_t)head_dim;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t recent_start = old_len;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t top_n = top_blocks < old_blocks ? top_blocks : old_blocks;

        float* token_slots = shmem;
        float* reduce = token_slots + selected_capacity;

        float selected_scores[64];
        uint32_t selected_blocks[64];
        for (uint32_t i = 0; i < 64; ++i) {
            selected_scores[i] = -1e30f;
            selected_blocks[i] = 0xffffffffu;
        }

        for (uint32_t b = 0; b < old_blocks; ++b) {
            const uint32_t block_start = b * block_size;
            const uint32_t block_end = block_start + block_size < old_len ? block_start + block_size : old_len;
            float dot_part = 0.0f;
            for (uint32_t t = block_start; t < block_end; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                const size_t scale_idx = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * (size_t)kv_heads + kvh_idx;
                const float ks = old_k_source == 1u ? 0.0f : k_scale[scale_idx];
                for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                    const float k_val = old_k_source == 1u && K
                        ? __half2float(K[base + d])
                        : (float)q8_k[base + d] * ks;
                    dot_part += qh[d] * k_val;
                }
            }
            reduce[tid] = dot_part;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float block_len = (float)(block_end - block_start);
                const float score = (reduce[0] / block_len) * inv_sqrt;
                uint32_t insert = top_n;
                for (uint32_t i = 0; i < top_n; ++i) {
                    if (score > selected_scores[i]) {
                        insert = i;
                        break;
                    }
                }
                if (insert < top_n) {
                    for (uint32_t j = top_n - 1; j > insert; --j) {
                        selected_scores[j] = selected_scores[j - 1];
                        selected_blocks[j] = selected_blocks[j - 1];
                    }
                    selected_scores[insert] = score;
                    selected_blocks[insert] = b;
                }
            }
            __syncthreads();
        }

        uint32_t selected_count = 0;
        if (tid == 0) {
            if (selected_block_order == 1u) {
                sort_selected_blocks_chronological(selected_blocks, top_n);
            } else if (selected_block_order == 2u) {
                sort_selected_blocks_descending(selected_blocks, top_n);
            }
            for (uint32_t i = 0; i < top_n; ++i) {
                const uint32_t b = selected_blocks[i];
                if (b == 0xffffffffu) continue;
                const uint32_t block_start = b * block_size;
                const uint32_t block_end = block_start + block_size < old_len ? block_start + block_size : old_len;
                for (uint32_t t = block_start; t < block_end; ++t) {
                    token_slots[selected_count++] = __uint_as_float(t);
                }
            }
            for (uint32_t t = recent_start; t < seq_len; ++t) {
                token_slots[selected_count++] = __uint_as_float(t);
            }
            reduce[0] = __uint_as_float(selected_count);
            staged_counts[qh_idx] = selected_count;
        }
        __syncthreads();
        selected_count = __float_as_uint(reduce[0]);

        for (uint32_t idx = 0; idx < selected_count; ++idx) {
            const uint32_t t = __float_as_uint(token_slots[idx]);
            if (tid == 0) {
                staged_positions[(size_t)qh_idx * selected_capacity + idx] = t;
            }
            const size_t dst = ((size_t)qh_idx * selected_capacity + idx) * head_dim;
            const bool old_token = t < recent_start;
            const size_t q8_src = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                + (size_t)kvh_idx * (size_t)head_dim;
            const size_t q4_src = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t)
                                * ((size_t)kv_heads * (size_t)(head_dim / 2u))
                                + (size_t)kvh_idx * (size_t)(head_dim / 2u);
            const size_t scale_idx = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * (size_t)kv_heads + kvh_idx;
            const float ks = old_token && old_k_source != 1u ? k_scale[scale_idx] : 0.0f;
            const float vs = old_token && old_v_source == 0u ? v_scale[scale_idx] : 0.0f;
            const float q4_vs = old_token && q4_v_scale ? q4_v_scale[scale_idx] : 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                if (old_token) {
                    staged_k[dst + d] = old_k_source == 1u && K
                        ? K[q8_src + d]
                        : __float2half_rn((float)q8_k[q8_src + d] * ks);
                    if (old_v_source == 1u && V) {
                        staged_v[dst + d] = V[q8_src + d];
                    } else if (old_v_source == 2u && q4_v && q4_v_scale) {
                        staged_v[dst + d] = __float2half_rn(unpack_q4(q4_v[q4_src + d / 2u], d & 1u, q4_vs));
                    } else {
                        staged_v[dst + d] = __float2half_rn((float)q8_v[q8_src + d] * vs);
                    }
                } else if (recent_k && recent_v) {
                    const uint32_t ring = t % recent_window;
                    const size_t recent_src = ((size_t)seq_id * (size_t)recent_window + (size_t)ring)
                                            * elems_per_token + (size_t)kvh_idx * (size_t)head_dim;
                    staged_k[dst + d] = recent_k[recent_src + d];
                    staged_v[dst + d] = recent_v[recent_src + d];
                } else {
                    const size_t src = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t)
                                     * elems_per_token + (size_t)kvh_idx * (size_t)head_dim;
                    staged_k[dst + d] = K[src + d];
                    staged_v[dst + d] = V[src + d];
                }
            }
        }
    }

    __global__ void attention_last_token_gqa_staged_exact_head64_kernel(
        const __half* __restrict__ staged_k,
        const __half* __restrict__ staged_v,
        const uint32_t* __restrict__ staged_counts,
        uint32_t selected_capacity,
        uint32_t q_heads,
        const float* __restrict__ Q,
        float* __restrict__ Out) {
        extern __shared__ float shmem[];
        const uint32_t qh_idx = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads) return;

        const uint32_t head_dim = 64;
        const float inv_sqrt = 0.125f;
        const float* qh = Q + (size_t)qh_idx * head_dim;
        const uint32_t selected_count = staged_counts[qh_idx];
        if (selected_count == 0 || selected_count > selected_capacity) return;
        float* scores = shmem;
        float* reduce = scores + selected_capacity;

        float max_score_local = -1e30f;
        for (uint32_t idx = 0; idx < selected_count; ++idx) {
            const size_t base = ((size_t)qh_idx * selected_capacity + idx) * head_dim;
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += qh[d] * __half2float(staged_k[base + d]);
            }
            reduce[tid] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[idx] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];

        float denom_part = 0.0f;
        for (uint32_t idx = tid; idx < selected_count; idx += blockDim.x) {
            denom_part += expf(scores[idx] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

        float* out_h = Out + (size_t)qh_idx * head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t idx = 0; idx < selected_count; ++idx) {
                const size_t base = ((size_t)qh_idx * selected_capacity + idx) * head_dim;
                const float p = expf(scores[idx] - max_score) / denom;
                acc += p * __half2float(staged_v[base + d]);
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_last_token_gqa_block_select_exact_q8_direct_head64_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        const __half* __restrict__ recent_k,
        const __half* __restrict__ recent_v,
        const int8_t* __restrict__ q8_k,
        const int8_t* __restrict__ q8_v,
        const float* __restrict__ k_scale,
        const float* __restrict__ v_scale,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t seq_id,
        const float* __restrict__ Q,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        uint32_t selected_capacity,
        uint32_t selected_block_order,
        uint32_t block_select_policy,
        float block_score_delta,
        uint32_t block_min_blocks,
        uint32_t block_max_blocks,
        uint64_t anchor_block_mask,
        uint64_t force_include_block_mask_low,
        uint64_t force_include_block_mask_high,
        uint64_t force_exclude_block_mask_low,
        uint64_t force_exclude_block_mask_high,
        float* __restrict__ Out) {
        extern __shared__ float shmem[];
        const uint32_t qh_idx = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const uint32_t head_dim = 64;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 0.125f;
        const float* qh = Q + (size_t)qh_idx * (size_t)head_dim;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t recent_start = old_len;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t block_capacity = block_max_blocks == 0u
            ? (old_blocks < 64u ? old_blocks : 64u)
            : (block_max_blocks < old_blocks ? block_max_blocks : old_blocks);
        const uint32_t top_n = top_blocks < old_blocks ? top_blocks : old_blocks;
        const uint32_t keep_n = (block_select_policy == 2u || block_select_policy == 6u || block_select_policy == 7u || block_select_policy == 8u) ? block_capacity : top_n;

        float* token_slots = shmem;
        float* scores = token_slots + selected_capacity;
        float* reduce = scores + selected_capacity;

        float selected_scores[64];
        uint32_t selected_blocks[64];
        uint32_t final_blocks[64];
        for (uint32_t i = 0; i < 64; ++i) {
            selected_scores[i] = -1e30f;
            selected_blocks[i] = 0xffffffffu;
            final_blocks[i] = 0xffffffffu;
        }

        for (uint32_t b = 0; b < old_blocks; ++b) {
            const uint32_t block_start = b * block_size;
            const uint32_t block_end = block_start + block_size < old_len ? block_start + block_size : old_len;
            float dot_part = 0.0f;
            for (uint32_t t = block_start; t < block_end; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                const size_t scale_idx = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * (size_t)kv_heads + kvh_idx;
                const float ks = k_scale[scale_idx];
                for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                    dot_part += qh[d] * ((float)q8_k[base + d] * ks);
                }
            }
            reduce[tid] = dot_part;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float block_len = (float)(block_end - block_start);
                const float score = (reduce[0] / block_len) * inv_sqrt;
                uint32_t insert = keep_n;
                for (uint32_t i = 0; i < keep_n; ++i) {
                    if (score > selected_scores[i]) {
                        insert = i;
                        break;
                    }
                }
                if (insert < keep_n) {
                    for (uint32_t j = keep_n - 1; j > insert; --j) {
                        selected_scores[j] = selected_scores[j - 1];
                        selected_blocks[j] = selected_blocks[j - 1];
                    }
                    selected_scores[insert] = score;
                    selected_blocks[insert] = b;
                }
            }
            __syncthreads();
        }

        uint32_t selected_count = 0;
        if (tid == 0) {
            uint32_t final_block_count = 0;
            const float best_score = top_n > 0 ? selected_scores[0] : -1e30f;
            const uint32_t base_min = block_min_blocks > top_n ? block_min_blocks : top_n;
            const uint32_t min_take = base_min < old_blocks ? base_min : old_blocks;
            for (uint32_t i = 0; i < top_n; ++i) {
                append_unique_block(final_blocks, &final_block_count, block_capacity, selected_blocks[i]);
            }
            if (block_select_policy == 1u || block_select_policy == 4u) {
                const uint32_t base_count = final_block_count;
                for (uint32_t i = 0; i < base_count; ++i) {
                    const uint32_t b = final_blocks[i];
                    if (b > 0u) append_unique_block(final_blocks, &final_block_count, block_capacity, b - 1u);
                    if (b + 1u < old_blocks) append_unique_block(final_blocks, &final_block_count, block_capacity, b + 1u);
                }
            }
            if (block_select_policy == 2u) {
                for (uint32_t i = 0; i < keep_n; ++i) {
                    if (i < min_take || selected_scores[i] >= best_score - block_score_delta) {
                        append_unique_block(final_blocks, &final_block_count, block_capacity, selected_blocks[i]);
                    }
                }
            }
            if (block_select_policy == 3u || block_select_policy == 4u) {
                for (uint32_t b = 0; b < old_blocks && b < 64u; ++b) {
                    if ((anchor_block_mask & (1ull << b)) != 0ull) {
                        append_unique_block(final_blocks, &final_block_count, block_capacity, b);
                    }
                }
            }
            if (block_select_policy == 5u) {
                for (uint32_t b = 0; b < old_blocks && b < 128u; ++b) {
                    if (block_mask_contains(force_include_block_mask_low, force_include_block_mask_high, b)) {
                        append_unique_block(final_blocks, &final_block_count, block_capacity, b);
                    }
                }
            }
            if (block_select_policy == 8u) {
                final_block_count = 0;
                for (uint32_t i = 0; i < keep_n; ++i) {
                    const uint32_t b = selected_blocks[i];
                    if (block_mask_contains(force_include_block_mask_low, force_include_block_mask_high, b)) {
                        append_unique_block(final_blocks, &final_block_count, block_capacity, b);
                    }
                }
            }
            if (block_select_policy == 6u || block_select_policy == 7u) {
                const float cutoff = top_n > 0 ? selected_scores[top_n - 1u] - block_score_delta : best_score;
                for (uint32_t i = top_n; i < keep_n; ++i) {
                    if (selected_scores[i] >= cutoff) {
                        append_unique_block(final_blocks, &final_block_count, block_capacity, selected_blocks[i]);
                    }
                }
            }
            for (uint32_t i = 0; final_block_count < min_take && i < keep_n; ++i) {
                append_unique_block(final_blocks, &final_block_count, block_capacity, selected_blocks[i]);
            }
            if (block_select_policy == 5u) {
                uint32_t write = 0;
                for (uint32_t i = 0; i < final_block_count; ++i) {
                    const uint32_t b = final_blocks[i];
                    if (!block_mask_contains(force_exclude_block_mask_low, force_exclude_block_mask_high, b)) {
                        final_blocks[write++] = b;
                    }
                }
                for (uint32_t i = write; i < final_block_count; ++i) final_blocks[i] = 0xffffffffu;
                final_block_count = write;
            }
            if (selected_block_order == 1u) {
                sort_selected_blocks_chronological(final_blocks, final_block_count);
            } else if (selected_block_order == 2u) {
                sort_selected_blocks_descending(final_blocks, final_block_count);
            }
            for (uint32_t i = 0; i < final_block_count; ++i) {
                const uint32_t b = final_blocks[i];
                if (b == 0xffffffffu) continue;
                const uint32_t block_start = b * block_size;
                const uint32_t block_end = block_start + block_size < old_len ? block_start + block_size : old_len;
                for (uint32_t t = block_start; t < block_end; ++t) {
                    token_slots[selected_count++] = __uint_as_float(t);
                }
            }
            for (uint32_t t = recent_start; t < seq_len; ++t) {
                token_slots[selected_count++] = __uint_as_float(t);
            }
            reduce[0] = __uint_as_float(selected_count);
        }
        __syncthreads();
        selected_count = __float_as_uint(reduce[0]);
        if (selected_count == 0 || selected_count > selected_capacity) return;

        float max_score_local = -1e30f;
        for (uint32_t idx = 0; idx < selected_count; ++idx) {
            const uint32_t t = __float_as_uint(token_slots[idx]);
            const bool old_token = t < recent_start;
            float dot = 0.0f;
            if (old_token) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                const size_t scale_idx = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * (size_t)kv_heads + kvh_idx;
                const float ks = k_scale[scale_idx];
                for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                    const __half k = __float2half_rn((float)q8_k[base + d] * ks);
                    dot += qh[d] * __half2float(k);
                }
            } else if (recent_k && recent_v) {
                const uint32_t ring = t % recent_window;
                const size_t base = ((size_t)seq_id * (size_t)recent_window + (size_t)ring) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                    dot += qh[d] * __half2float(recent_k[base + d]);
                }
            } else {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                    dot += qh[d] * __half2float(K[base + d]);
                }
            }
            reduce[tid] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[idx] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];

        float denom_part = 0.0f;
        for (uint32_t idx = tid; idx < selected_count; idx += blockDim.x) {
            denom_part += expf(scores[idx] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

        float* out_h = Out + (size_t)qh_idx * head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t idx = 0; idx < selected_count; ++idx) {
                const uint32_t t = __float_as_uint(token_slots[idx]);
                const float p = expf(scores[idx] - max_score) / denom;
                const bool old_token = t < recent_start;
                if (old_token) {
                    const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                       + (size_t)kvh_idx * (size_t)head_dim;
                    const size_t scale_idx = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * (size_t)kv_heads + kvh_idx;
                    const __half v = __float2half_rn((float)q8_v[base + d] * v_scale[scale_idx]);
                    acc += p * __half2float(v);
                } else if (recent_k && recent_v) {
                    const uint32_t ring = t % recent_window;
                    const size_t base = ((size_t)seq_id * (size_t)recent_window + (size_t)ring) * elems_per_token
                                       + (size_t)kvh_idx * (size_t)head_dim;
                    acc += p * __half2float(recent_v[base + d]);
                } else {
                    const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                       + (size_t)kvh_idx * (size_t)head_dim;
                    acc += p * __half2float(V[base + d]);
                }
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_last_token_gqa_block_select_exact_fp16_k_q4_v_direct_kernel(
        const __half* __restrict__ old_k,
        const __half* __restrict__ recent_k,
        const __half* __restrict__ recent_v,
        const uint8_t* __restrict__ q4_v,
        const float* __restrict__ q4_v_scale,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim,
        uint32_t seq_id,
        const float* __restrict__ Q,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        uint32_t selected_capacity,
        uint32_t selected_block_order,
        uint32_t block_select_policy,
        float block_score_delta,
        uint32_t block_min_blocks,
        uint32_t block_max_blocks,
        uint64_t anchor_block_mask,
        uint64_t force_include_block_mask_low,
        uint64_t force_include_block_mask_high,
        uint64_t force_exclude_block_mask_low,
        uint64_t force_exclude_block_mask_high,
        float* __restrict__ Out) {
        extern __shared__ float shmem[];
        const uint32_t qh_idx = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const size_t packed_per_token = (size_t)kv_heads * (size_t)(head_dim / 2u);
        const float inv_sqrt = head_dim == 64u ? 0.125f : 0.08838834764831845f;
        const float* qh = Q + (size_t)qh_idx * (size_t)head_dim;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t recent_start = old_len;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t block_capacity = block_max_blocks == 0u
            ? (old_blocks < 64u ? old_blocks : 64u)
            : (block_max_blocks < old_blocks ? block_max_blocks : old_blocks);
        const uint32_t top_n = top_blocks < old_blocks ? top_blocks : old_blocks;
        const uint32_t keep_n = (block_select_policy == 2u || block_select_policy == 6u || block_select_policy == 7u || block_select_policy == 8u) ? block_capacity : top_n;

        float* token_slots = shmem;
        float* scores = token_slots + selected_capacity;
        float* reduce = scores + selected_capacity;

        float selected_scores[64];
        uint32_t selected_blocks[64];
        uint32_t final_blocks[64];
        for (uint32_t i = 0; i < 64; ++i) {
            selected_scores[i] = -1e30f;
            selected_blocks[i] = 0xffffffffu;
            final_blocks[i] = 0xffffffffu;
        }

        for (uint32_t b = 0; b < old_blocks; ++b) {
            const uint32_t block_start = b * block_size;
            const uint32_t block_end = block_start + block_size < old_len ? block_start + block_size : old_len;
            float dot_part = 0.0f;
            for (uint32_t t = block_start; t < block_end; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                    dot_part += qh[d] * __half2float(old_k[base + d]);
                }
            }
            reduce[tid] = dot_part;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float block_len = (float)(block_end - block_start);
                const float score = (reduce[0] / block_len) * inv_sqrt;
                uint32_t insert = keep_n;
                for (uint32_t i = 0; i < keep_n; ++i) {
                    if (score > selected_scores[i]) {
                        insert = i;
                        break;
                    }
                }
                if (insert < keep_n) {
                    for (uint32_t j = keep_n - 1; j > insert; --j) {
                        selected_scores[j] = selected_scores[j - 1];
                        selected_blocks[j] = selected_blocks[j - 1];
                    }
                    selected_scores[insert] = score;
                    selected_blocks[insert] = b;
                }
            }
            __syncthreads();
        }

        uint32_t selected_count = 0;
        if (tid == 0) {
            uint32_t final_block_count = 0;
            const float best_score = top_n > 0 ? selected_scores[0] : -1e30f;
            const uint32_t base_min = block_min_blocks > top_n ? block_min_blocks : top_n;
            const uint32_t min_take = base_min < old_blocks ? base_min : old_blocks;
            for (uint32_t i = 0; i < top_n; ++i) {
                append_unique_block(final_blocks, &final_block_count, block_capacity, selected_blocks[i]);
            }
            if (block_select_policy == 1u || block_select_policy == 4u) {
                const uint32_t base_count = final_block_count;
                for (uint32_t i = 0; i < base_count; ++i) {
                    const uint32_t b = final_blocks[i];
                    if (b > 0u) append_unique_block(final_blocks, &final_block_count, block_capacity, b - 1u);
                    if (b + 1u < old_blocks) append_unique_block(final_blocks, &final_block_count, block_capacity, b + 1u);
                }
            }
            if (block_select_policy == 2u) {
                for (uint32_t i = 0; i < keep_n; ++i) {
                    if (i < min_take || selected_scores[i] >= best_score - block_score_delta) {
                        append_unique_block(final_blocks, &final_block_count, block_capacity, selected_blocks[i]);
                    }
                }
            }
            if (block_select_policy == 3u || block_select_policy == 4u) {
                for (uint32_t b = 0; b < old_blocks && b < 64u; ++b) {
                    if ((anchor_block_mask & (1ull << b)) != 0ull) {
                        append_unique_block(final_blocks, &final_block_count, block_capacity, b);
                    }
                }
            }
            if (block_select_policy == 5u) {
                for (uint32_t b = 0; b < old_blocks && b < 128u; ++b) {
                    if (block_mask_contains(force_include_block_mask_low, force_include_block_mask_high, b)) {
                        append_unique_block(final_blocks, &final_block_count, block_capacity, b);
                    }
                }
            }
            if (block_select_policy == 8u) {
                final_block_count = 0;
                for (uint32_t i = 0; i < keep_n; ++i) {
                    const uint32_t b = selected_blocks[i];
                    if (block_mask_contains(force_include_block_mask_low, force_include_block_mask_high, b)) {
                        append_unique_block(final_blocks, &final_block_count, block_capacity, b);
                    }
                }
            }
            if (block_select_policy == 6u || block_select_policy == 7u) {
                const float cutoff = top_n > 0 ? selected_scores[top_n - 1u] - block_score_delta : best_score;
                for (uint32_t i = top_n; i < keep_n; ++i) {
                    if (selected_scores[i] >= cutoff) {
                        append_unique_block(final_blocks, &final_block_count, block_capacity, selected_blocks[i]);
                    }
                }
            }
            for (uint32_t i = 0; final_block_count < min_take && i < keep_n; ++i) {
                append_unique_block(final_blocks, &final_block_count, block_capacity, selected_blocks[i]);
            }
            if (block_select_policy == 5u) {
                uint32_t write = 0;
                for (uint32_t i = 0; i < final_block_count; ++i) {
                    const uint32_t b = final_blocks[i];
                    if (!block_mask_contains(force_exclude_block_mask_low, force_exclude_block_mask_high, b)) {
                        final_blocks[write++] = b;
                    }
                }
                for (uint32_t i = write; i < final_block_count; ++i) final_blocks[i] = 0xffffffffu;
                final_block_count = write;
            }
            if (selected_block_order == 1u) {
                sort_selected_blocks_chronological(final_blocks, final_block_count);
            } else if (selected_block_order == 2u) {
                sort_selected_blocks_descending(final_blocks, final_block_count);
            }
            for (uint32_t i = 0; i < final_block_count; ++i) {
                const uint32_t b = final_blocks[i];
                if (b == 0xffffffffu) continue;
                const uint32_t block_start = b * block_size;
                const uint32_t block_end = block_start + block_size < old_len ? block_start + block_size : old_len;
                for (uint32_t t = block_start; t < block_end; ++t) {
                    token_slots[selected_count++] = __uint_as_float(t);
                }
            }
            for (uint32_t t = recent_start; t < seq_len; ++t) {
                token_slots[selected_count++] = __uint_as_float(t);
            }
            reduce[0] = __uint_as_float(selected_count);
        }
        __syncthreads();
        selected_count = __float_as_uint(reduce[0]);
        if (selected_count == 0 || selected_count > selected_capacity) return;

        float max_score_local = -1e30f;
        for (uint32_t idx = 0; idx < selected_count; ++idx) {
            const uint32_t t = __float_as_uint(token_slots[idx]);
            const bool old_token = t < recent_start;
            float dot = 0.0f;
            if (old_token) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                    dot += qh[d] * __half2float(old_k[base + d]);
                }
            } else {
                const uint32_t ring = t % recent_window;
                const size_t base = ((size_t)seq_id * (size_t)recent_window + (size_t)ring) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                    dot += qh[d] * __half2float(recent_k[base + d]);
                }
            }
            reduce[tid] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[idx] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];

        float denom_part = 0.0f;
        for (uint32_t idx = tid; idx < selected_count; idx += blockDim.x) {
            denom_part += expf(scores[idx] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

        float* out_h = Out + (size_t)qh_idx * head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t idx = 0; idx < selected_count; ++idx) {
                const uint32_t t = __float_as_uint(token_slots[idx]);
                const float p = expf(scores[idx] - max_score) / denom;
                const bool old_token = t < recent_start;
                if (old_token) {
                    const size_t q4_base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * packed_per_token
                                         + (size_t)kvh_idx * (size_t)(head_dim / 2u);
                    const size_t scale_idx = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t)
                                           * (size_t)kv_heads + kvh_idx;
                    const float v = unpack_q4(q4_v[q4_base + d / 2u], d & 1u, q4_v_scale[scale_idx]);
                    acc += p * __half2float(__float2half_rn(v));
                } else {
                    const uint32_t ring = t % recent_window;
                    const size_t base = ((size_t)seq_id * (size_t)recent_window + (size_t)ring) * elems_per_token
                                       + (size_t)kvh_idx * (size_t)head_dim;
                    acc += p * __half2float(recent_v[base + d]);
                }
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_last_token_gqa_block_select_exact_fp16_k_q4_v_direct_batched_kernel(
        const __half* __restrict__ old_k,
        const __half* __restrict__ recent_k,
        const __half* __restrict__ recent_v,
        const uint8_t* __restrict__ q4_v,
        const float* __restrict__ q4_v_scale,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim,
        const uint32_t* __restrict__ seq_ids,
        const uint32_t* __restrict__ seq_lens,
        uint32_t batch_size,
        const float* __restrict__ Q,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        uint32_t selected_capacity,
        uint32_t selected_block_order,
        uint32_t block_select_policy,
        float block_score_delta,
        uint32_t block_min_blocks,
        uint32_t block_max_blocks,
        uint64_t anchor_block_mask,
        uint64_t force_include_block_mask_low,
        uint64_t force_include_block_mask_high,
        uint64_t force_exclude_block_mask_low,
        uint64_t force_exclude_block_mask_high,
        float* __restrict__ Out) {
        extern __shared__ float shmem[];
        const uint32_t qh_idx = blockIdx.x;
        const uint32_t batch_idx = blockIdx.y;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads || batch_idx >= batch_size) return;

        const uint32_t seq_id = seq_ids[batch_idx];
        const uint32_t seq_len = seq_lens[batch_idx];
        if (seq_len == 0 || seq_len > max_seq_len) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const size_t packed_per_token = (size_t)kv_heads * (size_t)(head_dim / 2u);
        const float inv_sqrt = head_dim == 64u ? 0.125f : 0.08838834764831845f;
        const float* qh = Q + ((size_t)batch_idx * (size_t)q_heads + (size_t)qh_idx) * (size_t)head_dim;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t recent_start = old_len;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t block_capacity = block_max_blocks == 0u
            ? (old_blocks < 64u ? old_blocks : 64u)
            : (block_max_blocks < old_blocks ? block_max_blocks : old_blocks);
        const uint32_t top_n = top_blocks < old_blocks ? top_blocks : old_blocks;
        const uint32_t keep_n = (block_select_policy == 2u || block_select_policy == 6u || block_select_policy == 7u || block_select_policy == 8u) ? block_capacity : top_n;

        float* token_slots = shmem;
        float* scores = token_slots + selected_capacity;
        float* reduce = scores + selected_capacity;

        float selected_scores[64];
        uint32_t selected_blocks[64];
        uint32_t final_blocks[64];
        for (uint32_t i = 0; i < 64; ++i) {
            selected_scores[i] = -1e30f;
            selected_blocks[i] = 0xffffffffu;
            final_blocks[i] = 0xffffffffu;
        }

        for (uint32_t b = 0; b < old_blocks; ++b) {
            const uint32_t block_start = b * block_size;
            const uint32_t block_end = block_start + block_size < old_len ? block_start + block_size : old_len;
            float dot_part = 0.0f;
            for (uint32_t t = block_start; t < block_end; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                    dot_part += qh[d] * __half2float(old_k[base + d]);
                }
            }
            reduce[tid] = dot_part;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float block_len = (float)(block_end - block_start);
                const float score = (reduce[0] / block_len) * inv_sqrt;
                uint32_t insert = keep_n;
                for (uint32_t i = 0; i < keep_n; ++i) {
                    if (score > selected_scores[i]) {
                        insert = i;
                        break;
                    }
                }
                if (insert < keep_n) {
                    for (uint32_t j = keep_n - 1; j > insert; --j) {
                        selected_scores[j] = selected_scores[j - 1];
                        selected_blocks[j] = selected_blocks[j - 1];
                    }
                    selected_scores[insert] = score;
                    selected_blocks[insert] = b;
                }
            }
            __syncthreads();
        }

        uint32_t selected_count = 0;
        if (tid == 0) {
            uint32_t final_block_count = 0;
            const float best_score = top_n > 0 ? selected_scores[0] : -1e30f;
            const uint32_t base_min = block_min_blocks > top_n ? block_min_blocks : top_n;
            const uint32_t min_take = base_min < old_blocks ? base_min : old_blocks;
            for (uint32_t i = 0; i < top_n; ++i) {
                append_unique_block(final_blocks, &final_block_count, block_capacity, selected_blocks[i]);
            }
            if (block_select_policy == 1u || block_select_policy == 4u) {
                const uint32_t base_count = final_block_count;
                for (uint32_t i = 0; i < base_count; ++i) {
                    const uint32_t b = final_blocks[i];
                    if (b > 0u) append_unique_block(final_blocks, &final_block_count, block_capacity, b - 1u);
                    if (b + 1u < old_blocks) append_unique_block(final_blocks, &final_block_count, block_capacity, b + 1u);
                }
            }
            if (block_select_policy == 2u) {
                for (uint32_t i = 0; i < keep_n; ++i) {
                    if (i < min_take || selected_scores[i] >= best_score - block_score_delta) {
                        append_unique_block(final_blocks, &final_block_count, block_capacity, selected_blocks[i]);
                    }
                }
            }
            if (block_select_policy == 3u || block_select_policy == 4u) {
                for (uint32_t b = 0; b < old_blocks && b < 64u; ++b) {
                    if ((anchor_block_mask & (1ull << b)) != 0ull) {
                        append_unique_block(final_blocks, &final_block_count, block_capacity, b);
                    }
                }
            }
            if (block_select_policy == 5u) {
                for (uint32_t b = 0; b < old_blocks && b < 128u; ++b) {
                    if (block_mask_contains(force_include_block_mask_low, force_include_block_mask_high, b)) {
                        append_unique_block(final_blocks, &final_block_count, block_capacity, b);
                    }
                }
            }
            if (block_select_policy == 8u) {
                final_block_count = 0;
                for (uint32_t i = 0; i < keep_n; ++i) {
                    const uint32_t b = selected_blocks[i];
                    if (block_mask_contains(force_include_block_mask_low, force_include_block_mask_high, b)) {
                        append_unique_block(final_blocks, &final_block_count, block_capacity, b);
                    }
                }
            }
            if (block_select_policy == 6u || block_select_policy == 7u) {
                const float cutoff = top_n > 0 ? selected_scores[top_n - 1u] - block_score_delta : best_score;
                for (uint32_t i = top_n; i < keep_n; ++i) {
                    if (selected_scores[i] >= cutoff) {
                        append_unique_block(final_blocks, &final_block_count, block_capacity, selected_blocks[i]);
                    }
                }
            }
            for (uint32_t i = 0; final_block_count < min_take && i < keep_n; ++i) {
                append_unique_block(final_blocks, &final_block_count, block_capacity, selected_blocks[i]);
            }
            if (block_select_policy == 5u) {
                uint32_t write = 0;
                for (uint32_t i = 0; i < final_block_count; ++i) {
                    const uint32_t b = final_blocks[i];
                    if (!block_mask_contains(force_exclude_block_mask_low, force_exclude_block_mask_high, b)) {
                        final_blocks[write++] = b;
                    }
                }
                for (uint32_t i = write; i < final_block_count; ++i) final_blocks[i] = 0xffffffffu;
                final_block_count = write;
            }
            if (selected_block_order == 1u) {
                sort_selected_blocks_chronological(final_blocks, final_block_count);
            } else if (selected_block_order == 2u) {
                sort_selected_blocks_descending(final_blocks, final_block_count);
            }
            for (uint32_t i = 0; i < final_block_count; ++i) {
                const uint32_t b = final_blocks[i];
                if (b == 0xffffffffu) continue;
                const uint32_t block_start = b * block_size;
                const uint32_t block_end = block_start + block_size < old_len ? block_start + block_size : old_len;
                for (uint32_t t = block_start; t < block_end; ++t) {
                    token_slots[selected_count++] = __uint_as_float(t);
                }
            }
            for (uint32_t t = recent_start; t < seq_len; ++t) {
                token_slots[selected_count++] = __uint_as_float(t);
            }
            reduce[0] = __uint_as_float(selected_count);
        }
        __syncthreads();
        selected_count = __float_as_uint(reduce[0]);
        if (selected_count == 0 || selected_count > selected_capacity) return;

        float max_score_local = -1e30f;
        for (uint32_t idx = 0; idx < selected_count; ++idx) {
            const uint32_t t = __float_as_uint(token_slots[idx]);
            const bool old_token = t < recent_start;
            float dot = 0.0f;
            if (old_token) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                    dot += qh[d] * __half2float(old_k[base + d]);
                }
            } else {
                const uint32_t ring = t % recent_window;
                const size_t base = ((size_t)seq_id * (size_t)recent_window + (size_t)ring) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                    dot += qh[d] * __half2float(recent_k[base + d]);
                }
            }
            reduce[tid] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[idx] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];

        float denom_part = 0.0f;
        for (uint32_t idx = tid; idx < selected_count; idx += blockDim.x) {
            denom_part += expf(scores[idx] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

        float* out_h = Out + ((size_t)batch_idx * (size_t)q_heads + (size_t)qh_idx) * (size_t)head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t idx = 0; idx < selected_count; ++idx) {
                const uint32_t t = __float_as_uint(token_slots[idx]);
                const float p = expf(scores[idx] - max_score) / denom;
                const bool old_token = t < recent_start;
                if (old_token) {
                    const size_t q4_base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * packed_per_token
                                         + (size_t)kvh_idx * (size_t)(head_dim / 2u);
                    const size_t scale_idx = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t)
                                           * (size_t)kv_heads + kvh_idx;
                    const float v = unpack_q4(q4_v[q4_base + d / 2u], d & 1u, q4_v_scale[scale_idx]);
                    acc += p * __half2float(__float2half_rn(v));
                } else {
                    const uint32_t ring = t % recent_window;
                    const size_t base = ((size_t)seq_id * (size_t)recent_window + (size_t)ring) * elems_per_token
                                       + (size_t)kvh_idx * (size_t)head_dim;
                    acc += p * __half2float(recent_v[base + d]);
                }
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_last_token_gqa_block_summary_lossy_head64_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t seq_id,
        const float* __restrict__ Q,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        float* __restrict__ Out) {
        extern __shared__ float shmem[];
        const uint32_t qh_idx = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const uint32_t head_dim = 64;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 0.125f;
        const float* qh = Q + (size_t)qh_idx * (size_t)head_dim;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t recent_start = old_len;
        const uint32_t recent_count = seq_len < recent_window ? seq_len : recent_window;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t selected_summary_count =
            top_blocks == 0 || top_blocks > old_blocks ? old_blocks : top_blocks;
        const uint32_t selected_capacity = recent_count + selected_summary_count;
        float* scores = shmem;
        float* token_slots = scores + selected_capacity;
        float* reduce = token_slots + selected_capacity;

        float selected_scores[64];
        uint32_t selected_blocks[64];
        for (uint32_t i = 0; i < 64; ++i) {
            selected_scores[i] = -1e30f;
            selected_blocks[i] = 0xffffffffu;
        }

        uint32_t selected_count = 0;
        for (uint32_t b = 0; b < old_blocks; ++b) {
            const uint32_t block_start = b * block_size;
            const uint32_t block_end = block_start + block_size < old_len ? block_start + block_size : old_len;
            float dot_part = 0.0f;
            for (uint32_t t = block_start; t < block_end; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                    dot_part += qh[d] * __half2float(K[base + d]);
                }
            }
            reduce[tid] = dot_part;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float block_len = (float)(block_end - block_start);
                const float score = (reduce[0] / block_len) * inv_sqrt;
                if (top_blocks == 0) {
                    token_slots[selected_count] = __uint_as_float(0x80000000u | b);
                    scores[selected_count] = score;
                    selected_count++;
                } else {
                    uint32_t insert = selected_summary_count;
                    for (uint32_t i = 0; i < selected_summary_count; ++i) {
                        if (score > selected_scores[i]) {
                            insert = i;
                            break;
                        }
                    }
                    if (insert < selected_summary_count) {
                        for (uint32_t j = selected_summary_count - 1; j > insert; --j) {
                            selected_scores[j] = selected_scores[j - 1];
                            selected_blocks[j] = selected_blocks[j - 1];
                        }
                        selected_scores[insert] = score;
                        selected_blocks[insert] = b;
                    }
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            if (top_blocks != 0) {
                selected_count = 0;
                for (uint32_t i = 0; i < selected_summary_count; ++i) {
                    if (selected_blocks[i] == 0xffffffffu) continue;
                    token_slots[selected_count] = __uint_as_float(0x80000000u | selected_blocks[i]);
                    scores[selected_count] = selected_scores[i];
                    selected_count++;
                }
            }
            for (uint32_t t = recent_start; t < seq_len; ++t) {
                token_slots[selected_count++] = __uint_as_float(t);
            }
            reduce[0] = __uint_as_float(selected_count);
        }
        __syncthreads();
        selected_count = __float_as_uint(reduce[0]);
        if (selected_count == 0) return;

        float max_score_local = -1e30f;
        for (uint32_t idx = 0; idx < selected_count; ++idx) {
            const uint32_t slot = __float_as_uint(token_slots[idx]);
            if ((slot & 0x80000000u) != 0) {
                if (tid == 0 && scores[idx] > max_score_local) max_score_local = scores[idx];
                __syncthreads();
                continue;
            }
            const uint32_t t = slot;
            const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += qh[d] * __half2float(K[base + d]);
            }
            reduce[tid] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[idx] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];

        float denom_part = 0.0f;
        for (uint32_t idx = tid; idx < selected_count; idx += blockDim.x) {
            denom_part += expf(scores[idx] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

        float* out_h = Out + (size_t)qh_idx * (size_t)head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t idx = 0; idx < selected_count; ++idx) {
                const uint32_t slot = __float_as_uint(token_slots[idx]);
                const float p = expf(scores[idx] - max_score) / denom;
                if ((slot & 0x80000000u) != 0) {
                    const uint32_t b = slot & 0x7fffffffu;
                    const uint32_t block_start = b * block_size;
                    const uint32_t block_end = block_start + block_size < old_len ? block_start + block_size : old_len;
                    float mean_v = 0.0f;
                    for (uint32_t t = block_start; t < block_end; ++t) {
                        const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                           + (size_t)kvh_idx * (size_t)head_dim;
                        mean_v += __half2float(V[base + d]);
                    }
                    mean_v /= (float)(block_end - block_start);
                    acc += p * mean_v;
                } else {
                    const uint32_t t = slot;
                    const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                       + (size_t)kvh_idx * (size_t)head_dim;
                    acc += p * __half2float(V[base + d]);
                }
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_last_token_gqa_compressed_summary_head64_kernel(
        const __half* __restrict__ recent_k,
        const __half* __restrict__ recent_v,
        const __half* __restrict__ summary_k,
        const __half* __restrict__ summary_v,
        const __half* __restrict__ rep_k,
        const __half* __restrict__ rep_v,
        const uint32_t* __restrict__ rep_positions,
        const uint32_t* __restrict__ block_counts,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t max_blocks,
        uint32_t representatives,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t seq_id,
        const float* __restrict__ Q,
        uint32_t seq_len,
        uint32_t top_blocks,
        float* __restrict__ Out) {
        extern __shared__ float shmem[];
        const uint32_t qh_idx = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const uint32_t head_dim = 64;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 0.125f;
        const float* qh = Q + (size_t)qh_idx * (size_t)head_dim;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t recent_start = old_len;
        const uint32_t recent_count = seq_len < recent_window ? seq_len : recent_window;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t available_blocks = old_blocks < max_blocks ? old_blocks : max_blocks;
        const uint32_t selected_summary_count =
            top_blocks == 0 || top_blocks > available_blocks ? available_blocks : top_blocks;
        const uint32_t selected_capacity = recent_count + selected_summary_count * (1u + representatives);
        float* scores = shmem;
        float* slots = scores + selected_capacity;
        float* reduce = slots + selected_capacity;

        float selected_scores[64];
        uint32_t selected_blocks[64];
        for (uint32_t i = 0; i < 64; ++i) {
            selected_scores[i] = -1e30f;
            selected_blocks[i] = 0xffffffffu;
        }

        uint32_t selected_count = 0;
        for (uint32_t b = 0; b < available_blocks; ++b) {
            const uint32_t count = block_counts[(size_t)seq_id * (size_t)max_blocks + (size_t)b];
            if (count == 0) continue;
            const size_t base = ((size_t)seq_id * (size_t)max_blocks + (size_t)b) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += qh[d] * __half2float(summary_k[base + d]);
            }
            reduce[tid] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                if (top_blocks == 0) {
                    slots[selected_count] = __uint_as_float(0x80000000u | b);
                    scores[selected_count] = score;
                    selected_count++;
                } else {
                    uint32_t insert = selected_summary_count;
                    for (uint32_t i = 0; i < selected_summary_count; ++i) {
                        if (score > selected_scores[i]) {
                            insert = i;
                            break;
                        }
                    }
                    if (insert < selected_summary_count) {
                        for (uint32_t j = selected_summary_count - 1; j > insert; --j) {
                            selected_scores[j] = selected_scores[j - 1];
                            selected_blocks[j] = selected_blocks[j - 1];
                        }
                        selected_scores[insert] = score;
                        selected_blocks[insert] = b;
                    }
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            if (top_blocks != 0) {
                selected_count = 0;
                for (uint32_t i = 0; i < selected_summary_count; ++i) {
                    if (selected_blocks[i] == 0xffffffffu) continue;
                    slots[selected_count] = __uint_as_float(0x80000000u | selected_blocks[i]);
                    scores[selected_count] = selected_scores[i];
                    selected_count++;
                }
            }
            const uint32_t summary_count = selected_count;
            if (representatives > 0 && rep_k && rep_v && rep_positions) {
                for (uint32_t i = 0; i < summary_count; ++i) {
                    const uint32_t block_slot = __float_as_uint(slots[i]);
                    if ((block_slot & 0x80000000u) == 0) continue;
                    const uint32_t b = block_slot & 0x7fffffffu;
                    for (uint32_t r = 0; r < representatives; ++r) {
                        const size_t pos_idx = ((size_t)seq_id * (size_t)max_blocks + (size_t)b)
                            * (size_t)representatives + (size_t)r;
                        if (rep_positions[pos_idx] == 0xffffffffu) continue;
                        slots[selected_count++] = __uint_as_float(0x40000000u | (b * representatives + r));
                    }
                }
            }
            for (uint32_t t = recent_start; t < seq_len; ++t) {
                slots[selected_count++] = __uint_as_float(t);
            }
            reduce[0] = __uint_as_float(selected_count);
        }
        __syncthreads();
        selected_count = __float_as_uint(reduce[0]);
        if (selected_count == 0) return;

        float max_score_local = -1e30f;
        for (uint32_t idx = 0; idx < selected_count; ++idx) {
            const uint32_t slot = __float_as_uint(slots[idx]);
            if ((slot & 0x80000000u) != 0) {
                if (tid == 0 && scores[idx] > max_score_local) max_score_local = scores[idx];
                __syncthreads();
                continue;
            }
            if ((slot & 0x40000000u) != 0) {
                const uint32_t rep_id = slot & 0x3fffffffu;
                const uint32_t b = representatives > 0 ? rep_id / representatives : 0;
                const uint32_t r = representatives > 0 ? rep_id % representatives : 0;
                const size_t base = (((size_t)seq_id * (size_t)max_blocks + (size_t)b)
                    * (size_t)representatives + (size_t)r) * elems_per_token
                    + (size_t)kvh_idx * (size_t)head_dim;
                float dot = 0.0f;
                for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                    dot += qh[d] * __half2float(rep_k[base + d]);
                }
                reduce[tid] = dot;
                __syncthreads();
                for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                    if (tid < stride) reduce[tid] += reduce[tid + stride];
                    __syncthreads();
                }
                if (tid == 0) {
                    const float score = reduce[0] * inv_sqrt;
                    scores[idx] = score;
                    if (score > max_score_local) max_score_local = score;
                }
                __syncthreads();
                continue;
            }
            const uint32_t recent_idx = slot % recent_window;
            const size_t base = ((size_t)seq_id * (size_t)recent_window + (size_t)recent_idx) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += qh[d] * __half2float(recent_k[base + d]);
            }
            reduce[tid] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[idx] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];
        float denom_part = 0.0f;
        for (uint32_t idx = tid; idx < selected_count; idx += blockDim.x) {
            denom_part += expf(scores[idx] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;
        float* out_h = Out + (size_t)qh_idx * (size_t)head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t idx = 0; idx < selected_count; ++idx) {
                const uint32_t slot = __float_as_uint(slots[idx]);
                const float p = expf(scores[idx] - max_score) / denom;
                if ((slot & 0x80000000u) != 0) {
                    const uint32_t b = slot & 0x7fffffffu;
                    const size_t base = ((size_t)seq_id * (size_t)max_blocks + (size_t)b) * elems_per_token
                                       + (size_t)kvh_idx * (size_t)head_dim;
                    acc += p * __half2float(summary_v[base + d]);
                } else if ((slot & 0x40000000u) != 0) {
                    const uint32_t rep_id = slot & 0x3fffffffu;
                    const uint32_t b = representatives > 0 ? rep_id / representatives : 0;
                    const uint32_t r = representatives > 0 ? rep_id % representatives : 0;
                    const size_t base = (((size_t)seq_id * (size_t)max_blocks + (size_t)b)
                        * (size_t)representatives + (size_t)r) * elems_per_token
                        + (size_t)kvh_idx * (size_t)head_dim;
                    acc += p * __half2float(rep_v[base + d]);
                } else {
                    const uint32_t recent_idx = slot % recent_window;
                    const size_t base = ((size_t)seq_id * (size_t)recent_window + (size_t)recent_idx) * elems_per_token
                                       + (size_t)kvh_idx * (size_t)head_dim;
                    acc += p * __half2float(recent_v[base + d]);
                }
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_last_token_gqa_compressed_recent_only_head64_kernel(
        const __half* __restrict__ recent_k,
        const __half* __restrict__ recent_v,
        uint32_t recent_window,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t seq_id,
        const float* __restrict__ Q,
        uint32_t seq_len,
        float* __restrict__ Out) {
        extern __shared__ float shmem[];
        float* scores = shmem;
        float* reduce = scores + recent_window;
        const uint32_t qh_idx = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const uint32_t head_dim = 64;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 0.125f;
        const float* qh = Q + (size_t)qh_idx * (size_t)head_dim;
        const uint32_t recent_count = seq_len < recent_window ? seq_len : recent_window;
        const uint32_t recent_start = seq_len - recent_count;
        if (recent_count == 0) return;

        float max_score_local = -1e30f;
        for (uint32_t i = 0; i < recent_count; ++i) {
            const uint32_t pos = recent_start + i;
            const uint32_t ring = pos % recent_window;
            const size_t base = ((size_t)seq_id * (size_t)recent_window + (size_t)ring) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += qh[d] * __half2float(recent_k[base + d]);
            }
            reduce[tid] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[i] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];
        float denom_part = 0.0f;
        for (uint32_t i = tid; i < recent_count; i += blockDim.x) {
            denom_part += expf(scores[i] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

        float* out_h = Out + (size_t)qh_idx * (size_t)head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t i = 0; i < recent_count; ++i) {
                const uint32_t pos = recent_start + i;
                const uint32_t ring = pos % recent_window;
                const size_t base = ((size_t)seq_id * (size_t)recent_window + (size_t)ring) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                const float p = expf(scores[i] - max_score) / denom;
                acc += p * __half2float(recent_v[base + d]);
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_last_token_gqa_head64_ldg_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t seq_id,
        const float* __restrict__ Q,
        uint32_t seq_len,
        float* __restrict__ Out) {
        extern __shared__ float shmem[];
        float* scores = shmem;
        float* reduce = scores + seq_len;

        const uint32_t qh_idx = blockIdx.x;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const uint32_t head_dim = 64;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 0.125f;
        const float* qh = Q + (size_t)qh_idx * (size_t)head_dim;

        float max_score_local = -1e30f;
        for (uint32_t t = 0; t < seq_len; ++t) {
            const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += qh[d] * __half2float(__ldg(K + base + d));
            }
            reduce[tid] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[t] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];

        float denom_part = 0.0f;
        for (uint32_t t = tid; t < seq_len; t += blockDim.x) {
            denom_part += expf(scores[t] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

        float* out_h = Out + (size_t)qh_idx * (size_t)head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t t = 0; t < seq_len; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                const float p = expf(scores[t] - max_score) / denom;
                acc += p * __half2float(__ldg(V + base + d));
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_last_token_gqa_batched_head64_ldg_kernel(
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        uint32_t max_seq_len,
        uint32_t q_heads,
        uint32_t kv_heads,
        const uint32_t* __restrict__ seq_ids,
        const uint32_t* __restrict__ seq_lens,
        uint32_t batch_size,
        const float* __restrict__ Q,
        float* __restrict__ Out) {
        extern __shared__ float shmem[];
        const uint32_t qh_idx = blockIdx.x;
        const uint32_t batch_idx = blockIdx.y;
        const uint32_t tid = threadIdx.x;
        if (qh_idx >= q_heads || batch_idx >= batch_size) return;

        const uint32_t seq_id = seq_ids[batch_idx];
        const uint32_t seq_len = seq_lens[batch_idx];
        if (seq_len == 0) return;

        float* scores = shmem;
        float* reduce = scores + seq_len;
        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const uint32_t head_dim = 64;
        const size_t elems_per_token = (size_t)kv_heads * (size_t)head_dim;
        const float inv_sqrt = 0.125f;
        const float* qh = Q + ((size_t)batch_idx * (size_t)q_heads + (size_t)qh_idx) * head_dim;

        float max_score_local = -1e30f;
        for (uint32_t t = 0; t < seq_len; ++t) {
            const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                               + (size_t)kvh_idx * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += qh[d] * __half2float(__ldg(K + base + d));
            }
            reduce[tid] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[t] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];

        float denom_part = 0.0f;
        for (uint32_t t = tid; t < seq_len; t += blockDim.x) {
            denom_part += expf(scores[t] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

        float* out_h = Out + ((size_t)batch_idx * (size_t)q_heads + (size_t)qh_idx) * head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t t = 0; t < seq_len; ++t) {
                const size_t base = ((size_t)seq_id * (size_t)max_seq_len + (size_t)t) * elems_per_token
                                   + (size_t)kvh_idx * (size_t)head_dim;
                const float p = expf(scores[t] - max_score) / denom;
                acc += p * __half2float(__ldg(V + base + d));
            }
            out_h[d] = acc;
        }
    }

    __global__ void attention_prefill_gqa_varlen_kernel(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        const uint32_t* __restrict__ q_offsets,
        const uint32_t* __restrict__ kv_offsets,
        const uint32_t* __restrict__ q_lens,
        const uint32_t* __restrict__ kv_lens,
        uint32_t batch_size,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim,
        float* __restrict__ Out) {
        extern __shared__ float shmem[];
        const uint32_t q_idx = blockIdx.x;
        const uint32_t qh_idx = blockIdx.y;
        const uint32_t batch_idx = blockIdx.z;
        const uint32_t tid = threadIdx.x;
        if (batch_idx >= batch_size || qh_idx >= q_heads) return;

        const uint32_t q_len = q_lens[batch_idx];
        const uint32_t kv_len = kv_lens[batch_idx];
        if (q_idx >= q_len || q_len == 0 || kv_len == 0 || q_len > kv_len) return;

        const uint32_t group = q_heads / kv_heads;
        const uint32_t kvh_idx = qh_idx / group;
        const float inv_sqrt = head_dim == 64u ? 0.125f : 0.08838834764831845f;
        const uint32_t q_token_offset = q_offsets[batch_idx] + q_idx;
        const uint32_t kv_offset = kv_offsets[batch_idx];
        const uint32_t causal_end = kv_len - q_len + q_idx;
        float* scores = shmem;
        float* reduce = scores + 8192u;

        const float* qh = Q + ((size_t)q_token_offset * (size_t)q_heads + (size_t)qh_idx) * head_dim;

        float max_score_local = -1e30f;
        for (uint32_t t = 0; t <= causal_end; ++t) {
            const float* kh = K + ((size_t)(kv_offset + t) * (size_t)kv_heads + (size_t)kvh_idx) * head_dim;
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += qh[d] * kh[d];
            }
            reduce[tid] = dot;
            __syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const float score = reduce[0] * inv_sqrt;
                scores[t] = score;
                if (score > max_score_local) max_score_local = score;
            }
            __syncthreads();
        }

        if (tid == 0) reduce[0] = max_score_local;
        __syncthreads();
        const float max_score = reduce[0];

        float denom_part = 0.0f;
        for (uint32_t t = tid; t <= causal_end; t += blockDim.x) {
            denom_part += expf(scores[t] - max_score);
        }
        reduce[tid] = denom_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

        float* out_h = Out + ((size_t)q_token_offset * (size_t)q_heads + (size_t)qh_idx) * head_dim;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t t = 0; t <= causal_end; ++t) {
                const float* vh = V + ((size_t)(kv_offset + t) * (size_t)kv_heads + (size_t)kvh_idx) * head_dim;
                const float p = expf(scores[t] - max_score) / denom;
                acc += p * vh[d];
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
            reinterpret_cast<const __half*>(kv->d_k),
            reinterpret_cast<const __half*>(kv->d_v),
            kv->max_seq_len, kv->num_heads, kv->head_dim, seq_id,
            reinterpret_cast<const float*>(q_dev_f32), seq_len,
            reinterpret_cast<float*>(out_dev_f32)
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -4;
        err = cudaStreamSynchronize(ctx->decode_stream);
        if (err != cudaSuccess) return -5;
        return 0;
    }

    int m40llm_attention_last_token_f32_gqa(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        void* out_dev_f32) {
        int rc = m40llm_attention_last_token_f32_gqa_async(
            ctx, kv, seq_id, q_dev_f32, q_heads, seq_len, out_dev_f32);
        if (rc != 0) return rc;
        cudaError_t err = cudaStreamSynchronize(ctx->decode_stream);
        if (err != cudaSuccess) return -5;
        return 0;
    }

    int m40llm_attention_last_token_f32_gqa_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        void* out_dev_f32) {
        if (!ctx || !kv || !q_dev_f32 || !out_dev_f32) return -1;
        if (seq_id >= kv->max_batch_size) return -2;
        if (seq_len == 0 || seq_len > kv->max_seq_len) return -3;
        if (q_heads == 0 || kv->num_heads == 0 || q_heads % kv->num_heads != 0) return -6;

        const bool use_head64 = kv->head_dim == 64 && seq_len <= 8192;
        const bool use_head128 = kv->head_dim == 128 && seq_len <= 8192;
        if (use_head64 || use_head128) {
            const char* cache_experiment = getenv("M40LLM_CACHE_EXPERIMENT");
            const bool use_ldg_kv = cache_experiment && strcmp(cache_experiment, "ldg_kv") == 0;
            static int logged_head64 = 0;
            static int logged_head64_ldg = 0;
            const int blocks = (int)q_heads;
            const int threads = 128;
            const size_t shmem = ((size_t)seq_len + (size_t)threads) * sizeof(float);
            const char* log_env = getenv("M40LLM_ATTN_LOG");
            if (use_head64 && use_ldg_kv) {
                if (!logged_head64_ldg && log_env && strcmp(log_env, "1") == 0) {
                    fprintf(stderr, "[cuda] attention_gqa backend: head64 __ldg KV experiment\n");
                    logged_head64_ldg = 1;
                }
                attention_last_token_gqa_head64_ldg_kernel<<<blocks, threads, shmem, ctx->decode_stream>>>(
                    reinterpret_cast<const __half*>(kv->d_k),
                    reinterpret_cast<const __half*>(kv->d_v),
                    kv->max_seq_len, q_heads, kv->num_heads, seq_id,
                    reinterpret_cast<const float*>(q_dev_f32), seq_len,
                    reinterpret_cast<float*>(out_dev_f32)
                );
            } else if (use_head64) {
                if (!logged_head64 && log_env && strcmp(log_env, "1") == 0) {
                    fprintf(stderr, "[cuda] attention_gqa backend: head64 shared-score kernel\n");
                    logged_head64 = 1;
                }
                attention_last_token_gqa_head64_kernel<<<blocks, threads, shmem, ctx->decode_stream>>>(
                    reinterpret_cast<const __half*>(kv->d_k),
                    reinterpret_cast<const __half*>(kv->d_v),
                    kv->max_seq_len, q_heads, kv->num_heads, seq_id,
                    reinterpret_cast<const float*>(q_dev_f32), seq_len,
                    reinterpret_cast<float*>(out_dev_f32)
                );
            } else {
                static int logged_head128 = 0;
                if (!logged_head128 && log_env && strcmp(log_env, "1") == 0) {
                    fprintf(stderr, "[cuda] attention_gqa backend: head128 shared-score kernel\n");
                    logged_head128 = 1;
                }
                attention_last_token_gqa_head128_kernel<<<blocks, threads, shmem, ctx->decode_stream>>>(
                    reinterpret_cast<const __half*>(kv->d_k),
                    reinterpret_cast<const __half*>(kv->d_v),
                    kv->max_seq_len, q_heads, kv->num_heads, seq_id,
                    reinterpret_cast<const float*>(q_dev_f32), seq_len,
                    reinterpret_cast<float*>(out_dev_f32)
                );
            }
        } else {
            static int logged_fallback = 0;
            const char* log_env = getenv("M40LLM_ATTN_LOG");
            if (!logged_fallback && log_env && strcmp(log_env, "1") == 0) {
                fprintf(stderr, "[cuda] attention_gqa backend: generic fallback kernel\n");
                logged_fallback = 1;
            }
            const int blocks = (int)q_heads;
            const int threads = 1;
            attention_last_token_gqa_kernel<<<blocks, threads, 0, ctx->decode_stream>>>(
                reinterpret_cast<const __half*>(kv->d_k),
                reinterpret_cast<const __half*>(kv->d_v),
                kv->max_seq_len, q_heads, kv->num_heads, kv->head_dim, seq_id,
                reinterpret_cast<const float*>(q_dev_f32), seq_len,
                reinterpret_cast<float*>(out_dev_f32)
            );
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -4;
        return 0;
    }

    int m40llm_attention_last_token_f32_gqa_seq_len_dev_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        const uint32_t* seq_len_dev,
        void* out_dev_f32) {
        if (!ctx || !kv || !q_dev_f32 || !seq_len_dev || !out_dev_f32) return -1;
        if (seq_id >= kv->max_batch_size) return -2;
        if (q_heads == 0 || kv->num_heads == 0 || q_heads % kv->num_heads != 0) return -3;

        const int blocks = (int)q_heads;
        const int threads = 1;
        attention_last_token_gqa_seq_len_dev_kernel<<<blocks, threads, 0, ctx->decode_stream>>>(
            reinterpret_cast<const __half*>(kv->d_k),
            reinterpret_cast<const __half*>(kv->d_v),
            kv->max_seq_len, q_heads, kv->num_heads, kv->head_dim, seq_id,
            reinterpret_cast<const float*>(q_dev_f32), seq_len_dev,
            reinterpret_cast<float*>(out_dev_f32)
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -4;
        return 0;
    }

    int m40llm_attention_last_token_f32_gqa_dense_recent_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        uint32_t recent_window,
        void* out_dev_f32) {
        if (!ctx || !kv || !q_dev_f32 || !out_dev_f32) return -1;
        if (seq_id >= kv->max_batch_size) return -2;
        if (seq_len == 0 || seq_len > kv->max_seq_len) return -3;
        if (q_heads == 0 || kv->num_heads == 0 || q_heads % kv->num_heads != 0) return -4;
        if (kv->head_dim != 64) return -5;
        if (recent_window == 0) return -6;
        const uint32_t window_len = seq_len < recent_window ? seq_len : recent_window;
        const int blocks = (int)q_heads;
        const int threads = 256;
        const size_t shmem = ((size_t)window_len + (size_t)threads) * sizeof(float);
        attention_last_token_gqa_dense_recent_head64_kernel<<<blocks, threads, shmem, ctx->decode_stream>>>(
            reinterpret_cast<const __half*>(kv->d_k),
            reinterpret_cast<const __half*>(kv->d_v),
            kv->max_seq_len,
            q_heads,
            kv->num_heads,
            seq_id,
            reinterpret_cast<const float*>(q_dev_f32),
            seq_len,
            recent_window,
            reinterpret_cast<float*>(out_dev_f32)
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -7;
        return 0;
    }

    int m40llm_attention_last_token_f32_gqa_block_select_exact_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        void* out_dev_f32) {
        if (!ctx || !kv || !q_dev_f32 || !out_dev_f32) return -1;
        if (seq_id >= kv->max_batch_size) return -2;
        if (seq_len == 0 || seq_len > kv->max_seq_len) return -3;
        if (q_heads == 0 || kv->num_heads == 0 || q_heads % kv->num_heads != 0) return -4;
        if (kv->head_dim != 64) return -5;
        if (recent_window == 0 || block_size == 0 || top_blocks == 0 || top_blocks > 64) return -6;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t selected_old_blocks = top_blocks < old_blocks ? top_blocks : old_blocks;
        const uint32_t recent_count = seq_len < recent_window ? seq_len : recent_window;
        const uint32_t selected_capacity =
            exact_block_selected_token_capacity(recent_count, old_len, selected_old_blocks, block_size);
        if (selected_capacity == 0 || selected_capacity > seq_len) return -7;
        const uint32_t selected_block_order = selected_block_order_from_env();
        const int blocks = (int)q_heads;
        const int threads = 128;
        const size_t shmem = ((size_t)selected_capacity * 2u + (size_t)threads) * sizeof(float);
        attention_last_token_gqa_block_select_exact_head64_kernel<<<blocks, threads, shmem, ctx->decode_stream>>>(
            reinterpret_cast<const __half*>(kv->d_k),
            reinterpret_cast<const __half*>(kv->d_v),
            kv->max_seq_len,
            q_heads,
            kv->num_heads,
            seq_id,
            reinterpret_cast<const float*>(q_dev_f32),
            seq_len,
            recent_window,
            block_size,
            selected_old_blocks,
            selected_block_order,
            reinterpret_cast<float*>(out_dev_f32));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -8;
        return 0;
    }

    int m40llm_attention_last_token_f32_gqa_block_select_exact_staged_with_buffers_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        void* staged_k_dev,
        void* staged_v_dev,
        void* staged_positions_dev,
        void* staged_counts_dev,
        uint32_t staged_capacity_tokens,
        void* out_dev_f32);

    int m40llm_kvcache_build_q8_old_from_dense(
        M40llmCudaContext* ctx,
        M40llmKVCache* kv,
        uint32_t seq_id,
        uint32_t seq_len,
        uint32_t recent_window) {
        if (!ctx || !kv) return -1;
        if (!kv->q8_old_backing || !kv->d_q8_old_k || !kv->d_q8_old_v ||
            !kv->d_q8_old_k_scale || !kv->d_q8_old_v_scale) return -2;
        if (kv->compressed) return -3;
        if (kv->head_dim != 64 || kv->num_heads == 0) return -4;
        if (seq_id >= kv->max_batch_size) return -5;
        if (seq_len == 0 || seq_len > kv->max_seq_len || recent_window == 0) return -6;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        if (old_len == 0) return 0;
        const size_t vectors = (size_t)old_len * (size_t)kv->num_heads;
        const int threads = 128;
        const int blocks = (int)vectors;
        build_q8_old_from_dense_head64_kernel<<<blocks, threads, 0, ctx->decode_stream>>>(
            kv->d_k,
            kv->d_v,
            kv->d_q8_old_k,
            kv->d_q8_old_v,
            kv->d_q8_old_k_scale,
            kv->d_q8_old_v_scale,
            kv->max_seq_len,
            kv->num_heads,
            seq_id,
            seq_len,
            recent_window);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -7;
        return 0;
    }

    int m40llm_attention_last_token_f32_gqa_block_select_exact_staged_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        void* out_dev_f32) {
        if (!ctx || !kv || !q_dev_f32 || !out_dev_f32) return -1;
        if (seq_id >= kv->max_batch_size) return -2;
        if (seq_len == 0 || seq_len > kv->max_seq_len) return -3;
        if (q_heads == 0 || kv->num_heads == 0 || q_heads % kv->num_heads != 0) return -4;
        if (kv->head_dim != 64) return -5;
        if (recent_window == 0 || block_size == 0 || top_blocks == 0 || top_blocks > 64) return -6;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t selected_old_blocks = top_blocks < old_blocks ? top_blocks : old_blocks;
        const uint32_t recent_count = seq_len < recent_window ? seq_len : recent_window;
        const uint32_t selected_capacity =
            exact_block_selected_token_capacity(recent_count, old_len, selected_old_blocks, block_size);
        if (selected_capacity == 0 || selected_capacity > seq_len) return -7;

        const size_t staged_elems = (size_t)q_heads * (size_t)selected_capacity * 64u;
        const size_t positions_elems = (size_t)q_heads * (size_t)selected_capacity;
        __half* staged_k = nullptr;
        __half* staged_v = nullptr;
        uint32_t* staged_positions = nullptr;
        uint32_t* staged_counts = nullptr;
        cudaError_t err = cudaMalloc(&staged_k, staged_elems * sizeof(__half));
        if (err != cudaSuccess) return -8;
        err = cudaMalloc(&staged_v, staged_elems * sizeof(__half));
        if (err != cudaSuccess) { cudaFree(staged_k); return -9; }
        err = cudaMalloc(&staged_positions, positions_elems * sizeof(uint32_t));
        if (err != cudaSuccess) { cudaFree(staged_k); cudaFree(staged_v); return -10; }
        err = cudaMalloc(&staged_counts, (size_t)q_heads * sizeof(uint32_t));
        if (err != cudaSuccess) {
            cudaFree(staged_k); cudaFree(staged_v); cudaFree(staged_positions); return -11;
        }

        const int rc = m40llm_attention_last_token_f32_gqa_block_select_exact_staged_with_buffers_async(
            ctx, kv, seq_id, q_dev_f32, q_heads, seq_len, recent_window, block_size, top_blocks,
            staged_k, staged_v, staged_positions, staged_counts, selected_capacity, out_dev_f32);
        cudaFree(staged_k);
        cudaFree(staged_v);
        cudaFree(staged_positions);
        cudaFree(staged_counts);
        return rc == 0 ? 0 : rc - 20;
    }

    int m40llm_attention_last_token_f32_gqa_block_select_exact_staged_with_buffers_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        void* staged_k_dev,
        void* staged_v_dev,
        void* staged_positions_dev,
        void* staged_counts_dev,
        uint32_t staged_capacity_tokens,
        void* out_dev_f32) {
        if (!ctx || !kv || !q_dev_f32 || !out_dev_f32) return -1;
        if (!staged_k_dev || !staged_v_dev || !staged_positions_dev || !staged_counts_dev) return -14;
        if (seq_id >= kv->max_batch_size) return -2;
        if (seq_len == 0 || seq_len > kv->max_seq_len) return -3;
        if (q_heads == 0 || kv->num_heads == 0 || q_heads % kv->num_heads != 0) return -4;
        if (kv->head_dim != 64) return -5;
        if (recent_window == 0 || block_size == 0 || top_blocks == 0 || top_blocks > 64) return -6;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t selected_old_blocks = top_blocks < old_blocks ? top_blocks : old_blocks;
        const uint32_t recent_count = seq_len < recent_window ? seq_len : recent_window;
        const uint32_t selected_capacity =
            exact_block_selected_token_capacity(recent_count, old_len, selected_old_blocks, block_size);
        if (selected_capacity == 0 || selected_capacity > seq_len) return -7;
        if (selected_capacity > staged_capacity_tokens) return -15;
        const uint32_t selected_block_order = selected_block_order_from_env();
        __half* staged_k = reinterpret_cast<__half*>(staged_k_dev);
        __half* staged_v = reinterpret_cast<__half*>(staged_v_dev);
        uint32_t* staged_positions = reinterpret_cast<uint32_t*>(staged_positions_dev);
        uint32_t* staged_counts = reinterpret_cast<uint32_t*>(staged_counts_dev);

        const int blocks = (int)q_heads;
        const int threads = 128;
        const size_t gather_shmem = ((size_t)selected_capacity + (size_t)threads) * sizeof(float);
        gather_block_select_exact_head64_kernel<<<blocks, threads, gather_shmem, ctx->decode_stream>>>(
            reinterpret_cast<const __half*>(kv->d_k),
            reinterpret_cast<const __half*>(kv->d_v),
            kv->max_seq_len,
            q_heads,
            kv->num_heads,
            seq_id,
            reinterpret_cast<const float*>(q_dev_f32),
            seq_len,
            recent_window,
            block_size,
            selected_old_blocks,
            selected_capacity,
            selected_block_order,
            staged_k,
            staged_v,
            staged_positions,
            staged_counts);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -12;

        const size_t attention_shmem = ((size_t)selected_capacity + (size_t)threads) * sizeof(float);
        attention_last_token_gqa_staged_exact_head64_kernel<<<blocks, threads, attention_shmem, ctx->decode_stream>>>(
            staged_k,
            staged_v,
            staged_counts,
            selected_capacity,
            q_heads,
            reinterpret_cast<const float*>(q_dev_f32),
            reinterpret_cast<float*>(out_dev_f32));
        err = cudaGetLastError();
        if (err != cudaSuccess) return -13;
        return 0;
    }

    int m40llm_attention_last_token_f32_gqa_block_select_exact_staged_q8_old_with_buffers_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        void* staged_k_dev,
        void* staged_v_dev,
        void* staged_positions_dev,
        void* staged_counts_dev,
        uint32_t staged_capacity_tokens,
        void* out_dev_f32) {
        if (!ctx || !kv || !q_dev_f32 || !out_dev_f32) return -1;
        if (!staged_k_dev || !staged_v_dev || !staged_positions_dev || !staged_counts_dev) return -14;
        if (!kv->q8_old_backing || !kv->d_q8_old_k || !kv->d_q8_old_v ||
            !kv->d_q8_old_k_scale || !kv->d_q8_old_v_scale) return -16;
        if (seq_id >= kv->max_batch_size) return -2;
        if (seq_len == 0 || seq_len > kv->max_seq_len) return -3;
        if (q_heads == 0 || kv->num_heads == 0 || q_heads % kv->num_heads != 0) return -4;
        if (kv->head_dim != 64) return -5;
        if (recent_window == 0 || block_size == 0 || top_blocks == 0 || top_blocks > 64) return -6;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t selected_old_blocks = top_blocks < old_blocks ? top_blocks : old_blocks;
        const uint32_t recent_count = seq_len < recent_window ? seq_len : recent_window;
        const uint32_t selected_capacity =
            exact_block_selected_token_capacity(recent_count, old_len, selected_old_blocks, block_size);
        if (selected_capacity == 0 || selected_capacity > seq_len) return -7;
        if (selected_capacity > staged_capacity_tokens) return -15;
        const uint32_t selected_block_order = selected_block_order_from_env();
        const uint32_t old_k_source = q8_old_k_source_from_env();
        const uint32_t old_v_source = q8_old_v_source_from_env();
        if ((old_k_source == 1u || old_v_source == 1u) && (!kv->d_k || !kv->d_v)) return -17;
        if (old_v_source == 2u && (!kv->q4_old_v_backing || !kv->d_q4_old_v || !kv->d_q4_old_v_scale)) return -18;
        __half* staged_k = reinterpret_cast<__half*>(staged_k_dev);
        __half* staged_v = reinterpret_cast<__half*>(staged_v_dev);
        uint32_t* staged_positions = reinterpret_cast<uint32_t*>(staged_positions_dev);
        uint32_t* staged_counts = reinterpret_cast<uint32_t*>(staged_counts_dev);

        const int blocks = (int)q_heads;
        const int threads = 128;
        const size_t gather_shmem = ((size_t)selected_capacity + (size_t)threads) * sizeof(float);
        gather_block_select_exact_q8_old_head64_kernel<<<blocks, threads, gather_shmem, ctx->decode_stream>>>(
            reinterpret_cast<const __half*>(kv->d_k),
            reinterpret_cast<const __half*>(kv->d_v),
            kv->compressed ? kv->d_recent_k : nullptr,
            kv->compressed ? kv->d_recent_v : nullptr,
            kv->d_q8_old_k,
            kv->d_q8_old_v,
            kv->d_q8_old_k_scale,
            kv->d_q8_old_v_scale,
            kv->d_q4_old_v,
            kv->d_q4_old_v_scale,
            kv->max_seq_len,
            q_heads,
            kv->num_heads,
            seq_id,
            reinterpret_cast<const float*>(q_dev_f32),
            seq_len,
            recent_window,
            block_size,
            selected_old_blocks,
            selected_capacity,
            selected_block_order,
            old_k_source,
            old_v_source,
            staged_k,
            staged_v,
            staged_positions,
            staged_counts);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -12;

        const size_t attention_shmem = ((size_t)selected_capacity + (size_t)threads) * sizeof(float);
        attention_last_token_gqa_staged_exact_head64_kernel<<<blocks, threads, attention_shmem, ctx->decode_stream>>>(
            staged_k,
            staged_v,
            staged_counts,
            selected_capacity,
            q_heads,
            reinterpret_cast<const float*>(q_dev_f32),
            reinterpret_cast<float*>(out_dev_f32));
        err = cudaGetLastError();
        if (err != cudaSuccess) return -13;
        return 0;
    }

    int m40llm_attention_last_token_f32_gqa_block_select_exact_staged_fp16_k_q4_v_old_with_buffers_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        void* staged_k_dev,
        void* staged_v_dev,
        void* staged_positions_dev,
        void* staged_counts_dev,
        uint32_t staged_capacity_tokens,
        void* out_dev_f32) {
        if (!ctx || !kv || !q_dev_f32 || !out_dev_f32) return -1;
        if (!staged_k_dev || !staged_v_dev || !staged_positions_dev || !staged_counts_dev) return -14;
        if (!kv->fp16_k_q4_v_old_backing || !kv->d_fp16_old_k ||
            !kv->q4_old_v_backing || !kv->d_q4_old_v || !kv->d_q4_old_v_scale) return -16;
        if (seq_id >= kv->max_batch_size) return -2;
        if (seq_len == 0 || seq_len > kv->max_seq_len) return -3;
        if (q_heads == 0 || kv->num_heads == 0 || q_heads % kv->num_heads != 0) return -4;
        if (kv->head_dim != 64) return -5;
        if (recent_window == 0 || block_size == 0 || top_blocks == 0 || top_blocks > 64) return -6;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t selected_old_blocks = top_blocks < old_blocks ? top_blocks : old_blocks;
        const uint32_t recent_count = seq_len < recent_window ? seq_len : recent_window;
        const uint32_t selected_capacity =
            exact_block_selected_token_capacity(recent_count, old_len, selected_old_blocks, block_size);
        if (selected_capacity == 0 || selected_capacity > seq_len) return -7;
        if (selected_capacity > staged_capacity_tokens) return -15;

        __half* staged_k = reinterpret_cast<__half*>(staged_k_dev);
        __half* staged_v = reinterpret_cast<__half*>(staged_v_dev);
        uint32_t* staged_positions = reinterpret_cast<uint32_t*>(staged_positions_dev);
        uint32_t* staged_counts = reinterpret_cast<uint32_t*>(staged_counts_dev);
        const int blocks = (int)q_heads;
        const int threads = 128;
        const size_t gather_shmem = ((size_t)selected_capacity + (size_t)threads) * sizeof(float);
        gather_block_select_exact_q8_old_head64_kernel<<<blocks, threads, gather_shmem, ctx->decode_stream>>>(
            kv->d_fp16_old_k,
            nullptr,
            kv->compressed ? kv->d_recent_k : nullptr,
            kv->compressed ? kv->d_recent_v : nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            kv->d_q4_old_v,
            kv->d_q4_old_v_scale,
            kv->max_seq_len,
            q_heads,
            kv->num_heads,
            seq_id,
            reinterpret_cast<const float*>(q_dev_f32),
            seq_len,
            recent_window,
            block_size,
            selected_old_blocks,
            selected_capacity,
            selected_block_order_from_env(),
            1u,
            2u,
            staged_k,
            staged_v,
            staged_positions,
            staged_counts);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -12;

        const size_t attention_shmem = ((size_t)selected_capacity + (size_t)threads) * sizeof(float);
        attention_last_token_gqa_staged_exact_head64_kernel<<<blocks, threads, attention_shmem, ctx->decode_stream>>>(
            staged_k,
            staged_v,
            staged_counts,
            selected_capacity,
            q_heads,
            reinterpret_cast<const float*>(q_dev_f32),
            reinterpret_cast<float*>(out_dev_f32));
        err = cudaGetLastError();
        if (err != cudaSuccess) return -13;
        return 0;
    }

    int m40llm_attention_last_token_f32_gqa_block_select_exact_q8_old_direct_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        void* out_dev_f32) {
        if (!ctx || !kv || !q_dev_f32 || !out_dev_f32) return -1;
        if (!kv->q8_old_backing || !kv->d_q8_old_k || !kv->d_q8_old_v ||
            !kv->d_q8_old_k_scale || !kv->d_q8_old_v_scale) return -16;
        if (seq_id >= kv->max_batch_size) return -2;
        if (seq_len == 0 || seq_len > kv->max_seq_len) return -3;
        if (q_heads == 0 || kv->num_heads == 0 || q_heads % kv->num_heads != 0) return -4;
        if (kv->head_dim != 64) return -5;
        if (recent_window == 0 || block_size == 0 || top_blocks == 0 || top_blocks > 64) return -6;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t block_select_policy = block_select_policy_from_env();
        const float block_score_delta = block_score_delta_from_env();
        const uint32_t block_min_blocks = block_min_blocks_from_env();
        const uint32_t block_max_blocks = block_max_blocks_from_env(64u);
        const uint64_t anchor_block_mask = anchor_blocks_from_env();
        uint64_t force_include_low = 0ull, force_include_high = 0ull;
        uint64_t force_exclude_low = 0ull, force_exclude_high = 0ull;
        block_masks_from_env("M40LLM_KV_FORCE_INCLUDE_BLOCKS", &force_include_low, &force_include_high);
        block_masks_from_env("M40LLM_KV_FORCE_EXCLUDE_BLOCKS", &force_exclude_low, &force_exclude_high);
        const uint32_t policy_capacity = block_select_policy == 0u ? top_blocks : block_max_blocks;
        const uint32_t selected_old_blocks = policy_capacity < old_blocks ? policy_capacity : old_blocks;
        const uint32_t recent_count = seq_len < recent_window ? seq_len : recent_window;
        const uint32_t selected_capacity =
            exact_block_selected_token_capacity(recent_count, old_len, selected_old_blocks, block_size);
        if (selected_capacity == 0 || selected_capacity > seq_len) return -7;
        const uint32_t selected_block_order = selected_block_order_from_env();

        const int blocks = (int)q_heads;
        const int threads = 128;
        const size_t shmem = ((size_t)selected_capacity * 2u + (size_t)threads) * sizeof(float);
        attention_last_token_gqa_block_select_exact_q8_direct_head64_kernel<<<blocks, threads, shmem, ctx->decode_stream>>>(
            reinterpret_cast<const __half*>(kv->d_k),
            reinterpret_cast<const __half*>(kv->d_v),
            kv->compressed ? kv->d_recent_k : nullptr,
            kv->compressed ? kv->d_recent_v : nullptr,
            kv->d_q8_old_k,
            kv->d_q8_old_v,
            kv->d_q8_old_k_scale,
            kv->d_q8_old_v_scale,
            kv->max_seq_len,
            q_heads,
            kv->num_heads,
            seq_id,
            reinterpret_cast<const float*>(q_dev_f32),
            seq_len,
            recent_window,
            block_size,
            top_blocks,
            selected_capacity,
            selected_block_order,
            block_select_policy,
            block_score_delta,
            block_min_blocks,
            block_max_blocks,
            anchor_block_mask,
            force_include_low,
            force_include_high,
            force_exclude_low,
            force_exclude_high,
            reinterpret_cast<float*>(out_dev_f32));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -8;
        return 0;
    }

    int m40llm_attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        void* out_dev_f32) {
        if (!ctx || !kv || !q_dev_f32 || !out_dev_f32) return -1;
        if (!kv->fp16_k_q4_v_old_backing || !kv->d_fp16_old_k ||
            !kv->q4_old_v_backing || !kv->d_q4_old_v || !kv->d_q4_old_v_scale ||
            !kv->d_recent_k || !kv->d_recent_v) return -16;
        if (seq_id >= kv->max_batch_size) return -2;
        if (seq_len == 0 || seq_len > kv->max_seq_len) return -3;
        if (q_heads == 0 || kv->num_heads == 0 || q_heads % kv->num_heads != 0) return -4;
        if (kv->head_dim != 64 && kv->head_dim != 128) return -5;
        if (recent_window == 0 || block_size == 0 || top_blocks == 0 || top_blocks > 64) return -6;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t block_select_policy = block_select_policy_from_env();
        const float block_score_delta = block_score_delta_from_env();
        const uint32_t block_min_blocks = block_min_blocks_from_env();
        const uint32_t block_max_blocks = block_max_blocks_from_env(64u);
        const uint64_t anchor_block_mask = anchor_blocks_from_env();
        uint64_t force_include_low = 0ull, force_include_high = 0ull;
        uint64_t force_exclude_low = 0ull, force_exclude_high = 0ull;
        block_masks_from_env("M40LLM_KV_FORCE_INCLUDE_BLOCKS", &force_include_low, &force_include_high);
        block_masks_from_env("M40LLM_KV_FORCE_EXCLUDE_BLOCKS", &force_exclude_low, &force_exclude_high);
        const uint32_t policy_capacity = block_select_policy == 0u ? top_blocks : block_max_blocks;
        const uint32_t selected_old_blocks = policy_capacity < old_blocks ? policy_capacity : old_blocks;
        const uint32_t recent_count = seq_len < recent_window ? seq_len : recent_window;
        const uint32_t selected_capacity =
            exact_block_selected_token_capacity(recent_count, old_len, selected_old_blocks, block_size);
        if (selected_capacity == 0 || selected_capacity > seq_len) return -7;
        const uint32_t selected_block_order = selected_block_order_from_env();

        const int blocks = (int)q_heads;
        const int threads = 128;
        const size_t shmem = ((size_t)selected_capacity * 2u + (size_t)threads) * sizeof(float);
        attention_last_token_gqa_block_select_exact_fp16_k_q4_v_direct_kernel<<<blocks, threads, shmem, ctx->decode_stream>>>(
            kv->d_fp16_old_k,
            kv->d_recent_k,
            kv->d_recent_v,
            kv->d_q4_old_v,
            kv->d_q4_old_v_scale,
            kv->max_seq_len,
            q_heads,
            kv->num_heads,
            kv->head_dim,
            seq_id,
            reinterpret_cast<const float*>(q_dev_f32),
            seq_len,
            recent_window,
            block_size,
            top_blocks,
            selected_capacity,
            selected_block_order,
            block_select_policy,
            block_score_delta,
            block_min_blocks,
            block_max_blocks,
            anchor_block_mask,
            force_include_low,
            force_include_high,
            force_exclude_low,
            force_exclude_high,
            reinterpret_cast<float*>(out_dev_f32));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -8;
        return 0;
    }

    int m40llm_attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct_batched_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        const uint32_t* seq_ids_dev,
        const uint32_t* seq_lens_dev,
        uint32_t batch_size,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        void* out_dev_f32) {
        if (!ctx || !kv || !seq_ids_dev || !seq_lens_dev || !q_dev_f32 || !out_dev_f32) return -1;
        if (!kv->fp16_k_q4_v_old_backing || !kv->d_fp16_old_k ||
            !kv->q4_old_v_backing || !kv->d_q4_old_v || !kv->d_q4_old_v_scale ||
            !kv->d_recent_k || !kv->d_recent_v) return -16;
        if (batch_size == 0) return -2;
        if (q_heads == 0 || kv->num_heads == 0 || q_heads % kv->num_heads != 0) return -4;
        if (kv->head_dim != 64 && kv->head_dim != 128) return -5;
        if (recent_window == 0 || block_size == 0 || top_blocks == 0 || top_blocks > 64) return -6;

        uint32_t* seq_lens_host = new uint32_t[batch_size];
        uint32_t* seq_ids_host = new uint32_t[batch_size];
        cudaError_t err = cudaMemcpy(seq_lens_host, seq_lens_dev, (size_t)batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            delete[] seq_lens_host;
            delete[] seq_ids_host;
            return -7;
        }
        err = cudaMemcpy(seq_ids_host, seq_ids_dev, (size_t)batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            delete[] seq_lens_host;
            delete[] seq_ids_host;
            return -7;
        }

        const uint32_t block_select_policy = block_select_policy_from_env();
        const float block_score_delta = block_score_delta_from_env();
        const uint32_t block_min_blocks = block_min_blocks_from_env();
        const uint32_t block_max_blocks = block_max_blocks_from_env(64u);
        const uint64_t anchor_block_mask = anchor_blocks_from_env();
        uint64_t force_include_low = 0ull, force_include_high = 0ull;
        uint64_t force_exclude_low = 0ull, force_exclude_high = 0ull;
        block_masks_from_env("M40LLM_KV_FORCE_INCLUDE_BLOCKS", &force_include_low, &force_include_high);
        block_masks_from_env("M40LLM_KV_FORCE_EXCLUDE_BLOCKS", &force_exclude_low, &force_exclude_high);
        const uint32_t selected_block_order = selected_block_order_from_env();

        uint32_t max_selected_capacity = 0;
        for (uint32_t i = 0; i < batch_size; ++i) {
            const uint32_t seq_id = seq_ids_host[i];
            const uint32_t seq_len = seq_lens_host[i];
            if (seq_id >= kv->max_batch_size || seq_len == 0 || seq_len > kv->max_seq_len) {
                delete[] seq_lens_host;
                delete[] seq_ids_host;
                return -8;
            }
            const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
            const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
            const uint32_t policy_capacity = block_select_policy == 0u ? top_blocks : block_max_blocks;
            const uint32_t selected_old_blocks = policy_capacity < old_blocks ? policy_capacity : old_blocks;
            const uint32_t recent_count = seq_len < recent_window ? seq_len : recent_window;
            const uint32_t selected_capacity =
                exact_block_selected_token_capacity(recent_count, old_len, selected_old_blocks, block_size);
            if (selected_capacity == 0 || selected_capacity > seq_len) {
                delete[] seq_lens_host;
                delete[] seq_ids_host;
                return -9;
            }
            if (selected_capacity > max_selected_capacity) max_selected_capacity = selected_capacity;
        }
        delete[] seq_lens_host;
        delete[] seq_ids_host;
        if (max_selected_capacity == 0) return -9;

        dim3 grid((int)q_heads, (int)batch_size, 1);
        const int threads = 128;
        const size_t shmem = ((size_t)max_selected_capacity * 2u + (size_t)threads) * sizeof(float);
        attention_last_token_gqa_block_select_exact_fp16_k_q4_v_direct_batched_kernel<<<grid, threads, shmem, ctx->decode_stream>>>(
            kv->d_fp16_old_k,
            kv->d_recent_k,
            kv->d_recent_v,
            kv->d_q4_old_v,
            kv->d_q4_old_v_scale,
            kv->max_seq_len,
            q_heads,
            kv->num_heads,
            kv->head_dim,
            seq_ids_dev,
            seq_lens_dev,
            batch_size,
            reinterpret_cast<const float*>(q_dev_f32),
            recent_window,
            block_size,
            top_blocks,
            max_selected_capacity,
            selected_block_order,
            block_select_policy,
            block_score_delta,
            block_min_blocks,
            block_max_blocks,
            anchor_block_mask,
            force_include_low,
            force_include_high,
            force_exclude_low,
            force_exclude_high,
            reinterpret_cast<float*>(out_dev_f32));
        err = cudaGetLastError();
        if (err != cudaSuccess) return -10;
        return 0;
    }

    int m40llm_attention_last_token_f32_gqa_block_summary_lossy_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t top_blocks,
        void* out_dev_f32) {
        if (!ctx || !kv || !q_dev_f32 || !out_dev_f32) return -1;
        if (seq_id >= kv->max_batch_size) return -2;
        if (seq_len == 0 || seq_len > kv->max_seq_len) return -3;
        if (q_heads == 0 || kv->num_heads == 0 || q_heads % kv->num_heads != 0) return -4;
        if (kv->head_dim != 64) return -5;
        if (recent_window == 0 || block_size == 0 || top_blocks > 64) return -6;
        if (kv->compressed) {
            const uint32_t old_len = seq_len > kv->recent_window ? seq_len - kv->recent_window : 0;
            const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + kv->block_size - 1) / kv->block_size;
            const uint32_t selected_summary_count =
                top_blocks == 0 || top_blocks > old_blocks ? old_blocks : top_blocks;
            const uint32_t recent_count = seq_len < kv->recent_window ? seq_len : kv->recent_window;
            const uint32_t selected_capacity =
                recent_count + selected_summary_count * (1u + kv->representatives);
            if (selected_capacity == 0) return -7;
            const int blocks = (int)q_heads;
            const int threads = 128;
            const size_t shmem = ((size_t)selected_capacity * 2u + (size_t)threads) * sizeof(float);
            attention_last_token_gqa_compressed_summary_head64_kernel<<<blocks, threads, shmem, ctx->decode_stream>>>(
                kv->d_recent_k,
                kv->d_recent_v,
                kv->d_summary_k,
                kv->d_summary_v,
                kv->d_rep_k,
                kv->d_rep_v,
                kv->d_rep_positions,
                kv->d_block_counts,
                kv->recent_window,
                kv->block_size,
                kv->max_blocks,
                kv->representatives,
                q_heads,
                kv->num_heads,
                seq_id,
                reinterpret_cast<const float*>(q_dev_f32),
                seq_len,
                top_blocks,
                reinterpret_cast<float*>(out_dev_f32));
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) return -8;
            return 0;
        }
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + block_size - 1) / block_size;
        const uint32_t selected_summary_count =
            top_blocks == 0 || top_blocks > old_blocks ? old_blocks : top_blocks;
        const uint32_t recent_count = seq_len < recent_window ? seq_len : recent_window;
        const uint32_t selected_capacity = recent_count + selected_summary_count;
        if (selected_capacity == 0 || selected_capacity > seq_len) return -7;
        const int blocks = (int)q_heads;
        const int threads = 128;
        const size_t shmem = ((size_t)selected_capacity * 2u + (size_t)threads) * sizeof(float);
        attention_last_token_gqa_block_summary_lossy_head64_kernel<<<blocks, threads, shmem, ctx->decode_stream>>>(
            reinterpret_cast<const __half*>(kv->d_k),
            reinterpret_cast<const __half*>(kv->d_v),
            kv->max_seq_len,
            q_heads,
            kv->num_heads,
            seq_id,
            reinterpret_cast<const float*>(q_dev_f32),
            seq_len,
            recent_window,
            block_size,
            top_blocks,
            reinterpret_cast<float*>(out_dev_f32));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -8;
        return 0;
    }

    int m40llm_attention_last_token_f32_gqa_compressed_recent_only_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        uint32_t seq_id,
        const void* q_dev_f32,
        uint32_t q_heads,
        uint32_t seq_len,
        void* out_dev_f32) {
        if (!ctx || !kv || !q_dev_f32 || !out_dev_f32) return -1;
        if (!kv->compressed) return -2;
        if (seq_id >= kv->max_batch_size) return -3;
        if (seq_len == 0 || seq_len > kv->max_seq_len) return -4;
        if (q_heads == 0 || kv->num_heads == 0 || q_heads % kv->num_heads != 0) return -5;
        if (kv->head_dim != 64 || kv->recent_window == 0) return -6;
        const uint32_t recent_count = seq_len < kv->recent_window ? seq_len : kv->recent_window;
        if (recent_count == 0) return -7;
        const int blocks = (int)q_heads;
        const int threads = 128;
        const size_t shmem = ((size_t)kv->recent_window + (size_t)threads) * sizeof(float);
        attention_last_token_gqa_compressed_recent_only_head64_kernel<<<blocks, threads, shmem, ctx->decode_stream>>>(
            kv->d_recent_k,
            kv->d_recent_v,
            kv->recent_window,
            q_heads,
            kv->num_heads,
            seq_id,
            reinterpret_cast<const float*>(q_dev_f32),
            seq_len,
            reinterpret_cast<float*>(out_dev_f32));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -8;
        return 0;
    }

    static int attention_last_token_f32_gqa_batched_impl(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        const uint32_t* seq_ids_dev,
        const uint32_t* seq_lens_dev,
        uint32_t batch_size,
        const void* q_dev_f32,
        uint32_t q_heads,
        void* out_dev_f32,
        bool synchronize) {
        if (!ctx || !kv || !seq_ids_dev || !seq_lens_dev || !q_dev_f32 || !out_dev_f32) return -1;
        if (batch_size == 0) return -2;
        if (q_heads == 0 || kv->num_heads == 0 || q_heads % kv->num_heads != 0) return -3;
        if (kv->head_dim != 64 && kv->head_dim != 128) return -6;

        const char* cache_experiment = getenv("M40LLM_CACHE_EXPERIMENT");
        const bool use_ldg_kv = cache_experiment && strcmp(cache_experiment, "ldg_kv") == 0;
        static int logged = 0;
        static int logged_ldg = 0;
        static int logged_head128 = 0;
        const char* log_env = getenv("M40LLM_ATTN_LOG");
        if (use_ldg_kv && !logged_ldg && log_env && strcmp(log_env, "1") == 0) {
            fprintf(stderr, "[cuda] attention_gqa_batched backend: variable-length head64 __ldg KV experiment\n");
            logged_ldg = 1;
        } else if (kv->head_dim == 64 && !use_ldg_kv && !logged && log_env && strcmp(log_env, "1") == 0) {
            fprintf(stderr, "[cuda] attention_gqa_batched backend: variable-length head64 packed-q kernel\n");
            logged = 1;
        } else if (kv->head_dim == 128 && !logged_head128 && log_env && strcmp(log_env, "1") == 0) {
            fprintf(stderr, "[cuda] attention_gqa_batched backend: variable-length head128 packed-q kernel\n");
            logged_head128 = 1;
        }
        const int threads = 128;
        uint32_t max_seq_len_host = 0;
        uint32_t* seq_lens_host = new uint32_t[batch_size];
        uint32_t* seq_ids_host = new uint32_t[batch_size];
        cudaError_t err = cudaMemcpy(seq_lens_host, seq_lens_dev, (size_t)batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            delete[] seq_lens_host;
            delete[] seq_ids_host;
            return -7;
        }
        err = cudaMemcpy(seq_ids_host, seq_ids_dev, (size_t)batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            delete[] seq_lens_host;
            delete[] seq_ids_host;
            return -7;
        }
        for (uint32_t i = 0; i < batch_size; ++i) {
            if (seq_ids_host[i] >= kv->max_batch_size) {
                delete[] seq_lens_host;
                delete[] seq_ids_host;
                return -8;
            }
            if (seq_lens_host[i] == 0 || seq_lens_host[i] > kv->max_seq_len || seq_lens_host[i] > 8192) {
                delete[] seq_lens_host;
                delete[] seq_ids_host;
                return -8;
            }
            if (seq_lens_host[i] > max_seq_len_host) {
                max_seq_len_host = seq_lens_host[i];
            }
        }
        delete[] seq_lens_host;
        delete[] seq_ids_host;
        if (max_seq_len_host == 0 || max_seq_len_host > kv->max_seq_len || max_seq_len_host > 8192) return -8;

        dim3 grid((int)q_heads, (int)batch_size, 1);
        const size_t shmem = ((size_t)max_seq_len_host + (size_t)threads) * sizeof(float);
        if (use_ldg_kv && kv->head_dim == 64) {
            attention_last_token_gqa_batched_head64_ldg_kernel<<<grid, threads, shmem, ctx->decode_stream>>>(
                reinterpret_cast<const __half*>(kv->d_k),
                reinterpret_cast<const __half*>(kv->d_v),
                kv->max_seq_len,
                q_heads,
                kv->num_heads,
                seq_ids_dev,
                seq_lens_dev,
                batch_size,
                reinterpret_cast<const float*>(q_dev_f32),
                reinterpret_cast<float*>(out_dev_f32));
        } else if (kv->head_dim == 64) {
            attention_last_token_gqa_batched_head64_kernel<<<grid, threads, shmem, ctx->decode_stream>>>(
                reinterpret_cast<const __half*>(kv->d_k),
                reinterpret_cast<const __half*>(kv->d_v),
                kv->max_seq_len,
                q_heads,
                kv->num_heads,
                seq_ids_dev,
                seq_lens_dev,
                batch_size,
                reinterpret_cast<const float*>(q_dev_f32),
                reinterpret_cast<float*>(out_dev_f32));
        } else {
            attention_last_token_gqa_batched_head128_kernel<<<grid, threads, shmem, ctx->decode_stream>>>(
                reinterpret_cast<const __half*>(kv->d_k),
                reinterpret_cast<const __half*>(kv->d_v),
                kv->max_seq_len,
                q_heads,
                kv->num_heads,
                seq_ids_dev,
                seq_lens_dev,
                batch_size,
                reinterpret_cast<const float*>(q_dev_f32),
                reinterpret_cast<float*>(out_dev_f32));
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) return -4;
        if (synchronize) {
            err = cudaStreamSynchronize(ctx->decode_stream);
            if (err != cudaSuccess) return -5;
        }
        return 0;
    }

    int m40llm_attention_last_token_f32_gqa_batched(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        const uint32_t* seq_ids_dev,
        const uint32_t* seq_lens_dev,
        uint32_t batch_size,
        const void* q_dev_f32,
        uint32_t q_heads,
        void* out_dev_f32) {
        return attention_last_token_f32_gqa_batched_impl(
            ctx,
            kv,
            seq_ids_dev,
            seq_lens_dev,
            batch_size,
            q_dev_f32,
            q_heads,
            out_dev_f32,
            true);
    }

    int m40llm_attention_last_token_f32_gqa_batched_async(
        M40llmCudaContext* ctx,
        const M40llmKVCache* kv,
        const uint32_t* seq_ids_dev,
        const uint32_t* seq_lens_dev,
        uint32_t batch_size,
        const void* q_dev_f32,
        uint32_t q_heads,
        void* out_dev_f32) {
        return attention_last_token_f32_gqa_batched_impl(
            ctx,
            kv,
            seq_ids_dev,
            seq_lens_dev,
            batch_size,
            q_dev_f32,
            q_heads,
            out_dev_f32,
            false);
    }

    static int attention_prefill_f32_gqa_varlen_impl(
        M40llmCudaContext* ctx,
        const void* q_dev_f32,
        const void* k_dev_f32,
        const void* v_dev_f32,
        const uint32_t* q_offsets_dev,
        const uint32_t* kv_offsets_dev,
        const uint32_t* q_lens_dev,
        const uint32_t* kv_lens_dev,
        uint32_t batch_size,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim,
        void* out_dev_f32,
        bool synchronize) {
        if (!ctx || !q_dev_f32 || !k_dev_f32 || !v_dev_f32 || !q_offsets_dev || !kv_offsets_dev || !q_lens_dev || !kv_lens_dev || !out_dev_f32) return -1;
        if (batch_size == 0) return -2;
        if (q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0) return -3;
        if (head_dim != 64 && head_dim != 128) return -9;

        static int logged = 0;
        const char* log_env = getenv("M40LLM_ATTN_LOG");
        if (!logged && log_env && strcmp(log_env, "1") == 0) {
            fprintf(stderr, "[cuda] attention_prefill_gqa_varlen backend: head%u packed-f32 kernel\n", head_dim);
            logged = 1;
        }

        const int threads = 128;
        uint32_t max_q_len_host = 0;
        uint32_t max_kv_len_host = 0;
        uint32_t* q_lens_host = new uint32_t[batch_size];
        uint32_t* kv_lens_host = new uint32_t[batch_size];
        cudaError_t err = cudaMemcpy(q_lens_host, q_lens_dev, (size_t)batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            delete[] q_lens_host;
            delete[] kv_lens_host;
            return -7;
        }
        err = cudaMemcpy(kv_lens_host, kv_lens_dev, (size_t)batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            delete[] q_lens_host;
            delete[] kv_lens_host;
            return -7;
        }
        for (uint32_t i = 0; i < batch_size; ++i) {
            const uint32_t q_len = q_lens_host[i];
            const uint32_t kv_len = kv_lens_host[i];
            if (q_len == 0 || kv_len == 0 || q_len > kv_len || kv_len > 8192) {
                delete[] q_lens_host;
                delete[] kv_lens_host;
                return -8;
            }
            if (q_len > max_q_len_host) max_q_len_host = q_len;
            if (kv_len > max_kv_len_host) max_kv_len_host = kv_len;
        }
        delete[] q_lens_host;
        delete[] kv_lens_host;
        if (max_q_len_host == 0 || max_kv_len_host == 0 || max_kv_len_host > 8192) return -8;

        dim3 grid((int)max_q_len_host, (int)q_heads, (int)batch_size);
        const size_t shmem = (8192u + (size_t)threads) * sizeof(float);
        attention_prefill_gqa_varlen_kernel<<<grid, threads, shmem, ctx->prefill_stream>>>(
            reinterpret_cast<const float*>(q_dev_f32),
            reinterpret_cast<const float*>(k_dev_f32),
            reinterpret_cast<const float*>(v_dev_f32),
            q_offsets_dev,
            kv_offsets_dev,
            q_lens_dev,
            kv_lens_dev,
            batch_size,
            q_heads,
            kv_heads,
            head_dim,
            reinterpret_cast<float*>(out_dev_f32));
        err = cudaGetLastError();
        if (err != cudaSuccess) return -4;
        if (synchronize) {
            err = cudaStreamSynchronize(ctx->prefill_stream);
            if (err != cudaSuccess) return -5;
        }
        return 0;
    }

    int m40llm_attention_prefill_f32_gqa_varlen(
        M40llmCudaContext* ctx,
        const void* q_dev_f32,
        const void* k_dev_f32,
        const void* v_dev_f32,
        const uint32_t* q_offsets_dev,
        const uint32_t* kv_offsets_dev,
        const uint32_t* q_lens_dev,
        const uint32_t* kv_lens_dev,
        uint32_t batch_size,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim,
        void* out_dev_f32) {
        return attention_prefill_f32_gqa_varlen_impl(
            ctx,
            q_dev_f32,
            k_dev_f32,
            v_dev_f32,
            q_offsets_dev,
            kv_offsets_dev,
            q_lens_dev,
            kv_lens_dev,
            batch_size,
            q_heads,
            kv_heads,
            head_dim,
            out_dev_f32,
            true);
    }

    int m40llm_attention_prefill_f32_gqa_varlen_async(
        M40llmCudaContext* ctx,
        const void* q_dev_f32,
        const void* k_dev_f32,
        const void* v_dev_f32,
        const uint32_t* q_offsets_dev,
        const uint32_t* kv_offsets_dev,
        const uint32_t* q_lens_dev,
        const uint32_t* kv_lens_dev,
        uint32_t batch_size,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim,
        void* out_dev_f32) {
        return attention_prefill_f32_gqa_varlen_impl(
            ctx,
            q_dev_f32,
            k_dev_f32,
            v_dev_f32,
            q_offsets_dev,
            kv_offsets_dev,
            q_lens_dev,
            kv_lens_dev,
            batch_size,
            q_heads,
            kv_heads,
            head_dim,
            out_dev_f32,
            false);
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

    // Mixed GEMM for GGUF F16 tensors. GGUF stores tensor dimension 0 as the
    // fastest-moving dimension, so a logical [K, N] weight is laid out as
    // B[col * K + kk], not row-major B[kk * N + col].
    __global__ void gemm_f32xf16_gguf_f32_kernel(
        const float* __restrict__ A,  // MxK row-major activations
        const __half* __restrict__ B, // GGUF [K,N], K-fastest
        float* __restrict__ C,        // MxN row-major output
        int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= M || col >= N) return;
        float acc = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
            float a = A[row * K + kk];
            float b = __half2float(B[col * K + kk]);
            acc += a * b;
        }
        C[row * N + col] = acc;
    }

    __global__ void gemm_f32xf16_gguf_f32_decode_kernel(
        const float* __restrict__ A,  // 1xK row-major activations
        const __half* __restrict__ B, // GGUF [K,N], K-fastest
        float* __restrict__ C,        // 1xN row-major output
        int N, int K) {
        const int col = blockIdx.x;
        const int tid = threadIdx.x;
        if (col >= N) return;
        float acc = 0.0f;
        const size_t col_base = static_cast<size_t>(col) * static_cast<size_t>(K);
        if ((K & 1) == 0) {
            for (int kk = tid * 2; kk < K; kk += blockDim.x * 2) {
                const __half2 b2 = *reinterpret_cast<const __half2*>(B + col_base + kk);
                const float2 bf = __half22float2(b2);
                acc += A[kk] * bf.x + A[kk + 1] * bf.y;
            }
        } else {
            for (int kk = tid; kk < K; kk += blockDim.x) {
                acc += A[kk] * __half2float(B[col_base + kk]);
            }
        }
        extern __shared__ float f16_decode_reduce[];
        f16_decode_reduce[tid] = acc;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                f16_decode_reduce[tid] += f16_decode_reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            C[col] = f16_decode_reduce[0];
        }
    }

    __global__ void mlp_gate_up_swiglu_f32xf16_gguf_decode_kernel(
        const float* __restrict__ A,       // 1xK row-major activations
        const __half* __restrict__ W_gate, // GGUF [K,H], K-fastest
        const __half* __restrict__ W_up,   // GGUF [K,H], K-fastest
        float* __restrict__ C,             // 1xH row-major SwiGLU output
        int H, int K) {
        const int col = blockIdx.x;
        const int tid = threadIdx.x;
        if (col >= H) return;
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        const size_t col_base = static_cast<size_t>(col) * static_cast<size_t>(K);
        if ((K & 1) == 0) {
            for (int kk = tid * 2; kk < K; kk += blockDim.x * 2) {
                const float a0 = A[kk];
                const float a1 = A[kk + 1];
                const __half2 gate_h2 = *reinterpret_cast<const __half2*>(W_gate + col_base + kk);
                const __half2 up_h2 = *reinterpret_cast<const __half2*>(W_up + col_base + kk);
                const float2 gate_f2 = __half22float2(gate_h2);
                const float2 up_f2 = __half22float2(up_h2);
                gate_acc += a0 * gate_f2.x + a1 * gate_f2.y;
                up_acc += a0 * up_f2.x + a1 * up_f2.y;
            }
        } else {
            for (int kk = tid; kk < K; kk += blockDim.x) {
                const float a = A[kk];
                gate_acc += a * __half2float(W_gate[col_base + kk]);
                up_acc += a * __half2float(W_up[col_base + kk]);
            }
        }
        extern __shared__ float mlp_gate_up_reduce[];
        float* gate_reduce = mlp_gate_up_reduce;
        float* up_reduce = mlp_gate_up_reduce + blockDim.x;
        gate_reduce[tid] = gate_acc;
        up_reduce[tid] = up_acc;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                gate_reduce[tid] += gate_reduce[tid + stride];
                up_reduce[tid] += up_reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            const float gate = gate_reduce[0];
            const float up = up_reduce[0];
            C[col] = gate * (1.0f / (1.0f + expf(-gate))) * up;
        }
    }

    __global__ void qkv_f32xf16_gguf_decode_kernel(
        const float* __restrict__ A,      // 1xK row-major activations
        const __half* __restrict__ Wq,    // GGUF [K,Nq], K-fastest
        const __half* __restrict__ Wk,    // GGUF [K,Nk], K-fastest
        const __half* __restrict__ Wv,    // GGUF [K,Nv], K-fastest
        const float* __restrict__ bq,
        const float* __restrict__ bk,
        const float* __restrict__ bv,
        float* __restrict__ Q,
        float* __restrict__ Kout,
        float* __restrict__ Vout,
        int Nq, int Nk, int Nv, int Kdim) {
        const int out_col = blockIdx.x;
        const int tid = threadIdx.x;
        const int total = Nq + Nk + Nv;
        if (out_col >= total) return;

        const __half* W = Wq;
        const float* bias = bq;
        float* out = Q;
        int local_col = out_col;
        if (out_col >= Nq + Nk) {
            W = Wv;
            bias = bv;
            out = Vout;
            local_col = out_col - Nq - Nk;
        } else if (out_col >= Nq) {
            W = Wk;
            bias = bk;
            out = Kout;
            local_col = out_col - Nq;
        }

        float acc = 0.0f;
        const size_t col_base =
            static_cast<size_t>(local_col) * static_cast<size_t>(Kdim);
        if ((Kdim & 1) == 0) {
            for (int kk = tid * 2; kk < Kdim; kk += blockDim.x * 2) {
                const __half2 w2 = *reinterpret_cast<const __half2*>(W + col_base + kk);
                const float2 wf = __half22float2(w2);
                acc += A[kk] * wf.x + A[kk + 1] * wf.y;
            }
        } else {
            for (int kk = tid; kk < Kdim; kk += blockDim.x) {
                acc += A[kk] * __half2float(W[col_base + kk]);
            }
        }
        extern __shared__ float qkv_decode_reduce[];
        qkv_decode_reduce[tid] = acc;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                qkv_decode_reduce[tid] += qkv_decode_reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            float value = qkv_decode_reduce[0];
            if (bias) {
                value += bias[local_col];
            }
            out[local_col] = value;
        }
    }

    __global__ void gemm_f32xq8_0_gguf_f32_kernel(
        const float* __restrict__ A,           // MxK row-major activations
        const unsigned char* __restrict__ B,   // GGUF Q8_0 [K,N], K-fastest
        float* __restrict__ C,                 // MxN row-major output
        int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= M || col >= N) return;
        const int qk = 32;
        const int block_bytes = 34;
        const int blocks_per_col = (K + qk - 1) / qk;
        float acc = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
            const int block_idx = kk / qk;
            const int q_idx = kk - block_idx * qk;
            const size_t block_base =
                (static_cast<size_t>(col) * blocks_per_col + block_idx) * block_bytes;
            const __half scale_h = *reinterpret_cast<const __half*>(B + block_base);
            const float scale = __half2float(scale_h);
            const int8_t q = *reinterpret_cast<const int8_t*>(B + block_base + 2 + q_idx);
            acc += A[row * K + kk] * (static_cast<float>(q) * scale);
        }
        C[row * N + col] = acc;
    }

    __global__ void gemm_f32xq8_0_gguf_f32_blockloop_kernel(
        const float* __restrict__ A,           // MxK row-major activations
        const unsigned char* __restrict__ B,   // GGUF Q8_0 [K,N], K-fastest
        float* __restrict__ C,                 // MxN row-major output
        int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= M || col >= N) return;
        const int qk = 32;
        const int block_bytes = 34;
        const int blocks_per_col = (K + qk - 1) / qk;
        float acc = 0.0f;
        for (int block_idx = 0; block_idx < blocks_per_col; ++block_idx) {
            const int k_base = block_idx * qk;
            const int valid = min(qk, K - k_base);
            const size_t block_base =
                (static_cast<size_t>(col) * blocks_per_col + block_idx) * block_bytes;
            const __half scale_h = *reinterpret_cast<const __half*>(B + block_base);
            const float scale = __half2float(scale_h);
            #pragma unroll
            for (int q_idx = 0; q_idx < qk; ++q_idx) {
                if (q_idx < valid) {
                    const int8_t q = *reinterpret_cast<const int8_t*>(B + block_base + 2 + q_idx);
                    acc += A[row * K + k_base + q_idx] * (static_cast<float>(q) * scale);
                }
            }
        }
        C[row * N + col] = acc;
    }

    __global__ void gemm_f32xq8_0_gguf_f32_shared_activation_kernel(
        const float* __restrict__ A,           // MxK row-major activations
        const unsigned char* __restrict__ B,   // GGUF Q8_0 [K,N], K-fastest
        float* __restrict__ C,                 // MxN row-major output
        int M, int N, int K) {
        const int tile_rows = 8;
        const int tile_cols = 16;
        const int qk = 32;
        const int block_bytes = 34;
        const int row = blockIdx.y * tile_rows + threadIdx.y;
        const int col = blockIdx.x * tile_cols + threadIdx.x;
        const int tid = threadIdx.y * tile_cols + threadIdx.x;
        const int blocks_per_col = (K + qk - 1) / qk;
        __shared__ float a_tile[tile_rows][qk];
        float acc = 0.0f;
        for (int block_idx = 0; block_idx < blocks_per_col; ++block_idx) {
            const int k_base = block_idx * qk;
            for (int linear = tid; linear < tile_rows * qk; linear += tile_rows * tile_cols) {
                const int local_row = linear / qk;
                const int q_idx = linear - local_row * qk;
                const int global_row = blockIdx.y * tile_rows + local_row;
                const int k_idx = k_base + q_idx;
                a_tile[local_row][q_idx] =
                    (global_row < M && k_idx < K) ? A[global_row * K + k_idx] : 0.0f;
            }
            __syncthreads();
            if (row < M && col < N) {
                const int valid = min(qk, K - k_base);
                const size_t block_base =
                    (static_cast<size_t>(col) * blocks_per_col + block_idx) * block_bytes;
                const __half scale_h = *reinterpret_cast<const __half*>(B + block_base);
                const float scale = __half2float(scale_h);
                #pragma unroll
                for (int q_idx = 0; q_idx < qk; ++q_idx) {
                    if (q_idx < valid) {
                        const int8_t q = *reinterpret_cast<const int8_t*>(B + block_base + 2 + q_idx);
                        acc += a_tile[threadIdx.y][q_idx] * (static_cast<float>(q) * scale);
                    }
                }
            }
            __syncthreads();
        }
        if (row < M && col < N) {
            C[row * N + col] = acc;
        }
    }

    __global__ void gemm_f32xq8_0_gguf_f32_decode_kernel(
        const float* __restrict__ A,           // 1xK row-major activations
        const unsigned char* __restrict__ B,   // GGUF Q8_0 [K,N], K-fastest
        float* __restrict__ C,                 // 1xN row-major output
        int N, int K) {
        const int col = blockIdx.x;
        const int tid = threadIdx.x;
        if (col >= N) return;
        const int qk = 32;
        const int block_bytes = 34;
        const int blocks_per_col = K / qk;
        float acc = 0.0f;
        for (int block_idx = tid; block_idx < blocks_per_col; block_idx += blockDim.x) {
            const size_t block_base =
                (static_cast<size_t>(col) * blocks_per_col + block_idx) * block_bytes;
            const __half scale_h = *reinterpret_cast<const __half*>(B + block_base);
            const float scale = __half2float(scale_h);
            const int k_base = block_idx * qk;
            #pragma unroll
            for (int q_idx = 0; q_idx < qk; ++q_idx) {
                const int8_t q = *reinterpret_cast<const int8_t*>(B + block_base + 2 + q_idx);
                acc += A[k_base + q_idx] * (static_cast<float>(q) * scale);
            }
        }
        extern __shared__ float q8_decode_reduce[];
        q8_decode_reduce[tid] = acc;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                q8_decode_reduce[tid] += q8_decode_reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            C[col] = q8_decode_reduce[0];
        }
    }

    // Materialize GGUF [K,N] F16 weights into column-major [N,K] FP32.
    // This lets cublasSgemm compute row-major C=A*B via C_col=B^T*A^T.
    __global__ void materialize_gguf_f16_to_f32_colmajor_nt_kernel(
        const __half* __restrict__ B,
        float* __restrict__ Bt,
        int N, int K) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = N * K;
        if (idx >= total) return;
        const int n = idx % N;
        const int k = idx / N;
        Bt[idx] = __half2float(B[n * K + k]);
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

    // Expose mixed-dtype GEMM for GGUF F16 weights (dimension 0 fastest).
    int m40llm_gemm_f32xf16_gguf_f32(
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
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B_f16, CUDA_R_16F, K,
            d_A_f32, CUDA_R_32F, K,
            &beta,
            d_C_f32, CUDA_R_32F, N,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT);
        if (st == CUBLAS_STATUS_SUCCESS) {
            cudaError_t err = cudaStreamSynchronize(ctx->prefill_stream);
            return err == cudaSuccess ? 0 : -3;
        }
#endif
        const float* A = reinterpret_cast<const float*>(d_A_f32);
        const __half* B = reinterpret_cast<const __half*>(d_B_f16);
        float* C = reinterpret_cast<float*>(d_C_f32);
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        gemm_f32xf16_gguf_f32_kernel<<<grid, block, 0, ctx->prefill_stream>>>(A, B, C, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -2;
        err = cudaStreamSynchronize(ctx->prefill_stream);
        if (err != cudaSuccess) return -3;
        return 0;
    }

    int m40llm_gemm_f32xf16_gguf_f32_decode_async(
        M40llmCudaContext* ctx,
        const void* d_A_f32,
        const void* d_B_f16,
        void* d_C_f32,
        int M, int N, int K) {
        if (!ctx || !d_A_f32 || !d_B_f16 || !d_C_f32) return -1;
        if (M != 1 || N <= 0 || K <= 0) return -2;
        const int threads = 128;
        const size_t shmem = static_cast<size_t>(threads) * sizeof(float);
        const float* A = reinterpret_cast<const float*>(d_A_f32);
        const __half* B = reinterpret_cast<const __half*>(d_B_f16);
        float* C = reinterpret_cast<float*>(d_C_f32);
        gemm_f32xf16_gguf_f32_decode_kernel<<<
            N,
            threads,
            shmem,
            f16_decode_projection_stream(ctx)>>>(
            A, B, C, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "gemm_f32xf16_gguf_f32_decode kernel launch error: %s\n", cudaGetErrorString(err));
            return -3;
        }
        return 0;
    }

    int m40llm_mlp_gate_up_swiglu_f32xf16_gguf_decode_async(
        M40llmCudaContext* ctx,
        const void* d_A_f32,
        const void* d_W_gate_f16,
        const void* d_W_up_f16,
        void* d_C_f32,
        int H, int K) {
        if (!ctx || !d_A_f32 || !d_W_gate_f16 || !d_W_up_f16 || !d_C_f32) return -1;
        if (H <= 0 || K <= 0) return -2;
        const int threads = 128;
        const size_t shmem = static_cast<size_t>(threads) * 2 * sizeof(float);
        mlp_gate_up_swiglu_f32xf16_gguf_decode_kernel<<<
            H,
            threads,
            shmem,
            f16_decode_projection_stream(ctx)>>>(
            reinterpret_cast<const float*>(d_A_f32),
            reinterpret_cast<const __half*>(d_W_gate_f16),
            reinterpret_cast<const __half*>(d_W_up_f16),
            reinterpret_cast<float*>(d_C_f32),
            H,
            K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "mlp_gate_up_swiglu_f32xf16_gguf_decode kernel launch error: %s\n", cudaGetErrorString(err));
            return -3;
        }
        return 0;
    }

    int m40llm_qkv_f32xf16_gguf_decode_async(
        M40llmCudaContext* ctx,
        const void* d_A_f32,
        const void* d_Wq_f16,
        const void* d_Wk_f16,
        const void* d_Wv_f16,
        const void* d_bq_f32,
        const void* d_bk_f32,
        const void* d_bv_f32,
        void* d_Q_f32,
        void* d_K_f32,
        void* d_V_f32,
        int Nq, int Nk, int Nv, int Kdim) {
        if (!ctx || !d_A_f32 || !d_Wq_f16 || !d_Wk_f16 || !d_Wv_f16 ||
            !d_Q_f32 || !d_K_f32 || !d_V_f32) return -1;
        if (Nq <= 0 || Nk <= 0 || Nv <= 0 || Kdim <= 0) return -2;
        const int threads = 128;
        const size_t shmem = static_cast<size_t>(threads) * sizeof(float);
        qkv_f32xf16_gguf_decode_kernel<<<
            Nq + Nk + Nv,
            threads,
            shmem,
            f16_decode_projection_stream(ctx)>>>(
            reinterpret_cast<const float*>(d_A_f32),
            reinterpret_cast<const __half*>(d_Wq_f16),
            reinterpret_cast<const __half*>(d_Wk_f16),
            reinterpret_cast<const __half*>(d_Wv_f16),
            reinterpret_cast<const float*>(d_bq_f32),
            reinterpret_cast<const float*>(d_bk_f32),
            reinterpret_cast<const float*>(d_bv_f32),
            reinterpret_cast<float*>(d_Q_f32),
            reinterpret_cast<float*>(d_K_f32),
            reinterpret_cast<float*>(d_V_f32),
            Nq,
            Nk,
            Nv,
            Kdim);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "qkv_f32xf16_gguf_decode kernel launch error: %s\n", cudaGetErrorString(err));
            return -3;
        }
        return 0;
    }

    int m40llm_gemm_f32xq8_0_gguf_f32_generic_async(
        M40llmCudaContext* ctx,
        const void* d_A_f32,
        const void* d_B_q8_0,
        void* d_C_f32,
        int M, int N, int K) {
        if (!ctx) return -1;
        if (M <= 0 || N <= 0 || K <= 0) return -4;
        const float* A = reinterpret_cast<const float*>(d_A_f32);
        const unsigned char* B = reinterpret_cast<const unsigned char*>(d_B_q8_0);
        float* C = reinterpret_cast<float*>(d_C_f32);
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        gemm_f32xq8_0_gguf_f32_kernel<<<grid, block, 0, ctx->prefill_stream>>>(A, B, C, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -2;
        return 0;
    }

    int m40llm_gemm_f32xq8_0_gguf_f32_blockloop_async(
        M40llmCudaContext* ctx,
        const void* d_A_f32,
        const void* d_B_q8_0,
        void* d_C_f32,
        int M, int N, int K) {
        if (!ctx) return -1;
        if (M <= 0 || N <= 0 || K <= 0) return -4;
        const float* A = reinterpret_cast<const float*>(d_A_f32);
        const unsigned char* B = reinterpret_cast<const unsigned char*>(d_B_q8_0);
        float* C = reinterpret_cast<float*>(d_C_f32);
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        gemm_f32xq8_0_gguf_f32_blockloop_kernel<<<grid, block, 0, ctx->prefill_stream>>>(A, B, C, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -2;
        return 0;
    }

    int m40llm_gemm_f32xq8_0_gguf_f32_shared_activation_async(
        M40llmCudaContext* ctx,
        const void* d_A_f32,
        const void* d_B_q8_0,
        void* d_C_f32,
        int M, int N, int K) {
        if (!ctx) return -1;
        if (M <= 0 || N <= 0 || K <= 0) return -4;
        const float* A = reinterpret_cast<const float*>(d_A_f32);
        const unsigned char* B = reinterpret_cast<const unsigned char*>(d_B_q8_0);
        float* C = reinterpret_cast<float*>(d_C_f32);
        dim3 block(16, 8);
        dim3 grid((N + 15) / 16, (M + 7) / 8);
        gemm_f32xq8_0_gguf_f32_shared_activation_kernel<<<grid, block, 0, ctx->prefill_stream>>>(A, B, C, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -2;
        return 0;
    }

    int m40llm_gemm_f32xq8_0_gguf_f32_decode_async(
        M40llmCudaContext* ctx,
        const void* d_A_f32,
        const void* d_B_q8_0,
        void* d_C_f32,
        int M, int N, int K) {
        if (!ctx) return -1;
        if (M != 1 || N <= 0 || K <= 0 || (K % 32) != 0) return -4;
        const float* A = reinterpret_cast<const float*>(d_A_f32);
        const unsigned char* B = reinterpret_cast<const unsigned char*>(d_B_q8_0);
        float* C = reinterpret_cast<float*>(d_C_f32);
        const int threads = 128;
        const size_t shmem = threads * sizeof(float);
        gemm_f32xq8_0_gguf_f32_decode_kernel<<<N, threads, shmem, ctx->prefill_stream>>>(A, B, C, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -2;
        return 0;
    }

    int m40llm_gemm_f32xq8_0_gguf_f32_async(
        M40llmCudaContext* ctx,
        const void* d_A_f32,
        const void* d_B_q8_0,
        void* d_C_f32,
        int M, int N, int K) {
        if (M == 1 && K > 0 && (K % 32) == 0) {
            return m40llm_gemm_f32xq8_0_gguf_f32_decode_async(ctx, d_A_f32, d_B_q8_0, d_C_f32, M, N, K);
        }
        if (M >= 4 && K >= 64 && N >= 16) {
            return m40llm_gemm_f32xq8_0_gguf_f32_shared_activation_async(ctx, d_A_f32, d_B_q8_0, d_C_f32, M, N, K);
        }
        return m40llm_gemm_f32xq8_0_gguf_f32_blockloop_async(ctx, d_A_f32, d_B_q8_0, d_C_f32, M, N, K);
    }

    int m40llm_gemm_f32xq8_0_gguf_f32(
        M40llmCudaContext* ctx,
        const void* d_A_f32,
        const void* d_B_q8_0,
        void* d_C_f32,
        int M, int N, int K) {
        int rc = m40llm_gemm_f32xq8_0_gguf_f32_async(ctx, d_A_f32, d_B_q8_0, d_C_f32, M, N, K);
        if (rc != 0) return rc;
        cudaError_t err = cudaStreamSynchronize(ctx->prefill_stream);
        if (err != cudaSuccess) return -3;
        return 0;
    }

    int m40llm_gemm_f32xf32_f32_async(
        M40llmCudaContext* ctx,
        const void* d_A_f32,
        const void* d_B_f32_colmajor_nt,
        void* d_C_f32,
        int M, int N, int K);

    int m40llm_gemm_f32xf32_f32_stream_async(
        M40llmCudaContext* ctx,
        const void* d_A_f32,
        const void* d_B_f32_colmajor_nt,
        void* d_C_f32,
        int M, int N, int K,
        uint32_t stream_kind);

    int m40llm_gemm_f32xf32_f32(
        M40llmCudaContext* ctx,
        const void* d_A_f32,
        const void* d_B_f32_colmajor_nt,
        void* d_C_f32,
        int M, int N, int K) {
        int rc = m40llm_gemm_f32xf32_f32_async(
            ctx, d_A_f32, d_B_f32_colmajor_nt, d_C_f32, M, N, K);
        if (rc != 0) return rc;
        cudaError_t err = cudaStreamSynchronize(ctx->prefill_stream);
        return err == cudaSuccess ? 0 : -4;
    }

    int m40llm_gemm_f32xf32_f32_async(
        M40llmCudaContext* ctx,
        const void* d_A_f32,
        const void* d_B_f32_colmajor_nt,
        void* d_C_f32,
        int M, int N, int K) {
        return m40llm_gemm_f32xf32_f32_stream_async(
            ctx, d_A_f32, d_B_f32_colmajor_nt, d_C_f32, M, N, K, 0);
    }

    int m40llm_gemm_f32xf32_f32_stream_async(
        M40llmCudaContext* ctx,
        const void* d_A_f32,
        const void* d_B_f32_colmajor_nt,
        void* d_C_f32,
        int M, int N, int K,
        uint32_t stream_kind) {
        if (!ctx || !d_A_f32 || !d_B_f32_colmajor_nt || !d_C_f32) return -1;
        if (M <= 0 || N <= 0 || K <= 0) return -2;
#ifdef M40LLM_HAVE_CUBLAS
        cudaStream_t stream = select_stream(ctx, stream_kind);
        if (!stream) return -6;
        if (ctx->cublas_stream_kind != stream_kind) {
            cublasStatus_t stream_st = cublasSetStream(ctx->cublas, stream);
            if (stream_st != CUBLAS_STATUS_SUCCESS) return -7;
            ctx->cublas_stream_kind = stream_kind;
        }
        float alpha = 1.0f;
        float beta = 0.0f;
        if (M == 1) {
            cublasStatus_t st = cublasSgemv(
                ctx->cublas,
                CUBLAS_OP_N,
                N, K,
                &alpha,
                reinterpret_cast<const float*>(d_B_f32_colmajor_nt), N,
                reinterpret_cast<const float*>(d_A_f32), 1,
                &beta,
                reinterpret_cast<float*>(d_C_f32), 1);
            if (st != CUBLAS_STATUS_SUCCESS) return -8;
            return 0;
        }
        cublasStatus_t st = cublasSgemm(
            ctx->cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            reinterpret_cast<const float*>(d_B_f32_colmajor_nt), N,
            reinterpret_cast<const float*>(d_A_f32), K,
            &beta,
            reinterpret_cast<float*>(d_C_f32), N);
        if (st != CUBLAS_STATUS_SUCCESS) return -3;
        return 0;
#else
        (void)stream_kind;
        return -5;
#endif
    }

    int m40llm_materialize_gguf_f16_to_f32_colmajor_nt(
        M40llmCudaContext* ctx,
        const void* d_B_f16,
        void* d_B_f32_colmajor_nt,
        int N, int K) {
        if (!ctx || !d_B_f16 || !d_B_f32_colmajor_nt) return -1;
        if (N <= 0 || K <= 0) return -2;
        const int total = N * K;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        materialize_gguf_f16_to_f32_colmajor_nt_kernel<<<blocks, threads, 0, ctx->prefill_stream>>>(
            reinterpret_cast<const __half*>(d_B_f16),
            reinterpret_cast<float*>(d_B_f32_colmajor_nt),
            N,
            K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -3;
        err = cudaStreamSynchronize(ctx->prefill_stream);
        return err == cudaSuccess ? 0 : -4;
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

    static void init_q8_old_backing_fields(M40llmKVCache* kv) {
        kv->d_q8_old_k = nullptr;
        kv->d_q8_old_v = nullptr;
        kv->d_q8_old_k_scale = nullptr;
        kv->d_q8_old_v_scale = nullptr;
        kv->d_fp16_old_k = nullptr;
        kv->d_q4_old_v = nullptr;
        kv->d_q4_old_v_scale = nullptr;
        kv->q8_old_backing = 0;
        kv->fp16_k_q4_v_old_backing = 0;
        kv->q4_old_v_backing = 0;
    }

    static bool allocate_q8_old_backing(M40llmKVCache* kv) {
        const size_t elems = kv_storage_elems(
            kv->max_seq_len,
            kv->max_batch_size,
            kv->num_heads,
            kv->head_dim);
        const size_t vectors = (size_t)kv->max_seq_len * (size_t)kv->max_batch_size * (size_t)kv->num_heads;
        cudaError_t err = cudaMalloc(&kv->d_q8_old_k, elems * sizeof(int8_t));
        if (err != cudaSuccess) return false;
        err = cudaMalloc(&kv->d_q8_old_v, elems * sizeof(int8_t));
        if (err != cudaSuccess) return false;
        err = cudaMalloc(&kv->d_q8_old_k_scale, vectors * sizeof(float));
        if (err != cudaSuccess) return false;
        err = cudaMalloc(&kv->d_q8_old_v_scale, vectors * sizeof(float));
        if (err != cudaSuccess) return false;
        cudaMemset(kv->d_q8_old_k, 0, elems * sizeof(int8_t));
        cudaMemset(kv->d_q8_old_v, 0, elems * sizeof(int8_t));
        cudaMemset(kv->d_q8_old_k_scale, 0, vectors * sizeof(float));
        cudaMemset(kv->d_q8_old_v_scale, 0, vectors * sizeof(float));
        kv->q8_old_backing = 1;
        return true;
    }

    static bool allocate_fp16_old_k_backing(M40llmKVCache* kv) {
        const size_t elems = kv_storage_elems(
            kv->max_seq_len,
            kv->max_batch_size,
            kv->num_heads,
            kv->head_dim);
        cudaError_t err = cudaMalloc(&kv->d_fp16_old_k, elems * sizeof(__half));
        if (err != cudaSuccess) return false;
        cudaMemset(kv->d_fp16_old_k, 0, elems * sizeof(__half));
        kv->fp16_k_q4_v_old_backing = 1;
        return true;
    }

    static bool allocate_q4_old_v_backing(M40llmKVCache* kv) {
        if ((kv->head_dim != 64 && kv->head_dim != 128) || (kv->head_dim % 2u) != 0u) return false;
        const size_t elems = (size_t)kv->max_seq_len
            * (size_t)kv->max_batch_size
            * (size_t)kv->num_heads
            * (size_t)(kv->head_dim / 2u);
        const size_t vectors = (size_t)kv->max_seq_len * (size_t)kv->max_batch_size * (size_t)kv->num_heads;
        cudaError_t err = cudaMalloc(&kv->d_q4_old_v, elems * sizeof(uint8_t));
        if (err != cudaSuccess) return false;
        err = cudaMalloc(&kv->d_q4_old_v_scale, vectors * sizeof(float));
        if (err != cudaSuccess) return false;
        cudaMemset(kv->d_q4_old_v, 0, elems * sizeof(uint8_t));
        cudaMemset(kv->d_q4_old_v_scale, 0, vectors * sizeof(float));
        kv->q4_old_v_backing = 1;
        return true;
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
        kv->compressed = 0;
        kv->recent_window = 0;
        kv->block_size = 0;
        kv->max_blocks = 0;
        kv->top_blocks = 0;
        kv->representatives = 0;
        kv->representative_policy = 0;
        kv->d_recent_k = nullptr;
        kv->d_recent_v = nullptr;
        kv->d_summary_k_acc = nullptr;
        kv->d_summary_v_acc = nullptr;
        kv->d_summary_k = nullptr;
        kv->d_summary_v = nullptr;
        kv->d_block_counts = nullptr;
        kv->d_rep_k = nullptr;
        kv->d_rep_v = nullptr;
        kv->d_rep_positions = nullptr;
        init_q8_old_backing_fields(kv);

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

    M40llmKVCache* m40llm_kvcache_create_compressed(M40llmCudaContext* ctx,
                                         uint32_t max_seq_len,
                                         uint32_t max_batch_size,
                                         uint32_t num_heads,
                                         uint32_t head_dim,
                                         uint32_t recent_window,
                                         uint32_t block_size,
                                         uint32_t top_blocks,
                                         uint32_t representatives,
                                         uint32_t representative_policy,
                                         uint32_t exact_old_backing) {
        if (!ctx || recent_window == 0 || block_size == 0) return nullptr;
        if (ensure_device(ctx) != 0) return nullptr;
        M40llmKVCache* kv = new M40llmKVCache();
        kv->d_k = nullptr;
        kv->d_v = nullptr;
        kv->d_seq_map = nullptr;
        kv->d_recent_k = nullptr;
        kv->d_recent_v = nullptr;
        kv->d_summary_k_acc = nullptr;
        kv->d_summary_v_acc = nullptr;
        kv->d_summary_k = nullptr;
        kv->d_summary_v = nullptr;
        kv->d_block_counts = nullptr;
        kv->d_rep_k = nullptr;
        kv->d_rep_v = nullptr;
        kv->d_rep_positions = nullptr;
        init_q8_old_backing_fields(kv);
        kv->max_seq_len = max_seq_len;
        kv->max_batch_size = max_batch_size;
        kv->num_heads = num_heads;
        kv->head_dim = head_dim;
        kv->compressed = 1;
        kv->recent_window = recent_window;
        kv->block_size = block_size;
        kv->top_blocks = top_blocks;
        kv->representatives = representatives;
        kv->representative_policy = representative_policy;
        const uint32_t old_capacity = max_seq_len > recent_window ? max_seq_len - recent_window : 0;
        kv->max_blocks = old_capacity == 0 ? 1 : (old_capacity + block_size - 1) / block_size;

        const size_t elems_per_token = (size_t)num_heads * (size_t)head_dim;
        const size_t recent_elems = (size_t)max_batch_size * (size_t)recent_window * elems_per_token;
        const size_t summary_elems = (size_t)max_batch_size * (size_t)kv->max_blocks * elems_per_token;
        const size_t rep_elems = (size_t)max_batch_size * (size_t)kv->max_blocks * (size_t)representatives * elems_per_token;
        const size_t seq_map_size = (size_t)max_batch_size * sizeof(uint32_t);
        const size_t block_count_size = (size_t)max_batch_size * (size_t)kv->max_blocks * sizeof(uint32_t);
        const size_t rep_position_size = (size_t)max_batch_size * (size_t)kv->max_blocks * (size_t)representatives * sizeof(uint32_t);
        cudaError_t err = cudaMalloc(&kv->d_recent_k, recent_elems * sizeof(__half));
        if (err != cudaSuccess) { m40llm_kvcache_destroy(kv); return nullptr; }
        err = cudaMalloc(&kv->d_recent_v, recent_elems * sizeof(__half));
        if (err != cudaSuccess) { m40llm_kvcache_destroy(kv); return nullptr; }
        err = cudaMalloc(&kv->d_summary_k_acc, summary_elems * sizeof(float));
        if (err != cudaSuccess) { m40llm_kvcache_destroy(kv); return nullptr; }
        err = cudaMalloc(&kv->d_summary_v_acc, summary_elems * sizeof(float));
        if (err != cudaSuccess) { m40llm_kvcache_destroy(kv); return nullptr; }
        err = cudaMalloc(&kv->d_summary_k, summary_elems * sizeof(__half));
        if (err != cudaSuccess) { m40llm_kvcache_destroy(kv); return nullptr; }
        err = cudaMalloc(&kv->d_summary_v, summary_elems * sizeof(__half));
        if (err != cudaSuccess) { m40llm_kvcache_destroy(kv); return nullptr; }
        err = cudaMalloc(&kv->d_block_counts, block_count_size);
        if (err != cudaSuccess) { m40llm_kvcache_destroy(kv); return nullptr; }
        if (representatives > 0) {
            err = cudaMalloc(&kv->d_rep_k, rep_elems * sizeof(__half));
            if (err != cudaSuccess) { m40llm_kvcache_destroy(kv); return nullptr; }
            err = cudaMalloc(&kv->d_rep_v, rep_elems * sizeof(__half));
            if (err != cudaSuccess) { m40llm_kvcache_destroy(kv); return nullptr; }
            err = cudaMalloc(&kv->d_rep_positions, rep_position_size);
            if (err != cudaSuccess) { m40llm_kvcache_destroy(kv); return nullptr; }
        }
        if (exact_old_backing == 1u && !allocate_q8_old_backing(kv)) {
            m40llm_kvcache_destroy(kv);
            return nullptr;
        }
        if (exact_old_backing == 2u && (!allocate_fp16_old_k_backing(kv) || !allocate_q4_old_v_backing(kv))) {
            m40llm_kvcache_destroy(kv);
            return nullptr;
        }
        if (exact_old_backing == 1u && q4_old_v_diag_from_env() && !allocate_q4_old_v_backing(kv)) {
            m40llm_kvcache_destroy(kv);
            return nullptr;
        }
        if (exact_old_backing == 1u && q8_dense_shadow_from_env()) {
            const size_t dense_elems = kv_storage_elems(max_seq_len, max_batch_size, num_heads, head_dim);
            const size_t dense_bytes = dense_elems * sizeof(__half);
            err = cudaMalloc(&kv->d_k, dense_bytes);
            if (err != cudaSuccess) { m40llm_kvcache_destroy(kv); return nullptr; }
            err = cudaMalloc(&kv->d_v, dense_bytes);
            if (err != cudaSuccess) { m40llm_kvcache_destroy(kv); return nullptr; }
            cudaMemset(kv->d_k, 0, dense_bytes);
            cudaMemset(kv->d_v, 0, dense_bytes);
        }
        err = cudaMalloc(&kv->d_seq_map, seq_map_size);
        if (err != cudaSuccess) { m40llm_kvcache_destroy(kv); return nullptr; }
        cudaMemset(kv->d_recent_k, 0, recent_elems * sizeof(__half));
        cudaMemset(kv->d_recent_v, 0, recent_elems * sizeof(__half));
        cudaMemset(kv->d_summary_k_acc, 0, summary_elems * sizeof(float));
        cudaMemset(kv->d_summary_v_acc, 0, summary_elems * sizeof(float));
        cudaMemset(kv->d_summary_k, 0, summary_elems * sizeof(__half));
        cudaMemset(kv->d_summary_v, 0, summary_elems * sizeof(__half));
        cudaMemset(kv->d_block_counts, 0, block_count_size);
        if (representatives > 0) {
            cudaMemset(kv->d_rep_k, 0, rep_elems * sizeof(__half));
            cudaMemset(kv->d_rep_v, 0, rep_elems * sizeof(__half));
            cudaMemset(kv->d_rep_positions, 0xff, rep_position_size);
        }
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

    static constexpr uint32_t M40LLM_ROPE_LAYOUT_ADJACENT = 0u;
    static constexpr uint32_t M40LLM_ROPE_LAYOUT_NEOX = 1u;

    __device__ __forceinline__ void rope_pair_indices(
        uint32_t head,
        uint32_t offset_in_head,
        uint32_t head_dim,
        uint32_t rope_layout,
        size_t* i0,
        size_t* i1) {
        const size_t head_base = (size_t)head * (size_t)head_dim;
        if (rope_layout == M40LLM_ROPE_LAYOUT_NEOX) {
            *i0 = head_base + (size_t)offset_in_head;
            *i1 = *i0 + (size_t)(head_dim / 2u);
        } else {
            *i0 = head_base + (size_t)(2u * offset_in_head);
            *i1 = *i0 + 1u;
        }
    }

    __global__ void rope_k_append_f32_to_f16_h2_kernel(
        const float* __restrict__ k_in,
        const float* __restrict__ v_in,
        __half* __restrict__ k_out,
        __half* __restrict__ v_out,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t past_len,
        float freq_base,
        float freq_scale,
        uint32_t rope_layout,
        size_t pairs_per_token) {
        const size_t pair = blockIdx.x * blockDim.x + threadIdx.x;
        if (pair >= pairs_per_token) return;

        const uint32_t half_dim = head_dim / 2u;
        const uint32_t offset_in_head = (uint32_t)(pair % (size_t)half_dim);
        const uint32_t head = (uint32_t)(pair / (size_t)half_dim);
        if (head >= num_heads) return;

        size_t i0 = 0;
        size_t i1 = 0;
        rope_pair_indices(head, offset_in_head, head_dim, rope_layout, &i0, &i1);
        const float pos = static_cast<float>(past_len) * freq_scale;
        const float theta = pos * powf(
            freq_base,
            -2.0f * static_cast<float>(offset_in_head) / static_cast<float>(head_dim));
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float k0 = k_in[i0];
        const float k1 = k_in[i1];
        k_out[i0] = __float2half_rn(k0 * c - k1 * s);
        k_out[i1] = __float2half_rn(k0 * s + k1 * c);
        v_out[i0] = __float2half_rn(v_in[i0]);
        v_out[i1] = __float2half_rn(v_in[i1]);
    }

    __global__ void rope_k_append_f32_to_f16_h2_at_kernel(
        const float* __restrict__ k_in,
        const float* __restrict__ v_in,
        __half* __restrict__ k_base,
        __half* __restrict__ v_base,
        uint32_t* __restrict__ seq_map,
        uint32_t seq_id,
        uint32_t max_seq_len,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t position,
        const uint32_t* __restrict__ position_dev,
        float freq_base,
        float freq_scale,
        uint32_t rope_layout,
        size_t pairs_per_token) {
        const size_t pair = blockIdx.x * blockDim.x + threadIdx.x;
        if (pair >= pairs_per_token) return;
        if (position_dev) {
            position = *position_dev;
        }
        if (position >= max_seq_len) return;

        const size_t elems_per_token = (size_t)num_heads * (size_t)head_dim;
        const size_t token_offset =
            ((size_t)seq_id * (size_t)max_seq_len + (size_t)position) * elems_per_token;
        __half* k_out = k_base + token_offset;
        __half* v_out = v_base + token_offset;

        const uint32_t half_dim = head_dim / 2u;
        const uint32_t offset_in_head = (uint32_t)(pair % (size_t)half_dim);
        const uint32_t head = (uint32_t)(pair / (size_t)half_dim);
        if (head >= num_heads) return;

        size_t i0 = 0;
        size_t i1 = 0;
        rope_pair_indices(head, offset_in_head, head_dim, rope_layout, &i0, &i1);
        const float pos = static_cast<float>(position) * freq_scale;
        const float theta = pos * powf(
            freq_base,
            -2.0f * static_cast<float>(offset_in_head) / static_cast<float>(head_dim));
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float k0 = k_in[i0];
        const float k1 = k_in[i1];
        k_out[i0] = __float2half_rn(k0 * c - k1 * s);
        k_out[i1] = __float2half_rn(k0 * s + k1 * c);
        v_out[i0] = __float2half_rn(v_in[i0]);
        v_out[i1] = __float2half_rn(v_in[i1]);
        if (pair == 0) {
            seq_map[seq_id] = position + 1u;
        }
    }

    __device__ __forceinline__ uint32_t representative_slot_for_policy(
        uint32_t old_pos_in_block,
        uint32_t block_size,
        uint32_t representatives,
        uint32_t representative_policy) {
        if (representatives == 0) return 0xffffffffu;
        if (representative_policy == 1u) {
            const uint32_t slot = ((uint64_t)old_pos_in_block * (uint64_t)representatives) / (uint64_t)block_size;
            return slot < representatives ? slot : representatives - 1u;
        }
        return old_pos_in_block % representatives;
    }

    __global__ void compressed_rope_k_append_f32_to_f16_kernel(
        const float* __restrict__ k_in,
        const float* __restrict__ v_in,
        __half* __restrict__ recent_k,
        __half* __restrict__ recent_v,
        float* __restrict__ summary_k_acc,
        float* __restrict__ summary_v_acc,
        __half* __restrict__ summary_k,
        __half* __restrict__ summary_v,
        uint32_t* __restrict__ block_counts,
        __half* __restrict__ rep_k,
        __half* __restrict__ rep_v,
        uint32_t* __restrict__ rep_positions,
        uint32_t* __restrict__ seq_map,
        uint32_t seq_id,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t max_blocks,
        uint32_t representatives,
        uint32_t representative_policy,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t position,
        float freq_base,
        float freq_scale,
        uint32_t rope_layout,
        size_t elems_per_token) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= elems_per_token) return;
        const size_t recent_token = (size_t)position % (size_t)recent_window;
        const size_t recent_base =
            ((size_t)seq_id * (size_t)recent_window + recent_token) * elems_per_token;
        const uint32_t dim = (uint32_t)(i % (size_t)head_dim);
        const uint32_t head = (uint32_t)((i / (size_t)head_dim) % (size_t)num_heads);
        const uint32_t half_dim = head_dim / 2u;
        const uint32_t offset_in_head = rope_layout == M40LLM_ROPE_LAYOUT_NEOX
            ? (dim % half_dim)
            : (dim / 2u);
        const float pos = static_cast<float>(position) * freq_scale;
        const float theta = pos * powf(
            freq_base,
            -2.0f * static_cast<float>(offset_in_head) / static_cast<float>(head_dim));
        const float c = cosf(theta);
        const float s = sinf(theta);
        size_t i0 = 0;
        size_t i1 = 0;
        rope_pair_indices(head, offset_in_head, head_dim, rope_layout, &i0, &i1);
        const size_t token_head_base =
            (i / ((size_t)num_heads * (size_t)head_dim)) * ((size_t)num_heads * (size_t)head_dim);
        i0 += token_head_base;
        i1 += token_head_base;
        const float k0 = k_in[i0];
        const float k1 = k_in[i1];
        const float rotated = i == i0
            ? (k0 * c - k1 * s)
            : (k0 * s + k1 * c);
        const float v = v_in[i];

        if (position >= recent_window) {
            const uint32_t old_pos = position - recent_window;
            const uint32_t block = old_pos / block_size;
            if (block < max_blocks) {
                const uint32_t count = (old_pos % block_size) + 1u;
                const float old_k = __half2float(recent_k[recent_base + i]);
                const float old_v = __half2float(recent_v[recent_base + i]);
                const size_t summary_base =
                    ((size_t)seq_id * (size_t)max_blocks + (size_t)block) * elems_per_token;
                const float k_sum = summary_k_acc[summary_base + i] + old_k;
                const float v_sum = summary_v_acc[summary_base + i] + old_v;
                summary_k_acc[summary_base + i] = k_sum;
                summary_v_acc[summary_base + i] = v_sum;
                summary_k[summary_base + i] = __float2half_rn(k_sum / (float)count);
                summary_v[summary_base + i] = __float2half_rn(v_sum / (float)count);
                if (representatives > 0 && rep_k && rep_v && rep_positions) {
                    const uint32_t within = old_pos % block_size;
                    const uint32_t rep_slot = representative_slot_for_policy(
                        within, block_size, representatives, representative_policy);
                    if (rep_slot < representatives) {
                        const size_t rep_base =
                            (((size_t)seq_id * (size_t)max_blocks + (size_t)block)
                                * (size_t)representatives + (size_t)rep_slot) * elems_per_token;
                        rep_k[rep_base + i] = recent_k[recent_base + i];
                        rep_v[rep_base + i] = recent_v[recent_base + i];
                        if (i == 0) {
                            rep_positions[((size_t)seq_id * (size_t)max_blocks + (size_t)block)
                                * (size_t)representatives + (size_t)rep_slot] = old_pos;
                        }
                    }
                }
                if (i == 0) {
                    block_counts[(size_t)seq_id * (size_t)max_blocks + (size_t)block] = count;
                }
            }
        }

        recent_k[recent_base + i] = __float2half_rn(rotated);
        recent_v[recent_base + i] = __float2half_rn(v);
        if (i == 0) {
            seq_map[seq_id] = position + 1u;
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
        int rc = m40llm_kvcache_append_token_f32_async(ctx, kv, seq_id, k_dev_f32, v_dev_f32);
        if (rc != 0) return rc;
        cudaError_t err = cudaStreamSynchronize(ctx->decode_stream);
        if (err != cudaSuccess) return -7;
        return 0;
    }

    int m40llm_kvcache_append_token_f32_async(M40llmCudaContext* ctx,
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
            cudaGetLastError();
            cast_store_f32_to_f16_kernel_h2<<<blocks_h2, threads, 0, ctx->decode_stream>>>(k_in_h2, k_out_h2, pairs);
            cast_store_f32_to_f16_kernel_h2<<<blocks_h2, threads, 0, ctx->decode_stream>>>(v_in_h2, v_out_h2, pairs);
        }
        if (tail) {
            const float* k_tail_in = k_in_h2 + pairs * 2;
            const float* v_tail_in = v_in_h2 + pairs * 2;
            __half* k_tail_out = k_out_h2 + pairs * 2;
            __half* v_tail_out = v_out_h2 + pairs * 2;
            cudaGetLastError();
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

        return 0;
    }

    int m40llm_kvcache_append_token_f32_rope_k_layout_async(M40llmCudaContext* ctx,
                                         M40llmKVCache* kv,
                                         uint32_t seq_id,
                                         const void* k_dev_f32,
                                         const void* v_dev_f32,
                                         uint32_t past_len,
                                         float freq_base,
                                         float freq_scale,
                                         uint32_t rope_layout);
    int m40llm_kvcache_append_token_f32_rope_k_at_layout_async(M40llmCudaContext* ctx,
                                         M40llmKVCache* kv,
                                         uint32_t seq_id,
                                         const void* k_dev_f32,
                                         const void* v_dev_f32,
                                         uint32_t position,
                                         uint32_t past_len,
                                         float freq_base,
                                         float freq_scale,
                                         uint32_t rope_layout);
    int m40llm_kvcache_append_token_f32_rope_k_compressed_at_layout_async(M40llmCudaContext* ctx,
                                         M40llmKVCache* kv,
                                         uint32_t seq_id,
                                         const void* k_dev_f32,
                                         const void* v_dev_f32,
                                         uint32_t position,
                                         float freq_base,
                                         float freq_scale,
                                         uint32_t rope_layout);
    int m40llm_kvcache_append_token_f32_rope_k_position_dev_layout_async(M40llmCudaContext* ctx,
                                         M40llmKVCache* kv,
                                         uint32_t seq_id,
                                         const void* k_dev_f32,
                                         const void* v_dev_f32,
                                         const uint32_t* position_dev,
                                         float freq_base,
                                         float freq_scale,
                                         uint32_t rope_layout);

    int m40llm_kvcache_append_token_f32_rope_k(M40llmCudaContext* ctx,
                                         M40llmKVCache* kv,
                                         uint32_t seq_id,
                                         const void* k_dev_f32,
                                         const void* v_dev_f32,
                                         uint32_t past_len,
                                         float freq_base,
                                         float freq_scale) {
        int rc = m40llm_kvcache_append_token_f32_rope_k_async(
            ctx, kv, seq_id, k_dev_f32, v_dev_f32, past_len, freq_base, freq_scale);
        if (rc != 0) return rc;
        cudaError_t err = cudaStreamSynchronize(ctx->decode_stream);
        if (err != cudaSuccess) return -7;
        return 0;
    }

    int m40llm_kvcache_append_token_f32_rope_k_async(M40llmCudaContext* ctx,
                                         M40llmKVCache* kv,
                                         uint32_t seq_id,
                                         const void* k_dev_f32,
                                         const void* v_dev_f32,
                                         uint32_t past_len,
                                         float freq_base,
                                         float freq_scale) {
        return m40llm_kvcache_append_token_f32_rope_k_layout_async(
            ctx, kv, seq_id, k_dev_f32, v_dev_f32, past_len, freq_base, freq_scale,
            M40LLM_ROPE_LAYOUT_ADJACENT);
    }

    int m40llm_kvcache_append_token_f32_rope_k_layout_async(M40llmCudaContext* ctx,
                                         M40llmKVCache* kv,
                                         uint32_t seq_id,
                                         const void* k_dev_f32,
                                         const void* v_dev_f32,
                                         uint32_t past_len,
                                         float freq_base,
                                         float freq_scale,
                                         uint32_t rope_layout) {
        if (!ctx || !kv || !k_dev_f32 || !v_dev_f32) return -1;
        if (seq_id >= kv->max_batch_size) return -2;
        if (kv->head_dim == 0 || kv->num_heads == 0) return -3;
        if (kv->head_dim % 2 != 0) return -4;

        uint32_t cur_len = 0;
        cudaError_t err = cudaMemcpy(&cur_len, kv->d_seq_map + seq_id, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -5;
        if (cur_len >= kv->max_seq_len) return -6;

        const size_t elems_per_token = (size_t)kv->num_heads * (size_t)kv->head_dim;
        const size_t offset_elems = ((size_t)seq_id * (size_t)kv->max_seq_len + (size_t)cur_len) * elems_per_token;

        const float* k_in = reinterpret_cast<const float*>(k_dev_f32);
        const float* v_in = reinterpret_cast<const float*>(v_dev_f32);
        __half* k_out = kv->d_k + offset_elems;
        __half* v_out = kv->d_v + offset_elems;

        const size_t pairs_per_token = elems_per_token / 2u;
        const int threads = 256;
        const int blocks = (int)((pairs_per_token + threads - 1) / threads);
        rope_k_append_f32_to_f16_h2_kernel<<<blocks, threads, 0, ctx->decode_stream>>>(
            k_in,
            v_in,
            k_out,
            v_out,
            kv->num_heads,
            kv->head_dim,
            past_len,
            freq_base,
            freq_scale,
            rope_layout,
            pairs_per_token);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "append_token_f32_rope_k kernel launch error: %s\n", cudaGetErrorString(err));
            return -7;
        }

        cur_len += 1;
        err = cudaMemcpyAsync(kv->d_seq_map + seq_id, &cur_len, sizeof(uint32_t), cudaMemcpyHostToDevice, ctx->decode_stream);
        if (err != cudaSuccess) return -8;

        return 0;
    }

    int m40llm_kvcache_append_token_f32_rope_k_at_async(M40llmCudaContext* ctx,
                                         M40llmKVCache* kv,
                                         uint32_t seq_id,
                                         const void* k_dev_f32,
                                         const void* v_dev_f32,
                                         uint32_t position,
                                         uint32_t past_len,
                                         float freq_base,
                                         float freq_scale) {
        return m40llm_kvcache_append_token_f32_rope_k_at_layout_async(
            ctx, kv, seq_id, k_dev_f32, v_dev_f32, position, past_len, freq_base, freq_scale,
            M40LLM_ROPE_LAYOUT_ADJACENT);
    }

    int m40llm_kvcache_append_token_f32_rope_k_at_layout_async(M40llmCudaContext* ctx,
                                         M40llmKVCache* kv,
                                         uint32_t seq_id,
                                         const void* k_dev_f32,
                                         const void* v_dev_f32,
                                         uint32_t position,
                                         uint32_t past_len,
                                         float freq_base,
                                         float freq_scale,
                                         uint32_t rope_layout) {
        if (!ctx || !kv || !k_dev_f32 || !v_dev_f32) return -1;
        if (kv->compressed) {
            return m40llm_kvcache_append_token_f32_rope_k_compressed_at_layout_async(
                ctx, kv, seq_id, k_dev_f32, v_dev_f32, position, freq_base, freq_scale, rope_layout);
        }
        if (seq_id >= kv->max_batch_size) return -2;
        if (kv->head_dim == 0 || kv->num_heads == 0) return -3;
        if (kv->head_dim % 2 != 0) return -4;
        if (position >= kv->max_seq_len) return -5;
        (void)past_len;

        const size_t elems_per_token = (size_t)kv->num_heads * (size_t)kv->head_dim;
        const size_t pairs_per_token = elems_per_token / 2u;
        const int threads = 256;
        const int blocks = (int)((pairs_per_token + threads - 1) / threads);
        rope_k_append_f32_to_f16_h2_at_kernel<<<blocks, threads, 0, ctx->decode_stream>>>(
            reinterpret_cast<const float*>(k_dev_f32),
            reinterpret_cast<const float*>(v_dev_f32),
            kv->d_k,
            kv->d_v,
            kv->d_seq_map,
            seq_id,
            kv->max_seq_len,
            kv->num_heads,
            kv->head_dim,
            position,
            nullptr,
            freq_base,
            freq_scale,
            rope_layout,
            pairs_per_token);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "append_token_f32_rope_k_at kernel launch error: %s\n", cudaGetErrorString(err));
            return -6;
        }

        return 0;
    }

    int m40llm_kvcache_append_token_f32_rope_k_compressed_at_async(M40llmCudaContext* ctx,
                                         M40llmKVCache* kv,
                                         uint32_t seq_id,
                                         const void* k_dev_f32,
                                         const void* v_dev_f32,
                                         uint32_t position,
                                         float freq_base,
                                         float freq_scale) {
        return m40llm_kvcache_append_token_f32_rope_k_compressed_at_layout_async(
            ctx, kv, seq_id, k_dev_f32, v_dev_f32, position, freq_base, freq_scale,
            M40LLM_ROPE_LAYOUT_ADJACENT);
    }

    int m40llm_kvcache_append_token_f32_rope_k_compressed_at_layout_async(M40llmCudaContext* ctx,
                                         M40llmKVCache* kv,
                                         uint32_t seq_id,
                                         const void* k_dev_f32,
                                         const void* v_dev_f32,
                                         uint32_t position,
                                         float freq_base,
                                         float freq_scale,
                                         uint32_t rope_layout) {
        if (!ctx || !kv || !k_dev_f32 || !v_dev_f32) return -1;
        if (!kv->compressed) return -2;
        if (seq_id >= kv->max_batch_size) return -3;
        if ((kv->head_dim != 64 && kv->head_dim != 128) || kv->num_heads == 0 || kv->recent_window == 0 || kv->block_size == 0) return -4;
        if (position >= kv->max_seq_len) return -5;
        const size_t elems_per_token = (size_t)kv->num_heads * (size_t)kv->head_dim;
        const int threads = 256;
        if (kv->q8_old_backing && position >= kv->recent_window) {
            quantize_evicted_recent_to_q8_old_head64_kernel<<<(int)kv->num_heads, 128, 0, ctx->decode_stream>>>(
                kv->d_recent_k,
                kv->d_recent_v,
                kv->d_q8_old_k,
                kv->d_q8_old_v,
                kv->d_q8_old_k_scale,
                kv->d_q8_old_v_scale,
                kv->max_seq_len,
                kv->recent_window,
                kv->num_heads,
                seq_id,
                position);
            cudaError_t q8_err = cudaGetLastError();
            if (q8_err != cudaSuccess) {
                fprintf(stderr, "compressed q8 old append kernel launch error: %s\n", cudaGetErrorString(q8_err));
                return -7;
            }
            if (kv->q4_old_v_backing) {
                quantize_evicted_recent_v_to_q4_old_kernel<<<(int)kv->num_heads, 128, 0, ctx->decode_stream>>>(
                    kv->d_recent_v,
                    kv->d_q4_old_v,
                    kv->d_q4_old_v_scale,
                    kv->max_seq_len,
                    kv->recent_window,
                    kv->num_heads,
                    kv->head_dim,
                    seq_id,
                    position);
                cudaError_t q4_err = cudaGetLastError();
                if (q4_err != cudaSuccess) {
                    fprintf(stderr, "compressed q4 old V append kernel launch error: %s\n", cudaGetErrorString(q4_err));
                    return -9;
                }
            }
        }
        if (kv->fp16_k_q4_v_old_backing && position >= kv->recent_window) {
            copy_evicted_recent_k_to_fp16_old_kernel<<<(int)kv->num_heads, 128, 0, ctx->decode_stream>>>(
                kv->d_recent_k,
                kv->d_fp16_old_k,
                kv->max_seq_len,
                kv->recent_window,
                kv->num_heads,
                kv->head_dim,
                seq_id,
                position);
            cudaError_t old_k_err = cudaGetLastError();
            if (old_k_err != cudaSuccess) {
                fprintf(stderr, "compressed fp16 old K append kernel launch error: %s\n", cudaGetErrorString(old_k_err));
                return -10;
            }
            quantize_evicted_recent_v_to_q4_old_kernel<<<(int)kv->num_heads, 128, 0, ctx->decode_stream>>>(
                kv->d_recent_v,
                kv->d_q4_old_v,
                kv->d_q4_old_v_scale,
                kv->max_seq_len,
                kv->recent_window,
                kv->num_heads,
                kv->head_dim,
                seq_id,
                position);
            cudaError_t q4_err = cudaGetLastError();
            if (q4_err != cudaSuccess) {
                fprintf(stderr, "compressed fp16-k/q4-v old append kernel launch error: %s\n", cudaGetErrorString(q4_err));
                return -11;
            }
        }
        if (kv->d_k && kv->d_v) {
            const size_t pairs_per_token = elems_per_token / 2u;
            const int shadow_blocks = (int)((pairs_per_token + threads - 1) / threads);
            rope_k_append_f32_to_f16_h2_at_kernel<<<shadow_blocks, threads, 0, ctx->decode_stream>>>(
                reinterpret_cast<const float*>(k_dev_f32),
                reinterpret_cast<const float*>(v_dev_f32),
                kv->d_k,
                kv->d_v,
                kv->d_seq_map,
                seq_id,
                kv->max_seq_len,
                kv->num_heads,
                kv->head_dim,
                position,
                nullptr,
                freq_base,
                freq_scale,
                rope_layout,
                pairs_per_token);
            cudaError_t shadow_err = cudaGetLastError();
            if (shadow_err != cudaSuccess) {
                fprintf(stderr, "compressed dense-shadow append kernel launch error: %s\n", cudaGetErrorString(shadow_err));
                return -8;
            }
        }
        const int blocks = (int)((elems_per_token + threads - 1) / threads);
        compressed_rope_k_append_f32_to_f16_kernel<<<blocks, threads, 0, ctx->decode_stream>>>(
            reinterpret_cast<const float*>(k_dev_f32),
            reinterpret_cast<const float*>(v_dev_f32),
            kv->d_recent_k,
            kv->d_recent_v,
            kv->d_summary_k_acc,
            kv->d_summary_v_acc,
            kv->d_summary_k,
            kv->d_summary_v,
            kv->d_block_counts,
            kv->d_rep_k,
            kv->d_rep_v,
            kv->d_rep_positions,
            kv->d_seq_map,
            seq_id,
            kv->recent_window,
            kv->block_size,
            kv->max_blocks,
            kv->representatives,
            kv->representative_policy,
            kv->num_heads,
            kv->head_dim,
            position,
            freq_base,
            freq_scale,
            rope_layout,
            elems_per_token);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "compressed append kernel launch error: %s\n", cudaGetErrorString(err));
            return -6;
        }
        return 0;
    }

    int m40llm_kvcache_append_token_f32_rope_k_position_dev_async(M40llmCudaContext* ctx,
                                         M40llmKVCache* kv,
                                         uint32_t seq_id,
                                         const void* k_dev_f32,
                                         const void* v_dev_f32,
                                         const uint32_t* position_dev,
                                         float freq_base,
                                         float freq_scale) {
        return m40llm_kvcache_append_token_f32_rope_k_position_dev_layout_async(
            ctx, kv, seq_id, k_dev_f32, v_dev_f32, position_dev, freq_base, freq_scale,
            M40LLM_ROPE_LAYOUT_ADJACENT);
    }

    int m40llm_kvcache_append_token_f32_rope_k_position_dev_layout_async(M40llmCudaContext* ctx,
                                         M40llmKVCache* kv,
                                         uint32_t seq_id,
                                         const void* k_dev_f32,
                                         const void* v_dev_f32,
                                         const uint32_t* position_dev,
                                         float freq_base,
                                         float freq_scale,
                                         uint32_t rope_layout) {
        if (!ctx || !kv || !k_dev_f32 || !v_dev_f32 || !position_dev) return -1;
        if (seq_id >= kv->max_batch_size) return -2;
        if (kv->head_dim == 0 || kv->num_heads == 0) return -3;
        if (kv->head_dim % 2 != 0) return -4;

        const size_t elems_per_token = (size_t)kv->num_heads * (size_t)kv->head_dim;
        const size_t pairs_per_token = elems_per_token / 2u;
        const int threads = 256;
        const int blocks = (int)((pairs_per_token + threads - 1) / threads);
        rope_k_append_f32_to_f16_h2_at_kernel<<<blocks, threads, 0, ctx->decode_stream>>>(
            reinterpret_cast<const float*>(k_dev_f32),
            reinterpret_cast<const float*>(v_dev_f32),
            kv->d_k,
            kv->d_v,
            kv->d_seq_map,
            seq_id,
            kv->max_seq_len,
            kv->num_heads,
            kv->head_dim,
            0,
            position_dev,
            freq_base,
            freq_scale,
            rope_layout,
            pairs_per_token);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "append_token_f32_rope_k_position_dev kernel launch error: %s\n", cudaGetErrorString(err));
            return -5;
        }

        return 0;
    }

    int m40llm_kvcache_reset(M40llmCudaContext* ctx, M40llmKVCache* kv) {
        if (!ctx || !kv) return -1;
        if (ensure_device(ctx) != 0) return -2;
        const size_t seq_map_size = (size_t)kv->max_batch_size * sizeof(uint32_t);
        cudaError_t err = cudaMemsetAsync(kv->d_seq_map, 0, seq_map_size, ctx->decode_stream);
        if (err != cudaSuccess) return -3;
        if (kv->compressed) {
            const size_t elems_per_token = (size_t)kv->num_heads * (size_t)kv->head_dim;
            const size_t recent_elems = (size_t)kv->max_batch_size * (size_t)kv->recent_window * elems_per_token;
            const size_t summary_elems = (size_t)kv->max_batch_size * (size_t)kv->max_blocks * elems_per_token;
            const size_t block_count_size = (size_t)kv->max_batch_size * (size_t)kv->max_blocks * sizeof(uint32_t);
            const size_t rep_elems = (size_t)kv->max_batch_size * (size_t)kv->max_blocks * (size_t)kv->representatives * elems_per_token;
            const size_t rep_position_size = (size_t)kv->max_batch_size * (size_t)kv->max_blocks * (size_t)kv->representatives * sizeof(uint32_t);
            cudaMemsetAsync(kv->d_recent_k, 0, recent_elems * sizeof(__half), ctx->decode_stream);
            cudaMemsetAsync(kv->d_recent_v, 0, recent_elems * sizeof(__half), ctx->decode_stream);
            cudaMemsetAsync(kv->d_summary_k_acc, 0, summary_elems * sizeof(float), ctx->decode_stream);
            cudaMemsetAsync(kv->d_summary_v_acc, 0, summary_elems * sizeof(float), ctx->decode_stream);
            cudaMemsetAsync(kv->d_summary_k, 0, summary_elems * sizeof(__half), ctx->decode_stream);
            cudaMemsetAsync(kv->d_summary_v, 0, summary_elems * sizeof(__half), ctx->decode_stream);
            cudaMemsetAsync(kv->d_block_counts, 0, block_count_size, ctx->decode_stream);
            if (kv->representatives > 0) {
                cudaMemsetAsync(kv->d_rep_k, 0, rep_elems * sizeof(__half), ctx->decode_stream);
                cudaMemsetAsync(kv->d_rep_v, 0, rep_elems * sizeof(__half), ctx->decode_stream);
                cudaMemsetAsync(kv->d_rep_positions, 0xff, rep_position_size, ctx->decode_stream);
            }
            if (kv->d_fp16_old_k) {
                const size_t old_k_elems = kv_storage_elems(
                    kv->max_seq_len,
                    kv->max_batch_size,
                    kv->num_heads,
                    kv->head_dim);
                cudaMemsetAsync(kv->d_fp16_old_k, 0, old_k_elems * sizeof(__half), ctx->decode_stream);
            }
            if (kv->d_q4_old_v) {
                const size_t q4_elems = (size_t)kv->max_seq_len
                    * (size_t)kv->max_batch_size
                    * (size_t)kv->num_heads
                    * (size_t)(kv->head_dim / 2u);
                cudaMemsetAsync(kv->d_q4_old_v, 0, q4_elems * sizeof(uint8_t), ctx->decode_stream);
            }
            if (kv->d_q4_old_v_scale) {
                const size_t q4_scales = (size_t)kv->max_seq_len
                    * (size_t)kv->max_batch_size
                    * (size_t)kv->num_heads;
                cudaMemsetAsync(kv->d_q4_old_v_scale, 0, q4_scales * sizeof(float), ctx->decode_stream);
            }
        }
        err = cudaStreamSynchronize(ctx->decode_stream);
        if (err != cudaSuccess) return -4;
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

    int m40llm_kvcache_debug_read_compressed_state(M40llmCudaContext* ctx,
                                                    M40llmKVCache* kv,
                                                    uint32_t seq_id,
                                                    uint32_t* out_seq_len,
                                                    uint32_t* out_block_counts,
                                                    void* out_recent_k_f16,
                                                    void* out_recent_v_f16,
                                                    float* out_summary_k_acc,
                                                    float* out_summary_v_acc,
                                                    void* out_summary_k_f16,
                                                    void* out_summary_v_f16,
                                                    void* out_rep_k_f16,
                                                    void* out_rep_v_f16,
                                                    uint32_t* out_rep_positions) {
        if (!ctx) return -16;
        if (ensure_device(ctx) != 0) return -17;
        if (!kv || !kv->compressed) return -1;
        if (seq_id >= kv->max_batch_size) return -2;
        if (!out_seq_len || !out_block_counts || !out_recent_k_f16 || !out_recent_v_f16 ||
            !out_summary_k_acc || !out_summary_v_acc || !out_summary_k_f16 || !out_summary_v_f16) {
            return -3;
        }
        cudaError_t err = cudaStreamSynchronize(ctx->decode_stream);
        if (err != cudaSuccess) return -18;
        err = cudaStreamSynchronize(ctx->prefill_stream);
        if (err != cudaSuccess) return -19;
        const size_t elems_per_token = (size_t)kv->num_heads * (size_t)kv->head_dim;
        const size_t recent_elems = (size_t)kv->recent_window * elems_per_token;
        const size_t summary_elems = (size_t)kv->max_blocks * elems_per_token;
        const size_t rep_elems = (size_t)kv->max_blocks * (size_t)kv->representatives * elems_per_token;
        const size_t rep_positions = (size_t)kv->max_blocks * (size_t)kv->representatives;
        err = cudaMemcpy(out_seq_len, kv->d_seq_map + seq_id, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -4;
        err = cudaMemcpy(out_block_counts,
                         kv->d_block_counts + (size_t)seq_id * (size_t)kv->max_blocks,
                         (size_t)kv->max_blocks * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -5;
        const size_t recent_base = (size_t)seq_id * (size_t)kv->recent_window * elems_per_token;
        err = cudaMemcpy(out_recent_k_f16, kv->d_recent_k + recent_base, recent_elems * sizeof(__half), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -6;
        err = cudaMemcpy(out_recent_v_f16, kv->d_recent_v + recent_base, recent_elems * sizeof(__half), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -7;
        const size_t summary_base = (size_t)seq_id * (size_t)kv->max_blocks * elems_per_token;
        err = cudaMemcpy(out_summary_k_acc, kv->d_summary_k_acc + summary_base, summary_elems * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -8;
        err = cudaMemcpy(out_summary_v_acc, kv->d_summary_v_acc + summary_base, summary_elems * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -9;
        err = cudaMemcpy(out_summary_k_f16, kv->d_summary_k + summary_base, summary_elems * sizeof(__half), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -10;
        err = cudaMemcpy(out_summary_v_f16, kv->d_summary_v + summary_base, summary_elems * sizeof(__half), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -11;
        if (kv->representatives > 0) {
            if (!out_rep_k_f16 || !out_rep_v_f16 || !out_rep_positions) return -12;
            const size_t rep_base =
                (size_t)seq_id * (size_t)kv->max_blocks * (size_t)kv->representatives * elems_per_token;
            const size_t rep_pos_base =
                (size_t)seq_id * (size_t)kv->max_blocks * (size_t)kv->representatives;
            err = cudaMemcpy(out_rep_k_f16, kv->d_rep_k + rep_base, rep_elems * sizeof(__half), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) return -13;
            err = cudaMemcpy(out_rep_v_f16, kv->d_rep_v + rep_base, rep_elems * sizeof(__half), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) return -14;
            err = cudaMemcpy(out_rep_positions, kv->d_rep_positions + rep_pos_base, rep_positions * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) return -15;
        }
        return 0;
    }

    int m40llm_kvcache_debug_select_old_blocks(M40llmCudaContext* ctx,
                                                const M40llmKVCache* kv,
                                                uint32_t seq_id,
                                                const void* q_dev_f32,
                                                uint32_t q_heads,
                                                uint32_t seq_len,
                                                uint32_t recent_window,
                                                  uint32_t block_size,
                                                  uint32_t top_blocks,
                                                  uint32_t* out_blocks_host,
                                                  float* out_scores_host,
                                                  uint32_t* out_start_host,
                                                  uint32_t* out_end_host,
                                                  uint32_t* out_count,
                                                  uint32_t max_out,
                                                  uint32_t* out_total_old_blocks) {
        if (!ctx || !kv || !q_dev_f32 || !out_blocks_host || !out_scores_host ||
            !out_start_host || !out_end_host || !out_count || !out_total_old_blocks) return -1;
        if (seq_id >= kv->max_batch_size) return -2;
        if (q_heads == 0 || kv->num_heads == 0 || kv->head_dim == 0) return -3;
        if (seq_len == 0 || seq_len > kv->max_seq_len || block_size == 0 || max_out == 0) return -4;

        const uint32_t effective_recent = kv->compressed ? kv->recent_window : recent_window;
        const uint32_t effective_block = kv->compressed ? kv->block_size : block_size;
        if (effective_recent == 0 || effective_block == 0) return -5;
        const uint32_t old_len = seq_len > effective_recent ? seq_len - effective_recent : 0;
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + effective_block - 1) / effective_block;
        *out_total_old_blocks = old_blocks;
        *out_count = 0;
        if (old_blocks == 0) return 0;

        cudaError_t err = cudaStreamSynchronize(ctx->decode_stream);
        if (err != cudaSuccess) return -6;

        const uint32_t head_dim = kv->head_dim;
        const float scale = head_dim == 64u ? 0.125f : 0.08838834764831845f;
        std::vector<float> q(head_dim);
        err = cudaMemcpy(q.data(), q_dev_f32, q.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -7;

        std::vector<std::pair<float, uint32_t>> scored;
        scored.reserve(old_blocks);
        const size_t elems_per_token = (size_t)kv->num_heads * (size_t)kv->head_dim;
        std::vector<__half> k_buf(head_dim);
        const bool use_q8_old_host =
            kv->compressed && kv->q8_old_backing && kv->d_q8_old_k && kv->d_q8_old_k_scale;
        const bool use_fp16_old_host =
            kv->compressed && kv->fp16_k_q4_v_old_backing && kv->d_fp16_old_k;
        std::vector<int8_t> q8_old_k_host;
        std::vector<float> q8_old_k_scale_host;
        std::vector<__half> fp16_old_k_host;
        if (use_q8_old_host) {
            q8_old_k_host.resize((size_t)old_len * elems_per_token);
            q8_old_k_scale_host.resize((size_t)old_len * (size_t)kv->num_heads);
            const size_t q8_base = ((size_t)seq_id * (size_t)kv->max_seq_len) * elems_per_token;
            const size_t scale_base = ((size_t)seq_id * (size_t)kv->max_seq_len) * (size_t)kv->num_heads;
            err = cudaMemcpy(q8_old_k_host.data(),
                             kv->d_q8_old_k + q8_base,
                             q8_old_k_host.size() * sizeof(int8_t),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) return -10;
            err = cudaMemcpy(q8_old_k_scale_host.data(),
                             kv->d_q8_old_k_scale + scale_base,
                             q8_old_k_scale_host.size() * sizeof(float),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) return -11;
        }
        if (use_fp16_old_host) {
            fp16_old_k_host.resize((size_t)old_len * elems_per_token);
            const size_t old_k_base = ((size_t)seq_id * (size_t)kv->max_seq_len) * elems_per_token;
            err = cudaMemcpy(fp16_old_k_host.data(),
                             kv->d_fp16_old_k + old_k_base,
                             fp16_old_k_host.size() * sizeof(__half),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) return -12;
        }

        for (uint32_t block = 0; block < old_blocks; ++block) {
            float score = 0.0f;
            if (use_q8_old_host || use_fp16_old_host) {
                const uint32_t start = block * effective_block;
                const uint32_t end = std::min(start + effective_block, old_len);
                if (end <= start) continue;
                for (uint32_t pos = start; pos < end; ++pos) {
                    const size_t base = (size_t)pos * elems_per_token;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        const __half k = use_fp16_old_host
                            ? fp16_old_k_host[base + d]
                            : __float2half_rn(
                                (float)q8_old_k_host[base + d]
                                * q8_old_k_scale_host[(size_t)pos * (size_t)kv->num_heads]);
                        score += q[d] * __half2float(k);
                    }
                }
                score /= (float)(end - start);
            } else if (kv->compressed) {
                const size_t offset = ((size_t)seq_id * (size_t)kv->max_blocks + (size_t)block) * elems_per_token;
                err = cudaMemcpy(k_buf.data(), kv->d_summary_k + offset, k_buf.size() * sizeof(__half), cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) return -8;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    score += q[d] * __half2float(k_buf[d]);
                }
            } else {
                const uint32_t start = block * effective_block;
                const uint32_t end = std::min(start + effective_block, old_len);
                std::vector<float> mean_k(head_dim, 0.0f);
                for (uint32_t pos = start; pos < end; ++pos) {
                    const size_t offset = ((size_t)seq_id * (size_t)kv->max_seq_len + (size_t)pos) * elems_per_token;
                    err = cudaMemcpy(k_buf.data(), kv->d_k + offset, k_buf.size() * sizeof(__half), cudaMemcpyDeviceToHost);
                    if (err != cudaSuccess) return -9;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        mean_k[d] += __half2float(k_buf[d]);
                    }
                }
                const float inv_count = 1.0f / (float)(end - start);
                for (uint32_t d = 0; d < head_dim; ++d) {
                    score += q[d] * (mean_k[d] * inv_count);
                }
            }
            scored.emplace_back(score * scale, block);
        }

        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
            if (a.first == b.first) return a.second < b.second;
            return a.first > b.first;
        });
        const uint32_t requested = top_blocks == 0 ? old_blocks : std::min(top_blocks, old_blocks);
        const uint32_t policy = block_select_policy_from_env();
        const uint32_t policy_max = policy == 0u
            ? requested
            : std::min(block_max_blocks_from_env(64u), old_blocks);
        const uint32_t capacity = std::min(std::min(policy_max, max_out), 64u);
        const uint32_t base_count = std::min(requested, capacity);
        const uint32_t min_take = std::min(std::max(block_min_blocks_from_env(), base_count), old_blocks);
        const float best_score = scored.empty() ? -1e30f : scored[0].first;
        const float score_delta = block_score_delta_from_env();
        const uint64_t anchor_mask = anchor_blocks_from_env();
        uint64_t force_include_low = 0ull, force_include_high = 0ull;
        uint64_t force_exclude_low = 0ull, force_exclude_high = 0ull;
        block_masks_from_env("M40LLM_KV_FORCE_INCLUDE_BLOCKS", &force_include_low, &force_include_high);
        block_masks_from_env("M40LLM_KV_FORCE_EXCLUDE_BLOCKS", &force_exclude_low, &force_exclude_high);
        std::vector<uint32_t> selected_blocks;
        selected_blocks.reserve(capacity);
        auto append_unique = [&](uint32_t block) {
            if (block >= old_blocks || selected_blocks.size() >= capacity) return;
            if (std::find(selected_blocks.begin(), selected_blocks.end(), block) == selected_blocks.end()) {
                selected_blocks.push_back(block);
            }
        };
        for (uint32_t i = 0; i < base_count; ++i) {
            append_unique(scored[i].second);
        }
        if (policy == 1u || policy == 4u) {
            const size_t snapshot = selected_blocks.size();
            for (size_t i = 0; i < snapshot; ++i) {
                const uint32_t block = selected_blocks[i];
                if (block > 0u) append_unique(block - 1u);
                append_unique(block + 1u);
            }
        }
        if (policy == 2u) {
            for (const auto& entry : scored) {
                if (selected_blocks.size() < min_take || entry.first >= best_score - score_delta) {
                    append_unique(entry.second);
                }
            }
        }
        if (policy == 6u || policy == 7u) {
            const float cutoff = base_count > 0 && base_count <= scored.size()
                ? scored[base_count - 1u].first - score_delta
                : best_score;
            for (const auto& entry : scored) {
                if (entry.first >= cutoff) {
                    append_unique(entry.second);
                }
            }
        }
        if (policy == 3u || policy == 4u) {
            for (uint32_t block = 0; block < old_blocks && block < 64u; ++block) {
                if ((anchor_mask & (1ull << block)) != 0ull) append_unique(block);
            }
        }
        if (policy == 5u) {
            for (uint32_t block = 0; block < old_blocks && block < 128u; ++block) {
                if (block_mask_contains(force_include_low, force_include_high, block)) append_unique(block);
            }
        }
        if (policy == 8u) {
            selected_blocks.clear();
            for (const auto& entry : scored) {
                if (block_mask_contains(force_include_low, force_include_high, entry.second)) {
                    append_unique(entry.second);
                }
            }
        }
        for (uint32_t i = 0; selected_blocks.size() < min_take && i < scored.size(); ++i) {
            append_unique(scored[i].second);
        }
        if (policy == 5u) {
            selected_blocks.erase(
                std::remove_if(selected_blocks.begin(), selected_blocks.end(), [&](uint32_t block) {
                    return block_mask_contains(force_exclude_low, force_exclude_high, block);
                }),
                selected_blocks.end());
        }
        if (selected_block_order_from_env() == 1u) {
            std::sort(selected_blocks.begin(), selected_blocks.end());
        } else if (selected_block_order_from_env() == 2u) {
            std::sort(selected_blocks.begin(), selected_blocks.end(), std::greater<uint32_t>());
        }
        auto score_for_block = [&](uint32_t block) {
            for (const auto& entry : scored) {
                if (entry.second == block) return entry.first;
            }
            return -1e30f;
        };
        const uint32_t selected = (uint32_t)selected_blocks.size();
        for (uint32_t i = 0; i < selected; ++i) {
            const uint32_t block = selected_blocks[i];
            const uint32_t start = block * effective_block;
            const uint32_t end = std::min(start + effective_block, old_len);
            out_blocks_host[i] = block;
            out_scores_host[i] = score_for_block(block);
            out_start_host[i] = start;
            out_end_host[i] = end;
        }
        *out_count = selected;
        return 0;
    }

    int m40llm_kvcache_debug_attention_telemetry(M40llmCudaContext* ctx,
                                                  const M40llmKVCache* kv,
                                                  uint32_t mode,
                                                  uint32_t seq_id,
                                                  const void* q_dev_f32,
                                                  uint32_t q_heads,
                                                  uint32_t seq_len,
                                                  uint32_t recent_window,
                                                  uint32_t block_size,
                                                  uint32_t top_blocks,
                                                  uint32_t needle_block,
                                                  M40llmAttentionTelemetry* out) {
        if (!ctx || !kv || !q_dev_f32 || !out) return -1;
        if (seq_id >= kv->max_batch_size) return -2;
        if (q_heads == 0 || kv->num_heads == 0 || kv->head_dim != 64) return -3;
        if (seq_len == 0 || seq_len > kv->max_seq_len) return -4;

        memset(out, 0, sizeof(M40llmAttentionTelemetry));
        out->needle_block_mass = -1.0f;
        for (uint32_t i = 0; i < 8; ++i) {
            out->top_entries[i].block_index = 0xffffffffu;
            out->top_entries[i].token_position = 0xffffffffu;
        }
        for (uint32_t i = 0; i < 64; ++i) {
            out->selected_block_masses[i].block_index = 0xffffffffu;
        }

        cudaError_t err = cudaStreamSynchronize(ctx->decode_stream);
        if (err != cudaSuccess) return -5;

        float q[64];
        err = cudaMemcpy(q, q_dev_f32, sizeof(q), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -6;

        struct Candidate {
            uint32_t group;
            uint32_t block;
            uint32_t token;
            float score;
            float prob;
        };
        std::vector<Candidate> candidates;
        const uint32_t effective_recent = kv->compressed ? kv->recent_window : recent_window;
        const uint32_t effective_block = kv->compressed ? kv->block_size : block_size;
        if (effective_recent == 0 || effective_block == 0) return -7;
        const uint32_t old_len = seq_len > effective_recent ? seq_len - effective_recent : 0;
        const uint32_t recent_start = seq_len - (seq_len < effective_recent ? seq_len : effective_recent);
        const uint32_t old_blocks = old_len == 0 ? 0 : (old_len + effective_block - 1) / effective_block;
        const uint32_t available_blocks = kv->compressed && old_blocks > kv->max_blocks ? kv->max_blocks : old_blocks;
        const size_t elems_per_token = (size_t)kv->num_heads * 64u;
        const float scale = 0.125f;
        __half k_buf[64];
        const bool use_q8_old_host =
            kv->compressed && kv->q8_old_backing && kv->d_q8_old_k && kv->d_q8_old_k_scale;
        const bool use_fp16_old_host =
            kv->compressed && kv->fp16_k_q4_v_old_backing && kv->d_fp16_old_k;
        std::vector<int8_t> q8_old_k_host;
        std::vector<float> q8_old_k_scale_host;
        std::vector<__half> fp16_old_k_host;
        if (use_q8_old_host) {
            q8_old_k_host.resize((size_t)old_len * elems_per_token);
            q8_old_k_scale_host.resize((size_t)old_len * (size_t)kv->num_heads);
            const size_t q8_base = ((size_t)seq_id * (size_t)kv->max_seq_len) * elems_per_token;
            const size_t scale_base = ((size_t)seq_id * (size_t)kv->max_seq_len) * (size_t)kv->num_heads;
            err = cudaMemcpy(q8_old_k_host.data(),
                             kv->d_q8_old_k + q8_base,
                             q8_old_k_host.size() * sizeof(int8_t),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) return -20;
            err = cudaMemcpy(q8_old_k_scale_host.data(),
                             kv->d_q8_old_k_scale + scale_base,
                             q8_old_k_scale_host.size() * sizeof(float),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) return -21;
        }
        if (use_fp16_old_host) {
            fp16_old_k_host.resize((size_t)old_len * elems_per_token);
            const size_t old_k_base = ((size_t)seq_id * (size_t)kv->max_seq_len) * elems_per_token;
            err = cudaMemcpy(fp16_old_k_host.data(),
                             kv->d_fp16_old_k + old_k_base,
                             fp16_old_k_host.size() * sizeof(__half),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) return -22;
        }

        auto dot_q = [&](const __half* ptr) -> float {
            float dot = 0.0f;
            for (uint32_t d = 0; d < 64; ++d) dot += q[d] * __half2float(ptr[d]);
            return dot * scale;
        };
        auto dense_k_ptr = [&](uint32_t token) -> const __half* {
            const size_t offset = ((size_t)seq_id * (size_t)kv->max_seq_len + (size_t)token) * elems_per_token;
            return kv->d_k + offset;
        };
        auto recent_k_ptr = [&](uint32_t token) -> const __half* {
            const uint32_t ring = token % kv->recent_window;
            const size_t offset = ((size_t)seq_id * (size_t)kv->recent_window + (size_t)ring) * elems_per_token;
            return kv->d_recent_k + offset;
        };
        auto copy_and_score = [&](const __half* dptr, float* score) -> int {
            cudaError_t copy_err = cudaMemcpy(k_buf, dptr, sizeof(k_buf), cudaMemcpyDeviceToHost);
            if (copy_err != cudaSuccess) return -1;
            *score = dot_q(k_buf);
            return 0;
        };
        auto score_q8_token = [&](uint32_t token, float* score) -> int {
            if (!use_q8_old_host || token >= old_len) return -1;
            const size_t base = (size_t)token * elems_per_token;
            const float q8_scale = q8_old_k_scale_host[(size_t)token * (size_t)kv->num_heads];
            float dot = 0.0f;
            for (uint32_t d = 0; d < 64; ++d) {
                const __half k = __float2half_rn(
                    (float)q8_old_k_host[base + d] * q8_scale);
                dot += q[d] * __half2float(k);
            }
            *score = dot * scale;
            return 0;
        };
        auto score_q8_block_mean = [&](uint32_t block, float* score) -> int {
            const uint32_t start = block * effective_block;
            const uint32_t end = std::min(start + effective_block, old_len);
            if (end <= start) return -1;
            float sum = 0.0f;
            for (uint32_t token = start; token < end; ++token) {
                float token_score = 0.0f;
                if (score_q8_token(token, &token_score) != 0) return -1;
                sum += token_score;
            }
            *score = sum / (float)(end - start);
            return 0;
        };
        auto score_fp16_old_token = [&](uint32_t token, float* score) -> int {
            if (!use_fp16_old_host || token >= old_len) return -1;
            const size_t base = (size_t)token * elems_per_token;
            *score = dot_q(fp16_old_k_host.data() + base);
            return 0;
        };
        auto score_fp16_old_block_mean = [&](uint32_t block, float* score) -> int {
            const uint32_t start = block * effective_block;
            const uint32_t end = std::min(start + effective_block, old_len);
            if (end <= start) return -1;
            float sum = 0.0f;
            for (uint32_t token = start; token < end; ++token) {
                float token_score = 0.0f;
                if (score_fp16_old_token(token, &token_score) != 0) return -1;
                sum += token_score;
            }
            *score = sum / (float)(end - start);
            return 0;
        };
        auto score_dense_block_mean = [&](uint32_t block, float* score) -> int {
            const uint32_t start = block * effective_block;
            const uint32_t end = std::min(start + effective_block, old_len);
            float mean_k[64] = {0.0f};
            for (uint32_t token = start; token < end; ++token) {
                cudaError_t copy_err = cudaMemcpy(k_buf, dense_k_ptr(token), sizeof(k_buf), cudaMemcpyDeviceToHost);
                if (copy_err != cudaSuccess) return -1;
                for (uint32_t d = 0; d < 64; ++d) mean_k[d] += __half2float(k_buf[d]);
            }
            const float inv = 1.0f / (float)(end - start);
            float dot = 0.0f;
            for (uint32_t d = 0; d < 64; ++d) dot += q[d] * mean_k[d] * inv;
            *score = dot * scale;
            return 0;
        };
        auto selected_blocks_by_score = [&](bool compressed_source, bool q8_source) -> std::vector<uint32_t> {
            std::vector<std::pair<float, uint32_t>> scored;
            for (uint32_t block = 0; block < available_blocks; ++block) {
                if (q8_source && use_fp16_old_host) {
                    float score = 0.0f;
                    if (score_fp16_old_block_mean(block, &score) != 0) continue;
                    scored.emplace_back(score, block);
                } else if (q8_source) {
                    float score = 0.0f;
                    if (score_q8_block_mean(block, &score) != 0) continue;
                    scored.emplace_back(score, block);
                } else if (compressed_source) {
                    const uint32_t count = kv->d_block_counts ? 1u : 0u;
                    (void)count;
                    const size_t offset = ((size_t)seq_id * (size_t)kv->max_blocks + (size_t)block) * elems_per_token;
                    float score = 0.0f;
                    if (copy_and_score(kv->d_summary_k + offset, &score) != 0) continue;
                    scored.emplace_back(score, block);
                } else {
                    float score = 0.0f;
                    if (score_dense_block_mean(block, &score) != 0) continue;
                    scored.emplace_back(score, block);
                }
            }
            std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
                if (a.first == b.first) return a.second < b.second;
                return a.first > b.first;
            });
            const uint32_t take = top_blocks == 0 || top_blocks > scored.size()
                ? (uint32_t)scored.size()
                : top_blocks;
            std::vector<uint32_t> selected;
            for (uint32_t i = 0; i < take; ++i) selected.push_back(scored[i].second);
            if (selected_block_order_from_env() == 1u) {
                std::sort(selected.begin(), selected.end());
            } else if (selected_block_order_from_env() == 2u) {
                std::sort(selected.begin(), selected.end(), std::greater<uint32_t>());
            }
            return selected;
        };

        if (mode == 1) {
            if (!kv->d_k) return -8;
            std::vector<uint32_t> selected = selected_blocks_by_score(false, false);
            for (uint32_t block : selected) {
                const uint32_t start = block * effective_block;
                const uint32_t end = std::min(start + effective_block, old_len);
                for (uint32_t token = start; token < end; ++token) {
                    float score = 0.0f;
                    if (copy_and_score(dense_k_ptr(token), &score) != 0) return -9;
                    candidates.push_back({2u, block, token, score, 0.0f});
                }
            }
            for (uint32_t token = recent_start; token < seq_len; ++token) {
                float score = 0.0f;
                if (copy_and_score(dense_k_ptr(token), &score) != 0) return -10;
                candidates.push_back({1u, 0xffffffffu, token, score, 0.0f});
            }
        } else if (mode == 5) {
            if (!kv->compressed || (!use_q8_old_host && !use_fp16_old_host)) return -17;
            std::vector<uint32_t> selected = selected_blocks_by_score(false, true);
            for (uint32_t block : selected) {
                const uint32_t start = block * effective_block;
                const uint32_t end = std::min(start + effective_block, old_len);
                for (uint32_t token = start; token < end; ++token) {
                    float score = 0.0f;
                    if (use_fp16_old_host) {
                        if (score_fp16_old_token(token, &score) != 0) return -18;
                    } else if (score_q8_token(token, &score) != 0) {
                        return -18;
                    }
                    candidates.push_back({2u, block, token, score, 0.0f});
                }
            }
            for (uint32_t token = recent_start; token < seq_len; ++token) {
                float score = 0.0f;
                if (copy_and_score(recent_k_ptr(token), &score) != 0) return -19;
                candidates.push_back({1u, 0xffffffffu, token, score, 0.0f});
            }
        } else {
            if (!kv->compressed) return -11;
            if (mode == 3 || mode == 4) {
                const uint32_t selected_top = mode == 3 ? 0u : top_blocks;
                const uint32_t saved_top = top_blocks;
                top_blocks = selected_top;
                std::vector<uint32_t> selected = selected_blocks_by_score(true, false);
                top_blocks = saved_top;
                for (uint32_t block : selected) {
                    const size_t offset = ((size_t)seq_id * (size_t)kv->max_blocks + (size_t)block) * elems_per_token;
                    float score = 0.0f;
                    if (copy_and_score(kv->d_summary_k + offset, &score) != 0) return -12;
                    candidates.push_back({3u, block, 0xffffffffu, score, 0.0f});
                    if (kv->representatives > 0 && kv->d_rep_k && kv->d_rep_positions) {
                        for (uint32_t r = 0; r < kv->representatives; ++r) {
                            const size_t pos_idx = ((size_t)seq_id * (size_t)kv->max_blocks + (size_t)block)
                                * (size_t)kv->representatives + (size_t)r;
                            uint32_t rep_pos = 0xffffffffu;
                            err = cudaMemcpy(&rep_pos, kv->d_rep_positions + pos_idx, sizeof(uint32_t), cudaMemcpyDeviceToHost);
                            if (err != cudaSuccess) return -13;
                            if (rep_pos == 0xffffffffu) continue;
                            const size_t rep_offset = (((size_t)seq_id * (size_t)kv->max_blocks + (size_t)block)
                                * (size_t)kv->representatives + (size_t)r) * elems_per_token;
                            if (copy_and_score(kv->d_rep_k + rep_offset, &score) != 0) return -14;
                            candidates.push_back({4u, block, rep_pos, score, 0.0f});
                        }
                    }
                }
            }
            for (uint32_t token = recent_start; token < seq_len; ++token) {
                float score = 0.0f;
                if (copy_and_score(recent_k_ptr(token), &score) != 0) return -15;
                candidates.push_back({1u, 0xffffffffu, token, score, 0.0f});
            }
        }

        if (candidates.empty()) return -16;
        float max_score = -1e30f;
        for (const Candidate& c : candidates) max_score = std::max(max_score, c.score);
        float denom = 0.0f;
        for (Candidate& c : candidates) {
            c.prob = expf(c.score - max_score);
            denom += c.prob;
        }
        if (denom <= 0.0f) denom = 1.0f;
        for (Candidate& c : candidates) c.prob /= denom;

        struct MutableStats {
            float mass = 0.0f;
            float max_logit = -1e30f;
            float sum_logit = 0.0f;
            uint32_t count = 0;
        };
        MutableStats stats[5];
        std::vector<MutableStats> block_stats(available_blocks);
        float needle_mass = 0.0f;
        bool have_needle = needle_block != 0xffffffffu;
        for (const Candidate& c : candidates) {
            const uint32_t idx = c.group <= 4 ? c.group : 0;
            stats[idx].mass += c.prob;
            stats[idx].max_logit = std::max(stats[idx].max_logit, c.score);
            stats[idx].sum_logit += c.score;
            stats[idx].count++;
            if (have_needle && c.block == needle_block) needle_mass += c.prob;
            if (c.group == 2u && c.block < block_stats.size()) {
                MutableStats& b = block_stats[c.block];
                b.mass += c.prob;
                b.max_logit = std::max(b.max_logit, c.score);
                b.sum_logit += c.score;
                b.count++;
            }
        }
        auto finish = [](const MutableStats& s) -> M40llmAttentionGroupStats {
            M40llmAttentionGroupStats out_stats;
            out_stats.prob_mass = s.mass;
            out_stats.logit_max = s.count ? s.max_logit : 0.0f;
            out_stats.logit_mean = s.count ? s.sum_logit / (float)s.count : 0.0f;
            out_stats.count = s.count;
            return out_stats;
        };
        out->other = finish(stats[0]);
        out->recent = finish(stats[1]);
        out->selected_old_exact = finish(stats[2]);
        out->summary = finish(stats[3]);
        out->representatives = finish(stats[4]);
        if (have_needle) out->needle_block_mass = needle_mass;
        for (uint32_t block = 0;
             block < block_stats.size() && out->selected_block_mass_count < 64;
             ++block) {
            const MutableStats& b = block_stats[block];
            if (b.count == 0) continue;
            M40llmAttentionBlockMass& dst =
                out->selected_block_masses[out->selected_block_mass_count++];
            dst.block_index = block;
            dst.prob_mass = b.mass;
            dst.logit_max = b.max_logit;
            dst.logit_mean = b.sum_logit / (float)b.count;
            dst.count = b.count;
        }

        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            if (a.prob == b.prob) return a.score > b.score;
            return a.prob > b.prob;
        });
        out->top_entry_count = (uint32_t)std::min<size_t>(8, candidates.size());
        for (uint32_t i = 0; i < out->top_entry_count; ++i) {
            out->top_entries[i].group = candidates[i].group;
            out->top_entries[i].block_index = candidates[i].block;
            out->top_entries[i].token_position = candidates[i].token;
            out->top_entries[i].score = candidates[i].score;
            out->top_entries[i].probability = candidates[i].prob;
        }
        return 0;
    }

    __global__ void build_compressed_recent_from_dense_kernel(
        const __half* __restrict__ dense_k,
        const __half* __restrict__ dense_v,
        __half* __restrict__ recent_k,
        __half* __restrict__ recent_v,
        uint32_t* __restrict__ seq_map,
        uint32_t max_seq_len,
        uint32_t recent_window,
        uint32_t seq_len,
        size_t elems_per_token,
        size_t total) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= total) return;
        const size_t recent_elems = (size_t)recent_window * elems_per_token;
        const uint32_t seq = (uint32_t)(i / recent_elems);
        const size_t rem = i % recent_elems;
        const uint32_t recent_idx = (uint32_t)(rem / elems_per_token);
        const size_t elem = rem % elems_per_token;
        const uint32_t recent_count = seq_len < recent_window ? seq_len : recent_window;
        if (recent_idx < recent_count) {
            const uint32_t pos = seq_len - recent_count + recent_idx;
            const uint32_t ring = pos % recent_window;
            const size_t dst = ((size_t)seq * (size_t)recent_window + (size_t)ring) * elems_per_token + elem;
            const size_t src = ((size_t)seq * (size_t)max_seq_len + (size_t)pos) * elems_per_token + elem;
            recent_k[dst] = dense_k[src];
            recent_v[dst] = dense_v[src];
        }
        if (rem == 0) {
            seq_map[seq] = seq_len;
        }
    }

    __global__ void build_compressed_summaries_from_dense_kernel(
        const __half* __restrict__ dense_k,
        const __half* __restrict__ dense_v,
        float* __restrict__ summary_k_acc,
        float* __restrict__ summary_v_acc,
        __half* __restrict__ summary_k,
        __half* __restrict__ summary_v,
        uint32_t* __restrict__ block_counts,
        uint32_t max_seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t max_blocks,
        uint32_t seq_len,
        size_t elems_per_token,
        size_t total) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= total) return;
        const size_t summary_elems_per_seq = (size_t)max_blocks * elems_per_token;
        const uint32_t seq = (uint32_t)(i / summary_elems_per_seq);
        const size_t rem = i % summary_elems_per_seq;
        const uint32_t block = (uint32_t)(rem / elems_per_token);
        const size_t elem = rem % elems_per_token;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        if (block >= max_blocks || block * block_size >= old_len) return;
        const uint32_t start = block * block_size;
        const uint32_t end = min(start + block_size, old_len);
        float k_sum = 0.0f;
        float v_sum = 0.0f;
        for (uint32_t pos = start; pos < end; ++pos) {
            const size_t src = ((size_t)seq * (size_t)max_seq_len + (size_t)pos) * elems_per_token + elem;
            k_sum += __half2float(dense_k[src]);
            v_sum += __half2float(dense_v[src]);
        }
        const uint32_t count = end - start;
        const size_t dst = ((size_t)seq * (size_t)max_blocks + (size_t)block) * elems_per_token + elem;
        summary_k_acc[dst] = k_sum;
        summary_v_acc[dst] = v_sum;
        summary_k[dst] = __float2half_rn(k_sum / (float)count);
        summary_v[dst] = __float2half_rn(v_sum / (float)count);
        if (elem == 0) {
            block_counts[(size_t)seq * (size_t)max_blocks + (size_t)block] = count;
        }
    }

    __global__ void build_compressed_representatives_from_dense_kernel(
        const __half* __restrict__ dense_k,
        const __half* __restrict__ dense_v,
        __half* __restrict__ rep_k,
        __half* __restrict__ rep_v,
        uint32_t* __restrict__ rep_positions,
        uint32_t max_seq_len,
        uint32_t recent_window,
        uint32_t block_size,
        uint32_t max_blocks,
        uint32_t representatives,
        uint32_t representative_policy,
        uint32_t seq_len,
        size_t elems_per_token,
        size_t total) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= total || representatives == 0) return;
        const size_t rep_elems_per_seq = (size_t)max_blocks * (size_t)representatives * elems_per_token;
        const uint32_t seq = (uint32_t)(i / rep_elems_per_seq);
        const size_t rem = i % rep_elems_per_seq;
        const uint32_t block = (uint32_t)(rem / ((size_t)representatives * elems_per_token));
        const size_t block_rem = rem % ((size_t)representatives * elems_per_token);
        const uint32_t rep_slot = (uint32_t)(block_rem / elems_per_token);
        const size_t elem = block_rem % elems_per_token;
        const uint32_t old_len = seq_len > recent_window ? seq_len - recent_window : 0;
        if (block >= max_blocks || rep_slot >= representatives || block * block_size >= old_len) return;

        const uint32_t start = block * block_size;
        const uint32_t end = min(start + block_size, old_len);
        const uint32_t count = end - start;
        uint32_t old_pos = 0xffffffffu;
        if (representative_policy == 1u) {
            const uint32_t range_start = (uint32_t)(((uint64_t)rep_slot * (uint64_t)block_size + (uint64_t)representatives - 1ull) / (uint64_t)representatives);
            const uint32_t range_end = min(block_size, (uint32_t)((((uint64_t)rep_slot + 1ull) * (uint64_t)block_size + (uint64_t)representatives - 1ull) / (uint64_t)representatives));
            if (range_start < count) {
                old_pos = start + min(count, range_end) - 1u;
            }
        } else {
            if (rep_slot < count) {
                const uint32_t last_with_slot =
                    (count - 1u) - ((count - 1u + representatives - rep_slot) % representatives);
                old_pos = start + last_with_slot;
            }
        }
        const size_t rep_pos_idx = ((size_t)seq * (size_t)max_blocks + (size_t)block) * (size_t)representatives + (size_t)rep_slot;
        if (old_pos == 0xffffffffu) {
            if (elem == 0) rep_positions[rep_pos_idx] = 0xffffffffu;
            return;
        }
        const size_t src = ((size_t)seq * (size_t)max_seq_len + (size_t)old_pos) * elems_per_token + elem;
        const size_t dst = (((size_t)seq * (size_t)max_blocks + (size_t)block)
            * (size_t)representatives + (size_t)rep_slot) * elems_per_token + elem;
        rep_k[dst] = dense_k[src];
        rep_v[dst] = dense_v[src];
        if (elem == 0) {
            rep_positions[rep_pos_idx] = old_pos;
        }
    }

    int m40llm_kvcache_build_compressed_from_dense(M40llmCudaContext* ctx,
                                                    M40llmKVCache* compressed,
                                                    const M40llmKVCache* dense,
                                                    uint32_t seq_len) {
        if (!ctx || !compressed || !dense) return -1;
        if (!compressed->compressed || dense->compressed) return -2;
        if (compressed->max_batch_size != dense->max_batch_size ||
            compressed->num_heads != dense->num_heads ||
            compressed->head_dim != dense->head_dim) return -3;
        if (seq_len > compressed->max_seq_len || seq_len > dense->max_seq_len) return -4;
        const size_t elems_per_token = (size_t)compressed->num_heads * (size_t)compressed->head_dim;
        const size_t recent_elems = (size_t)compressed->max_batch_size * (size_t)compressed->recent_window * elems_per_token;
        const size_t summary_elems = (size_t)compressed->max_batch_size * (size_t)compressed->max_blocks * elems_per_token;
        const size_t rep_elems = (size_t)compressed->max_batch_size * (size_t)compressed->max_blocks * (size_t)compressed->representatives * elems_per_token;
        const size_t block_count_size = (size_t)compressed->max_batch_size * (size_t)compressed->max_blocks * sizeof(uint32_t);
        const size_t rep_position_size = (size_t)compressed->max_batch_size * (size_t)compressed->max_blocks * (size_t)compressed->representatives * sizeof(uint32_t);
        cudaMemsetAsync(compressed->d_recent_k, 0, recent_elems * sizeof(__half), ctx->decode_stream);
        cudaMemsetAsync(compressed->d_recent_v, 0, recent_elems * sizeof(__half), ctx->decode_stream);
        cudaMemsetAsync(compressed->d_summary_k_acc, 0, summary_elems * sizeof(float), ctx->decode_stream);
        cudaMemsetAsync(compressed->d_summary_v_acc, 0, summary_elems * sizeof(float), ctx->decode_stream);
        cudaMemsetAsync(compressed->d_summary_k, 0, summary_elems * sizeof(__half), ctx->decode_stream);
        cudaMemsetAsync(compressed->d_summary_v, 0, summary_elems * sizeof(__half), ctx->decode_stream);
        cudaMemsetAsync(compressed->d_block_counts, 0, block_count_size, ctx->decode_stream);
        if (compressed->representatives > 0) {
            cudaMemsetAsync(compressed->d_rep_k, 0, rep_elems * sizeof(__half), ctx->decode_stream);
            cudaMemsetAsync(compressed->d_rep_v, 0, rep_elems * sizeof(__half), ctx->decode_stream);
            cudaMemsetAsync(compressed->d_rep_positions, 0xff, rep_position_size, ctx->decode_stream);
        }
        cudaMemsetAsync(compressed->d_seq_map, 0, (size_t)compressed->max_batch_size * sizeof(uint32_t), ctx->decode_stream);
        const int threads = 256;
        if (recent_elems > 0) {
            const int blocks = (int)((recent_elems + threads - 1) / threads);
            build_compressed_recent_from_dense_kernel<<<blocks, threads, 0, ctx->decode_stream>>>(
                dense->d_k, dense->d_v, compressed->d_recent_k, compressed->d_recent_v,
                compressed->d_seq_map, dense->max_seq_len, compressed->recent_window,
                seq_len, elems_per_token, recent_elems);
        }
        if (summary_elems > 0) {
            const int blocks = (int)((summary_elems + threads - 1) / threads);
            build_compressed_summaries_from_dense_kernel<<<blocks, threads, 0, ctx->decode_stream>>>(
                dense->d_k, dense->d_v, compressed->d_summary_k_acc, compressed->d_summary_v_acc,
                compressed->d_summary_k, compressed->d_summary_v, compressed->d_block_counts,
                dense->max_seq_len, compressed->recent_window, compressed->block_size,
                compressed->max_blocks, seq_len, elems_per_token, summary_elems);
        }
        if (rep_elems > 0) {
            const int blocks = (int)((rep_elems + threads - 1) / threads);
            build_compressed_representatives_from_dense_kernel<<<blocks, threads, 0, ctx->decode_stream>>>(
                dense->d_k, dense->d_v, compressed->d_rep_k, compressed->d_rep_v,
                compressed->d_rep_positions, dense->max_seq_len, compressed->recent_window,
                compressed->block_size, compressed->max_blocks, compressed->representatives,
                compressed->representative_policy, seq_len, elems_per_token, rep_elems);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -5;
        err = cudaStreamSynchronize(ctx->decode_stream);
        if (err != cudaSuccess) return -6;
        return 0;
    }

    void m40llm_kvcache_destroy(M40llmKVCache* kv) {
        if (!kv) return;
        if (kv->d_k) cudaFree(kv->d_k);
        if (kv->d_v) cudaFree(kv->d_v);
        if (kv->d_seq_map) cudaFree(kv->d_seq_map);
        if (kv->d_recent_k) cudaFree(kv->d_recent_k);
        if (kv->d_recent_v) cudaFree(kv->d_recent_v);
        if (kv->d_summary_k_acc) cudaFree(kv->d_summary_k_acc);
        if (kv->d_summary_v_acc) cudaFree(kv->d_summary_v_acc);
        if (kv->d_summary_k) cudaFree(kv->d_summary_k);
        if (kv->d_summary_v) cudaFree(kv->d_summary_v);
        if (kv->d_block_counts) cudaFree(kv->d_block_counts);
        if (kv->d_rep_k) cudaFree(kv->d_rep_k);
        if (kv->d_rep_v) cudaFree(kv->d_rep_v);
        if (kv->d_rep_positions) cudaFree(kv->d_rep_positions);
        if (kv->d_q8_old_k) cudaFree(kv->d_q8_old_k);
        if (kv->d_q8_old_v) cudaFree(kv->d_q8_old_v);
        if (kv->d_q8_old_k_scale) cudaFree(kv->d_q8_old_k_scale);
        if (kv->d_q8_old_v_scale) cudaFree(kv->d_q8_old_v_scale);
        if (kv->d_fp16_old_k) cudaFree(kv->d_fp16_old_k);
        if (kv->d_q4_old_v) cudaFree(kv->d_q4_old_v);
        if (kv->d_q4_old_v_scale) cudaFree(kv->d_q4_old_v_scale);
        delete kv;
    }

    __global__ void rope_f32_kernel(
        float* __restrict__ q,
        float* __restrict__ k,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t past_len,
        float freq_base,
        float freq_scale,
        uint32_t rope_layout) {
        const uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t pairs_per_row = (num_heads * head_dim) / 2;
        const uint32_t total_pairs = rows * pairs_per_row;
        if (pair_idx >= total_pairs) return;

        const uint32_t row = pair_idx / pairs_per_row;
        const uint32_t pair_in_row = pair_idx % pairs_per_row;
        const uint32_t head = pair_in_row / (head_dim / 2);
        const uint32_t offset_in_head = pair_in_row % (head_dim / 2);

        size_t i0 = 0;
        size_t i1 = 0;
        rope_pair_indices(head, offset_in_head, head_dim, rope_layout, &i0, &i1);
        const size_t base = (size_t)row * (size_t)num_heads * (size_t)head_dim;
        const float pos = static_cast<float>(past_len + row) * freq_scale;
        const float theta = pos * powf(
            freq_base,
            -2.0f * static_cast<float>(offset_in_head) / static_cast<float>(head_dim));
        const float c = cosf(theta);
        const float s = sinf(theta);

        const float q0 = q[base + i0];
        const float q1 = q[base + i1];
        q[base + i0] = q0 * c - q1 * s;
        q[base + i1] = q0 * s + q1 * c;

        const float k0 = k[base + i0];
        const float k1 = k[base + i1];
        k[base + i0] = k0 * c - k1 * s;
        k[base + i1] = k0 * s + k1 * c;
    }

    __global__ void rope_f32_inplace_kernel(
        float* __restrict__ x,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t past_len,
        float freq_base,
        float freq_scale,
        uint32_t rope_layout) {
        const uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t pairs_per_row = (num_heads * head_dim) / 2;
        const uint32_t total_pairs = rows * pairs_per_row;
        if (pair_idx >= total_pairs) return;

        const uint32_t row = pair_idx / pairs_per_row;
        const uint32_t pair_in_row = pair_idx % pairs_per_row;
        const uint32_t head = pair_in_row / (head_dim / 2);
        const uint32_t offset_in_head = pair_in_row % (head_dim / 2);
        size_t i0 = 0;
        size_t i1 = 0;
        rope_pair_indices(head, offset_in_head, head_dim, rope_layout, &i0, &i1);
        const size_t base = (size_t)row * (size_t)num_heads * (size_t)head_dim;
        const float pos = static_cast<float>(past_len + row) * freq_scale;
        const float theta = pos * powf(
            freq_base,
            -2.0f * static_cast<float>(offset_in_head) / static_cast<float>(head_dim));
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x0 = x[base + i0];
        const float x1 = x[base + i1];
        x[base + i0] = x0 * c - x1 * s;
        x[base + i1] = x0 * s + x1 * c;
    }

    __global__ void rope_f32_inplace_position_dev_kernel(
        float* __restrict__ x,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        const uint32_t* __restrict__ past_len_dev,
        float freq_base,
        float freq_scale,
        uint32_t rope_layout) {
        if (!past_len_dev) return;
        const uint32_t past_len = *past_len_dev;
        const uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t pairs_per_row = (num_heads * head_dim) / 2;
        const uint32_t total_pairs = rows * pairs_per_row;
        if (pair_idx >= total_pairs) return;

        const uint32_t row = pair_idx / pairs_per_row;
        const uint32_t pair_in_row = pair_idx % pairs_per_row;
        const uint32_t head = pair_in_row / (head_dim / 2);
        const uint32_t offset_in_head = pair_in_row % (head_dim / 2);
        size_t i0 = 0;
        size_t i1 = 0;
        rope_pair_indices(head, offset_in_head, head_dim, rope_layout, &i0, &i1);
        const size_t base = (size_t)row * (size_t)num_heads * (size_t)head_dim;
        const float pos = static_cast<float>(past_len + row) * freq_scale;
        const float theta = pos * powf(
            freq_base,
            -2.0f * static_cast<float>(offset_in_head) / static_cast<float>(head_dim));
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x0 = x[base + i0];
        const float x1 = x[base + i1];
        x[base + i0] = x0 * c - x1 * s;
        x[base + i1] = x0 * s + x1 * c;
    }

    int m40llm_rope_f32_layout_async(
        M40llmCudaContext* ctx,
        float* q,
        float* k,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t past_len,
        float freq_base,
        float freq_scale,
        uint32_t rope_layout);
    int m40llm_rope_f32_inplace_layout_async(
        M40llmCudaContext* ctx,
        float* x,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t past_len,
        float freq_base,
        float freq_scale,
        uint32_t rope_layout);
    int m40llm_rope_f32_inplace_position_dev_layout_async(
        M40llmCudaContext* ctx,
        float* x,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        const uint32_t* past_len_dev,
        float freq_base,
        float freq_scale,
        uint32_t rope_layout);

    int m40llm_rope_f32(
        M40llmCudaContext* ctx,
        float* q,
        float* k,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t past_len,
        float freq_base,
        float freq_scale) {
        int rc = m40llm_rope_f32_async(
            ctx, q, k, rows, num_heads, head_dim, past_len, freq_base, freq_scale);
        if (rc != 0) return rc;
        cudaError_t err = cudaStreamSynchronize(ctx->decode_stream);
        return err == cudaSuccess ? 0 : -4;
    }

    int m40llm_rope_f32_async(
        M40llmCudaContext* ctx,
        float* q,
        float* k,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t past_len,
        float freq_base,
        float freq_scale) {
        return m40llm_rope_f32_layout_async(
            ctx, q, k, rows, num_heads, head_dim, past_len, freq_base, freq_scale,
            M40LLM_ROPE_LAYOUT_ADJACENT);
    }

    int m40llm_rope_f32_layout_async(
        M40llmCudaContext* ctx,
        float* q,
        float* k,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t past_len,
        float freq_base,
        float freq_scale,
        uint32_t rope_layout) {
        if (!ctx || !q || !k || head_dim == 0 || num_heads == 0) return -1;
        if (head_dim % 2 != 0) return -2;
        const uint32_t pairs_per_row = (num_heads * head_dim) / 2;
        const uint32_t total_pairs = rows * pairs_per_row;
        const int threads_per_block = 256;
        const int blocks = (total_pairs + threads_per_block - 1) / threads_per_block;
        rope_f32_kernel<<<blocks, threads_per_block, 0, ctx->decode_stream>>>(
            q, k, rows, num_heads, head_dim, past_len, freq_base, freq_scale, rope_layout);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -3;
        return 0;
    }

    int m40llm_rope_f32_inplace(
        M40llmCudaContext* ctx,
        float* x,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t past_len,
        float freq_base,
        float freq_scale) {
        int rc = m40llm_rope_f32_inplace_async(
            ctx, x, rows, num_heads, head_dim, past_len, freq_base, freq_scale);
        if (rc != 0) return rc;
        cudaError_t err = cudaStreamSynchronize(ctx->decode_stream);
        return err == cudaSuccess ? 0 : -4;
    }

    int m40llm_rope_f32_inplace_async(
        M40llmCudaContext* ctx,
        float* x,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t past_len,
        float freq_base,
        float freq_scale) {
        return m40llm_rope_f32_inplace_layout_async(
            ctx, x, rows, num_heads, head_dim, past_len, freq_base, freq_scale,
            M40LLM_ROPE_LAYOUT_ADJACENT);
    }

    int m40llm_rope_f32_inplace_layout_async(
        M40llmCudaContext* ctx,
        float* x,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t past_len,
        float freq_base,
        float freq_scale,
        uint32_t rope_layout) {
        if (!ctx || !x || head_dim == 0 || num_heads == 0) return -1;
        if (head_dim % 2 != 0) return -2;
        const uint32_t pairs_per_row = (num_heads * head_dim) / 2;
        const uint32_t total_pairs = rows * pairs_per_row;
        const int threads_per_block = 256;
        const int blocks = (total_pairs + threads_per_block - 1) / threads_per_block;
        rope_f32_inplace_kernel<<<blocks, threads_per_block, 0, ctx->decode_stream>>>(
            x, rows, num_heads, head_dim, past_len, freq_base, freq_scale, rope_layout);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -3;
        return 0;
    }

    int m40llm_rope_f32_inplace_position_dev_async(
        M40llmCudaContext* ctx,
        float* x,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        const uint32_t* past_len_dev,
        float freq_base,
        float freq_scale) {
        return m40llm_rope_f32_inplace_position_dev_layout_async(
            ctx, x, rows, num_heads, head_dim, past_len_dev, freq_base, freq_scale,
            M40LLM_ROPE_LAYOUT_ADJACENT);
    }

    int m40llm_rope_f32_inplace_position_dev_layout_async(
        M40llmCudaContext* ctx,
        float* x,
        uint32_t rows,
        uint32_t num_heads,
        uint32_t head_dim,
        const uint32_t* past_len_dev,
        float freq_base,
        float freq_scale,
        uint32_t rope_layout) {
        if (!ctx || !x || !past_len_dev || head_dim == 0 || num_heads == 0) return -1;
        if (head_dim % 2 != 0) return -2;
        const uint32_t pairs_per_row = (num_heads * head_dim) / 2;
        const uint32_t total_pairs = rows * pairs_per_row;
        const int threads_per_block = 256;
        const int blocks = (total_pairs + threads_per_block - 1) / threads_per_block;
        rope_f32_inplace_position_dev_kernel<<<blocks, threads_per_block, 0, ctx->decode_stream>>>(
            x, rows, num_heads, head_dim, past_len_dev, freq_base, freq_scale, rope_layout);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return -3;
        return 0;
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
    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    residual_add_f32<<<blocks, threads_per_block>>>(a, b, out, size);
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

    struct M40llmPersistentDecodeState {
        volatile uint32_t stop;
        volatile uint32_t command;
        volatile uint32_t status;
        volatile uint32_t command_id;
        const float* in;
        float* out;
        uint32_t n;
        uint32_t iterations;
        float scale;
        float bias;
    };

    struct M40llmPersistentDecode {
        M40llmPersistentDecodeState* host_state;
        M40llmPersistentDecodeState* device_state;
        cudaStream_t stream;
        uint32_t next_command_id;
        bool running;
    };

    __global__ void persistent_decode_vec_kernel(M40llmPersistentDecodeState* state) {
        while (state->stop == 0) {
            if (state->command == 1 && state->status == 1) {
                const float* in = state->in;
                float* out = state->out;
                const uint32_t n = state->n;
                const uint32_t iterations = state->iterations == 0 ? 1 : state->iterations;
                const float scale = state->scale;
                const float bias = state->bias;

                for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
                    float x = in[i];
                    for (uint32_t it = 0; it < iterations; ++it) {
                        x = x * scale + bias;
                    }
                    out[i] = x;
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    __threadfence_system();
                    state->command = 0;
                    state->status = 2;
                    __threadfence_system();
                }
            } else {
                for (volatile int spin = 0; spin < 64; ++spin) {}
            }
        }
    }

    static int persistent_decode_stop_impl(M40llmCudaContext* ctx) {
        if (!ctx) return -1;
        M40llmPersistentDecode* session = ctx->persistent_decode;
        if (!session) return 0;
        if (session->host_state) {
            session->host_state->stop = 1;
            __sync_synchronize();
        }
        if (session->running) {
            cudaError_t err = cudaStreamSynchronize(session->stream);
            if (err != cudaSuccess) return -2;
            session->running = false;
        }
        if (session->stream) {
            cudaStreamDestroy(session->stream);
        }
        if (session->host_state) {
            cudaFreeHost(session->host_state);
        }
        delete session;
        ctx->persistent_decode = nullptr;
        return 0;
    }

    int m40llm_start_persistent_decode(M40llmCudaContext* ctx) {
        if (!ctx) return -1;
        if (ensure_device(ctx) != 0) return -2;
        if (ctx->persistent_decode) return 0;

        M40llmPersistentDecode* session = new M40llmPersistentDecode();
        memset(session, 0, sizeof(M40llmPersistentDecode));
        session->next_command_id = 1;

        cudaError_t err = cudaHostAlloc(
            reinterpret_cast<void**>(&session->host_state),
            sizeof(M40llmPersistentDecodeState),
            cudaHostAllocMapped);
        if (err != cudaSuccess) {
            delete session;
            return -3;
        }
        memset(session->host_state, 0, sizeof(M40llmPersistentDecodeState));
        err = cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&session->device_state),
            session->host_state,
            0);
        if (err != cudaSuccess) {
            cudaFreeHost(session->host_state);
            delete session;
            return -4;
        }
        err = cudaStreamCreateWithPriority(
            &session->stream,
            cudaStreamNonBlocking,
            ctx->decode_priority);
        if (err != cudaSuccess) {
            cudaFreeHost(session->host_state);
            delete session;
            return -5;
        }

        persistent_decode_vec_kernel<<<1, 256, 0, session->stream>>>(session->device_state);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaStreamDestroy(session->stream);
            cudaFreeHost(session->host_state);
            delete session;
            return -6;
        }
        session->running = true;
        ctx->persistent_decode = session;
        return 0;
    }

    int m40llm_stop_persistent_decode(M40llmCudaContext* ctx) {
        return persistent_decode_stop_impl(ctx);
    }

    int m40llm_persistent_decode_submit_vec(
        M40llmCudaContext* ctx,
        const void* d_in_f32,
        void* d_out_f32,
        uint32_t n,
        float scale,
        float bias,
        uint32_t iterations,
        uint32_t* out_command_id) {
        if (!ctx || !d_in_f32 || !d_out_f32 || !out_command_id) return -1;
        if (n == 0) return -2;
        if (ensure_device(ctx) != 0) return -3;
        M40llmPersistentDecode* session = ctx->persistent_decode;
        if (!session || !session->host_state || !session->running) return -4;
        M40llmPersistentDecodeState* state = session->host_state;
        if (state->status == 1 || state->command == 1) return -5;

        const uint32_t command_id = session->next_command_id++;
        state->in = reinterpret_cast<const float*>(d_in_f32);
        state->out = reinterpret_cast<float*>(d_out_f32);
        state->n = n;
        state->iterations = iterations == 0 ? 1 : iterations;
        state->scale = scale;
        state->bias = bias;
        state->command_id = command_id;
        state->status = 1;
        __sync_synchronize();
        state->command = 1;
        __sync_synchronize();
        *out_command_id = command_id;
        return 0;
    }

    int m40llm_persistent_decode_poll(
        M40llmCudaContext* ctx,
        uint32_t* out_status,
        uint32_t* out_command_id) {
        if (!ctx || !out_status || !out_command_id) return -1;
        M40llmPersistentDecode* session = ctx->persistent_decode;
        if (!session || !session->host_state) return -2;
        __sync_synchronize();
        *out_status = session->host_state->status;
        *out_command_id = session->host_state->command_id;
        return 0;
    }

    void m40llm_destroy_context(M40llmCudaContext* ctx) {
        if (!ctx) return;
        ensure_device(ctx);
        if (ctx->prefill_stream) cudaStreamSynchronize(ctx->prefill_stream);
        if (ctx->decode_stream) cudaStreamSynchronize(ctx->decode_stream);
        persistent_decode_stop_impl(ctx);
    #ifdef M40LLM_HAVE_CUBLAS
        cublasDestroy(ctx->cublas);
    #endif
        if (ctx->prefill_waits_decode_event) cudaEventDestroy(ctx->prefill_waits_decode_event);
        if (ctx->decode_waits_prefill_event) cudaEventDestroy(ctx->decode_waits_prefill_event);
        for (cudaEvent_t event : ctx->retained_wait_events) {
            if (event) cudaEventDestroy(event);
        }
        cudaStreamDestroy(ctx->prefill_stream);
        cudaStreamDestroy(ctx->decode_stream);
        delete ctx;
    }
} // extern "C"
