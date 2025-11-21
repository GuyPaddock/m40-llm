// cuda/stub.c - stubbed CUDA FFI for environments without NVCC/CUDA
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct M40llmCudaContext { int device_id; } M40llmCudaContext;
typedef struct M40llmKVCache { int _stub; } M40llmKVCache;

int m40llm_device_malloc(M40llmCudaContext* ctx, size_t bytes, void** out_ptr) {
    (void)ctx; (void)bytes; (void)out_ptr; return -1;
}
int m40llm_device_free(M40llmCudaContext* ctx, void* ptr) {
    (void)ctx; (void)ptr; return -1;
}
int m40llm_memcpy_h2d(M40llmCudaContext* ctx, void* dst_device, const void* src_host, size_t bytes) {
    (void)ctx; (void)dst_device; (void)src_host; (void)bytes; return -1;
}
int m40llm_memcpy_d2h(M40llmCudaContext* ctx, void* dst_host, const void* src_device, size_t bytes) {
    (void)ctx; (void)dst_host; (void)src_device; (void)bytes; return -1;
}

M40llmCudaContext* m40llm_create_context(int device_id) {
    (void)device_id; return (M40llmCudaContext*)0x1; // non-null stub
}
void m40llm_destroy_context(M40llmCudaContext* ctx) { (void)ctx; }

int m40llm_upload_weights(M40llmCudaContext* ctx, const void* host_ptr, size_t num_bytes, void** out_device_ptr) {
    (void)ctx; (void)host_ptr; (void)num_bytes; (void)out_device_ptr; return -1;
}

int m40llm_gemm_f16_storage_f32_compute(M40llmCudaContext* ctx, const void* d_A, const void* d_B, void* d_C, int M, int N, int K) {
    (void)ctx; (void)d_A; (void)d_B; (void)d_C; (void)M; (void)N; (void)K; return -1;
}

M40llmKVCache* m40llm_kvcache_create(M40llmCudaContext* ctx, uint32_t max_seq_len, uint32_t max_batch_size, uint32_t num_heads, uint32_t head_dim) {
    (void)ctx; (void)max_seq_len; (void)max_batch_size; (void)num_heads; (void)head_dim; return (M40llmKVCache*)0x1;
}
int m40llm_kvcache_append_token(M40llmCudaContext* ctx, M40llmKVCache* kv, uint32_t seq_id, const void* k_dev, const void* v_dev) {
    (void)ctx; (void)kv; (void)seq_id; (void)k_dev; (void)v_dev; return -1;
}
void m40llm_kvcache_destroy(M40llmKVCache* kv) { (void)kv; }

int m40llm_start_persistent_decode(M40llmCudaContext* ctx) { (void)ctx; return -1; }
int m40llm_stop_persistent_decode(M40llmCudaContext* ctx) { (void)ctx; return -1; }

#ifdef __cplusplus
}
#endif
