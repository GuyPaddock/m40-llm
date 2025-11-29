// cuda/stub.c - stubbed CUDA FFI for environments without NVCC/CUDA
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct M40llmCudaContext { int device_id; } M40llmCudaContext;
typedef struct M40llmKVCache { int _stub; } M40llmKVCache;

int m40llm_current_device_props(char* name_buf, size_t buf_len, int* major, int* minor, int* device_id) {
    if (name_buf && buf_len) {
        const char* name = "stub";
        size_t n = strlen(name);
        if (n + 1 > buf_len) n = buf_len - 1;
        memcpy(name_buf, name, n);
        name_buf[n] = '\0';
    }
    if (major) *major = 0;
    if (minor) *minor = 0;
    if (device_id) *device_id = -1;
    return -1; // indicate unavailable in stub
}

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

int m40llm_gemm_f32xf16_f32(M40llmCudaContext* ctx, const void* d_A_f32, const void* d_B_f16, void* d_C_f32, int M, int N, int K) {
    (void)ctx; (void)d_A_f32; (void)d_B_f16; (void)d_C_f32; (void)M; (void)N; (void)K; return -1;
}


int m40llm_gemm_f16xf16_f32(M40llmCudaContext* ctx, const void* d_A_f16, const void* d_B_f16, void* d_C_f32, int M, int N, int K) {
    (void)ctx; (void)d_A_f16; (void)d_B_f16; (void)d_C_f32; (void)M; (void)N; (void)K; return -1;
}

M40llmKVCache* m40llm_kvcache_create(M40llmCudaContext* ctx, uint32_t max_seq_len, uint32_t max_batch_size, uint32_t num_heads, uint32_t head_dim) {
    (void)ctx; (void)max_seq_len; (void)max_batch_size; (void)num_heads; (void)head_dim; return (M40llmKVCache*)0x1;
}
int m40llm_kvcache_append_token(M40llmCudaContext* ctx, M40llmKVCache* kv, uint32_t seq_id, const void* k_dev, const void* v_dev) {
    (void)ctx; (void)kv; (void)seq_id; (void)k_dev; (void)v_dev; return -1;
}
void m40llm_kvcache_destroy(M40llmKVCache* kv) { (void)kv; }

int m40llm_kvcache_append_token_f32(M40llmCudaContext* ctx, M40llmKVCache* kv, uint32_t seq_id, const void* k_dev_f32, const void* v_dev_f32) {
    (void)ctx; (void)kv; (void)seq_id; (void)k_dev_f32; (void)v_dev_f32; return -1;
}
int m40llm_kvcache_debug_read_token(M40llmCudaContext* ctx, M40llmKVCache* kv, uint32_t seq_id, uint32_t token, void* out_k_host, void* out_v_host) {
    (void)ctx; (void)kv; (void)seq_id; (void)token; (void)out_k_host; (void)out_v_host; return -1;
}


int m40llm_attention_last_token_f32(
    M40llmCudaContext* ctx,
    const M40llmKVCache* kv,
    uint32_t seq_id,
    const void* q_dev_f32,
    uint32_t seq_len,
    void* out_dev_f32) {
    (void)ctx; (void)kv; (void)seq_id; (void)q_dev_f32; (void)seq_len; (void)out_dev_f32; return -1;
}

int m40llm_start_persistent_decode(M40llmCudaContext* ctx) { (void)ctx; return -1; }
int m40llm_stop_persistent_decode(M40llmCudaContext* ctx) { (void)ctx; return -1; }


  int m40llm_f16_to_f32(M40llmCudaContext* ctx, const void* d_in_f16, void* d_out_f32, size_t n) {
      (void)ctx; (void)d_in_f16; (void)d_out_f32; (void)n; return -1;
  }
  int m40llm_q80_to_f32(M40llmCudaContext* ctx, const void* d_in_q80, void* d_out_f32, size_t n) {
      (void)ctx; (void)d_in_q80; (void)d_out_f32; (void)n; return -1;
  }

#ifdef __cplusplus
}
#endif
