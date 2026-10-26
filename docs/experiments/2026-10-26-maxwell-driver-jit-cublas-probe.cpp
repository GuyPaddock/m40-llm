// Identify context initialization failures without model inference or timing.
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
    const char *forced = std::getenv("CUDA_FORCE_PTX_JIT");
    const char *cache = std::getenv("CUDA_CACHE_DISABLE");
    if (!forced || !cache || (std::strcmp(forced, "0") != 0 && std::strcmp(forced, "1") != 0)
        || std::strcmp(cache, "1") != 0) return 2;
    const auto selected = cudaSetDevice(0);
    const auto initialized = selected == cudaSuccess ? cudaFree(nullptr) : selected;
    cublasHandle_t handle = nullptr;
    const int created = initialized == cudaSuccess ? static_cast<int>(cublasCreate(&handle)) : -1;
    const int destroyed = created == CUBLAS_STATUS_SUCCESS ? static_cast<int>(cublasDestroy(handle)) : -1;
    std::printf("{\"force_ptx_jit\":\"%s\",\"cache_disable\":\"%s\","
                "\"cuda_select_status\":%d,\"cuda_initialize_status\":%d,"
                "\"cublas_create_status\":%d,\"cublas_destroy_status\":%d}\n",
                forced, cache, static_cast<int>(selected), static_cast<int>(initialized), created, destroyed);
    return initialized == cudaSuccess && created == CUBLAS_STATUS_SUCCESS && destroyed == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}
