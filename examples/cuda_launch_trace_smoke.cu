// The observer must preserve execution and report standard/extended launches,
// including an API rejection. Run explicitly on an idle sm_52 device.
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void m40_trace_increment(int *value) {
    atomicAdd(value, 1);
}

static bool check_cuda(cudaError_t result, const char *operation) {
    if (result == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
    return false;
}

// A passing process proves the observed kernels ran correctly, not just that
// the library loaded or a callback could be subscribed.
int main() {
    cudaDeviceProp properties{};
    if (!check_cuda(cudaGetDeviceProperties(&properties, 0), "device") ||
        properties.major != 5 || properties.minor != 2) return 1;
    int *value = nullptr;
    if (!check_cuda(cudaMalloc(&value, sizeof(int)), "allocate")) return 1;
    if (!check_cuda(cudaMemset(value, 0, sizeof(int)), "initialize")) return 1;
    m40_trace_increment<<<2, 16, 64>>>(value);
    if (!check_cuda(cudaGetLastError(), "standard launch")) return 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(1);
    config.blockDim = dim3(64);
    void *arguments[] = {&value};
    if (!check_cuda(cudaLaunchKernelExC(&config,
                    reinterpret_cast<const void *>(m40_trace_increment), arguments), "extended launch")) return 1;
    m40_trace_increment<<<0, 32>>>(value);
    if (cudaGetLastError() != cudaErrorInvalidConfiguration) return 1;
    int actual = 0;
    if (!check_cuda(cudaMemcpy(&actual, value, sizeof(int), cudaMemcpyDeviceToHost), "result")) return 1;
    if (!check_cuda(cudaFree(value), "free")) return 1;
    if (actual != 96) return 1;
    std::printf("{\"actual\":%d,\"expected\":96,\"is_pass\":true}\n", actual);
    return 0;
}
