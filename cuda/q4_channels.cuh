#pragma once

// Fixed research geometry. All arithmetic accumulates in FP32 on sm_52.
namespace m40llm_q4_channels {
constexpr int down_splits = 32;
constexpr int zigzag_inputs = 11008;
constexpr int zigzag_outputs = 2048;

__device__ __forceinline__ float read_native(
    const unsigned char* weights, int output, int input, int width) {
    const size_t base = (static_cast<size_t>(output) * (width / 32) + input / 32) * 18;
    const unsigned char byte = weights[base + 2 + input % 16];
    const int code = ((byte >> (4 * ((input % 32) / 16))) & 15) - 8;
    return __half2float(*reinterpret_cast<const __half*>(weights + base)) * float(code);
}

__global__ void gate_up(
    const float* input, const unsigned char* gate, const unsigned char* up,
    const unsigned* indices, float* hidden, int k, int count) {
    const int selected = blockIdx.x;
    const int row = blockIdx.y;
    const int channel = indices[static_cast<size_t>(row) * count + selected];
    float g = 0.0f, u = 0.0f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const float x = input[static_cast<size_t>(row) * k + i];
        g += x * read_native(gate, channel, i, k);
        u += x * read_native(up, channel, i, k);
    }
    __shared__ float sums[256];
    sums[threadIdx.x] = g;
    sums[128 + threadIdx.x] = u;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
            sums[128 + threadIdx.x] += sums[128 + threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        hidden[static_cast<size_t>(row) * count + selected] =
            sums[0] * (1.0f / (1.0f + expf(-sums[0]))) * sums[128];
    }
}

template<bool packed, bool device_count = false>
__global__ void down_partial(
    const float* hidden, const unsigned* indices,
    const unsigned char* native, const __half* scales,
    const unsigned char* codes, float* partial, int n, int h, int count,
    const unsigned* active_count = nullptr) {
    const int output = blockIdx.x * blockDim.x + threadIdx.x;
    const int split = blockIdx.y;
    const int row = blockIdx.z;
    if (output >= n) return;
    float sum = 0.0f;
    const int limit = device_count ? static_cast<int>(*active_count) : count;
    for (int selected = split; selected < limit; selected += down_splits) {
        const int channel = indices[static_cast<size_t>(row) * count + selected];
        float weight;
        if (packed) {
            const unsigned char byte = codes[static_cast<size_t>(channel) * ((n + 1) / 2) + output / 2];
            const int code = ((byte >> (4 * (output % 2))) & 15) - 8;
            weight = __half2float(scales[static_cast<size_t>(channel / 32) * n + output]) * float(code);
        } else {
            weight = read_native(native, output, channel, h);
        }
        sum += hidden[static_cast<size_t>(row) * count + selected] * weight;
    }
    partial[(static_cast<size_t>(row) * down_splits + split) * n + output] = sum;
}

// Layout [output tile][input][FP16 scale, 16 Q4 bytes], 32 outputs per tile.
template<bool dense, bool charged>
__global__ void zigzag_partial(
    const float* input, const unsigned* indices, const unsigned* active_count,
    int oracle_count, const unsigned char* weights, float* partial) {
    const int output = blockIdx.x * blockDim.x + threadIdx.x;
    const int split = blockIdx.y;
    const int limit = dense ? zigzag_inputs : (charged ? int(*active_count) : oracle_count);
    float sum = 0.0f;
    for (int i = split; i < limit; i += down_splits) {
        const int channel = dense ? i : int(indices[i]);
        const size_t base = (size_t(output / 32) * zigzag_inputs + channel) * 18;
        const int lane = output % 32;
        const int code = ((weights[base + 2 + lane % 16] >> (4 * (lane / 16))) & 15) - 8;
        const float scale = __half2float(*reinterpret_cast<const __half*>(weights + base));
        sum += input[i] * (scale * float(code));
    }
    partial[split * zigzag_outputs + output] = sum;
}

__global__ void down_reduce(const float* partial, float* output, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (col >= n) return;
    float sum = 0.0f;
    for (int split = 0; split < down_splits; ++split) {
        sum += partial[(static_cast<size_t>(row) * down_splits + split) * n + col];
    }
    output[static_cast<size_t>(row) * n + col] = sum;
}
}
