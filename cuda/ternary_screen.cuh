#pragma once
#include <climits>

// Fixed geometry for the isolated Maxwell screening-cost experiment.
namespace m40llm_ternary_screen {
constexpr int width = 128;
constexpr int selected = 80;

__device__ __forceinline__ bool precedes(float a, int ai, float b, int bi) {
    return a > b || (a == b && ai < bi);
}

__global__ void rotate_select(
    const float* input, const signed char* signs, float* rotated,
    unsigned char* coordinates) {
    const int i = threadIdx.x;
    const size_t base = static_cast<size_t>(blockIdx.x) * width;
    __shared__ float values[width];
    __shared__ int order[width];
    values[i] = input[base + i] * float(signs[base + i]);
    __syncthreads();
    for (int stride = 1; stride < width; stride <<= 1) {
        const float a = values[i];
        const float b = values[i ^ stride];
        __syncthreads();
        values[i] = (i & stride) ? b - a : a + b;
        __syncthreads();
    }
    const float value = values[i] * 0.08838834764831844f;
    rotated[base + i] = value;
    values[i] = fabsf(value);
    order[i] = i;
    __syncthreads();
    for (int span = 2; span <= width; span <<= 1) {
        for (int step = span >> 1; step; step >>= 1) {
            const int peer = i ^ step;
            const float a = values[i], b = values[peer];
            const int ai = order[i], bi = order[peer];
            const bool wants_first = ((i & span) == 0) == ((i & step) == 0);
            const bool swap = wants_first ? precedes(b, bi, a, ai) : precedes(a, ai, b, bi);
            __syncthreads();
            values[i] = swap ? b : a;
            order[i] = swap ? bi : ai;
            __syncthreads();
        }
    }
    if (i < selected) {
        coordinates[static_cast<size_t>(blockIdx.x) * selected + i] = order[i];
    }
}

__device__ __forceinline__ float read_code(
    const unsigned char* codes, size_t offset, int shift, float value) {
    const unsigned code = (codes[offset] >> shift) & 3;
    return code == 1 ? value : (code == 2 ? -value : 0.0f);
}

__global__ void score_channels(
    const float* rotated, const unsigned char* coordinates,
    const __half* gate_scales, const unsigned char* gate_codes,
    const __half* up_scales, const unsigned char* up_codes,
    const float* norms, float* output, int k, int h) {
    const int lane = threadIdx.x % 32;
    const int part = threadIdx.x / 32;
    const int channel = blockIdx.x * 32 + lane;
    const int stride = (h + 3) / 4;
    const int shift = 2 * (channel % 4);
    float gate = 0.0f, up = 0.0f;
    if (channel < h) {
        for (int block = 0; block < k / width; ++block) {
            float g = 0.0f, u = 0.0f;
            for (int rank = part; rank < selected; rank += 8) {
                const int coordinate = block * width + coordinates[block * selected + rank];
                const float x = rotated[coordinate];
                const size_t offset = static_cast<size_t>(coordinate) * stride + channel / 4;
                g += read_code(gate_codes, offset, shift, x);
                u += read_code(up_codes, offset, shift, x);
            }
            const size_t offset = static_cast<size_t>(block) * h + channel;
            gate += __half2float(gate_scales[offset]) * g;
            up += __half2float(up_scales[offset]) * u;
        }
    }
    __shared__ float sums[512];
    sums[threadIdx.x] = gate;
    sums[256 + threadIdx.x] = up;
    __syncthreads();
    if (part == 0 && channel < h) {
        gate = 0.0f;
        up = 0.0f;
        for (int p = 0; p < 8; ++p) {
            gate += sums[p * 32 + lane];
            up += sums[256 + p * 32 + lane];
        }
        const float activation = gate / (1.0f + expf(-gate)) * up;
        output[channel] = gate;
        output[static_cast<size_t>(h) + channel] = up;
        output[static_cast<size_t>(2) * h + channel] = activation;
        output[static_cast<size_t>(3) * h + channel] = fabsf(activation) * norms[channel];
    }
}
}
