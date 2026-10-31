// Experiment-only FP16 codec and matched full-context attention consumers.
#pragma once
#include <math_constants.h>

__device__ __forceinline__ uint32_t lossless_kv_read_u32(const unsigned char* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

// Input streams have passed the CPU codec's complete validation. Byte loads
// handle unaligned payloads and never read beyond the last packed field.
__device__ __forceinline__ unsigned short lossless_kv_read_word(
    const unsigned char* stream, uint32_t index) {
    const unsigned char* entry = stream + 8 + size_t(index / 512) * 8;
    const unsigned char* payload = stream + lossless_kv_read_u32(entry);
    const uint32_t within = index % 512;
    if (entry[7] == 255) {
        return uint16_t(payload[within * 2]) | (uint16_t(payload[within * 2 + 1]) << 8);
    }
    const uint32_t width = 11 + entry[7];
    const uint32_t bit = within * width;
    const uint32_t shift = bit % 8;
    const unsigned char* first = payload + bit / 8;
    uint32_t packed = uint32_t(first[0]) | (uint32_t(first[1]) << 8);
    if (shift + width > 16) packed |= uint32_t(first[2]) << 16;
    const uint32_t code = (packed >> shift) & ((1u << width) - 1);
    return static_cast<unsigned short>((code & 1023) | ((code & 1024) << 5) |
                                       ((uint32_t(entry[6]) + (code >> 11)) << 10));
}

__global__ void lossless_kv_decode_kernel(
    const unsigned char* const* streams, uint32_t values,
    unsigned short* output) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < values) {
        output[size_t(blockIdx.y) * values + index] =
            lossless_kv_read_word(streams[blockIdx.y], index);
    }
}

extern "C" int m40llm_lossless_kv_decode_async(
    M40llmCudaContext* ctx, const void* streams, uint32_t values,
    uint32_t stream_count, void* output) {
    if (!ctx || !streams || !output) return -1;
    if (!values || values > UINT32_MAX - 255 || !stream_count || stream_count > 64) return -4;
    lossless_kv_decode_kernel<<<dim3((values + 255) / 256, stream_count), 256, 0, ctx->decode_stream>>>(
        static_cast<const unsigned char* const*>(streams), values,
        static_cast<unsigned short*>(output));
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

template<bool IsPacked>
__device__ __forceinline__ float lossless_attention_read(
    const void* storage, uint32_t stream, uint32_t index, uint32_t values) {
    unsigned short word;
    if (IsPacked) {
        word = lossless_kv_read_word(
            static_cast<const unsigned char* const*>(storage)[stream], index);
    } else {
        word = static_cast<const unsigned short*>(storage)[size_t(stream) * values + index];
    }
    return __half2float(__ushort_as_half(word));
}

// One warp per query head and contiguous context partition. Both storage
// variants use the same FP32 dot tree and online softmax. Partial state stays
// proportional to head count, not context length; no decoded KV is materialized.
template<int Dimension, bool IsPacked>
__global__ void lossless_attention_partition_kernel(
    const void* keys, const void* values, const float* query, float* partial,
    uint32_t length, uint32_t query_heads, uint32_t kv_heads, uint32_t partitions) {
    const uint32_t lane = threadIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t sequence = blockIdx.z;
    const uint32_t stream = sequence * kv_heads + head / (query_heads / kv_heads);
    const uint32_t begin = uint64_t(length) * blockIdx.x / partitions;
    const uint32_t end = uint64_t(length) * (blockIdx.x + 1) / partitions;
    const size_t query_base = (size_t(sequence) * query_heads + head) * Dimension;
    float q[Dimension / 32], acc[Dimension / 32] = {};
    #pragma unroll
    for (int c = 0; c < Dimension / 32; ++c) q[c] = query[query_base + lane + c * 32];
    float maximum = -CUDART_INF_F;
    float denominator = 0.0f;
    for (uint32_t token = begin; token < end; ++token) {
        float dot = 0.0f;
        #pragma unroll
        for (int c = 0; c < Dimension / 32; ++c) {
            dot += q[c] * lossless_attention_read<IsPacked>(
                keys, stream, token * Dimension + lane + c * 32, length * Dimension);
        }
        #pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, delta);
        const float score = __shfl_sync(0xffffffffu, dot, 0) * rsqrtf(float(Dimension));
        const float next_maximum = fmaxf(maximum, score);
        const float old_weight = expf(maximum - next_maximum);
        const float weight = expf(score - next_maximum);
        denominator = denominator * old_weight + weight;
        #pragma unroll
        for (int c = 0; c < Dimension / 32; ++c) {
            acc[c] = acc[c] * old_weight + weight * lossless_attention_read<IsPacked>(
                values, stream, token * Dimension + lane + c * 32, length * Dimension);
        }
        maximum = next_maximum;
    }
    const size_t base = ((size_t(sequence) * query_heads + head) * partitions + blockIdx.x) * (Dimension + 2);
    if (lane == 0) {
        partial[base] = maximum;
        partial[base + 1] = denominator;
    }
    #pragma unroll
    for (int c = 0; c < Dimension / 32; ++c) partial[base + 2 + lane + c * 32] = acc[c];
}

template<int Dimension>
__global__ void lossless_attention_merge_kernel(
    const float* partial, float* output, uint32_t query_heads, uint32_t partitions) {
    const uint32_t lane = threadIdx.x;
    const size_t row = size_t(blockIdx.y) * query_heads + blockIdx.x;
    const size_t base = row * partitions * (Dimension + 2);
    float maximum = -CUDART_INF_F;
    for (uint32_t p = 0; p < partitions; ++p)
        maximum = fmaxf(maximum, partial[base + p * (Dimension + 2)]);
    float denominator = 0.0f;
    float acc[Dimension / 32] = {};
    for (uint32_t p = 0; p < partitions; ++p) {
        const size_t offset = base + p * (Dimension + 2);
        const float weight = expf(partial[offset] - maximum);
        denominator += weight * partial[offset + 1];
        #pragma unroll
        for (int c = 0; c < Dimension / 32; ++c)
            acc[c] += weight * partial[offset + 2 + lane + c * 32];
    }
    #pragma unroll
    for (int c = 0; c < Dimension / 32; ++c)
        output[row * Dimension + lane + c * 32] = acc[c] / denominator;
}

template<int Dimension, bool IsPacked>
static int lossless_attention_launch(
    M40llmCudaContext* ctx, const void* keys, const void* values,
    const float* query, float* partial, float* output, uint32_t length,
    uint32_t query_heads, uint32_t kv_heads, uint32_t batch, uint32_t partitions) {
    lossless_attention_partition_kernel<Dimension, IsPacked>
        <<<dim3(partitions, query_heads, batch), 32, 0, ctx->decode_stream>>>(
            keys, values, query, partial, length, query_heads, kv_heads, partitions);
    if (cudaGetLastError() != cudaSuccess) return -2;
    lossless_attention_merge_kernel<Dimension>
        <<<dim3(query_heads, batch), 32, 0, ctx->decode_stream>>>(partial, output, query_heads, partitions);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

extern "C" int m40llm_lossless_kv_attention_async(
    M40llmCudaContext* ctx, const void* keys, const void* values,
    const void* query, void* partial, void* output, uint32_t length,
    uint32_t query_heads, uint32_t kv_heads, uint32_t batch,
    uint32_t dimension, uint32_t partitions, uint32_t is_packed) {
    if (!ctx || !keys || !values || !query || !partial || !output) return -1;
    if (!length || length > 8192 || !query_heads || query_heads > 64 ||
        !kv_heads || query_heads % kv_heads || !batch || batch > 8 || batch * kv_heads > 64 ||
        !partitions || partitions > 32 || partitions > length || is_packed > 1 ||
        (dimension != 64 && dimension != 128)) return -4;
    const float* q = static_cast<const float*>(query);
    float* p = static_cast<float*>(partial);
    float* o = static_cast<float*>(output);
    if (dimension == 128) {
        return is_packed
            ? lossless_attention_launch<128, true>(ctx, keys, values, q, p, o, length, query_heads, kv_heads, batch, partitions)
            : lossless_attention_launch<128, false>(ctx, keys, values, q, p, o, length, query_heads, kv_heads, batch, partitions);
    }
    return is_packed
        ? lossless_attention_launch<64, true>(ctx, keys, values, q, p, o, length, query_heads, kv_heads, batch, partitions)
        : lossless_attention_launch<64, false>(ctx, keys, values, q, p, o, length, query_heads, kv_heads, batch, partitions);
}
