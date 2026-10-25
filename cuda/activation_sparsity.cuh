// Dense Q8 weights, sparse completed SwiGLU inputs. Research-only decode path.
#pragma once

__device__ __forceinline__ bool retain_sparse_input(float value, float threshold) {
    return !isfinite(value) || fabsf(value) > threshold;
}

__global__ void select_sparse_inputs_kernel(
    const float* __restrict__ input, uint32_t* __restrict__ indices,
    float* __restrict__ values, uint32_t* count, int K, float threshold) {
    constexpr int warps = 8;
    __shared__ int warp_counts[warps];
    __shared__ int warp_offsets[warps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int chunk = (K + 255) / 256;
    const int begin = min(static_cast<int>(threadIdx.x) * chunk, K);
    const int end = min(begin + chunk, K);
    int local_count = 0;
    for (int i = begin; i < end; ++i) {
        local_count += retain_sparse_input(input[i], threshold);
    }
    int prefix = local_count;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int previous = __shfl_up_sync(0xffffffffu, prefix, offset);
        if (lane >= offset) prefix += previous;
    }
    if (lane == 31) warp_counts[warp] = prefix;
    __syncthreads();
    if (warp == 0) {
        const int own = lane < warps ? warp_counts[lane] : 0;
        int total = own;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int previous = __shfl_up_sync(0xffffffffu, total, offset);
            if (lane >= offset) total += previous;
        }
        if (lane < warps) warp_offsets[lane] = total - own;
        if (lane == warps - 1) *count = static_cast<uint32_t>(total);
    }
    __syncthreads();
    int destination = warp_offsets[warp] + prefix - local_count;
    for (int i = begin; i < end; ++i) {
        const float value = input[i];
        if (retain_sparse_input(value, threshold)) {
            indices[destination] = static_cast<uint32_t>(i);
            values[destination++] = value;
        }
    }
}

extern "C" int m40llm_select_sparse_inputs_f32_async(
    M40llmCudaContext* ctx, const void* input, void* indices, void* values,
    void* count, int K, float threshold) {
    if (!ctx || !input || !indices || !values || !count) return -1;
    if (K <= 0 || K > INT32_MAX - 255 || !isfinite(threshold) || threshold < 0.0f ||
        input == indices || input == values || input == count || indices == values ||
        indices == count || values == count) return -4;
    select_sparse_inputs_kernel<<<1, 256, 0, ctx->decode_stream>>>(
        static_cast<const float*>(input), static_cast<uint32_t*>(indices),
        static_cast<float*>(values), static_cast<uint32_t*>(count), K, threshold);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

template<int Columns>
__global__ void project_sparse_inputs_q8_kernel(
    const uint32_t* __restrict__ indices, const float* __restrict__ values,
    const uint32_t* count, const unsigned char* __restrict__ tiles,
    float* __restrict__ output, int N) {
    constexpr int values_offset = ((Columns * 2 + 15) / 16) * 16;
    constexpr int record_bytes = values_offset + Columns * 32;
    const int tile = blockIdx.x;
    const size_t output_tiles = (static_cast<size_t>(N) + Columns - 1) / Columns;
    const uint32_t retained = *count;
    float acc[Columns] = {};
    for (uint32_t position = threadIdx.x; position < retained; position += blockDim.x) {
        const uint32_t k = indices[position];
        const unsigned char* record = tiles +
            (static_cast<size_t>(k / 32) * output_tiles + tile) * record_bytes;
        accumulate_q8_tiled_values(record, k % 32, values[position], acc);
    }
    write_q8_tiled_projection(acc, output, N, tile);
}

template<int Columns>
static void launch_sparse_inputs_q8(
    M40llmCudaContext* ctx, const void* indices, const void* values, const void* count,
    const void* tiles, void* output, int N, int threads) {
    const dim3 grid((static_cast<unsigned>(N) + Columns - 1) / Columns);
    const size_t shared = (threads / 32) * Columns * sizeof(float);
    project_sparse_inputs_q8_kernel<Columns><<<grid, threads, shared, ctx->decode_stream>>>(
        static_cast<const uint32_t*>(indices), static_cast<const float*>(values),
        static_cast<const uint32_t*>(count), static_cast<const unsigned char*>(tiles),
        static_cast<float*>(output), N);
}

extern "C" int m40llm_project_sparse_inputs_q8_async(
    M40llmCudaContext* ctx, const void* indices, const void* values, const void* count,
    const void* tiles, void* output, int N, int K, int columns, int threads) {
    if (!ctx || !indices || !values || !count || !tiles || !output) return -1;
    if (N <= 0 || N > INT32_MAX - 15 || K <= 0 || K > INT32_MAX - 255 ||
        (columns != 8 && columns != 16) || (threads != 128 && threads != 256) ||
        output == indices || output == values || output == count || output == tiles) return -4;
    if (columns == 8) {
        launch_sparse_inputs_q8<8>(ctx, indices, values, count, tiles, output, N, threads);
    } else {
        launch_sparse_inputs_q8<16>(ctx, indices, values, count, tiles, output, N, threads);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
