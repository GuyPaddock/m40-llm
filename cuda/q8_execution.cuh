// Lossless Q8 execution layouts. Included after the context type.
#pragma once

template<int Columns>
__device__ __forceinline__ void q8_separate_accumulate(
    const __half* __restrict__ scales,
    const int8_t* __restrict__ values,
    const float (&activation)[8], int col0, int N, int blocks_per_col,
    int qblock, int offset, float (&acc)[Columns]) {
    #pragma unroll
    for (int c = 0; c < Columns; ++c) {
        if (col0 + c < N) {
            const size_t block = static_cast<size_t>(col0 + c) * blocks_per_col + qblock;
            const float scale = __half2float(scales[block]);
            const uint2 packed = *reinterpret_cast<const uint2*>(values + block * 32 + offset);
            const int8_t* q = reinterpret_cast<const int8_t*>(&packed);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[c] += activation[i] * (static_cast<float>(q[i]) * scale);
            }
        }
    }
}

template<int Columns>
__device__ __forceinline__ void q8_separate_reduce(float (&acc)[Columns]) {
    const int tid = threadIdx.x;
    extern __shared__ float reduction[];
    #pragma unroll
    for (int c = 0; c < Columns; ++c) {
        reduction[c * blockDim.x + tid] = acc[c];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int c = 0; c < Columns; ++c) {
                reduction[c * blockDim.x + tid] += reduction[c * blockDim.x + tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            acc[c] = reduction[c * blockDim.x];
        }
    }
}

template<int Columns>
__device__ __forceinline__ void q8_separate_dot(
    const float* input, const __half* scales, const int8_t* values,
    int col0, int N, int K, float (&acc)[Columns]) {
    const int blocks_per_col = (K + 31) / 32;
    for (int task = threadIdx.x; task < blocks_per_col * 4; task += blockDim.x) {
        const int qblock = task / 4;
        const int offset = (task % 4) * 8;
        const int k0 = qblock * 32 + offset;
        float activation[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            activation[i] = k0 + i < K ? input[k0 + i] : 0.0f;
        }
        q8_separate_accumulate<Columns>(scales, values, activation, col0, N,
                                        blocks_per_col, qblock, offset, acc);
    }
    q8_separate_reduce<Columns>(acc);
}

__global__ void q8_separate_mlp_kernel(
    const float* input, const __half* gate_scales, const int8_t* gate_values,
    const __half* up_scales, const int8_t* up_values, float* output, int H, int K) {
    const int col = blockIdx.x;
    const int blocks = (K + 31) / 32;
    float gate[1] = {};
    float up[1] = {};
    for (int task = threadIdx.x; task < blocks * 4; task += blockDim.x) {
        const int qblock = task / 4;
        const int offset = (task % 4) * 8;
        const int k0 = qblock * 32 + offset;
        float activation[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            activation[i] = k0 + i < K ? input[k0 + i] : 0.0f;
        }
        q8_separate_accumulate<1>(gate_scales, gate_values, activation, col, H, blocks,
                                 qblock, offset, gate);
        q8_separate_accumulate<1>(up_scales, up_values, activation, col, H, blocks,
                                 qblock, offset, up);
    }
    float acc[2] = {gate[0], up[0]};
    q8_separate_reduce<2>(acc);
    if (threadIdx.x == 0) output[col] = acc[0] * (1.0f / (1.0f + expf(-acc[0]))) * acc[1];
}

extern "C" int m40llm_mlp_gate_up_swiglu_f32xq8_separate_async(
    M40llmCudaContext* ctx, const void* input, const void* gate_scales,
    const void* gate_values, const void* up_scales, const void* up_values,
    void* output, int H, int K, uint32_t stream_kind) {
    if (!ctx || !input || !gate_scales || !gate_values || !up_scales || !up_values || !output) return -1;
    if (H <= 0 || K <= 0 || K > INT32_MAX - 31 || stream_kind > 1) return -4;
    const cudaStream_t stream = stream_kind == 1 ? ctx->decode_stream : ctx->prefill_stream;
    q8_separate_mlp_kernel<<<H, 128, 2 * 128 * sizeof(float), stream>>>(
        static_cast<const float*>(input), static_cast<const __half*>(gate_scales),
        static_cast<const int8_t*>(gate_values), static_cast<const __half*>(up_scales),
        static_cast<const int8_t*>(up_values), static_cast<float*>(output), H, K);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

template<int Columns>
__global__ void q8_separate_projection_kernel(
    const float* __restrict__ input, const __half* __restrict__ scales,
    const int8_t* __restrict__ values, float* __restrict__ output, int N, int K) {
    const int col0 = blockIdx.x * Columns;
    const int row = blockIdx.y;
    float acc[Columns] = {};
    q8_separate_dot<Columns>(input + static_cast<size_t>(row) * K, scales, values,
                             col0, N, K, acc);
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            if (col0 + c < N) output[static_cast<size_t>(row) * N + col0 + c] = acc[c];
        }
    }
}

__global__ void q8_separate_qkv_kernel(
    const float* input,
    const __half* sq, const int8_t* vq, const __half* sk, const int8_t* vk,
    const __half* sv, const int8_t* vv,
    const float* bq, const float* bk, const float* bv,
    float* oq, float* ok, float* ov, int nq, int nk, int nv, int k) {
    const int q_tiles = (nq + 3) / 4;
    const int tile = blockIdx.x;
    if (tile < q_tiles) {
        const int col0 = tile * 4;
        float acc[4] = {};
        q8_separate_dot<4>(input, sq, vq, col0, nq, k, acc);
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                if (col0 + c < nq) oq[col0 + c] = acc[c] + (bq ? bq[col0 + c] : 0.0f);
            }
        }
    } else {
        const bool is_value = tile - q_tiles >= nk;
        const int col = tile - q_tiles - (is_value ? nk : 0);
        float acc[1] = {};
        q8_separate_dot<1>(input, is_value ? sv : sk, is_value ? vv : vk,
                             col, is_value ? nv : nk, k, acc);
        if (threadIdx.x == 0) {
            const float* bias = is_value ? bv : bk;
            (is_value ? ov : ok)[col] = acc[0] + (bias ? bias[col] : 0.0f);
        }
    }
}

extern "C" int m40llm_qkv_f32xq8_separate_async(
    M40llmCudaContext* ctx, const void* input, const void* const* scales,
    const void* const* values, const void* const* biases, void* const* outputs,
    const int* widths, int K, uint32_t stream_kind) {
    if (!ctx || !input || !scales || !values || !biases || !outputs || !widths) return -1;
    if (K <= 0 || K > INT32_MAX - 31 || stream_kind > 1) return -4;
    for (int i = 0; i < 3; ++i) {
        if (!scales[i] || !values[i] || !outputs[i]) return -1;
        if (widths[i] <= 0 || widths[i] > INT32_MAX - 3) return -4;
    }
    const int64_t blocks = (static_cast<int64_t>(widths[0]) + 3) / 4 + widths[1] + widths[2];
    if (blocks > INT32_MAX) return -4;
    const cudaStream_t stream = stream_kind == 1 ? ctx->decode_stream : ctx->prefill_stream;
    // Host metadata arrays are unpacked before enqueue; the device never reads them.
    q8_separate_qkv_kernel<<<static_cast<unsigned>(blocks), 128, 4 * 128 * sizeof(float), stream>>>(
        static_cast<const float*>(input),
        static_cast<const __half*>(scales[0]), static_cast<const int8_t*>(values[0]),
        static_cast<const __half*>(scales[1]), static_cast<const int8_t*>(values[1]),
        static_cast<const __half*>(scales[2]), static_cast<const int8_t*>(values[2]),
        static_cast<const float*>(biases[0]), static_cast<const float*>(biases[1]),
        static_cast<const float*>(biases[2]), static_cast<float*>(outputs[0]),
        static_cast<float*>(outputs[1]), static_cast<float*>(outputs[2]),
        widths[0], widths[1], widths[2], K);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

extern "C" int m40llm_gemm_f32xq8_separate_f32_async(
    M40llmCudaContext* ctx, const void* input, const void* scales,
    const void* values, void* output, int M, int N, int K, int columns, int threads, uint32_t stream_kind) {
    if (!ctx || !input || !scales || !values || !output) return -1;
    if (M <= 0 || M > 65535 || N <= 0 || K <= 0 || K > INT32_MAX - 31 ||
        (columns != 1 && columns != 4) || (threads != 128 && threads != 256) || stream_kind > 1) return -4;
    const cudaStream_t stream = stream_kind == 1 ? ctx->decode_stream : ctx->prefill_stream;
    const dim3 grid((static_cast<unsigned>(N) + columns - 1) / columns, M);
    const size_t shared = columns * threads * sizeof(float);
    if (columns == 1) {
        q8_separate_projection_kernel<1><<<grid, threads, shared, stream>>>(
            static_cast<const float*>(input), static_cast<const __half*>(scales),
            static_cast<const int8_t*>(values), static_cast<float*>(output), N, K);
    } else {
        q8_separate_projection_kernel<4><<<grid, threads, shared, stream>>>(
            static_cast<const float*>(input), static_cast<const __half*>(scales),
            static_cast<const int8_t*>(values), static_cast<float*>(output), N, K);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

template<int Columns>
__global__ void q8_input_tiled_projection_kernel(
    const float* __restrict__ input,
    const unsigned char* __restrict__ tiles,
    float* __restrict__ output, int N, int K) {
    constexpr int values_offset = ((Columns * 2 + 15) / 16) * 16;
    constexpr int record_bytes = values_offset + Columns * 32;
    const int tile = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x / 32;
    const size_t output_tiles = (static_cast<size_t>(N) + Columns - 1) / Columns;
    const int qblocks = (K + 31) / 32;
    float acc[Columns] = {};
    for (int qb = warp; qb < qblocks; qb += warps) {
        const unsigned char* record = tiles + (qb * output_tiles + tile) * record_bytes;
        const int k = qb * 32 + lane;
        const float activation = k < K ? input[static_cast<size_t>(row) * K + k] : 0.0f;
        const unsigned char* values = record + values_offset + lane * Columns;
        uint4 packed = {};
        if constexpr (Columns == 4) {
            packed.x = *reinterpret_cast<const unsigned int*>(values);
        } else if constexpr (Columns == 8) {
            const uint2 pair = *reinterpret_cast<const uint2*>(values);
            packed.x = pair.x;
            packed.y = pair.y;
        } else {
            packed = *reinterpret_cast<const uint4*>(values);
        }
        const int8_t* quant = reinterpret_cast<const int8_t*>(&packed);
        #pragma unroll
        for (int col = 0; col < Columns; ++col) {
            const float scale = __half2float(reinterpret_cast<const __half*>(record)[col]);
            acc[col] += activation * (static_cast<float>(quant[col]) * scale);
        }
    }
    extern __shared__ float partial[];
    #pragma unroll
    for (int col = 0; col < Columns; ++col) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[col] += __shfl_down_sync(0xffffffffu, acc[col], offset);
        }
        if (lane == 0) partial[warp * Columns + col] = acc[col];
    }
    __syncthreads();
    if (threadIdx.x < Columns && tile * Columns + threadIdx.x < N) {
        float sum = 0.0f;
        for (int w = 0; w < warps; ++w) sum += partial[w * Columns + threadIdx.x];
        output[static_cast<size_t>(row) * N + tile * Columns + threadIdx.x] = sum;
    }
}

template<int Columns>
__global__ void q8_input_tiled_lanes_kernel(
    const float* __restrict__ input, const unsigned char* __restrict__ tiles,
    float* __restrict__ output, int N, int K) {
    constexpr int values_offset = ((Columns * 2 + 15) / 16) * 16;
    constexpr int record_bytes = values_offset + Columns * 32;
    constexpr int groups = 32 / Columns;
    const int tile = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x / 32;
    const int col = lane % Columns;
    const int first_q = lane / Columns;
    const size_t output_tiles = (static_cast<size_t>(N) + Columns - 1) / Columns;
    float acc = 0.0f;
    for (int qb = warp; qb < (K + 31) / 32; qb += warps) {
        const unsigned char* record = tiles + (qb * output_tiles + tile) * record_bytes;
        const float scale = __half2float(reinterpret_cast<const __half*>(record)[col]);
        #pragma unroll
        for (int i = 0; i < Columns; ++i) {
            const int q = first_q + i * groups;
            const int k = qb * 32 + q;
            const float a = k < K ? input[static_cast<size_t>(blockIdx.y) * K + k] : 0.0f;
            const int8_t value = reinterpret_cast<const int8_t*>(record + values_offset)[q * Columns + col];
            acc += a * (static_cast<float>(value) * scale);
        }
    }
    for (int offset = 16; offset >= Columns; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    extern __shared__ float partial[];
    if (lane < Columns) partial[warp * Columns + lane] = acc;
    __syncthreads();
    if (threadIdx.x < Columns && tile * Columns + threadIdx.x < N) {
        float sum = 0.0f;
        for (int w = 0; w < warps; ++w) sum += partial[w * Columns + threadIdx.x];
        output[static_cast<size_t>(blockIdx.y) * N + tile * Columns + threadIdx.x] = sum;
    }
}

template<int Columns>
static void launch_q8_input_tiled(
    const void* input, const void* tiles, void* output,
    int M, int N, int K, int threads, int access, cudaStream_t stream) {
    const dim3 grid((static_cast<unsigned>(N) + Columns - 1) / Columns, M);
    const size_t shared = (threads / 32) * Columns * sizeof(float);
    if (access == 0) {
        q8_input_tiled_projection_kernel<Columns><<<grid, threads, shared, stream>>>(
            static_cast<const float*>(input), static_cast<const unsigned char*>(tiles),
            static_cast<float*>(output), N, K);
    } else {
        q8_input_tiled_lanes_kernel<Columns><<<grid, threads, shared, stream>>>(
            static_cast<const float*>(input), static_cast<const unsigned char*>(tiles),
            static_cast<float*>(output), N, K);
    }
}

extern "C" int m40llm_gemm_f32xq8_tiled_f32_async(
    M40llmCudaContext* ctx, const void* input, const void* tiles, void* output,
    int M, int N, int K, int columns, int threads, int access, uint32_t stream_kind) {
    if (!ctx || !input || !tiles || !output) return -1;
    if (M <= 0 || M > 65535 || N <= 0 || K <= 0 || K > INT32_MAX - 31 ||
        (columns != 4 && columns != 8 && columns != 16) ||
        (threads != 128 && threads != 256) || (access != 0 && access != 1) || stream_kind > 1) return -4;
    const cudaStream_t stream = stream_kind == 1 ? ctx->decode_stream : ctx->prefill_stream;
    if (columns == 4) {
        launch_q8_input_tiled<4>(input,tiles,output,M,N,K,threads,access,stream);
    } else if (columns == 8) {
        launch_q8_input_tiled<8>(input,tiles,output,M,N,K,threads,access,stream);
    } else {
        launch_q8_input_tiled<16>(input,tiles,output,M,N,K,threads,access,stream);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
