// Public CUB API reference only; no inference outputs or copied CUB internals.
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <cmath>
// libstdc++ places these C99 classifications only in std; libcudacxx also
// imports the global names. Re-export the standard overloads, not new math.
#if defined(__GLIBCXX__)
using std::fpclassify;
using std::isgreater;
using std::isgreaterequal;
using std::isless;
using std::islessequal;
using std::islessgreater;
using std::isnormal;
using std::isunordered;
#endif
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

namespace m40llm_q8_cub_read {
struct ReadPair {
    const uint16_t* gate_scales;
    const uint32_t* gate_values;
    const uint16_t* up_scales;
    const uint32_t* up_values;
    int tensor_words;

    __device__ uint2 operator()(int index) const {
        const int word = index < tensor_words ? index : index % tensor_words;
        return make_uint2(gate_values[word] + static_cast<uint32_t>(gate_scales[word / 8]),
                          up_values[word] + static_cast<uint32_t>(up_scales[word / 8]));
    }
};

struct AddPair {
    __host__ __device__ uint2 operator()(uint2 a, uint2 b) const {
        return make_uint2(a.x + b.x, a.y + b.y);
    }
};

struct SegmentOffset {
    int tasks;
    __host__ __device__ int operator()(int index) const { return index * tasks; }
};

using Counter = cub::CountingInputIterator<int>;
using Input = cub::TransformInputIterator<uint2, ReadPair, Counter>;
using Offset = cub::TransformInputIterator<int, SegmentOffset, Counter>;

cudaError_t reduce(void* scratch, size_t& bytes, const void* const* scales,
                   const void* const* values, void* output, int rows, int hidden,
                   int k, cudaStream_t stream) {
    const int tasks = k / 4;
    const ReadPair read{static_cast<const uint16_t*>(scales[0]),
                        static_cast<const uint32_t*>(values[0]),
                        static_cast<const uint16_t*>(scales[1]),
                        static_cast<const uint32_t*>(values[1]), hidden * tasks};
    return cub::DeviceSegmentedReduce::Reduce(
        scratch, bytes, Input(Counter(0), read), static_cast<uint2*>(output),
        rows * hidden, Offset(Counter(0), SegmentOffset{tasks}),
        Offset(Counter(1), SegmentOffset{tasks}), AddPair{}, make_uint2(0, 0), stream);
}
} // namespace m40llm_q8_cub_read

extern "C" uint32_t m40llm_q8_cub_checksum_version() { return CUB_VERSION; }

extern "C" int m40llm_cub_checksum_scratch_bytes(void* stream, size_t* bytes) {
    if (!bytes) return -1;
    const void* empty[2] = {nullptr, nullptr};
    return m40llm_q8_cub_read::reduce(nullptr, *bytes, empty, empty, nullptr,
                                     1, 1, 32, static_cast<cudaStream_t>(stream)) == cudaSuccess ? 0 : -4;
}

extern "C" int m40llm_cub_checksum_launch(
    void* stream, const void* const* scales, const void* const* values,
    void* output, void* scratch, size_t bytes, int rows, int hidden, int k) {
    size_t required = 0;
    const int query = m40llm_cub_checksum_scratch_bytes(stream, &required);
    if (query != 0) return query;
    if (bytes < required) return -3;
    return m40llm_q8_cub_read::reduce(scratch, bytes, scales, values, output,
                                     rows, hidden, k, static_cast<cudaStream_t>(stream)) == cudaSuccess ? 0 : -4;
}
