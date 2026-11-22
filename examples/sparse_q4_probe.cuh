#pragma once
// Standalone experiment TU so the three Phase 4 control kernels remain exact.
#include "../cuda/kernels.cu"
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace {
constexpr int inputs = 11008, outputs = 2048, splits = 32, repeats = 20;
using Clock = std::chrono::steady_clock;
using Bytes = std::vector<unsigned char>;

void require(bool value, const char* error) {
    if (!value) throw std::runtime_error(error);
}
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> struct Buffer {
    T* ptr = nullptr;
    size_t count;
    explicit Buffer(size_t length) : count(length) { check(cudaMalloc(&ptr, count * sizeof(T))); }
    ~Buffer() { cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const T* source) { check(cudaMemcpy(ptr, source, count * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> read() const {
        std::vector<T> result(count);
        check(cudaMemcpy(result.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }
};
struct Event {
    cudaEvent_t event;
    Event() { check(cudaEventCreate(&event)); }
    ~Event() { cudaEventDestroy(event); }
};
Bytes read_plane(const std::string& folder, const char* name, size_t bytes) {
    std::ifstream file(folder + "/" + name, std::ios::binary | std::ios::ate);
    require(file.good() && file.tellg() == std::streampos(bytes), "invalid plane");
    Bytes result(bytes);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(result.data()), bytes);
    require(file.good(), "incomplete plane");
    return result;
}

struct Fixture {
    Bytes native, alternate, scales, codes;
    Buffer<unsigned char> d_native, d_alternate, d_scales, d_codes;
    Buffer<float> x{inputs}, compact{inputs}, partial{outputs*splits}, result{outputs};
    Buffer<unsigned> indices{inputs}, count{1};
    cudaStream_t stream;
    explicit Fixture(const std::string& folder) :
        native(read_plane(folder,"native",inputs*outputs/32*18)),
        alternate(read_plane(folder,"zigzag",native.size())),
        scales(read_plane(folder,"scales",inputs*outputs/32*2)),
        codes(read_plane(folder,"codes",inputs*outputs/2)),
        d_native(native.size()), d_alternate(alternate.size()),
        d_scales(scales.size()), d_codes(codes.size()) {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        d_native.upload(native.data());
        d_alternate.upload(alternate.data());
        d_scales.upload(scales.data());
        d_codes.upload(codes.data());
    }
    ~Fixture() { cudaStreamDestroy(stream); }
};

enum Arm { dense_native, dense_zigzag, native_oracle, packed_oracle, zigzag_oracle,
           native_charged, packed_charged, zigzag_charged, arm_count };
const char* names[] = {"dense_native","dense_zigzag","native_oracle","packed_oracle",
                       "zigzag_oracle","native_charged","packed_charged","zigzag_charged"};

void compact(Fixture& f, float cutoff = 0.0f) {
    select_sparse_inputs_kernel<<<1,256,0,f.stream>>>(f.x.ptr,f.indices.ptr,
        f.compact.ptr,f.count.ptr,inputs,cutoff,nullptr);
}
template<bool packed, bool charged>
void selected(Fixture& f, int count) {
    m40llm_q4_channels::down_partial<packed,charged><<<dim3(outputs/256,splits),256,0,f.stream>>>(
        f.compact.ptr,f.indices.ptr,f.d_native.ptr,reinterpret_cast<const __half*>(f.d_scales.ptr),
        f.d_codes.ptr,f.partial.ptr,outputs,inputs,charged ? inputs : count,f.count.ptr);
}
void enqueue(Fixture& f, Arm arm, int count, float cutoff = 0.0f) {
    if (arm == dense_native) {
        const int threads = q8_decode_threads_for_k(inputs);
        gemm_f32xq4_0_gguf_f32_decode_kernel<<<outputs,threads,threads*sizeof(float),f.stream>>>(
            f.x.ptr,f.d_native.ptr,f.result.ptr,outputs,inputs);
        return;
    }
    if (arm >= native_charged) compact(f, cutoff);
    const dim3 grid(outputs/256,splits);
    switch (arm) {
    case dense_zigzag:
        m40llm_q4_channels::zigzag_partial<true,false><<<grid,256,0,f.stream>>>(f.x.ptr,f.indices.ptr,f.count.ptr,count,
            f.d_alternate.ptr,f.partial.ptr);
        break;
    case native_oracle: selected<false,false>(f,count); break;
    case packed_oracle: selected<true,false>(f,count); break;
    case native_charged: selected<false,true>(f,count); break;
    case packed_charged: selected<true,true>(f,count); break;
    case zigzag_oracle:
        m40llm_q4_channels::zigzag_partial<false,false><<<grid,256,0,f.stream>>>(f.compact.ptr,f.indices.ptr,f.count.ptr,
            count,f.d_alternate.ptr,f.partial.ptr);
        break;
    case zigzag_charged:
        m40llm_q4_channels::zigzag_partial<false,true><<<grid,256,0,f.stream>>>(f.compact.ptr,f.indices.ptr,f.count.ptr,
            0,f.d_alternate.ptr,f.partial.ptr);
        break;
    default: throw std::runtime_error("unknown arm");
    }
    m40llm_q4_channels::down_reduce<<<outputs/256,256,0,f.stream>>>(f.partial.ptr,f.result.ptr,outputs);
}

float decode(const Bytes& bytes, size_t block, int lane) {
    __half_raw bits;
    memcpy(&bits.x, bytes.data() + 18 * block, 2);
    const unsigned char byte = bytes[18 * block + 2 + lane % 16];
    return __half2float(__half(bits)) * float(int((byte >> (4*(lane/16))) & 15) - 8);
}

struct Reference {
    std::vector<double> sum, absolute;
    Reference(const Bytes& plane, bool alternate, const std::vector<float>& x,
              const std::vector<unsigned>& indices) : sum(outputs), absolute(outputs) {
        for (int output = 0; output < outputs; ++output) {
            for (unsigned channel : indices) {
                const size_t block = alternate ? size_t(output/32)*inputs + channel :
                                                 size_t(output)*(inputs/32) + channel/32;
                const double product = double(x[channel]) * decode(plane,block,alternate ? output%32 : channel%32);
                sum[output] += product;
                absolute[output] += std::abs(product);
            }
        }
    }
};

} // namespace
