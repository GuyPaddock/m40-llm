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

// Layout [output tile][input][FP16 scale, 16 Q4 bytes], 32 outputs per tile.
template<bool dense, bool charged>
__global__ void zigzag_partial(const float* x, const unsigned* indices,
                               const unsigned* actual, int oracle_count,
                               const unsigned char* weights, float* partial) {
    const int output = blockIdx.x * blockDim.x + threadIdx.x;
    const int split = blockIdx.y;
    const int limit = dense ? inputs : (charged ? int(*actual) : oracle_count);
    float sum = 0;
    for (int i = split; i < limit; i += splits) {
        const int channel = dense ? i : int(indices[i]);
        const size_t base = (size_t(output / 32) * inputs + channel) * 18;
        const int lane = output % 32;
        const int code = ((weights[base+2+lane%16] >> (4*(lane/16))) & 15) - 8;
        const float scale = __half2float(*reinterpret_cast<const __half*>(weights+base));
        sum += x[i] * (scale * float(code));
    }
    partial[split * outputs + output] = sum;
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

void compact(Fixture& f) {
    select_sparse_inputs_kernel<<<1,256,0,f.stream>>>(f.x.ptr,f.indices.ptr,
        f.compact.ptr,f.count.ptr,inputs,0.0f,nullptr);
}
template<bool packed, bool charged>
void selected(Fixture& f, int count) {
    m40llm_q4_channels::down_partial<packed,charged><<<dim3(outputs/256,splits),256,0,f.stream>>>(
        f.compact.ptr,f.indices.ptr,f.d_native.ptr,reinterpret_cast<const __half*>(f.d_scales.ptr),
        f.d_codes.ptr,f.partial.ptr,outputs,inputs,charged ? inputs : count,f.count.ptr);
}
void enqueue(Fixture& f, Arm arm, int count) {
    if (arm == dense_native) {
        const int threads = q8_decode_threads_for_k(inputs);
        gemm_f32xq4_0_gguf_f32_decode_kernel<<<outputs,threads,threads*sizeof(float),f.stream>>>(
            f.x.ptr,f.d_native.ptr,f.result.ptr,outputs,inputs);
        return;
    }
    if (arm >= native_charged) compact(f);
    const dim3 grid(outputs/256,splits);
    switch (arm) {
    case dense_zigzag:
        zigzag_partial<true,false><<<grid,256,0,f.stream>>>(f.x.ptr,f.indices.ptr,f.count.ptr,count,
            f.d_alternate.ptr,f.partial.ptr);
        break;
    case native_oracle: selected<false,false>(f,count); break;
    case packed_oracle: selected<true,false>(f,count); break;
    case native_charged: selected<false,true>(f,count); break;
    case packed_charged: selected<true,true>(f,count); break;
    case zigzag_oracle:
        zigzag_partial<false,false><<<grid,256,0,f.stream>>>(f.compact.ptr,f.indices.ptr,f.count.ptr,
            count,f.d_alternate.ptr,f.partial.ptr);
        break;
    case zigzag_charged:
        zigzag_partial<false,true><<<grid,256,0,f.stream>>>(f.compact.ptr,f.indices.ptr,f.count.ptr,
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

void run_case(Fixture& f, int layer, int count, int rounds, int start) {
    std::vector<unsigned> order(inputs);
    std::iota(order.begin(),order.end(),0);
    std::mt19937 generator(1121 + layer);
    std::shuffle(order.begin(),order.end(),generator);
    order.resize(count);
    std::sort(order.begin(),order.end());
    std::vector<float> x(inputs,0.0f);
    for (unsigned i : order) x[i] = (i%2 ? -1.f : 1.f) * float(1+(i*73)%997)/997.f;
    f.x.upload(x.data());
    compact(f);
    check(cudaStreamSynchronize(f.stream));
    auto indices = f.indices.read();
    auto values = f.compact.read();
    require(f.count.read()[0] == unsigned(count), "device count mismatch");
    for (int i=0; i<count; ++i)
        require(indices[i] == order[i] && values[i] == x[order[i]], "unstable compaction");
    Reference native(f.native,false,x,order), alternate(f.alternate,true,x,order);
    std::vector<float> selected_native, selected_zigzag;
    for (int a=0; a<arm_count; ++a) {
        const Arm arm = Arm(a);
        enqueue(f,arm,count);
        check(cudaStreamSynchronize(f.stream));
        const auto result = f.result.read();
        const bool zigzag = arm==dense_zigzag || arm==zigzag_oracle || arm==zigzag_charged;
        const auto& ref = zigzag ? alternate : native;
        double max_error=0, max_layout_error=0;
        for (int out=0; out<outputs; ++out) {
            const double error = std::abs(result[out] - ref.sum[out]);
            require(std::isfinite(result[out]) && error <= 0.0005*(1+ref.absolute[out]), "FP64 disagreement");
            max_error = std::max(max_error,error);
            max_layout_error = std::max(max_layout_error,std::abs(ref.sum[out]-native.sum[out]));
        }
        if (arm==native_oracle) selected_native=result;
        if (arm==zigzag_oracle) selected_zigzag=result;
        if (arm==packed_oracle || arm==native_charged || arm==packed_charged)
            require(memcmp(result.data(),selected_native.data(),outputs*sizeof(float))==0, "selected layout disagreement");
        if (arm==zigzag_charged)
            require(memcmp(result.data(),selected_zigzag.data(),outputs*sizeof(float))==0,"zigzag count disagreement");
        printf("{\"kind\":\"check\",\"layer\":%d,\"count\":%d,\"arm\":\"%s\",\"max_error\":%.9g,\"max_layout_delta\":%.9g}\n",
            layer,count,names[arm],max_error,max_layout_error);
    }
    if (!count) return;
    Event begin,end;
    for (int round=-1; round<rounds; ++round) {
        for (int index=0; index<arm_count; ++index) {
            const Arm arm = Arm((index+round+1+start)%arm_count);
            const auto wall = Clock::now();
            check(cudaEventRecord(begin.event,f.stream));
            const auto submit = Clock::now();
            for (int i=0; i<repeats; ++i) enqueue(f,arm,count);
            check(cudaPeekAtLastError());
            const auto submitted = Clock::now();
            check(cudaEventRecord(end.event,f.stream));
            check(cudaEventSynchronize(end.event));
            const auto done = Clock::now();
            float ms;
            check(cudaEventElapsedTime(&ms,begin.event,end.event));
            using Micros = std::chrono::duration<double,std::micro>;
            printf("{\"kind\":\"trial\",\"layer\":%d,\"count\":%d,\"round\":%d,\"order\":%d,\"arm\":\"%s\",\"repeats\":%d,\"event_us\":%.6f,\"submit_us\":%.6f,\"wall_us\":%.6f}\n",
                layer,count,round < 0 ? -1 : round+start,index,names[arm],repeats,ms*1000/repeats,
                Micros(submitted-submit).count()/repeats,Micros(done-wall).count()/repeats);
        }
        fflush(stdout);
    }
}
} // namespace

int main(int argc,char** argv) {
    try {
        require(argc==4, "usage: probe PLANES ROUNDS START");
        const int rounds=std::stoi(argv[2]);
        const int start=std::stoi(argv[3]);
        require(start==0 || start==5, "invalid start");
        require(rounds==5 || rounds==10 || rounds==20, "invalid rounds");
        cudaDeviceProp props;
        check(cudaGetDeviceProperties(&props,0));
        require(props.major==5 && props.minor==2 && strstr(props.name,"M40"),"Tesla M40 required");
        for (int layer : {0,17,35}) {
            size_t before,live,after,total;
            check(cudaMemGetInfo(&before,&total));
            {
                Fixture f(std::string(argv[1])+"/"+std::to_string(layer));
                check(cudaMemGetInfo(&live,&total));
                printf("{\"kind\":\"residency\",\"layer\":%d,\"free_before\":%zu,\"free_live\":%zu,\"all_weight_bytes\":%zu,\"common_workspace_bytes\":%zu}\n",
                    layer,before,live,f.native.size()+f.alternate.size()+f.scales.size()+f.codes.size(),
                    (3*size_t(inputs)+1+size_t(outputs)*(splits+1))*4);
                for (int count : {0, inputs/8, inputs/4, inputs/2, inputs}) run_case(f,layer,count,rounds,start);
            }
            check(cudaMemGetInfo(&after,&total));
            printf("{\"kind\":\"released\",\"layer\":%d,\"free_after\":%zu}\n",layer,after);
        }
        puts("{\"kind\":\"complete\"}");
    } catch (const std::exception& err) {
        fprintf(stderr,"%s\n",err.what());
        return 1;
    }
}
