#include "sparse_q4_probe.cuh"

namespace {
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
