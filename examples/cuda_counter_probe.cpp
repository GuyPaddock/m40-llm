// Untimed legacy CUPTI capability probe. No inference or kernel launches.
#include <cuda.h>
#include <cupti.h>

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace {
void require_cuda(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS) return;
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    std::fprintf(stderr, "%s failed: %s\n", operation, message ? message : "unknown");
    std::exit(1);
}

void print_status(const char* operation, CUptiResult status) {
    const char* message = nullptr;
    cuptiGetResultString(status, &message);
    std::printf("%s: %d %s\n", operation, static_cast<int>(status), message ? message : "unknown");
}
}

int main() {
    require_cuda(cuInit(0), "cuInit");
    CUdevice device;
    require_cuda(cuDeviceGet(&device, 0), "cuDeviceGet");
    char name[256]{};
    require_cuda(cuDeviceGetName(name, sizeof(name), device), "cuDeviceGetName");
    CUcontext context;
    require_cuda(cuCtxCreate(&context, 0, device), "cuCtxCreate");
    uint32_t version = 0, domains = 0, metrics = 0;
    print_status("cuptiGetVersion", cuptiGetVersion(&version));
    std::printf("device: %s; CUPTI version: %u; is_counter_measurement: false\n", name, version);
    print_status("cuptiDeviceGetNumEventDomains", cuptiDeviceGetNumEventDomains(device, &domains));
    print_status("cuptiDeviceGetNumMetrics", cuptiDeviceGetNumMetrics(device, &metrics));
    std::printf("reported event domains: %u; reported metrics: %u\n", domains, metrics);
    for (const char* metric : {"dram_read_throughput", "gld_efficiency", "achieved_occupancy"}) {
        std::printf("metric: %s\n", metric);
        CUpti_MetricID id;
        const auto lookup = cuptiMetricGetIdFromName(device, metric, &id);
        print_status("cuptiMetricGetIdFromName", lookup);
        if (lookup != CUPTI_SUCCESS) continue;
        CUpti_EventGroupSets* sets = nullptr;
        const auto creation = cuptiMetricCreateEventGroupSets(context, sizeof(id), &id, &sets);
        print_status("cuptiMetricCreateEventGroupSets", creation);
        if (creation != CUPTI_SUCCESS) continue;
        for (uint32_t pass = 0; pass < sets->numSets; ++pass) {
            for (uint32_t group = 0; group < sets->sets[pass].numEventGroups; ++group) {
                const auto handle = sets->sets[pass].eventGroups[group];
                const auto enabled = cuptiEventGroupEnable(handle);
                print_status("cuptiEventGroupEnable", enabled);
                if (enabled == CUPTI_SUCCESS) print_status("cuptiEventGroupDisable", cuptiEventGroupDisable(handle));
            }
        }
        print_status("cuptiEventGroupSetsDestroy", cuptiEventGroupSetsDestroy(sets));
    }
    require_cuda(cuCtxDestroy(context), "cuCtxDestroy");
    return 0;
}
