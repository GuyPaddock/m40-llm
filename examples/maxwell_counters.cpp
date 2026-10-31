// Optional legacy CUPTI preflight, deliberately outside the Cargo build.
#include <cuda.h>
#include <cupti.h>

#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

static std::string json_string(const std::string &value) {
    std::ostringstream out;
    out << '"';
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char byte : value) {
        if (byte == '"' || byte == '\\') {
            out << '\\' << byte;
        } else if (byte < 0x20) {
            out << "\\u00" << hex[byte >> 4] << hex[byte & 15];
        } else {
            out << byte;
        }
    }
    out << '"';
    return out.str();
}

static std::string status_json(CUptiResult result) {
    const char *name = nullptr;
    cuptiGetResultString(result, &name);
    return "{\"code\":" + std::to_string(static_cast<int>(result)) +
           ",\"name\":" + json_string(name ? name : "unknown") + "}";
}

static void check_driver(CUresult result, const char *operation) {
    if (result != CUDA_SUCCESS) {
        const char *name = nullptr;
        cuGetErrorName(result, &name);
        throw std::runtime_error(std::string(operation) + ": " +
                                 (name ? name : "unknown CUDA error"));
    }
}

class Context {
  public:
    explicit Context(CUdevice device) {
        check_driver(cuCtxCreate(&context_, 0, device), "cuCtxCreate");
    }
    ~Context() { cuCtxDestroy(context_); }
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
    CUcontext get() const { return context_; }

  private:
    CUcontext context_ = nullptr;
};

class EventGroup {
  public:
    explicit EventGroup(CUcontext context)
        : result_(cuptiEventGroupCreate(context, &group_, 0)) {}
    ~EventGroup() {
        if (result_ == CUPTI_SUCCESS) {
            cuptiEventGroupDestroy(group_);
        }
    }
    EventGroup(const EventGroup &) = delete;
    EventGroup &operator=(const EventGroup &) = delete;
    CUptiResult get_result() const { return result_; }

  private:
    CUpti_EventGroup group_ = nullptr;
    CUptiResult result_;
};

static int probe(int device_index) {
    check_driver(cuInit(0), "cuInit");
    CUdevice device = 0;
    check_driver(cuDeviceGet(&device, device_index), "cuDeviceGet");
    int major = 0, minor = 0, driver_version = 0;
    char device_name[256] = {};
    check_driver(cuDeviceGetName(device_name, sizeof(device_name), device),
                 "cuDeviceGetName");
    check_driver(cuDeviceGetAttribute(&major,
                 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
                 "compute capability major");
    check_driver(cuDeviceGetAttribute(&minor,
                 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
                 "compute capability minor");
    check_driver(cuDriverGetVersion(&driver_version), "cuDriverGetVersion");
    if (major != 5 || minor != 2) {
        throw std::runtime_error("this probe requires a Maxwell sm_52 device");
    }
    Context context(device);
    uint32_t cupti_version = 0, metric_count = 0, domain_count = 0;
    auto version_status = cuptiGetVersion(&cupti_version);
    auto catalog_status = cuptiDeviceGetNumMetrics(device, &metric_count);
    std::vector<CUpti_MetricID> metrics;
    if (catalog_status == CUPTI_SUCCESS) {
        metrics.resize(metric_count);
        size_t bytes = metrics.size() * sizeof(CUpti_MetricID);
        catalog_status = cuptiDeviceEnumMetrics(device, &bytes, metrics.data());
        if (catalog_status == CUPTI_SUCCESS) {
            metrics.resize(bytes / sizeof(CUpti_MetricID));
        } else {
            metrics.clear();
        }
    }
    auto domain_status = cuptiDeviceGetNumEventDomains(device, &domain_count);
    EventGroup group(context.get());

    std::cout << "{\"kind\":\"maxwell_counter_preflight\",\"device\":"
              << json_string(device_name) << ",\"device_index\":" << device_index
              << ",\"sm\":52,\"driver_api_version\":" << driver_version
              << ",\"cuda_headers_version\":" << CUDA_VERSION
              << ",\"cupti_api_version\":" << cupti_version
              << ",\"effective_uid\":" << geteuid()
              << ",\"version_status\":" << status_json(version_status)
              << ",\"catalog_status\":" << status_json(catalog_status)
              << ",\"domain_status\":" << status_json(domain_status)
              << ",\"domain_count\":";
    if (domain_status == CUPTI_SUCCESS) {
        std::cout << domain_count;
    } else {
        std::cout << "null";
    }
    std::cout << ",\"event_group_status\":" << status_json(group.get_result())
              << ",\"collection_status\":"
              << json_string(group.get_result() == CUPTI_SUCCESS
                     ? "setup-only-not-measured" : "unavailable")
              << ",\"catalog\":[";
    bool is_first = true;
    for (auto metric : metrics) {
        std::array<char, 4096> name{};
        size_t bytes = name.size();
        auto result = cuptiMetricGetAttribute(metric, CUPTI_METRIC_ATTR_NAME,
                                              &bytes, name.data());
        if (!is_first) {
            std::cout << ',';
        }
        is_first = false;
        std::cout << "{\"id\":" << metric << ",\"name\":";
        if (result == CUPTI_SUCCESS) {
            name.back() = '\0';
            std::cout << json_string(name.data());
        } else {
            std::cout << "null";
        }
        std::cout << ",\"status\":" << status_json(result) << '}';
    }
    std::cout << "],\"target_metrics\":[";
    is_first = true;
    for (const char *name : {"dram_read_bytes", "dram_write_bytes",
                            "achieved_occupancy", "sm_efficiency",
                            "gld_efficiency", "gst_efficiency"}) {
        CUpti_MetricID metric = 0;
        auto result = cuptiMetricGetIdFromName(device, name, &metric);
        if (!is_first) {
            std::cout << ',';
        }
        is_first = false;
        std::cout << "{\"name\":" << json_string(name)
                  << ",\"lookup_status\":" << status_json(result)
                  << ",\"value\":null}";
    }
    std::cout << "],\"kernel_measurements\":null}\n";
    // A catalog or successful empty-group setup is never a measurement.
    return group.get_result() == CUPTI_SUCCESS ? 0 : 2;
}

int main(int argc, char **argv) {
    try {
        if (argc > 2) {
            throw std::runtime_error("usage: maxwell_counters [device_index]");
        }
        int device_index = 0;
        if (argc == 2) {
            std::string argument(argv[1]);
            size_t consumed = 0;
            device_index = std::stoi(argument, &consumed);
            if (consumed != argument.size() || device_index < 0) {
                throw std::runtime_error("device_index must be nonnegative");
            }
        }
        return probe(device_index);
    } catch (const std::exception &error) {
        std::cout << "{\"kind\":\"maxwell_counter_preflight\",\"error\":"
                  << json_string(error.what())
                  << ",\"kernel_measurements\":null}\n";
        return 1;
    }
}
