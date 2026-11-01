// Untimed capability preflight only. No kernels, model buffers or samples.
#include <cuda.h>
#include <cupti.h>

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

namespace {
void print_json_string(const char* value) {
    std::putchar('"');
    for (const auto* cursor = reinterpret_cast<const unsigned char*>(value);
         *cursor != 0; ++cursor) {
        switch (*cursor) {
        case '"': std::fputs("\\\"", stdout); break;
        case '\\': std::fputs("\\\\", stdout); break;
        default:
            if (*cursor < 32) {
                std::printf("\\u%04x", static_cast<unsigned int>(*cursor));
            } else {
                std::putchar(*cursor);
            }
        }
    }
    std::putchar('"');
}

void print_status(const char* api, int code, const char* message) {
    std::fputs("{\"kind\":\"api\",\"api\":", stdout);
    print_json_string(api);
    std::printf(",\"status\":%d,\"message\":", code);
    print_json_string(message ? message : "unknown");
    std::puts("}");
}

bool has_cuda_success(CUresult status, const char* api) {
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    print_status(api, static_cast<int>(status), message);
    return status == CUDA_SUCCESS;
}

int classify_cupti_status(CUptiResult status, const char* api) {
    const char* message = nullptr;
    cuptiGetResultString(status, &message);
    print_status(api, static_cast<int>(status), message);
    switch (status) {
    case CUPTI_SUCCESS: return 0;
    case CUPTI_ERROR_NOT_SUPPORTED:
    case CUPTI_ERROR_API_NOT_IMPLEMENTED:
    case CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED:
    case CUPTI_ERROR_NOT_COMPATIBLE:
    case CUPTI_ERROR_INVALID_KIND:
    case CUPTI_ERROR_INSUFFICIENT_PRIVILEGES: return 2;
    default: return 1;
    }
}

int finish_probe(CUcontext context, int exit_code, bool has_setup_support) {
    bool has_clean_context = true;
    if (context != nullptr) {
        has_clean_context = has_cuda_success(cuCtxDestroy(context), "cuCtxDestroy");
        if (!has_clean_context) {
            exit_code = 1;
        }
    }
    std::printf("{\"kind\":\"result\",\"exit_code\":%d,"
                "\"has_setup_support\":%s,\"has_clean_context\":%s,"
                "\"has_kernel_measurements\":false,\"kernel_launches\":0,"
                "\"samples\":null}\n",
                exit_code, has_setup_support ? "true" : "false",
                has_clean_context ? "true" : "false");
    return std::fflush(stdout) == 0 ? exit_code : 1;
}

bool parse_device_index(int argc, char** argv, int& device_index) {
    if (argc == 1) {
        device_index = 0;
        return true;
    }
    if (argc != 2 || argv[1][0] == '\0') {
        return false;
    }
    for (const char* cursor = argv[1]; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            return false;
        }
    }
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(argv[1], &end, 10);
    if (errno != 0 || *end != '\0' || value > INT_MAX) {
        return false;
    }
    device_index = static_cast<int>(value);
    return true;
}
} // namespace

int main(int argc, char** argv) {
    int device_index = 0;
    if (!parse_device_index(argc, argv, device_index)) {
        std::fputs("Usage: cuda-pc-sampling-probe [nonnegative-device-index]\n", stderr);
        return 1;
    }
    std::printf("{\"kind\":\"preflight\",\"device_index\":%d,\"effective_uid\":%u,"
                "\"compiled_cupti_api\":%d,\"sampling_period\":\"MID\","
                "\"sampling_period2\":0}\n",
                device_index, static_cast<unsigned int>(geteuid()), CUPTI_API_VERSION);
    int driver_version = 0;
    CUdevice device = 0;
    if (!has_cuda_success(cuInit(0), "cuInit") ||
        !has_cuda_success(cuDriverGetVersion(&driver_version), "cuDriverGetVersion") ||
        !has_cuda_success(cuDeviceGet(&device, device_index), "cuDeviceGet")) {
        return finish_probe(nullptr, 1, false);
    }
    char name[256]{};
    int major = 0, minor = 0;
    if (!has_cuda_success(cuDeviceGetName(name, sizeof(name), device), "cuDeviceGetName") ||
        !has_cuda_success(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                             device), "cuDeviceGetAttribute.major") ||
        !has_cuda_success(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                             device), "cuDeviceGetAttribute.minor")) {
        return finish_probe(nullptr, 1, false);
    }
    std::fputs("{\"kind\":\"device\",\"name\":", stdout);
    print_json_string(name);
    std::printf(",\"major\":%d,\"minor\":%d,\"driver_api_version\":%d}\n",
                major, minor, driver_version);
    if (major != 5 || minor != 2) {
        std::fputs("This bounded probe requires sm_52.\n", stderr);
        return finish_probe(nullptr, 2, false);
    }
    CUcontext context = nullptr;
    if (!has_cuda_success(cuCtxCreate(&context, 0, device), "cuCtxCreate")) {
        return finish_probe(nullptr, 1, false);
    }
    uint32_t version = 0;
    int exit_code = classify_cupti_status(cuptiGetVersion(&version), "cuptiGetVersion");
    if (exit_code != 0) {
        return finish_probe(context, exit_code, false);
    }
    std::printf("{\"kind\":\"sdk\",\"runtime_cupti_api\":%u}\n", version);
    CUpti_ActivityPCSamplingConfig configuration{};
    configuration.size = sizeof(configuration);
    configuration.samplingPeriod = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MID;
    configuration.samplingPeriod2 = 0;
    exit_code = classify_cupti_status(cuptiActivityConfigurePCSampling(context, &configuration),
                                      "cuptiActivityConfigurePCSampling");
    if (exit_code != 0) {
        return finish_probe(context, exit_code, false);
    }
    exit_code = classify_cupti_status(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING),
                                      "cuptiActivityEnable.PC_SAMPLING");
    if (exit_code != 0) {
        return finish_probe(context, exit_code, false);
    }
    const int disable_code = classify_cupti_status(
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_PC_SAMPLING), "cuptiActivityDisable.PC_SAMPLING");
    // A cleanup failure is an error, not an unavailable setup or passing skip.
    return finish_probe(context, disable_code == 0 ? 0 : 1, disable_code == 0);
}
