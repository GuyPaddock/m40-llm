// Optional untimed CUPTI callback observer; never linked into inference.
#include <cuda_runtime_api.h>
#include <cupti.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <unistd.h>

namespace {

void write_json_string(FILE *file, const char *value) {
    std::fputc('"', file);
    if (value) {
        for (const unsigned char *p = reinterpret_cast<const unsigned char *>(value); *p; ++p) {
            if (*p == '"' || *p == '\\') {
                std::fputc('\\', file);
                std::fputc(*p, file);
            } else if (*p < 0x20) {
                std::fprintf(file, "\\u%04x", *p);
            } else {
                std::fputc(*p, file);
            }
        }
    }
    std::fputc('"', file);
}

struct Geometry {
    std::array<unsigned, 3> grid;
    std::array<unsigned, 3> block;
    size_t shared_bytes;
    const void *stream;
};

bool read_geometry(CUpti_CallbackDomain domain, CUpti_CallbackId id,
                   const void *params, Geometry &out) {
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        if (id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
            id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000) {
            const auto &p = *static_cast<const cudaLaunchKernel_v7000_params *>(params);
            out = {{p.gridDim.x, p.gridDim.y, p.gridDim.z},
                   {p.blockDim.x, p.blockDim.y, p.blockDim.z}, p.sharedMem, p.stream};
            return true;
        }
        if (id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060 ||
            id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060) {
            const auto &p = *static_cast<const cudaLaunchKernelExC_v11060_params *>(params);
            if (!p.config) return false;
            const auto &c = *p.config;
            out = {{c.gridDim.x, c.gridDim.y, c.gridDim.z},
                   {c.blockDim.x, c.blockDim.y, c.blockDim.z}, c.dynamicSmemBytes, c.stream};
            return true;
        }
    }
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        if (id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
            id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz) {
            const auto &p = *static_cast<const cuLaunchKernel_params *>(params);
            out = {{p.gridDimX, p.gridDimY, p.gridDimZ},
                   {p.blockDimX, p.blockDimY, p.blockDimZ}, p.sharedMemBytes, p.hStream};
            return true;
        }
        if (id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx ||
            id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz) {
            const auto &p = *static_cast<const cuLaunchKernelEx_params *>(params);
            if (!p.config) return false;
            const auto &c = *p.config;
            out = {{c.gridDimX, c.gridDimY, c.gridDimZ},
                   {c.blockDimX, c.blockDimY, c.blockDimZ}, c.sharedMemBytes, c.hStream};
            return true;
        }
    }
    return false;
}

class LaunchTrace {
  public:
    LaunchTrace() {
        const char *prefix = std::getenv("M40LLM_CUDA_TRACE_PREFIX");
        if (!prefix || !*prefix) return;
        const std::string path = std::string(prefix) + "." + std::to_string(getpid()) + ".jsonl";
        const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) {
            std::perror("create new CUDA launch trace");
            std::_Exit(2);
        }
        file_ = fdopen(fd, "w");
        if (!file_) {
            close(fd);
            std::perror("open CUDA launch trace");
            std::_Exit(2);
        }
        prefix_ = prefix;
        const char *modules = std::getenv("M40LLM_CUDA_TRACE_MODULES");
        if (modules && std::strcmp(modules, "0") != 0 && std::strcmp(modules, "1") != 0) {
            fail_module("M40LLM_CUDA_TRACE_MODULES must be 0 or 1");
        }
        captures_modules_ = modules && std::strcmp(modules, "1") == 0;
        require(cuptiSubscribe(&subscriber_, callback, this), "subscribe");
        if (captures_modules_) {
            require(cuptiEnableCallback(1, subscriber_, CUPTI_CB_DOMAIN_RESOURCE,
                                       CUPTI_CBID_RESOURCE_MODULE_LOADED), "module capture");
        }
        const CUpti_CallbackId runtime_ids[] = {
            CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
            CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000,
            CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060,
            CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060,
        };
        const CUpti_CallbackId driver_ids[] = {
            CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
            CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
            CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx,
            CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
        };
        for (auto id : runtime_ids) {
            require(cuptiEnableCallback(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API, id), "runtime");
        }
        for (auto id : driver_ids) {
            require(cuptiEnableCallback(1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API, id), "driver");
        }
        std::fprintf(file_, "{\"kind\":\"cuda_trace_start\",\"pid\":%ld,"
                     "\"is_timing_measurement\":false,\"record_limit\":%zu,"
                     "\"captures_modules\":%s,\"module_record_limit\":%zu,\"module_byte_limit\":%zu}\n",
                     static_cast<long>(getpid()), record_limit,
                     captures_modules_ ? "true" : "false", module_limit, total_byte_limit);
        std::fputs("{\"kind\":\"cuda_trace_environment\",\"cuda_force_ptx_jit\":", file_);
        const auto write_env = [this](const char *name) {
            const char *value = std::getenv(name);
            if (value) write_json_string(file_, value);
            else std::fputs("null", file_);
        };
        write_env("CUDA_FORCE_PTX_JIT");
        std::fputs(",\"cuda_disable_ptx_jit\":", file_);
        write_env("CUDA_DISABLE_PTX_JIT");
        std::fputs(",\"cuda_cache_disable\":", file_);
        write_env("CUDA_CACHE_DISABLE");
        std::fputs("}\n", file_);
        std::fflush(file_);
    }

    ~LaunchTrace() {
        if (!file_) return;
        const auto status = cuptiUnsubscribe(subscriber_);
        std::fprintf(file_, "{\"kind\":\"cuda_trace_complete\",\"launch_records\":%zu,"
                     "\"dropped_launches\":%zu,\"failed_launches\":%zu,"
                     "\"unknown_symbols\":%zu,\"unsubscribe_status\":%d,"
                     "\"module_records\":%zu,\"module_bytes\":%zu}\n",
                     records_, dropped_, failed_, unknown_, static_cast<int>(status),
                     module_records_, module_bytes_);
        if (std::fclose(file_) != 0) std::perror("finish CUDA launch trace");
    }

    LaunchTrace(const LaunchTrace &) = delete;
    LaunchTrace &operator=(const LaunchTrace &) = delete;

  private:
    static constexpr size_t record_limit = 100000;
    static constexpr size_t module_limit = 64;
    static constexpr size_t image_byte_limit = 64 * 1024 * 1024;
    static constexpr size_t total_byte_limit = 256 * 1024 * 1024;
    FILE *file_ = nullptr;
    std::string prefix_;
    bool captures_modules_ = false;
    CUpti_SubscriberHandle subscriber_ = nullptr;
    std::mutex mutex_;
    size_t records_ = 0, dropped_ = 0, failed_ = 0, unknown_ = 0;
    size_t module_records_ = 0, module_bytes_ = 0;

    [[noreturn]] void fail_module(const char *message) {
        std::fputs("{\"kind\":\"cuda_module_error\",\"message\":", file_);
        write_json_string(file_, message);
        std::fputs("}\n", file_);
        std::fflush(file_);
        std::fprintf(stderr, "CUDA module capture failed: %s\n", message);
        std::_Exit(2);
    }

    void record_module(const CUpti_ModuleResourceData *module) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!module || !module->pCubin || module->cubinSize < 4 ||
            module->cubinSize > image_byte_limit) fail_module("invalid or oversized module image");
        if (std::memcmp(module->pCubin, "\x7f" "ELF", 4) != 0)
            fail_module("module image is not an ELF cubin");
        if (module_records_ == module_limit || module->cubinSize > total_byte_limit - module_bytes_)
            fail_module("module capture limit exceeded");
        const std::string path = prefix_ + "." + std::to_string(getpid()) + ".module-"
            + std::to_string(module_records_) + ".cubin";
        const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) fail_module("cannot exclusively create module image");
        FILE *image = fdopen(fd, "wb");
        if (!image) {
            close(fd);
            fail_module("cannot open module image stream");
        }
        // Resource image pointers expire when this callback returns.
        const size_t written = std::fwrite(module->pCubin, 1, module->cubinSize, image);
        const int status = std::fclose(image);
        if (written != module->cubinSize || status != 0) fail_module("cannot finish module image");
        std::fprintf(file_, "{\"kind\":\"cuda_module\",\"module_id\":%u,\"image_index\":%zu,"
                     "\"cubin_bytes\":%zu,\"path\":", module->moduleId, module_records_, module->cubinSize);
        write_json_string(file_, path.c_str());
        std::fputs("}\n", file_);
        ++module_records_;
        module_bytes_ += module->cubinSize;
        if (std::fflush(file_) != 0) fail_module("cannot flush module record");
    }

    void require(CUptiResult status, const char *operation) {
        if (status == CUPTI_SUCCESS) return;
        const char *name = nullptr;
        cuptiGetResultString(status, &name);
        std::fprintf(file_, "{\"kind\":\"cuda_trace_error\",\"operation\":");
        write_json_string(file_, operation);
        std::fprintf(file_, ",\"status\":%d,\"message\":", static_cast<int>(status));
        write_json_string(file_, name);
        std::fputs("}\n", file_);
        std::fflush(file_);
        std::fprintf(stderr, "CUDA launch tracing failed: %s: %s\n", operation, name ? name : "unknown");
        std::_Exit(2);
    }

    static void CUPTIAPI callback(void *user, CUpti_CallbackDomain domain,
                                  CUpti_CallbackId id, const void *data) {
        auto &trace = *static_cast<LaunchTrace *>(user);
        if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
            if (id == CUPTI_CBID_RESOURCE_MODULE_LOADED && trace.captures_modules_) {
                const auto *resource = static_cast<const CUpti_ResourceData *>(data);
                trace.record_module(resource
                    ? static_cast<const CUpti_ModuleResourceData *>(resource->resourceDescriptor) : nullptr);
            }
            return;
        }
        const auto &call = *static_cast<const CUpti_CallbackData *>(data);
        if (call.callbackSite != CUPTI_API_EXIT) return;
        Geometry geometry{};
        if (!call.functionParams || !read_geometry(domain, id, call.functionParams, geometry)) return;
        trace.record(domain, call, geometry);
    }

    void record(CUpti_CallbackDomain domain, const CUpti_CallbackData &call, const Geometry &g) {
        const int status = call.functionReturnValue
            ? (domain == CUPTI_CB_DOMAIN_RUNTIME_API
                ? static_cast<int>(*static_cast<const cudaError_t *>(call.functionReturnValue))
                : static_cast<int>(*static_cast<const CUresult *>(call.functionReturnValue)))
            : -1;
        std::lock_guard<std::mutex> lock(mutex_);
        if (records_ == record_limit) {
            ++dropped_;
            return;
        }
        ++records_;
        if (status != 0) ++failed_;
        if (!call.symbolName || !*call.symbolName) ++unknown_;
        int demangle_status = 0;
        char *demangled = call.symbolName
            ? abi::__cxa_demangle(call.symbolName, nullptr, nullptr, &demangle_status) : nullptr;
        const auto observed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        std::fprintf(file_, "{\"kind\":\"cuda_launch\",\"domain\":\"%s\",\"api\":",
                     domain == CUPTI_CB_DOMAIN_RUNTIME_API ? "runtime" : "driver");
        write_json_string(file_, call.functionName);
        std::fputs(",\"symbol\":", file_);
        write_json_string(file_, call.symbolName);
        std::fputs(",\"demangled_symbol\":", file_);
        write_json_string(file_, demangle_status == 0 ? demangled : call.symbolName);
        std::fprintf(file_, ",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shared_bytes\":%zu,"
                     "\"stream\":\"%p\",\"context_id\":%u,\"correlation_id\":%u,"
                     "\"launch_status\":%d,\"host_observed_ns\":%lld}\n",
                     g.grid[0], g.grid[1], g.grid[2], g.block[0], g.block[1], g.block[2],
                     g.shared_bytes, g.stream, call.contextUid, call.correlationId, status,
                     static_cast<long long>(observed));
        std::free(demangled);
        if (std::fflush(file_) != 0) {
            std::perror("write CUDA launch trace");
            std::_Exit(2);
        }
    }
};

LaunchTrace launch_trace;

} // namespace
