// External CUPTI diagnostics only; never included in the inference build.
#pragma once
#include <cupti.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>

class KernelActivityTrace {
  public:
    void enable(FILE *file, std::mutex &mutex, void (*write_string)(FILE *, const char *)) {
        file_ = file;
        mutex_ = &mutex;
        write_string_ = write_string;
        instance_ = this;
        require(cuptiActivityRegisterCallbacks(request_buffer, complete_buffer), "register activities");
        require(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "enable concurrent kernels");
        is_enabled_ = true;
    }

    void finish() {
        if (!is_enabled_) return;
        require(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "disable concurrent kernels");
        require(cuptiActivityFlushAll(0), "flush completed activities");
        std::lock_guard<std::mutex> lock(*mutex_);
        std::fprintf(file_, "{\"kind\":\"cuda_activity_complete\",\"kernel_records\":%zu,"
                     "\"dropped_records\":%zu,\"invalid_timestamps\":%zu,"
                     "\"buffer_bytes\":%zu,\"maximum_buffers\":%zu,"
                     "\"record_limit\":%zu}\n", records_, dropped_, invalid_timestamps_,
                     buffer_bytes, maximum_buffers, record_limit);
        is_enabled_ = false;
    }

  private:
    static constexpr size_t buffer_bytes = 1024 * 1024;
    static constexpr size_t maximum_buffers = 8;
    static constexpr size_t record_limit = 100000;
    inline static KernelActivityTrace *instance_ = nullptr;
    FILE *file_ = nullptr;
    std::mutex *mutex_ = nullptr;
    void (*write_string_)(FILE *, const char *) = nullptr;
    bool is_enabled_ = false;
    std::atomic<size_t> buffers_{0};
    size_t records_ = 0, dropped_ = 0, invalid_timestamps_ = 0;

    [[noreturn]] void fail(const char *operation, int status, const char *message) {
        std::lock_guard<std::mutex> lock(*mutex_);
        std::fputs("{\"kind\":\"cuda_activity_error\",\"operation\":", file_);
        write_string_(file_, operation);
        std::fprintf(file_, ",\"status\":%d,\"message\":", status);
        write_string_(file_, message);
        std::fputs("}\n", file_);
        std::fflush(file_);
        std::fprintf(stderr, "CUDA activity observation failed: %s: %s\n", operation, message);
        std::_Exit(2);
    }

    void require(CUptiResult status, const char *operation) {
        if (status == CUPTI_SUCCESS) return;
        const char *message = nullptr;
        cuptiGetResultString(status, &message);
        fail(operation, static_cast<int>(status), message ? message : "unknown CUPTI error");
    }

    static void CUPTIAPI request_buffer(uint8_t **buffer, size_t *size, size_t *maximum_records) {
        auto &trace = *instance_;
        if (trace.buffers_.fetch_add(1) >= maximum_buffers)
            trace.fail("allocate activity buffer", -1, "activity buffer limit exceeded");
        // malloc supplies the alignment required by CUPTI activity records.
        *buffer = static_cast<uint8_t *>(std::malloc(buffer_bytes));
        if (!*buffer) trace.fail("allocate activity buffer", -1, "host allocation failed");
        *size = buffer_bytes;
        *maximum_records = 0;
    }

    static void CUPTIAPI complete_buffer(CUcontext context, uint32_t stream_id,
                                         uint8_t *buffer, size_t, size_t valid_bytes) {
        auto &trace = *instance_;
        CUpti_Activity *record = nullptr;
        while (valid_bytes != 0) {
            const auto status = cuptiActivityGetNextRecord(buffer, valid_bytes, &record);
            if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
            trace.require(status, "read activity record");
            if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
                trace.fail("read activity kind", -1, "unexpected activity kind");
            trace.record(*reinterpret_cast<const CUpti_ActivityKernel9 *>(record));
        }
        size_t dropped = 0;
        trace.require(cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped),
                      "read dropped activity count");
        {
            std::lock_guard<std::mutex> lock(*trace.mutex_);
            trace.dropped_ += dropped;
        }
        std::free(buffer);
        trace.buffers_.fetch_sub(1);
    }

    void record(const CUpti_ActivityKernel9 &kernel) {
        std::lock_guard<std::mutex> lock(*mutex_);
        if (records_ == record_limit) {
            ++dropped_;
            return;
        }
        ++records_;
        const bool has_valid_timestamps = kernel.start != 0 && kernel.end > kernel.start;
        if (!has_valid_timestamps) ++invalid_timestamps_;
        std::fputs("{\"kind\":\"cuda_kernel_activity\",\"symbol\":", file_);
        write_string_(file_, kernel.name);
        std::fprintf(file_, ",\"start_ns\":%llu,\"end_ns\":%llu,"
                     "\"has_valid_timestamps\":%s,\"device_id\":%u,\"context_id\":%u,"
                     "\"stream_id\":%u,\"correlation_id\":%u,\"grid\":[%d,%d,%d],"
                     "\"block\":[%d,%d,%d],\"registers_per_thread\":%u,"
                     "\"static_shared_bytes\":%d,\"dynamic_shared_bytes\":%d}\n",
                     static_cast<unsigned long long>(kernel.start),
                     static_cast<unsigned long long>(kernel.end),
                     has_valid_timestamps ? "true" : "false", kernel.deviceId, kernel.contextId,
                     kernel.streamId, kernel.correlationId, kernel.gridX, kernel.gridY, kernel.gridZ,
                     kernel.blockX, kernel.blockY, kernel.blockZ, kernel.registersPerThread,
                     kernel.staticSharedMemory, kernel.dynamicSharedMemory);
        if (std::fflush(file_) != 0) {
            std::perror("write CUDA kernel activity");
            std::_Exit(2);
        }
    }
};
