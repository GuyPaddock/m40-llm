// Charged Q4 down projections on held-out, native post-SwiGLU rows.
#include "sparse_q4_probe.cuh"
#include <array>
#include <memory>

namespace {
enum class LiveArm { native, packed, zigzag, threshold_native, threshold_packed, threshold_zigzag };
const std::array<LiveArm, 6> arms = {LiveArm::native,           LiveArm::packed,
                                     LiveArm::zigzag,           LiveArm::threshold_native,
                                     LiveArm::threshold_packed, LiveArm::threshold_zigzag};
const char *labels[] = {"native",           "packed",           "zigzag",
                        "threshold_native", "threshold_packed", "threshold_zigzag"};

void project(Fixture &f, const unsigned *identity, LiveArm arm, float cutoff) {
    switch (arm) {
    case LiveArm::native:
        enqueue(f, dense_native, inputs);
        break;
    case LiveArm::zigzag:
        enqueue(f, dense_zigzag, inputs);
        break;
    case LiveArm::packed:
        m40llm_q4_channels::down_partial<true, false><<<dim3(outputs / 256, splits), 256, 0, f.stream>>>(
            f.x.ptr, identity, f.d_native.ptr, reinterpret_cast<const __half *>(f.d_scales.ptr),
            f.d_codes.ptr, f.partial.ptr, outputs, inputs, inputs, f.count.ptr);
        m40llm_q4_channels::down_reduce<<<outputs / 256, 256, 0, f.stream>>>(f.partial.ptr, f.result.ptr,
                                                                             outputs);
        break;
    case LiveArm::threshold_native:
        enqueue(f, native_charged, 0, cutoff);
        break;
    case LiveArm::threshold_packed:
        enqueue(f, packed_charged, 0, cutoff);
        break;
    case LiveArm::threshold_zigzag:
        enqueue(f, zigzag_charged, 0, cutoff);
        break;
    }
}

struct Error {
    double maximum = 0, relative_rms = 0, cosine = 1;
};

Error check_output(const std::vector<float> &actual, const Reference &matched, const Reference &native) {
    double reference_sq = 0, actual_sq = 0, error_sq = 0, inner = 0;
    Error error;
    for (int i = 0; i < outputs; ++i) {
        const double observed = actual[i], expected = native.sum[i];
        const double delta = std::abs(observed - matched.sum[i]);
        require(std::isfinite(observed) && delta <= 0.0005 * (1 + matched.absolute[i]),
                "matched FP64 disagreement");
        error.maximum = std::max(error.maximum, delta);
        error_sq += (observed - expected) * (observed - expected);
        actual_sq += observed * observed;
        reference_sq += expected * expected;
        inner += observed * expected;
    }
    if (reference_sq) {
        error.relative_rms = std::sqrt(error_sq / reference_sq);
        error.cosine = actual_sq ? inner / std::sqrt(actual_sq * reference_sq) : 0;
    } else {
        require(actual_sq == 0, "nonzero output for zero native reference");
    }
    return error;
}

void time_row(Fixture &f, const unsigned *identity, int layer, int row, int level, float cutoff) {
    Event begin, end;
    for (int round = -1; round < 5; ++round) {
        for (int order = 0; order < int(arms.size()); ++order) {
            const int index = (order + row + level + round + 1) % arms.size();
            const auto wall = Clock::now();
            check(cudaEventRecord(begin.event, f.stream));
            const auto submit = Clock::now();
            for (int i = 0; i < repeats; ++i)
                project(f, identity, arms[index], cutoff);
            check(cudaPeekAtLastError());
            const auto submitted = Clock::now();
            check(cudaEventRecord(end.event, f.stream));
            check(cudaEventSynchronize(end.event));
            const auto done = Clock::now();
            float ms;
            check(cudaEventElapsedTime(&ms, begin.event, end.event));
            using Micros = std::chrono::duration<double, std::micro>;
            printf(
                "{\"kind\":\"trial\",\"layer\":%d,\"row\":%d,\"level\":%d,\"round\":%d,\"order\":%d,"
                "\"arm\":\"%s\",\"repeats\":%d,\"event_us\":%.6f,\"submit_us\":%.6f,\"wall_us\":%.6f}\n",
                layer, row, level, round, order, labels[index], repeats, ms * 1000 / repeats,
                Micros(submitted - submit).count() / repeats, Micros(done - wall).count() / repeats);
        }
    }
}

void measure_row(Fixture &f, const Buffer<unsigned> &identity, const std::vector<unsigned> &all,
                 const std::vector<float> &x, int layer, int row, const std::array<float, 3> &cutoffs) {
    f.x.upload(x.data());
    const Reference dense(f.native, false, x, all), zigzag(f.alternate, true, x, all);
    for (int level = 0; level < 3; ++level) {
        const float cutoff = cutoffs[level];
        std::vector<unsigned> selected;
        double mass = 0, retained = 0;
        for (unsigned i : all) {
            require(std::isfinite(x[i]), "nonfinite captured activation");
            const double magnitude = std::abs(double(x[i]));
            mass += magnitude;
            if (magnitude > cutoff) {
                selected.push_back(i);
                retained += magnitude;
            }
        }
        compact(f, cutoff);
        check(cudaStreamSynchronize(f.stream));
        require(f.count.read()[0] == selected.size(), "live count mismatch");
        const auto indices = f.indices.read();
        const auto values = f.compact.read();
        for (size_t i = 0; i < selected.size(); ++i)
            require(indices[i] == selected[i] && values[i] == x[selected[i]], "compacted row mismatch");
        // A zero cutoff elides exact zeros only, so its full reference is valid.
        std::unique_ptr<Reference> masked, masked_zigzag;
        if (cutoff > 0) {
            masked.reset(new Reference(f.native, false, x, selected));
            masked_zigzag.reset(new Reference(f.alternate, true, x, selected));
        }
        std::vector<float> native_selected;
        for (auto arm : arms) {
            project(f, identity.ptr, arm, cutoff);
            check(cudaStreamSynchronize(f.stream));
            const auto actual = f.result.read();
            const bool alternate = arm == LiveArm::zigzag || arm == LiveArm::threshold_zigzag;
            const Reference *matched = alternate ? &zigzag : &dense;
            if (arm >= LiveArm::threshold_native && masked)
                matched = alternate ? masked_zigzag.get() : masked.get();
            const Error error = check_output(actual, *matched, dense);
            const Error layout_error = alternate ? check_output(actual, *matched, zigzag) : error;
            if (arm == LiveArm::threshold_native)
                native_selected = actual;
            if (arm == LiveArm::threshold_packed)
                require(memcmp(actual.data(), native_selected.data(), outputs * sizeof(float)) == 0,
                        "lossless output mismatch");
            printf(
                "{\"kind\":\"check\",\"layer\":%d,\"row\":%d,\"level\":%d,\"cutoff\":%.9g,\"count\":%zu,"
                "\"mass_fraction\":%.12g,\"arm\":\"%s\",\"matched_max_error\":%.12g,"
                "\"native_relative_rms\":%.12g,\"native_cosine\":%.12g,\"layout_relative_rms\":%.12g,"
                "\"layout_cosine\":%.12g}\n",
                layer, row, level, cutoff, selected.size(), mass ? retained / mass : 1.,
                labels[int(arm)], error.maximum, error.relative_rms, error.cosine,
                layout_error.relative_rms, layout_error.cosine);
        }
        if (row >= 0)
            time_row(f, identity.ptr, layer, row, level, cutoff);
        fflush(stdout);
    }
}
} // namespace

int main(int argc, char **argv) {
    try {
        require(argc == 7, "usage: live PLANES ROWS LAYER ROW_COUNT MEDIAN_CUTOFF UPPER_CUTOFF");
        const int layer = std::stoi(argv[3]), count = std::stoi(argv[4]);
        require((layer == 0 || layer == 17 || layer == 35) && count >= 1 && count <= 48,
                "invalid dimensions");
        const std::array<float, 3> cutoffs = {0.f, std::stof(argv[5]), std::stof(argv[6])};
        require(std::isfinite(cutoffs[2]) && 0 <= cutoffs[1] && cutoffs[1] <= cutoffs[2],
                "invalid cutoffs");
        cudaDeviceProp props;
        check(cudaGetDeviceProperties(&props, 0));
        require(props.major == 5 && props.minor == 2 && strstr(props.name, "M40"), "Tesla M40 required");
        std::vector<unsigned> all(inputs);
        std::iota(all.begin(), all.end(), 0);
        const Bytes bytes =
            read_plane(argv[2], (std::to_string(layer) + ".f32le").c_str(), size_t(count) * inputs * 4);
        size_t before, live, after, total;
        check(cudaMemGetInfo(&before, &total));
        {
            Buffer<unsigned> identity(inputs);
            identity.upload(all.data());
            Fixture f(std::string(argv[1]) + "/" + std::to_string(layer));
            check(cudaMemGetInfo(&live, &total));
            std::vector<float> x(inputs, 0.f);
            measure_row(f, identity, all, x, layer, -1, cutoffs);
            for (int row = 0; row < count; ++row) {
                memcpy(x.data(), bytes.data() + size_t(row) * inputs * 4, inputs * 4);
                measure_row(f, identity, all, x, layer, row, cutoffs);
            }
        }
        check(cudaMemGetInfo(&after, &total));
        printf("{\"kind\":\"complete\",\"layer\":%d,\"rows\":%d,\"free_before\":%zu,\"free_live\":%zu,"
               "\"free_after\":%zu}\n",
               layer, count, before, live, after);
    } catch (const std::exception &error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
