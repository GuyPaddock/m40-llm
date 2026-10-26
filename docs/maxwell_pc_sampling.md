# Maxwell PC-Sampling Preflight

This SDK-only tool checks setup of the legacy CUPTI PC-sampling activity
API in a fresh sm_52 context. It does not collect samples, launch kernels,
or link CUPTI into inference. Event/metric counters remain independently
[unavailable on the current stack](maxwell_counters.md).

```bash
source scripts/dev-env.sh
export CUPTI_ROOT=/path/to/cuda_cupti-linux-x86_64-12.8.90-archive
export CUDA_HOME=/opt/cuda
bash scripts/probe_maxwell_pc_sampling.sh 0 > /tmp/m40-pc-sampling.jsonl
```

Request hardware access and ensure the M40 is idle. The wrapper compiles
original C++17 code with warnings as errors, prints its binary SHA-256 to
stderr, executes it once and removes the temporary binary. No packages,
system path, driver settings or toolkit installations are modified.

Exit 0 means configure/enable/disable and context cleanup succeeded, not
that sampling works. Exit 2 means unavailable hardware/API/permissions or
missing SDK. Exit 1 means invalid input, unexpected API error, compilation,
output or cleanup failure. Runtime records include individual API statuses;
all results keep `samples=null` and `has_kernel_measurements=false`.
An invalid argument is rejected before CUDA initialization. An invisible or
invalid device is an error, not successful coverage.

The fixed configuration uses period MID with `samplingPeriod2=0` and
configures before enabling, as required on Pascal and older architectures.
There are no buffer callbacks because this preflight does not collect
activities. Success still requires a separately validated real-kernel
collector with buffer handling, sample/drop counts and PC/function mapping.
Do not interpret setup as measured memory traffic, occupancy or stalls.

NVIDIA documents the legacy activity API for compute 5.2 and newer;
additional latency-sample semantics require compute 6.0. Legacy sampling
serializes kernels, so sampled runs cannot stand in for normal throughput.
The newer continuous API is not an M40 fallback. See the
[CUPTI 12.8 usage guide](https://docs.nvidia.com/cupti/12.8.1/main/main.html#pc-sampling)
and [fixed observation contract](experiments/2026-10-26-maxwell-pc-sampling-hypothesis.md).

An explicit unsupported response closes this SDK/driver route without
root retries or period/SDK changes. Only an explicit permissions error
could justify one separately recorded privileged diagnostic. No inference
kernel or runtime/default promotion follows from capability setup alone.

## Current Result

The [2026-10-26 observation](experiments/2026-10-26-maxwell-pc-sampling-capability.md)
returns `27 / CUPTI_ERROR_NOT_SUPPORTED` during configuration, before
enabling sampling. Context cleanup succeeds and the GPU is empty afterward.
No root retry or collector follows; there are no measured stall samples.
