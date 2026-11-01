# Legacy Maxwell PC-Sampling Capability

## Purpose

The qualified FFN accounts for about 65% of sampled middle-token GPU time,
but no measured stall evidence explains its remaining cost. Legacy CUPTI
event/metric group creation is already unsupported on this stack, including
as root. Do not retry those APIs or infer stalls from read-only surrogates.

NVIDIA documents a different PC-sampling activity API for compute 5.2 and
newer. It samples active warps and reports PC/stall categories. Additional
latency-sample semantics are documented for compute 6.0 and newer, not M40.
The newer continuous PC-sampling API is not an sm_52 fallback. See the
[CUDA 12.8 usage documentation](https://docs.nvidia.com/cupti/12.8.1/main/main.html#pc-sampling).

## Fixed Preflight

- Use the existing CUPTI 12.8.90 SDK, CUDA 12.8 headers and installed driver.
  No download, installation, driver setting or inference dependency changes.
- Build an original standalone SDK-only executable with warnings as errors.
  Preserve API statuses, SDK/driver versions, device identity and effective
  UID in JSONL. Invalid arguments fail before CUDA initialization.
- Require an idle M40 and a fresh owned Driver API context on device 0.
  Accept only sm_52; unsupported hardware is not a passing observation.
- Configure `CUpti_ActivityPCSamplingConfig` with its full SDK size,
  `CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MID` and `samplingPeriod2=0`, before
  enabling `CUPTI_ACTIVITY_KIND_PC_SAMPLING`, as required on Maxwell.
- Only enable after configuration succeeds; disable if enabled and destroy
  the context on every path. Do not enable event/metric collection or the
  record-info kind independently. Do not launch kernels or allocate model
  buffers. This tests setup, not activity-buffer collection.
- Exit 0 only for successful configure/enable/disable and context cleanup;
  2 for documented unavailable/permission statuses; 1 for invalid inputs or
  unexpected CUDA/CUPTI/cleanup failures. Include explicit null samples and
  `has_kernel_measurements=false`, even on successful setup.
- Make one hardware observation. An unsupported status closes this route
  without privileged retries or alternate periods/SDKs. Only an explicit
  insufficient-privileges status could justify a separate privileged check.
- No other inference, builds, tests or MemPalace work may overlap hardware
  observation. Verify final GPU state and preserve process exit status.

## Decision And Atomic Delivery

1. `docs(research): bound PC sampling preflight`: this contract.
2. `feat(trace): probe legacy PC sampling setup`: original standalone tool,
   optional build wrapper and input/error behavior checks. Commit before
   collecting the hardware result.
3. `docs(research): record PC sampling capability`: raw statuses, independent
   checks, checksums, limits, current status and MemPalace checkpoint.

Successful setup is not stall data. A later, separately bounded collector
would need real-kernel PC records, function/instruction correlation, sample
and drop counts, cleanup and perturbation evidence before attribution.
Legacy PC sampling serializes kernels, so those runs cannot replace normal
E2E timing. Unsupported setup yields no traffic/occupancy/stall conclusions
and no new kernel candidate.

Qualified exact runtime remains 49.146 HTTP tok/s versus 43.106 official
Ollama. The original 53.06 and 56.037 HTTP tok/s requirements remain unmet.
Preserve FP32 activations, prepared down/vocabulary, dense attention for
this workload, runtime defaults and all closed experiment gates.
