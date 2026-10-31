# Legacy Maxwell PC Sampling Is Unavailable Here

## Result

The fixed preflight returns **27 / CUPTI_ERROR_NOT_SUPPORTED** from
`cuptiActivityConfigurePCSampling`. It never enables sampling. Context
creation/destruction succeed, the executable exits 2 and final GPU compute
state is empty. This is not an insufficient-privileges response; no root
retry, different sampling period, toolkit or driver change follows.

This check uses a different API from the already-unavailable legacy
event/metric groups. Neither route yields measured stalls, occupancy,
memory traffic or bandwidth on this SDK/driver combination. It does not
establish that all Maxwell/driver/toolkit combinations lack PC sampling.

## Fixed Observation And Provenance

- Clean implementation `37b2def338f0f9185bda56b469d668fb12e37602`, after
  contract `5edf0ca`. No model, kernel launch, timing trial or allocation of
  model buffers occurs. One user observation runs at UID 1000.
- Tesla M40 24GB, sm_52, UUID
  `GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`, driver 580.159.04;
  Driver API reports 13000. This is not use of a CUDA 13 build toolkit.
- Built with the existing CUPTI 12.8.90 SDK and CUDA 12.8 headers;
  compiled/runtime CUPTI API both report 26. Fixed MID period,
  `samplingPeriod2=0`, full SDK configuration structure size.
- Source SHA-256
  `b41723cc3708dd37466220686e4375199eded0f874532e0cd38e19268cd758d6`;
  standalone binary SHA-256
  `e9bf42338ff9264a9e9ea71c8838e023d8375ed37c941ff47daee92df1823c61`.
  Binary retained in ignored `target/research-tools/cuda-pc-sampling-probe`.
- No other GPU process precedes or follows the probe. No inference,
  builds, tests or MemPalace overlap observation. Process is terminal;
  there is no outstanding session or delayed collection to wait for.

The original tool is outside Cargo/inference and uses SDK declarations,
not copied NVIDIA sample implementation. Its wrapper compiles with
`-std=c++17 -O2 -Wall -Wextra -Werror`, embeds the SDK library rpath and
does not install dependencies or alter the system path.

## Checks And Limits

Standalone compilation, shell syntax and Rust formatting pass. Seven
malformed device arguments and extra arguments return 1 before CUDA;
missing SDK returns 2. A separate hidden-device negative check returns 1,
with null measurements and no setup support, rather than passing coverage.
The independent artifact audit validates the exact API sequence, fixed
device/settings, unsupported status, successful cleanup and null samples.
Ten corruption controls reject incomplete cleanup, altered failure/config,
false success and invented measurements.

The preflight intentionally has no activity-buffer callbacks and launches
no kernels. Configuration fails before enabling or attempting collection,
so this is evidence of setup unavailability, not an empty sampling run or
a finding about which instructions stall. NVIDIA's
[legacy PC-sampling documentation](https://docs.nvidia.com/cupti/12.8.1/main/main.html#pc-sampling)
describes sm_52 support and serialized execution. Setup success would still
have required buffer/sample/drop validation before real-kernel attribution.
Compute 6.0 latency-sample semantics and newer continuous APIs must not be
applied to this M40.

## Evidence And Decision

- [Contract](2026-10-26-maxwell-pc-sampling-hypothesis.md)
- [Raw API records](2026-10-26-maxwell-pc-sampling-capability.jsonl)
- [Process, hash and GPU log](2026-10-26-maxwell-pc-sampling-capability.log)
- [Independent audit](2026-10-26-maxwell-pc-sampling-audit.sh),
  [derived result](2026-10-26-maxwell-pc-sampling-audit.json) and
  [checksums](2026-10-26-maxwell-pc-sampling.sha256)
- [Tool usage](../maxwell_pc_sampling.md)

Close this capability route without a collector or counter/SDK rescue.
Continue with ordinary CUDA-event stage timings and compiled/address-model
evidence, explicitly labeling unknown stalls and traffic. A new kernel
experiment still needs an independent dominant-cost hypothesis; lack of
PC sampling does not validate retuning or combining closed candidates.

No new throughput result or runtime/default change. Qualified exact status
stays **49.146 HTTP tok/s** versus **43.106 official Ollama**; both original
**53.06** and **56.037** HTTP tok/s requirements remain unmet and active.
