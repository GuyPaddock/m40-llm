# Bounded Concurrent-Kernel Activity Diagnostic

## Question

Actual official Ollama launch and loaded-code traces establish dispatch,
not GPU stage costs. The qualified M40 profile is available, but static
instruction counts cannot explain comparative execution cost. Test whether
CUDA 12.8 CUPTI concurrent-kernel activities provide usable GPU timestamps
on this M40/driver before building another inference candidate.

## Contract

- Extend only the external SDK observer, behind explicit
  `M40LLM_CUDA_TRACE_ACTIVITY=1`. Unset/zero retains launch-only behavior.
  No inference, kernel, model, runtime-default or Cargo change.
- Enable only `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL`, never serialized
  kernel profiling, PC sampling or hardware counters. Unsupported setup
  is a reported error, not a passing trace or permission to change SDKs.
- Use bounded host buffers and record counts. Preserve kernel names,
  correlation/context/stream IDs, GPU start/end timestamps, geometry and
  reported launch resources. Count dropped records and invalid timestamps;
  incomplete or dropped captures cannot support attribution.
- First run the existing sm_52 two-launch smoke. Require actual output 96,
  exactly two completed activities with valid durations, correlation to
  successful observed launches, and no record for the rejected launch.
  Check unset/zero and invalid-setting behavior separately.
- Only after that passes, trace one isolated official Ollama 0.40.11
  request using the canonical 65 prompt IDs and sixteen generated tokens.
  Verify original model/backend hashes, fresh KV and no other GPU work.
  Separate startup/warmup, prefill and decode by actual executed shapes and
  ordered output-projection boundaries. Reject ambiguous attribution rather
  than dividing whole-process totals by sixteen.
- Preserve actual output, natural stopping, GPU/process evidence, clean
  shutdown, complete callback/activity counts and the instrumented wall
  interval. No builds, tests or MemPalace during GPU observation.

## Limits And Delivery

Timestamp spans are observer-perturbed GPU execution, not unprofiled
latency, end-to-end qualification, achieved occupancy or measured DRAM
traffic. Different activation precision remains explicit. This short trace
cannot replace the 512-token workload or its exact-output requirements.
No geometry/cache/layout rescue or runtime change follows automatically.

Atomic sequence:
1. `docs(research): bound GPU activity attribution`: this contract.
2. `feat(tools): observe concurrent kernel activity`: SDK observer option,
   bounded ownership and behavioral smoke validation.
3. `docs(perf): record GPU activity checkpoint`: raw evidence, independent
   audit, limitations and next action. If unavailable, record that result
   without a new collector or SDK/privilege workaround.

Qualified exact status stays 49.146 HTTP tok/s versus 43.106 official
Ollama. The 53.06 and 56.037 HTTP targets remain unmet. Global compressed
KV defaults, qualified dense control and all original gates are unchanged.
