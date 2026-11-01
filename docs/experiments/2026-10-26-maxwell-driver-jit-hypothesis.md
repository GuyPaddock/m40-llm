# Driver JIT Code-Generation Check

## Independent Question

The pinned official Ollama backend ships sm_52 PTX but no sm_52 cubin.
M01 inspected an offline ptxas assembly, explicitly not the driver's loaded
code. M40-LLM instead embeds native sm_52 SASS plus compatible PTX.
Check whether the installed driver's compilation is materially different
before proposing another exact FFN implementation or compiler experiment.

Extend the existing untimed CUPTI observer to optionally copy loaded module
cubin images from resource callbacks. Use the SDK's documented callback
types, with original implementation code, no borrowed NVIDIA sample body.
Capture synchronously while callback data remains valid. Never call CUDA
from the callback, modify tensors, or use observer durations for speed claims.

## Fixed Diagnostic

- Opt-in `M40LLM_CUDA_TRACE_MODULES=1`; existing launch-only behavior remains
  the default. Require valid 0/1 values, exclusive new output files and
  explicit completeness/error records.
- Bound captures to 64 images, 64 MiB per image and 256 MiB total per process.
  Fail clearly on invalid images, limits or IO errors; never silently omit
  requested module data. Keep cubins in ignored research storage; commit
  hashes, extracted relevant SASS/resources and trace evidence, not binaries.
- Validate ordinary observer behavior and module capture using the existing
  GPU output smoke. Test hidden-device and invalid-setting failures and an
  image-write failure. No new CUDA inference kernel or loader is introduced.
- Commit verified observer code before the research observation. On idle
  M40, run the existing FFN primitive parity tests under forced PTX JIT with
  driver disk caching disabled. Serialize all GPU work and capture module
  images; this is correctness/code observation, not a throughput benchmark.
- Compare original qualified gate/up and prepared down/head loaded code
  against embedded sm_52 words/resources from that exact executable.
  Do not compare a rejected experimental kernel or infer performance from
  register counts/static instruction length alone.

If relevant instructions/resources match, close this standalone compiler
hypothesis without another timing matrix. If different, record the exact
differences and plan a separate bounded experiment that changes only gate/up
while retaining the prepared down/head control. A changed module does not
authorize a runtime selector or global forced-JIT recommendation.

If resource capture is unavailable, report the actual limitation. Do not
reuse offline ptxas as proof of loaded JIT code or retry unavailable Maxwell
hardware metric APIs. No geometry/cache/register/closed-kernel rescue follows.

## Gates And Delivery

1. `docs(research): bound driver JIT inspection`: this contract.
2. `feat(trace): capture loaded CUDA modules`: optional observer capture,
   behavior/error validation and instructions; depends on 1.
3. `docs(research): record driver JIT evidence`: clean source/build/module
   identities, relevant compiled-code comparisons, artifacts and decision;
   depends on 2. Update performance/status and MemPalace without claiming TPS.

Runtime defaults, original Q8_0 bytes, FP32 arithmetic, SGEMM prefill, dense
reference and exact-output gates remain fixed. Completion still requires
53.06 HTTP tok/s and 1.30x fresh official Ollama. Both are currently unmet.
