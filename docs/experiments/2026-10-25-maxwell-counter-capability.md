# Maxwell Counter Capability Check

The installed CUPTI 12.8.90 SDK can enumerate metric names for Tesla M40,
but cannot create their event groups on this machine. Do not mistake the
154 enumerated names for available measurements. The original SDK
`cupti_query -device 0 -getdomains` returns `CUPTI_ERROR_NOT_SUPPORTED`.
The repository's bounded, untimed [probe](../../examples/cuda_counter_probe.cpp)
also obtains that error when creating groups for `dram_read_throughput`,
`gld_efficiency` and `achieved_occupancy`. This is not a permissions error.
No kernel launches or throughput measurements occur in the probe.

[Raw probe output](2026-10-25-maxwell-counter-capability.txt) records each API
status. Metric-name lookup succeeds; event-group creation fails for all three.
No DRAM traffic, achieved occupancy or load-efficiency result is available.
CUDA-event elapsed time and compiler resource/disassembly evidence remain
usable, but neither measures those counters.

Reproduce with the existing SDK, without downloading packages or changing
the inference build:

```bash
source scripts/dev-env.sh
CUPTI_ROOT=/tmp/m40-llm-tools/cupti-12.8.90/cuda_cupti-linux-x86_64-12.8.90-archive
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I/opt/cuda/include -I"$CUPTI_ROOT/include" \
  examples/cuda_counter_probe.cpp -L"$CUPTI_ROOT/lib" \
  -Wl,-rpath,"$CUPTI_ROOT/lib" -lcupti -lcuda \
  -o target/research-tools/cuda-counter-probe
target/research-tools/cuda-counter-probe
```

The probe is original code using SDK declarations, not adapted NVIDIA sample
code. It compiles warning-free and completes on the M40. It does not link
into inference, install anything or modify driver settings. The available
SDK sample query was compiled unmodified; compiling its larger CUDA callback
sample encountered host-header feature-macro conflicts, so its output is not
used as evidence. The small probe directly tests event-group creation.

Next use an explicitly labeled warp-instruction/32-byte-sector address model
to compare the qualified split4 mapping with one adjacent-pair mapping.
Count each issued request separately; do not describe those static request
footprints as physical DRAM reads or assume that cache reuse is absent.
Only a matched real-input complete-FFN screen can establish a speed benefit.
Prepared down/vocabulary, FP32 activations and runtime defaults remain fixed.
