# Completed transfer and stream handoff probe

The `cooperative_transfer_probe` example prices the runtime's existing pageable
H2D/D2H methods and explicit producer/consumer stream dependencies. Its isolated
M40 hardware test and seven protocol tests pass, with no ignored or skipped
case. Formatting and the previously recorded CUDA/server all-target Clippy
checks pass. Both probe lifetimes release every tracked allocation; external
residency returns to zero MiB with no compute process. The
[qualification manifest](2026-10-31-dualpace-transfer-qualification.sha256)
retains the executed script, logs and source hashes. Full-grid transfer timing
remains the next step.

## Scope and placement

Use CPU 15 with node 0 memory, then CPU 30 with node 1 memory, in separate
serial runs. `examples/common/cpu_placement.rs` now contains the original CPU
projection probe's placement checks. Both probes require exactly one logical
CPU, a bound local NUMA node and matching host inventory. Extraction preserves
the CPU probe's admission and report fields. Twelve existing CPU/protocol tests,
including explicit AVX2 execution, pass after the change.

The transfer manifest records process status after CUDA context creation so
runtime-created threads remain visible. It also records CPU/NUMA placement,
GPU UUID and PCI bus, topology, driver, clocks, temperature, memory state,
source revision and executable hash. Per-case NUMA maps accompany allocations.
The application submits transfers from one pinned thread; this does not imply
that CUDA creates no internal threads.

Eight activation-size cases use FP32 widths 2048/11008 at batches 1/2/4/8.
Five additional cases cover 1,024-byte single-layer KV append data,
36,864-byte 36-layer append data, 1 MiB and 8 MiB dense per-layer KV caches,
and one 23,953,408-byte original Q8 projection matrix. Sizes describe the
selected model geometry. Contents are deterministic byte patterns, not model
activations or weight values. Every observation changes the fixture generation
so stale copied data fails validation.

All cases retain reusable source, intermediate and destination device buffers
with guards, plus guarded pageable host input/output. Setup, first touch,
per-trial preparation and output verification remain outside timing. The
probe reports persistent bytes and allocation counts; CUDA's internal pageable
staging is included in service latency but is not separately instrumented.

## Cost boundaries

Each operation completes before the next begins:

- H2D copies host bytes through the existing ready-on-return API.
- D2H copies device bytes through the existing completed-readback API.
- The same-stream control performs two D2D copies and waits for decode.
- The handoff case copies on prefill, orders decode after the producer,
  copies the intermediate result on decode, and waits for decode.
- Idle decode/prefill synchronization calls run with no pending work.

The two D2D cases perform identical byte movement. Completing every operation
prevents the next producer from overwriting an intermediate buffer still in
use by the preceding consumer. Payloads, inactive buffers and every guard
are checked after each observation. The ignored hardware test covers two
allocation lifetimes, lengths 1/17/8192/44032 and all six paths.

Host wall timing encloses only repeated API calls and their required
completion. CUDA-event timestamps form an ordered device timeline around that
interval. The pageable APIs do not expose their default stream to the existing
event wrapper, so this timeline includes host gaps and event submission. It
must not be reported as active DMA time. Host-ready wall cost is the primary
input to the scheduling model. No PCIe hardware counters are collected.

Three warm observations per arm precede five alternating pairs. Reuse the
common variation rule to extend unstable series to ten pairs. Directional
and synchronization comparisons are cost profiles, so no adoption threshold
or borderline speed confirmation applies. Raw observations, nanosecond
latency distributions and paired operation-rate statistics are retained.
The default is twenty completed operations per transfer interval and one
thousand calls per idle synchronization interval.

The grid has thirteen transfer cases with two comparison groups each, plus
one idle synchronization group. These costs do not measure GPU projection,
complete decoding or a concurrent serving workload. Initial matrix/KV movement
is recorded for prerequisite pricing; the simulator's decode-ready boundary
still excludes initial migration and prefill.

Use measured completed H2D/D2H costs with
[schema-2 serialized copies](2026-10-31-dualpace-copy-policy.md). Their completion
cost is already included, so additional synchronization can be zero with that
evidence. Idle synchronization is a separate diagnostic, not another mandatory
charge. D2D handoff costs apply only when the proposed execution actually uses
that handoff. Pinned asynchronous transfers and copy/compute overlap remain
unmeasured prerequisites; this probe does not silently assume them.

## Commands and remaining work

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/dualpace/target
# Only after capture completes and GPU process inspection shows idle:
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --example cooperative_transfer_probe -- --include-ignored --test-threads=1
# Commit the hardware-verified probe before real-grid execution.
M40LLM_ENABLE_CUBLAS=1 cargo build --release --features cuda --locked \
  --example cooperative_transfer_probe -j 4
numactl --physcpubind=15 --membind=0 \
  target/release/examples/cooperative_transfer_probe \
  --cpu 15 --numa-node 0 --validate-only --output target/transfers-node0-validation-new.jsonl
numactl --physcpubind=15 --membind=0 \
  target/release/examples/cooperative_transfer_probe \
  --cpu 15 --numa-node 0 --iterations 20 --output target/transfers-node0-timed-new.jsonl
```

Repeat validation and timing with CPU 30/node 1 and new outputs. The executable
requires a clean worktree; timings require release. It rejects runtime/logging
overrides, checks the selected M40 for other compute processes before creating
its context and before each case, and verifies tracked cleanup. Inspect driver
residency externally after process exit too. Timed runs remain serial with CPU
projection pricing and all other hardware experiments.

The [validation manifest](2026-10-31-dualpace-transfer-probe.sha256) retains twelve
passing CPU/protocol tests, non-CUDA and CUDA/server all-target Clippy, and the
hardware executable's compile/link result. The shared placement extraction is
independent of hardware qualification. The transfer example is now qualified
on M40 and ready for its clean release run.
Real-grid validation, both NUMA timing sets, sourced simulation inputs and the
priced full-CPU-decode prerequisite report remain outstanding.
