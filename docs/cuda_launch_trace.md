# Untimed CUDA Launch Observation

This optional Linux CUPTI callback observer runs outside Cargo and inference.
It records standard and extended driver/runtime launch API exits: symbol,
demangled name, grid/block dimensions, dynamic shared memory, stream, context,
correlation ID, API return status and a host monotonic observation timestamp.
By default it does not synchronize, read tensors, collect hardware counters
or measure GPU elapsed time. Logging perturbs execution; never use it for speed claims.
No NVIDIA sample implementation is copied.

## Concurrent-Kernel Activities

Explicit `M40LLM_CUDA_TRACE_ACTIVITY=1` additionally enables CUPTI concurrent
kernel activity timestamps. Unset/zero preserves launch-only behavior; other
values fail. This does not enable serialized kernel profiling or counters.
Setup failure is an error rather than a successful empty capture.

`cuda_kernel_activity` rows contain GPU start/end nanoseconds, launch
geometry, reported resources and context/stream/correlation IDs. Join them
to callback launch records using context/correlation IDs, not record order
or the callback's host clock. Activity buffers arrive asynchronously and
may not be chronological. There is one activity per executed kernel, not
one per runtime/driver callback. Invalid timestamps are reported explicitly.

Host buffers are bounded to eight outstanding 1 MiB allocations, each
released by its completion callback. The activity cap is 100,000 per
process, independent of the existing callback cap. Graceful shutdown
disables collection, flushes completed activities without forced incomplete
records, then emits `cuda_activity_complete` with counts before unsubscribing.
Require zero drops, no invalid timestamps, a complete final record and
expected executed-kernel coverage. A killed process is incomplete.
`is_timing_measurement=false` continues to identify a diagnostic trace, not
an authoritative throughput trial; `captures_kernel_activity` identifies
whether GPU timestamps were requested.

Spans measure instrumented GPU execution. Logging, collection and host
scheduling perturb the workload; these are not unprofiled kernel latency,
HTTP throughput, occupancy or DRAM traffic. Overlapping spans cannot be
summed as elapsed time. Keep authoritative throughput runs observer-free.

Absolute clocks exceed the exact-integer range of double-precision JSON
tools. Before jq analysis, run `perl scripts/normalize_cuda_activity.pl
TRACE.jsonl > NEW-normalized.jsonl`. This requires 64-bit Perl integers;
it subtracts exact clocks to add integer `duration_ns` and relative start/end
fields, and changes absolute timestamps to strings. Relative times use the
first delivered activity as origin, not necessarily the earliest execution.
Preserve the original trace separately. Invalid ranges fail, not round.
`bash scripts/test_cuda_launch_activity.sh /absolute/new-result-directory`
checks this precision boundary and the real two-launch opt-in behavior.

## Run

Use an idle GPU and the separately installed CUDA 12.x CUPTI SDK:

```bash
source scripts/dev-env.sh
export CUPTI_ROOT=/path/to/cuda_cupti-linux-x86_64-12.8.90-archive
export CUDA_HOME=/opt/cuda
bash scripts/trace_cuda_launches.sh /tmp/new-launch-trace -- command args
```

The wrapper builds a standalone library in ignored `target/research-tools/`
with warnings as errors and sets `LD_PRELOAD` only for the requested process
and its children. The parent shell and ordinary benchmark environment are
unchanged. Do not rebuild that library while traced processes remain live.
Output is `/tmp/new-launch-trace.PID.jsonl`, created exclusively per process.
Existing files, missing SDKs, subscription errors and write failures are errors,
not successful skips. Without `M40LLM_CUDA_TRACE_PREFIX`, the library is inert.

Driver and runtime records can describe the **same** launch. Group by process,
context and correlation ID, or use one domain, rather than summing both.
Successful API return is not proof of completed GPU execution. Kernel arguments,
cooperative launches and graph-launch/graph-node activity are not captured.
Unknown symbols and the 100,000-record process cap are reported explicitly.
Neither an empty trace nor unobserved API families prove absence of GPU work.

## Loaded Module Images

Set `M40LLM_CUDA_TRACE_MODULES=1` for an untimed compiled-code diagnostic.
The observer additionally copies loaded cubin images from CUPTI resource
callbacks to `PREFIX.PID.module-INDEX.cubin` and records their paths, sizes
and module IDs in `cuda_module` rows. Callback-owned bytes are copied before
the callback returns. This requires no CUDA call from the callback.

Capture is off by default; only unset, `0` and `1` are accepted. Images use
exclusive files with mode 0600. Each process permits 64 images, at most
64 MiB each and 256 MiB total. Invalid images, limits, setting errors and IO
failures terminate with a `cuda_module_error`, not a successful partial trace.
The completion record includes module count and bytes. A requested capture
with zero modules is not proof of kernel/code coverage.

For a driver-JIT inspection, launch a separate process with
`CUDA_FORCE_PTX_JIT=1 CUDA_CACHE_DISABLE=1`; preserve those settings with
the trace and compare relevant functions against the exact executable's
embedded SASS. Module capture itself does not prove JIT origin.
The `cuda_trace_environment` row records driver force/disable-JIT and
cache-disable settings inside the observed process; unset values are null.
Keep binary images in ignored research storage. Commit hashes, extracted
relevant SASS/resources and provenance. Do not use traced durations for timing claims,
infer performance from register counts, or globally enable JIT from a source
inspection. Existing launch-only usage and inference builds are unchanged.

Only a normal process exit invokes the final unsubscribe/completeness record.
A killed process, or a runtime that exits without C++ destructors, can leave a
flushed but incomplete trace. For the isolated C++ Ollama runner, finish the
HTTP request, verify its executable/PID, and request graceful SIGINT shutdown.
Accept a complete run only with the final record, zero drops, zero unexpected
API errors and successful unsubscribe. Preserve partial traces as partial.

## Hardware Smoke

The fixture checks actual GPU output after both ordinary and extended launches
and deliberately submits an invalid grid. It fails if sm_52 hardware is absent.
Compile through the configured project environment:

```bash
source scripts/dev-env.sh
mkdir -p target/research-tools
nvcc -std=c++17 -O3 -I"$M40LLM_SYSROOT/usr/include" \
  -gencode=arch=compute_52,code=sm_52 \
  -gencode=arch=compute_52,code=compute_52 \
  examples/cuda_launch_trace_smoke.cu \
  -o target/research-tools/cuda-launch-trace-smoke
bash scripts/trace_cuda_launches.sh /tmp/new-smoke -- \
  target/research-tools/cuda-launch-trace-smoke
```

Expected fixture output is `actual=96`, `expected=96`, `is_pass=true` in JSON.
The trace has ordinary grid `[2,1,1]`, block `[16,1,1]`, 64 shared bytes;
extended grid `[1,1,1]`, block `[64,1,1]`; and an invalid runtime grid with
status 9. On the validated stack, this gives five records for two actual
launches plus one rejected API call, zero drops/unknown symbols and exactly
one deliberate failure. The complete M40 smoke and an explicit no-device
failure pass; this does not validate inference speed or numerical parity.
