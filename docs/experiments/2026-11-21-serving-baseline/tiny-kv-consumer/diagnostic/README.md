# Historical KV dispatch traced through every TinyLlama layer

Both mixed KV paths completed real TinyLlama F16 generation on the M40.
The staged path and the direct path consumed the same 1,390 prompt IDs
and generated the same 32 IDs, in both the CLI and HTTP diagnostics.
Each trace contains one invocation of its intended attention consumer
at every input position through position 1,421 and every one of the 22
layers. All historical blocks fit within the 64-block selection budget.

| Scope | Wrapper rows per path | Staged median, µs | Direct median, µs |
|---|---:|---:|---:|
| Recent-only prefill | 22,528 | 10.020 | 7.460 |
| Historical prefill | 8,052 | 9.910 | 7.707 |
| Decode after first sample | 682 | 10.064 | 7.760 |

The values are profiler wrapper spans from a single instrumented request
per arm. The wrapper's launch counter identifies the operation. Its
elapsed span does not measure GPU service or enumerate the kernels
inside the wrapper. In the instrumented server processes, sampled GPU
memory peaked at 6,224 MiB staged and 6,208 MiB direct. Each process
owned its own cache. The warmed tracing-off series determines the
complete-request performance verdict and its variability.

The pinned GGUF header has 32 query heads, four KV heads, embedding
width 2,048 and model context 2,048. The local llama.cpp GGUF reader at
d327bf1 independently reproduced those values. On Phase 4,
`DecodeSession::new_for_sequence_with_kv_config` clips staging capacity
to `min(1024 + 64 × 32, 2048) = 2048`. `ExactBlockStagingWorkspace`
requests a per-query-head FP16 key plane, FP16 value plane, 32-bit token
positions and 32-bit counts. With head dimension 64, these add up to
17,039,488 bytes per staged session. The direct arm omits that
workspace and reads the existing mixed cache. The allocation
calculation explains the scale of the observed 16 MiB difference.
Process memory includes CUDA allocator rounding and other allocations.

[Dispatch replay](dispatch.json) · [Staged response](calibrate-0.json) ·
[Direct response](calibrate-1.json) · [Build and device identity](preflight.txt).
The full stage logs remain in `research-1121/raw/tiny-kv-consumer-01`
and are identified by [their hashes](full-traces.sha256). From the serving
baseline directory, run `python3 kv-trace.py RAW_DIRECTORY`.
