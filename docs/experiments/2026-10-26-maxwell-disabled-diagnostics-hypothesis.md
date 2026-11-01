# Disabled Forward Diagnostics Cost Study

## Scope

Source `src/infer/forward.rs::log_profiled_op` formats two operation labels
and calls timing/profile logging even when logs are disabled. The caller
also checks snapshot enablement and reads the CPU clock around each op.
The qualified middle-token profile records 396 layer operation boundaries.

Measure this existing CPU wrapper before considering changes. Replay the
396 ordered layer/stage identities from the frozen qualified profile,
repeat for 512 nominal token positions and compare against traversal only.
This is not model inference, the full host enqueue path, a GPU timing or
a rigorous removable-latency bound. Labels are preconstructed; no CUDA
enqueue, counter recording, device call or model allocation is simulated.
Do not confuse 396 repeated boundaries with an actual 512-token generation.

Use the actual private logging function, snapshot check and CPU clocks,
not a rewritten approximation. Keep all log variables disabled, GPUs hidden,
release build, CPU core 0 affinity and five alternating paired trials.
Report raw totals and loop counts, per-512-replay cost, per-token cost,
environment, source/executable/capture hashes and unchanged counters/bytes.
Bound each trial to eight replay iterations and preserve both arm checksums.
The process is not CPU-exclusive and CPU frequency is not controlled.

The profile SHA is
`4eeec0020ed346341cfd816cc3ef28ae1e98ae88f2b7a6f24a378b75927b126c`.
Require ordered 36 layers with all eleven expected stages and a single
selected session/step. Reject malformed or incomplete input. Add behavioral
tests for traversal/counter preservation and rejected stage coverage.

## Decision

Compare observed wrapper cost with the qualified 10417.843301 ms HTTP
interval as context only. CPU/GPU overlap prevents interpreting that ratio
as an E2E speedup. If measured cost is below 3%, close this standalone
performance hypothesis without a runtime fast path. If larger, require a
separate controlled runtime diagnostic before any optimization is promoted.
Never aggregate it with closed kernel probes to bypass their gates.

No CUDA code, inference arithmetic, environment caching, runtime defaults,
weight layouts, prefill, attention or scheduling changes are authorized.
Both original 53.06 HTTP tok/s and 1.30x fresh Ollama remain required.

## Atomic Delivery

1. `docs(research): bound diagnostics CPU study`: this contract.
2. `test(bench): measure disabled diagnostics`: private-function CPU replay
   and behavioral checks; depends on 1. Commit before release timing.
3. `docs(perf): record diagnostics CPU cost`: raw pairs, audited scope and
   decision; depends on 2. Record the checkpoint in MemPalace.
