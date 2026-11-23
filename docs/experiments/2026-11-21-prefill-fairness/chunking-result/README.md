# Dense prefill chunking on the M40

**PASS for the frozen mixed-request fairness gate.** The 16-position
policy reduced the short request's maximum inter-token gap by a paired
median of **99.028%**, with a 95% bootstrap interval of
**99.024–99.030%**. Every pair cleared the 80% gate. The medium
request's median client-to-first-sample time rose **4.632%**, within
the 5% limit. Median group makespan changed **−0.287%** with tracing off.

| Median across five repetitions per arm | Existing scheduler | 16-position policy |
|---|---:|---:|
| Short request maximum sampled gap | 347.470 s | 3.377 s |
| Short request median sampled gap | 136.534 ms | 2,793.311 ms |
| Short request P95 sampled gap | 378.831 ms | 3,317.946 ms |
| Short outputs sampled during the medium request's wait for its first sample | 1 | 124 |
| Medium request time to first sample | 347.430 s | 363.522 s |
| Buffered short request wall time, tracing off | 380.377 s | 354.003 s |
| Buffered medium request wall time, tracing off | 359.019 s | 370.364 s |
| Group makespan, tracing off | 380.377 s | 379.284 s |
| Group makespan, diagnostic tracing | 380.624 s | 378.895 s |
| Aggregate HTTP throughput, tracing off | 0.4206 tok/s | 0.4218 tok/s |
| Sampled process peak across all mixed arms | 3,564 MiB | 3,564 MiB |

The staggered 2,060-token prefill arrived during active decode in all
ten instrumented mixed-request groups. Each used a 65-input,
128-output short request and a 2,060-input, 32-output medium request.
All 800 generated IDs matched across the five diagnostic pairs.
Timing-off outputs, token counts and stop reasons matched each
diagnostic in all 20 responses. The four short client controls also
matched all 512 IDs across the two configurations. The separate
[cuBLAS-on and off fixtures](../chunking-guards/README.md) establish
exact valid FP16 KV and cancellation behavior on their guarded state
case.

The policy spreads decode progress across the prefill. It reduces
the longest gap and short request completion time while increasing
the usual token gap and the medium request's first-sample wait. Keep
it opt-in for the tested dense buffered use case. A port needs
independent state and scheduling checks on the selected main-based
runtime, plus long/short mixes with applicable latency budgets.

The source change is `092dcc4b5cccdb6afe5f525f26e9a0303dd8f7d0` in
`src/server.rs` with `src/server/prefill_chunk_tests.rs`. The server
was built from `8133884f04fb5024a3223281a8a0220debe0680d`. The run
used the same `sm_52` server binary for both arms with
`M40LLM_RESEARCH_DENSE_PREFILL_CHUNKS=1` only in the candidate.
The corrected readiness function from `3ac7bb4` kept the pair labels
intact. The [stopped first attempt](../chunking-attempt-01/README.md)
remains independent of these results. The host [replay](host-replay.json)
observed 17.24–19.80 QEMU logical CPUs across the mixed arm intervals.

The [raw archive](measurements.tar.gz), [checksum](archive.sha256),
[gate output](summary.json) and [independent replay](replay.json)
retain the matched model/inputs, server command lines, request and
token clocks, all test gates and sampled device state. Extract the
archive and use the exact `research_fairness_report` executable from
preflight with [the replay](../replay-chunking.sh) to reproduce the
receipt. It rechecks every raw output, counts short-request samples
during the medium wait and recomputes the measured peaks with PID
and time windows.
