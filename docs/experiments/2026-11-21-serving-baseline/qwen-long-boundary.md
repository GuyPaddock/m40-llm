# Long dense-KV request feasibility and attention boundary

The long serving case uses 8,197 frozen prompt IDs, an 8,448-position
capacity and 128 greedy output slots. Both cache modes use the original
Qwen2.5-3B Q8_0 bytes and the pinned `sm_52` Phase 4 executable. The
default CLI and buffered diagnostic completed with matching 128 IDs.
Default streaming also completed. At 16:40 EST, the dense CLI had run
46 minutes without a sampled token. Keep both settings and inputs fixed.

Apply the existing `research_http_load` limit of 3,600 seconds per
request as the operational feasibility bound for the dense CLI
diagnostic. The CLI has no configured deadline. If its own process is
still running at the bound, retain the elapsed time, emitted IDs, model
and executable identities and last device and process snapshots. Stop
that process and record the exit in the long-cohort ledger. The paired
series requires a completed workload from both arms. Preserve default
diagnostics and any partial dense evidence separately on a stop.

The ordinary dense head128 launcher uses a 128-thread specialized
kernel through 8,192 positions. For longer sequences it launches a
single thread for each query head. The generic kernel recomputes the
context's query/key dot products for every output dimension. High
device utilization alone gives no launch-time or numerical evidence.

Run one bounded operator check on the same M40 under the shared lock.
Use the existing head128 test fixture with 16 query heads, two KV
heads, FP16 cache and a fixed query. Compare the ordinary dense launcher
at 8,192 and 8,193 valid positions. Exclude fixture construction and
host readbacks from the device interval. Record one warmed call per
sample, three alternating pairs, CUDA-event span, host wall interval,
the selected launch log and agreement with the fixture's FP64 reference.
Use the established attention test tolerance and retain all rows.
Stop on a numerical failure or a 15-minute operator-run limit. The
diagnostic establishes the boundary cost only. A full-model throughput
claim requires a complete matched model request.
