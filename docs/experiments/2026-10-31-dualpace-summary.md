# Cooperative scheduling result summaries

The [summary extractor](2026-10-31-dualpace-summarize.jq) reads one complete
simulator JSONL report. It retains every scheduling mode, including rejected
capacity admissions, and reports local assistance savings before queueing.
Global completion windows, simulated throughput, memory reservations and
resource use remain the simulator's original results.

Per-owner results follow the request ownership declared in the input trace.
This keeps a CPU-owned request identifiable when the GPU-only control executes
it on the GPU. First-decode latency subtracts the request's arrival time.
Inter-token latency uses differences between consecutive completions within
each request and pools those intervals by declared owner. Percentiles use
nearest rank and retain sample counts. One-token requests have no ITL sample.

Queueing summaries retain the sum and distribution of recorded stage waits
for each resource, globally and by owner. These sums describe request waiting;
they are not extra elapsed time to add to the completion window. Local assist
saving subtracts GPU service, both transfers and additional synchronization
from CPU stage service. Queueing and displaced GPU work still determine the
simulated outcome. A positive local saving alone is insufficient for adoption.

Throughput ratios use the same trace's GPU-only completion window. If that
mode fails memory admission, its error remains visible and ratios are null.
Incomplete reports, duplicate modes, inconsistent admission counts and missing
request completions reject without emitting a summary. The extractor preserves
trace/executable hashes and source evidence; original reports remain the
authority for interval-level analysis.

The [CPU-only check](2026-10-31-dualpace-summary-check.sh) passes four positive
cases and four malformed-report cases. It covers the retained one-token
fixture, delayed arrivals, memory rejection and seven different ITLs. In the
last case, a 1,000 ns first token is excluded from the seven samples 10 through
70 ns; p50 is 40 ns and p95/p99 are 70 ns. These are synthetic correctness
checks, not hardware latency measurements. An initial test-fixture jq binding
error was corrected before the successful complete run.

```bash
bash docs/experiments/2026-10-31-dualpace-summary-check.sh \
  target/debug/examples/cooperative_decode
jq -s -f docs/experiments/2026-10-31-dualpace-summarize.jq COMPLETE_RESULT.jsonl
sha256sum --check docs/experiments/2026-10-31-dualpace-summary.sha256
```

The [retained example](2026-10-31-dualpace-summary-example.json) summarizes the
existing synthetic schema-3 fixture. Actual CPU/transfer pricing and the
108-input simulation remain pending. No complete CPU engine, learned model
quality or serving throughput is qualified by this reporting tool.
