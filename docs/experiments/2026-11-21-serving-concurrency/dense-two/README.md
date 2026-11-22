# Two concurrent buffered requests with dense KV

**PASS for aggregate HTTP throughput in the measured cohort.** Five
alternating M40 pairs used Qwen2.5-3B Q8_0 at capacity 1,024 with explicit
dense FP16 KV. Each of the two clients submitted the same 65 prompt IDs
and returned the same 64 generated IDs. Batched makespan improved
from 33.933 to 21.275 seconds. Aggregate throughput rose from 3.772
to 6.016 tok/s, a 59.495% gain with a paired 95% interval of
58.899–60.491%. Within-arm MAD/median was 0.150% serialized and
0.188% batched. All ten timing sets matched their companion responses.

| Measure | Serialized | Batched |
|---|---:|---:|
| Median aggregate throughput, tok/s | 3.772 | 6.016 |
| Median makespan, s | 33.933 | 21.275 |
| Diagnostic client TTFT, s | 8.514 / 25.371 | 7.405 / 7.413 |
| Diagnostic client median token gap, ms | 133.215 / 133.655 | 218.827 / 219.022 |
| Diagnostic client P95 token gap, ms | 135.229 / 136.432 | 226.031 / 226.049 |
| Peak process NVML sample, MiB | 3,275 | 3,409 |

The instrumented candidate completed 64 width-two decode forwards.
Both clients started receiving tokens sooner and finished closer
together. Each saw a slower token cadence, and the sampled process
peak rose by 134 MiB. The timings use buffered client wall across
the two requests. The per-client first-token and token-gap data come
from separate diagnostic requests. The 250-ms samples include
model load and each arm's requests.

Keep this as a Phase 4 two-client dense baseline for future serving
work. The [two-client default-KV comparison](../default-two/README.md)
uses a different cache configuration and has its own controls.
The dense result covers two equal short prompts. Mixed-length
fairness and additional client counts have separate workloads.

The [raw archive](measurements.tar.gz), [checksum](archive.sha256)
and [replay receipt](replay.json) retain the request order, traces,
paired intervals, builds and resources. Run
`bash ../replay.sh EXTRACTED_DIRECTORY RESEARCH_FAIRNESS_REPORT RESEARCH_HTTP_PAIRS`
from this report directory. The replay matches both original token
traces, all timed records and the [paired screen](screen.json).
