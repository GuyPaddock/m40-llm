# Four concurrent buffered requests with default KV

**PASS for aggregate throughput.** Five warmed pairs on the M40 used
the same Qwen2.5-3B Q8_0 checkpoint with four simultaneous requests,
each with 65 prompt tokens and 64 outputs. All 256 generated IDs
matched in the companions. The ten uninstrumented responses matched
those outputs, counts and stops.

On the normal request path, median makespan fell from 52.264 to
34.975 seconds and aggregate throughput improved from 4.898 to
7.320 tok/s. This is a 49.435% improvement with paired 95% interval
49.174–50.224%. The two MAD/median values were 0.153% and 0.101%.

| Measure | Serialized | Batched |
|---|---:|---:|
| Diagnostic client TTFT range, s | 3.814–42.999 | 14.319–14.343 |
| Diagnostic median token gap range, ms | 145.053–146.219 | 327.528–327.648 |
| Diagnostic P95 token gap range, ms | 149.913–150.758 | 331.441–331.606 |
| Peak sampled process VRAM, MiB | 3,312 | 3,528 |

The instrumented candidate completed 64 width-four target forwards.
The first serialized client started sooner, and the last waited
nearly 43 seconds. The four batched clients started at nearly the
same time and observed slower per-token cadence. Their shared model
process used 216 MiB more at its sampled peak. This qualifies the
measured aggregate-throughput mode for equal short prompts. Every
request remains in the recent window of the default mixed cache.

The [raw archive](measurements.tar.gz), [checksum](archive.sha256),
[paired screen](screen.json) and [replay receipt](replay.json)
retain the endpoint and per-client evidence. Run
`bash ../replay.sh EXTRACTED_DIRECTORY RESEARCH_FAIRNESS_REPORT RESEARCH_HTTP_PAIRS`
from the report directory to reproduce the receipt.
