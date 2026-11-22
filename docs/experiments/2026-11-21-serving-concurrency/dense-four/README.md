# Four concurrent buffered requests with dense KV

**PASS for aggregate throughput in this short dense-KV workload.**
Five warmed pairs on the M40 used four simultaneous Qwen2.5-3B Q8_0
requests with 65 input and 64 output IDs each. All 256 generated IDs
matched in the diagnostic. Every timing-off response reproduced its
companion output, count and length stop.

| Measure | Serialized | Batched |
|---|---:|---:|
| Median group completion time | 67.629 s | 34.729 s |
| Median aggregate throughput | 3.785 tok/s | 7.371 tok/s |
| Throughput MAD over median | 0.126% | 0.082% |
| Diagnostic client TTFT range | 8.510–59.195 s | 14.241–14.265 s |
| Diagnostic median inter-token gap range | 132.292–133.392 ms | 323.598–323.736 ms |
| Diagnostic P95 inter-token gap range | 134.535–135.601 ms | 327.133–327.300 ms |
| Maximum sampled timed-process peak | 3,275 MiB | 3,436 MiB |

The median paired throughput gain was **94.733%** with a 95% interval
of **94.574–95.077%**. The candidate recorded 64 completed width-four
target forwards. Three of the four clients reached their first sample
sooner. The earliest serialized client had lower TTFT and every
serialized client had shorter token gaps. The batched peak cost 161 MiB
more. Keep the result scoped to four equal short prompts with dense KV.

The [archive](measurements.tar.gz), [checksum](archive.sha256),
[screen](screen.json) and [independent replay](replay.json) retain the
original request and token clocks. To replay, verify the checksum,
extract the archive and invoke
`bash ../replay.sh EXTRACTED_DIRECTORY RESEARCH_FAIRNESS_REPORT RESEARCH_HTTP_PAIRS`.
The replay verifies the executable identities, actual scheduler width,
diagnostic IDs, timing-off outputs and all paired statistics.
