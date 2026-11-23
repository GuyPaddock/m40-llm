# Single-client batching control with default KV

**PASS for equivalent work. FAIL for the 5% throughput gate.** Five
warmed pairs on the M40 used one buffered Qwen2.5-3B Q8_0 request with
65 input and 64 output IDs. The diagnostic matched all 64 generated IDs
and recorded zero packed decode forwards. All timing-off responses
matched the corresponding trace and length stop.

| Measure | Batching off | Batching enabled |
|---|---:|---:|
| Median request time | 12.807 s | 12.762 s |
| Median throughput | 4.997 tok/s | 5.015 tok/s |
| Throughput MAD over median | 0.167% | 0.124% |
| Diagnostic client time to first sample | 3.811 s | 3.802 s |
| Diagnostic median inter-token gap | 142.767 ms | 142.432 ms |
| Diagnostic P95 inter-token gap | 146.211 ms | 146.145 ms |
| Maximum sampled timed-process peak | 3,312 MiB | 3,488 MiB |

The median paired gain was 0.349% with a 95% interval of −0.082% to
+0.349%. Enabling scheduling with four allowed slots cost 176 MiB
of sampled process peak in this one-request control. Its single-sequence
fallback is a useful check for the separate multi-client cohorts.

The clean runner source was `3d10ccaa44be00fa7fd3d66ea7b8b836836ad45e`.
The [archive](measurements.tar.gz), [checksum](archive.sha256),
[screen](screen.json) and [independent replay](replay.json) retain the
original requests, diagnostics, timing statistics and process samples.
Verify the checksum and extract the archive. Run
`bash ../replay.sh EXTRACTED_DIRECTORY RESEARCH_FAIRNESS_REPORT RESEARCH_HTTP_PAIRS`
to replay the workload, width, IDs and timing comparisons. Recover each
arm's sampled peak by selecting the recorded PIDs under
`server-pair-*-ARM.pid` in `processes.csv`.
