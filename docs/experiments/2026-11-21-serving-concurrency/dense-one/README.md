# Single-client batching control with dense KV

**PASS for equivalent work. FAIL for the 5% throughput gate.** Five
warmed M40 pairs used one buffered Qwen2.5-3B Q8_0 request with explicit
dense FP16 KV, the same 65 prompt IDs and 64 output IDs. All diagnostic
and timing-off outputs matched. The enabled scheduler completed zero
packed decode forwards and used the single-sequence path.

| Measure | Batching off | Batching enabled |
|---|---:|---:|
| Median completion time | 16.919 s | 16.969 s |
| Median throughput | 3.783 tok/s | 3.772 tok/s |
| Throughput MAD over median | 0.063% | 0.034% |
| Diagnostic client time to first sample | 8.466 s | 8.506 s |
| Diagnostic median inter-token gap | 133.613 ms | 133.685 ms |
| Diagnostic P95 inter-token gap | 135.520 ms | 135.322 ms |
| Maximum sampled timed-process peak | 3,275 MiB | 3,383 MiB |

The paired throughput change was −0.295%, with a 95% interval from
−0.376% to +0.307%. The four-slot configuration cost 108 MiB of
sampled peak process memory in this one-request control. With default
KV, the corresponding cohort changed throughput +0.349% and added
176 MiB. Both intervals span zero. Use the controls to preserve the
scope of the aggregate wins measured at two and four concurrent clients.
[Default one-client control](../default-one/README.md).

The clean runner source was `5891b68cb9f9c6662ad1b3c575ccb10e306a7cc1`.
The [archive](measurements.tar.gz), [checksum](archive.sha256),
[screen](screen.json) and [independent replay](replay.json) retain the
original requests, scheduler logs, process samples, clocks and paired
statistics. After verifying and extracting the archive, run
`bash ../replay.sh EXTRACTED_DIRECTORY RESEARCH_FAIRNESS_REPORT RESEARCH_HTTP_PAIRS`.
The replay output matched before archiving and after extraction. The
sampled peaks use the recorded `server-pair-*-ARM.pid` values in
`processes.csv`.
