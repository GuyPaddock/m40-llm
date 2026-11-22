# Two concurrent buffered requests with direct mixed KV

**PASS for aggregate HTTP throughput on this workload.** Five alternating
pairs on the M40 used the same Qwen2.5-3B Q8_0 checkpoint, 65 prompt IDs
and 64 generated IDs per client. The serialized server delivered a median
4.900 aggregate tok/s. Buffered batching delivered 5.956 tok/s, a
21.557% gain with a paired 95% interval of 20.918–24.095%. Within-arm
MAD/median was 0.164% and 0.249%, respectively. The two diagnostic clients
returned the same 128 sampled IDs across arms. All ten timed response
sets matched their companion output, count and stop.

| Measure | Serialized | Batched |
|---|---:|---:|
| Median makespan, s | 26.123 | 21.490 |
| Client TTFT, diagnostic, s | 3.819 / 16.817 | 7.441 / 7.433 |
| Median ITL, diagnostic, ms | 144.920 / 147.453 | 222.586 / 222.591 |
| P95 ITL, diagnostic, ms | 149.807 / 150.079 | 227.780 / 227.720 |
| Sampled primary and diagnostic process peak, MiB | 3,312 | 3,501 |

The timed server used its default direct FP16-K/Q4-V cache with a four-slot
pool in the batched arm. The measured companion recorded 64 successful
width-two target forwards. The candidate increased sampled per-process
peak by 189 MiB. The aggregate speed improvement came with longer
first-client TTFT and inter-token intervals. The second client entered
generation earlier and finished much closer to the first. The workload
fits in the cache's recent window, so this comparison does not establish
an improvement for batched historical block selection.

The [raw archive](measurements.tar.gz) retains model/build identity,
GPU and host samples, server diagnostics, paired HTTP timings and all
outputs. Verify it with `sha256sum -c archive.sha256`. The
[paired calculation](screen.json) can be reproduced from the extracted
`timings.jsonl` with the CPU `research_http_pairs` example. To
reconstruct those rows, run `research_fairness_report EXTRACTED_DIRECTORY
--client-run LABEL` and pass the emitted summary, matching diagnostic
JSON and primary response to [observe.jq](../observe.jq), as in [run.sh](../run.sh).
