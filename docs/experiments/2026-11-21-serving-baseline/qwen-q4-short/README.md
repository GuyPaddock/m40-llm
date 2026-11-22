# Qwen Q4_0 on the short buffered workload

**PASS for Phase 4 checkpoint admission and matched outputs. FAIL for
a 5% complete-request benefit from switching to dense KV.** The local
checkpoint mapped all 36 layers and used its dedicated Q6_K output
matrix. Both cache arms passed the same 65 prompt IDs and returned
the same 32 generated IDs through CLI and buffered HTTP.

| Measure | Default cache | Explicit dense |
|---|---:|---:|
| Median buffered request, s | 11.085 | 11.308 |
| Median buffered tokens/s | 2.887 | 2.830 |
| Median generation admission to first sample, s | 7.131 | 7.584 |
| Median decode tokens/s | 7.986 | 8.415 |
| Peak process NVML observation, MiB | 2,084 | 2,047 |
| P95 inter-token gap, diagnostic, ms | 131.319 | 121.340 |

Dense improved the decode portion while increasing total request time.
Its paired throughput change was −1.970% with a 95% interval of
−3.001% to −1.653%. The default prompt fits entirely in the recent ring.
This short contrast cannot price serving with historical KV. The NVML
peak covers each warmed server process at the recorded sample cadence.

The matched cold CLI observations took 23.34 s and 23.42 s respectively.
The separate streaming calls emitted 32 nonterminal records, with
their first records at 9.044 s and 8.683 s. Neither stream observation
is a replacement for the buffered request boundary.

This is an additional GGUF baseline on Phase 4. The main admission
probe rejects the Q4_0 projection at layer zero. The comparison also
supplies an unchanged-model control for the live sparse-Q4 study.
Its data provides no sparse execution speed or quality result.

The [raw archive](measurements.tar.gz) keeps the frozen model/inputs,
preflight, output IDs, five pairs, cold runs, and process telemetry.
After `sha256sum -c archive.sha256`, extract into a fresh directory
and run `bash ../replay.sh EXTRACTED_DIRECTORY`. The output matches
[the retained replay](replay.json).
