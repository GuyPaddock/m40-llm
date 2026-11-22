# Dense prefill disrupts active buffered decode

**PASS for admission to a bounded scheduling experiment.** All five
alternating pairs cleared the observed-stall gate and matched their
tracing-off output, stop and token-count checks. The 2,060-token
prefill coincided with a 344.171–350.970 s decode gap, compared with
8.585–8.749 s for the 65-token prefill. The long request entered the
scheduler within 4.9–129.0 ms. Its prefill and first sample occupied
almost all of the gap. Every response in the diagnostic pairs matched
the corresponding isolated IDs. [Aligned intervals](../diagnostic/README.md).

| Concurrent workload | Tracing-off makespan, s | Aggregate sampled tokens/s | First request TTFT, s | First request median ITL, ms | First request P95 ITL, ms |
|---|---:|---:|---:|---:|---:|
| One short | 25.718 | 4.977 | 8.506 | 133.851 | 137.465 |
| One medium | 354.022 | 0.090 | 345.384 | 206.176 | 210.259 |
| Two short, staggered | 45.677 | 5.605 | 8.562 | 220.118 | 228.017 |
| Short then medium | 380.604 | 0.420 | 8.572 | 136.210 | 379.293 |
| Four short | 55.831 | 9.171 | 14.256 | 323.314 | 330.805 |

The two-client rows are medians across five trials. Other rows are
single controls. The TTFT and ITL columns come from the token clocks.
The medium request in the mixed pair reached its first sample after a
median 346.454 s. Clients finished within 0.485–0.758 s of each other
in tracing-off short/short and within 12.256–12.531 s in short/medium.
Every request completed. The multi-minute gap affects the worst token
interval much more than P95. Aggregate throughput includes prefill and
uses the actual, different generated counts for each workload.

On matching workloads, sampled-clock makespan was between 0.86% lower
and 0.45% higher than tracing-off makespan. The two series were run
sequentially, so time drift can contribute to this comparison. Each
server reached a sampled 3,616 MiB process peak at 250 ms cadence.
Only the two successive server PIDs appear in process samples. The
final M40 snapshot showed 0 MiB used. Portage remained stopped in the
sampled diagnostic intervals and the Windows VM ran throughout.

The baseline covers the opt-in dense buffered scheduler and the frozen
Qwen Q8_0 workload. [Bounded chunking protocol](../chunking-plan.md).

The [raw archive](measurements.tar.gz) includes HTTP records, prompt
bytes, sampler timelines, binary/model hashes, build and GPU identity,
both server logs, and the final device snapshot. Verify it with
`sha256sum -c archive.sha256`. The [runner summary](summary.json) and
[independent response replay](independent.json) agree on the completed
work and sampled throughput for all 13 scenarios.

Extract into a fresh directory and set `d` to its path. From the
study worktree, replay with the existing release example and jq:

```bash
study=docs/experiments/2026-11-21-prefill-fairness
target/cpu/release/examples/research_fairness_report "$d" |
    diff - "$study/result/summary.json"
for label in single-short single-medium four {0..4}-{control,treatment}; do
    jq -ces -f "$study/responses.jq" "$d"/{diagnostic,primary}-"$label".json |
        jq -c --arg label "$label" '{scenario:$label}+.'
done | jq -s . | diff - "$study/result/independent.json"
```
