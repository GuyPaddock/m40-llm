# First prefill chunking HTTP attempt

**FAIL for benchmark orchestration.** The shared server-readiness loop
used the caller's `i` variable as its probe counter. The first diagnostic
baseline ran under `diagnostic-0-baseline`. The matching candidate ran
under `diagnostic-25-candidate`. After both requests finished, the runner
searched for `diagnostic-25-baseline.json` and exited 2. The raw labels,
outputs and actual measurement intervals are preserved unchanged.

**PASS for both four-client short controls and the first diagnostic ID
comparison.** The four-client control matched all 512 sampled IDs.
The staggered medium request arrived while the short request was
decoding in both mixed arms. All 128 and 32 sampled IDs matched their
respective requests, along with prompt lengths and stops. The original
research clock tool regenerated the saved time summaries for each arm.

| First mixed diagnostic | Baseline | 16-position candidate |
|---|---:|---:|
| Short request maximum inter-token gap | 346.807 s | 3.393 s |
| Short request median inter-token gap | 136.787 ms | 2,789.875 ms |
| Short request P95 inter-token gap | 378.753 ms | 3,312.977 ms |
| Medium request client-to-first-sample | 346.802 s | 363.304 s |
| Group completion wall time | 379.992 s | 378.675 s |
| Sampled process peak | 3,564 MiB | 3,564 MiB |

The attempt contains one instrumented mixed pair. The frozen gate calls
for five pairs and a tracing-off series. The runner fix isolates the
server-readiness counter at consolidation commit
`3ac7bb4ee0dd5e6e80f9451c6b56035f4be73535`. An offline fixture verifies
that three readiness probes preserve the caller index. The earlier D3
baseline started the server before entering each pair loop. The A2 and
concurrency runners use a separate `trial` counter. The corrected attempt
retains the original model, binary, input IDs, order and gates.

Verify the [archive SHA-256](archive.sha256) and extract
[the original data](measurements.tar.gz). The
[diagnostic receipt](diagnostic-replay.json) records ID counts, actual
arrival overlap and process peaks with their scope. Run the pinned
`research_fairness_report` on the extracted directory with
`--client-run diagnostic-0-baseline --diagnostic` and with
`--client-run diagnostic-25-candidate --diagnostic`. Compare each result
byte for byte with the corresponding `.clocks.json` file.
