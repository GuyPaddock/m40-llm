# Isolated Q8 process residency on the normal server

**PASS for the matched residency observations.** Each arm started
a fresh release server on the same M40. Each ran the frozen Qwen
65-token prompt twice and matched all 512 sampled IDs on each
request. Server and client ownership, dispatch and process peaks
replayed from the raw records. Each process was isolated from the
other arm. The 250-ms NVML samples include startup and requests.

| Separate comparison | Control peak | Candidate peak | Difference | Control launch-to-health | Candidate launch-to-health |
|---|---:|---:|---:|---:|---:|
| Native Q8 → prepared down and vocabulary | 3,275 MiB | 4,414 MiB | +1,139 MiB | 6.233 s | 17.322 s |
| Compound retain-split → compound joint-only | 7,807 MiB | 6,160 MiB | −1,647 MiB | 19.377 s | 18.334 s |

Selective preparation retained 1,192,935,424 additional requested
weight bytes and cost about 11.1 seconds in this one fresh-process
setup comparison. Host high-water RSS was 6,676,524 KiB native and
6,998,452 KiB prepared. The separately repeated, tracing-off
[whole-request Q8 comparison](../../2026-11-21-native-split/README.md)
measured 11.776% higher HTTP throughput. The two diagnostic
requests per arm establish residency with token accounting. Their
wall times do not replace the paired speed endpoint.

Compound's loaded ownership was the only intended difference
in its own cohort. Both arms used packed QKV/output, the same
joint FFN arithmetic and dequant-SGEMM prefill with an explicit
128-token bound. Joint-only removed redundant split gate/up
weights, matching the earlier 1,647 MiB peak reduction on the
normal server. Host RSS high-water was approximately 7.0 million
KiB in both arms. The
[paired ownership study](../README.md) supplies the exact logit
and throughput non-regression evidence.

## Retained attempts

- Compound attempt 01 booted with sequential native prefill and
  stopped on the required packed-dispatch assertion.
- Compound attempt 02 enabled dequant-SGEMM prefill and returned
  HTTP 500 before generation because its packed request bound was
  absent. The final cohort specified both settings.
- Native/split attempt 01 completed both request sets, but the live
  script was edited during the run. Bash stopped before writing
  the final snapshot. Attempt 02 stopped before preflight evidence.
  Attempt 03 is the complete comparison.

The [archive](measurements.tar.gz) preserves all six directories.
Verify the [checksum](archive.sha256), then run
`bash ../server-peak-replay.sh EXTRACTED/q8-owner-http-peaks-03 compound`
and repeat for `EXTRACTED/native-split-server-peak-03 native-split`.
The [compound](compound.json) and [native/split](native-split.json)
receipts matched before and after extraction.
