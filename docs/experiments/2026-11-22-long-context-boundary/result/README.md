# Dense head128 dispatch at 8,192 and 8,193 tokens

**PASS for the dispatch and numerical diagnostic. FAIL for practical
latency above the fast-path limit.** On the Tesla M40, the ordinary
dense attention launcher switched from the head128 fast warp-reduce
kernel to the generic fallback when the cache grew by one token.

| Active cache | Decode-stream CUDA event median and range | Synchronized host median and range |
|---|---:|---:|
| 8,192 | 7.921 ms, 7.919–7.942 ms | 7.932 ms, 7.927–7.959 ms |
| 8,193 | 9,045.630 ms, 9,036.137–9,047.599 ms | 9,045.621 ms, 9,036.125–9,047.586 ms |

The event medians differ by about 1,142×. Each of three alternating
pairs used one warmed call per length, the same query, 16 query heads,
two KV heads and 25,952,256 bytes of FP16 K/V capacity. All 12,288
observed output values met the fixture's FP64 reference tolerance.
Maximum absolute difference across the six calls was 1.29e-8. Tracked
allocation returned to its initial value. The test exited zero with
one executed pass.

The CUDA events bracket the ordinary decode-stream call. Host wall
also includes submission and the stream wait. These are single-layer
operator observations on fixed synthetic inputs. The existing
[8,197-token Qwen diagnostic](https://github.com/GuyPaddock/m40-llm/blob/721a52bea0ddc72858342c41ea19ef20f98c5f62/docs/experiments/2026-11-21-serving-baseline/qwen-long/README.md)
separately reached its one-hour dense limit before the first generated
token. The matching geometry and observed launcher switch identify a
concrete bottleneck for an exact long-context extension. Require FP16
KV, logit and greedy parity across the boundary and sustained request
measurements for any replacement.

The clean revision was `86e15affca6975a44689156cb584f2ff8918e9d1`.
The CUDA 12.8/cuBLAS release test contains `sm_52` and has SHA-256
`1d3dcef6111a043138dcf8890854cd6962c3f528d04971a85c70d4ff6d076dcf`.
The [archive](measurements.tar.gz) and [checksum](archive.sha256)
retain the preflight, all observations, build artifacts, kernel names,
exit code and host snapshots. Extract the archive and use
`jq -s 'map(select(.kind == "boundary_trial")) | group_by(.cache_length)' observations.jsonl`
to inspect all six calls. The [parent protocol](../README.md) retains
the build and M40 reproduction command.
