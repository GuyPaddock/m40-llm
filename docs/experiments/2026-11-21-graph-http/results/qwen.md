# Qwen CUDA Graph replay on a complete HTTP request

**PASS for exact sampled IDs and actual graph admission. FAIL for the 5%
whole-request improvement.** Five alternating M40 pairs with dense KV,
native head128 attention, the same 65 prompt IDs and 512 generated IDs
gave a median throughput improvement of **0.361%**. The paired 95%
bootstrap interval was **0.269–0.477%**. Within-arm MAD/median was
0.051% ordinary and 0.023% graph. Neither extension nor confirmation
was called for. Retain the correctness repair and keep graphs opt-in.

| Median measure | Ordinary | Repaired graph |
|---|---:|---:|
| HTTP tokens/s | 21.886 | 21.965 |
| Complete client wall, seconds | 23.394 | 23.310 |
| Generation admission through first sample, ms | 3,708.31 | 3,734.64 |
| Decode ms per interval, 511 intervals/request | 38.437 | 38.212 |
| Process-wide server user + system CPU, seconds | 23.42 | 23.37 |
| Peak single-process NVML sample in full trace companion, MiB | 3,288 | 3,288 |
| Device busy, median of samples across the five timed requests | 99% | 100% |
| Device temperature across those samples, °C | 57–68 | 59–68 |

The 32-token diagnostic measured one 396-node capture at 2.830 ms and
instantiation at 1.239 ms. Median graph launch CPU time was 36.57 µs
and median synchronized CUDA-event replay span was 26.416 ms. Dividing
the two setup costs by the median paired decode saving of 0.2325 ms per
interval gives a setup-only estimate of 17.5 intervals. That estimate
excludes warmup, parameter updates and other first-token costs. The
actual paired 512-token request is the promotion endpoint. Server CPU
includes queueing, synchronization and generation. The replay event
covers the captured forward pass. Per-kernel service time and physical
SM occupancy were unavailable.

Both single-server traces emitted identical 512 sampled IDs and the
graph server reported successful capture. Every timed output, count
and stop matched those traces. Two models were resident during timing,
with only one request in flight. Each companion's memory peak includes
model load and comes from 500-ms NVML samples. Driver memory below that
cadence can differ. The snapshots enclosing the ten timed requests
contained 235 ordinary and 234 graph device samples. All showed an
SM clock of 1,113 MHz. With both servers resident the board used
6,561–6,574 MiB in either arm. The busy percentage measures time with
GPU activity and does not measure the number of resident warps.
Background CPU-only research compilation ran at idle
scheduling priority during part of the cohort. Each request retains host
load and GPU observations. This result applies to the common native
attention backend. Faster ordinary packed/prepared Q8 configurations
have independent controls and cannot inherit this comparison.

Source `ad3b9516e54092720b4a3ebfdd12017e05e7e13f`,
release executable SHA-256
`4cd3c2d98fc0c50562c52afb08eafb4cb6744f58c3ec0f59b48e2e87a5c68eb3`.
The [raw archive](qwen-http.tar.gz) contains the model's SHA-256, flags, input,
timings, token IDs, graph diagnostics and per-request CPU and GPU samples.
SHA-256 `e8a6cde289b189106bc957e7455c2d181809de7bd3a92374730ea80360b6db26`.
Run `python3 ../audit.py EXTRACTED_DIRECTORY` from this results directory
to reproduce the [independent endpoint summary](qwen-summary.json). The
archive's screen-5.json contains the 10,000 paired bootstrap calculation.
