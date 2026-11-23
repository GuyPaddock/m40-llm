# Packed sparse Q4 missed the request-speed gate

**PASS for equivalent work. FAIL for the frozen 5% throughput gate.**
Five alternating warm M40 pairs used the same 65 input IDs and 32
output IDs on Qwen2.5-3B Q4_0/Q6_K. Native and thresholded packed
returned the same continuation and length stop in all ten fresh
processes. The candidate exercised layers 0, 17 and 35. Each process
returned tracked device bytes to zero.

| Measure | Native Q4 | Thresholded packed Q4 |
|---|---:|---:|
| Warm resident request median | 9.528 s | 9.484 s |
| Warm output rate median | 3.359 tok/s | 3.374 tok/s |
| Rate MAD over median | 0.102% | 0.284% |
| Median cold model load | 11.911 s | 18.363 s |
| Median packed preparation within load | 0 s | 6.676 s |
| Maximum sampled process VRAM | 2,047 MiB | 2,084 MiB |

Paired gain was **0.456%** with a 95% interval of **−0.172% to
+0.818%**. The owner used another 38,043,648 requested device bytes
and 350,212 bytes of selection scratch. The sampled peak rose
37 MiB and median cold load rose 6.452 s. The charged operator
advantage and narrow real-model quality result do not support
integrating this configuration. Retain the kernels and measurements
as controls. Close the frozen configuration without adjusting its
cutoff or layer count.
[Expanded quality](https://github.com/GuyPaddock/m40-llm/blob/2269a0aa1f6fb3f61e4834a3d55cc2ddd82eeb25/docs/experiments/2026-11-22-sparse-q4-live/extended-result/README.md).

The timed call exercises normal prefill, greedy decoding and text
generation with one resident model. It excludes cold load and HTTP.
The preflight pins all three input sources and the `sm_52` release
binary. Source `1880f66e01ce3844ef3fbe54cdb9f9ff7395dbab` passed
formatting, two release quality arithmetic and evidence tests, and
targeted CUDA/server Clippy with warnings denied. The build and tests
ran with the GPU hidden. The NVCC build logs include the compiler's
offline-target deprecation notice.

Verify the [archive](measurements.tar.gz) with its [checksum](archive.sha256).
Then run `bash replay.sh EXTRACTED/sparse-q4-request-01 RESEARCH_HTTP_PAIRS`.
The replay reconstructs every trial from its primary JSONL, checks all
outputs and released allocations, and regenerates the common paired
statistics. The [screen](screen.json) and [replay receipt](replay.txt)
retain the result. The complete raw and build directories are in the
archive. The [parent protocol](../README.md) gives the M40 run command.
