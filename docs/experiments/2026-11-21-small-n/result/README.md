# Four-row Q8 weight reuse on the M40

**PASS for admission to complete-target verification.** The release
operator comparison passed 108 numerical controls and completed 540
alternating timing intervals. The new projection matched the serial
prepared result bit for bit at all six widths on all three layers. Its
four-row CUDA-event saving against the actual serial down schedule was
0.15069, 0.15041 and 0.15017 ms on layers 0, 17 and 35 respectively.
Across 36 layers this projects to 5.415 ms, above the frozen 2.891 ms
admission threshold. The corresponding wall estimate is 5.424 ms.
The operator uses the existing split planes and allocates no extra weight.

| Input rows | Serial split | Existing multi-row split | Shared weights | Native GGUF | Cached SGEMM | Dequant + SGEMM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.137 | 0.137 | 0.137 | 0.191 | 0.465 | 1.207 |
| 2 | 0.273 | 0.258 | 0.198 | 1.223 | 0.422 | 1.156 |
| 3 | 0.391 | 0.357 | 0.257 | 1.391 | 0.420 | 1.146 |
| 4 | 0.521 | 0.471 | 0.370 | 1.662 | 0.425 | 1.152 |
| 8 | 1.041 | 0.917 | 0.730 | 2.601 | 0.421 | 1.144 |
| 16 | 2.082 | 1.812 | 1.419 | 4.753 | 0.493 | 1.216 |

Entries are layer-0 median CUDA-event milliseconds per call. At width
four, cached SGEMM was the fastest control but required a 90,177,536-byte
expanded tensor. Shared weights still saved 0.054–0.055 ms per call
against it. Existing multi-row split uses the same weight storage as the
candidate, reproduces the serial bits and saves launches. The candidate
saved another 0.0997–0.1002 ms per layer against that control. Include
it alongside the current direct verifier in the complete-target comparison.
At widths 8 and 16 the pre-expanded SGEMM was faster. No width change
or new layout sweep is justified by the four-row admission.

The [audit](operator-audit.json) retains all six widths and three layers,
two timing methods, variability, individual paired differences and
numerical controls. The raw [report](report.jsonl.gz), [guard test](guard.log.gz),
[host load](host-load.log.gz), [device resources](resources.txt.gz) and
[preflight](preflight.txt.gz) retain the original measurements. Use the
[independent replay](../README.md) on the decompressed report. Complete
target, rollback and request timings must qualify independently.
