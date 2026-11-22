# Cost of the extra attention node on the M40

**PASS for a measurable operator replay cost. FAIL for admitting a
runtime pruning change.** The head128 graph captured one attention
node at capacity 8,192 and two at 8,193. Each measured live length was
small enough to take the specialized path. After subtracting the
ordinary capacity effect, the added synchronized wall cost was
1.569 µs at length 65 and 3.019 µs at length 2,048. Head7 captured
one node at both capacities and its interval included zero.

| Shape | Additional CUDA-event span, µs | Additional completion wall, µs (paired 95% interval) |
|---|---:|---:|
| Head128, length 65 | 1.563 | 1.569 (1.523–1.606) |
| Head128, length 2,048 | 2.837 | 3.019 (1.365–3.191) |
| Head7, length 65 | −0.016 | −0.012 (−0.118–0.041) |

The first and second five rounds each showed a positive head128 wall
effect. The six controls preserved exact outputs and the valid cache length.
The 120 timed rows each represent 100 invocations. The paired enqueue
interval includes zero for every shape. Capture and instantiation are
reported separately and were sampled once per configuration. The event
span includes the replayed nodes and does not identify their individual
kernel service times.

At 36 layers, 3.019 µs × 36 is about 0.109 ms per token if the same cost
applied to each layer. This is less than 0.3% of the 38.212 ms decode
interval from the separate Qwen graph HTTP comparison. This estimate
compares two workloads and covers only the measured live lengths.
The evidence misses
the 5% headroom gate and does not support removal of the required
8,193-position fallback.

The [raw archive](measurements.tar.gz) includes the exact configuration,
node diagnostics, device and host timelines, and a final allocator count
of zero. Verify with `sha256sum -c archive.sha256`. Recompute the
[summary](summary.json) with `python3 ../analyze.py
EXTRACTED_DIRECTORY/report.jsonl EXTRACTED_DIRECTORY/run.log`.
