# Charged sparse Q4 operator on held-out activations

**PASS for operator admission in both layouts.** On actual Qwen2.5-3B
Q4_0 post-SwiGLU vectors, charged packed Q4 reduced synchronized
operator wall time by 41.63–42.95% against full packed Q4. Charged
zigzag reduced it by 42.89–47.19% against full zigzag. Both also
beat the original dense native projection, and both passed the
paired criteria on all three frozen layers. The device selection,
compaction, count, projection and final reduction are inside the
timed interval.

| Layer | Packed 50% cutoff vs native / packed | Zigzag 25% cutoff vs native / zigzag | Achieved median density, packed / zigzag |
|---:|---:|---:|---:|
| 0 | 64.75% / 42.95% | 84.16% / 47.19% | 51.30% / 25.17% |
| 17 | 63.80% / 41.63% | 82.77% / 42.89% | 50.58% / 25.22% |
| 35 | 63.83% / 41.71% | 83.18% / 44.10% | 50.20% / 24.45% |

Each interval weighs the six held-out tasks equally after their
per-row aggregation. All wall and CUDA-event lower bounds support
the frozen threshold. The archive keeps all 1,998 numerical controls
and 11,664 timing records, including warmups and the two diagnostic
cutoffs. The zero-input fixture produced finite zero outputs.
Thresholded packed and thresholded native matched bitwise. Matched
FP64 checks passed on all layouts. The largest relative RMS error
against the original dense result was 0.1722 for the primary packed
cutoff and 0.4223 for the primary zigzag cutoff. Changing the
quantization layout adds its own approximation before masking.
These errors warrant measuring the two dense-repacked controls
alongside their thresholded variants.

The GPU operator was built at
`5dde384139db7e7b5206bec4c0bac75aba9748aa`. The
[capture provenance](../../2026-11-22-sparse-q4-live.md#captured-inputs)
binds the original weights, exact held-out observations and frozen
cutoffs. The microbenchmark includes exactly one down projection.
The frozen next gate covers three layers, held-out free generations
and likelihood passages with the same model weights. Whole-request
and model-quality dispositions remain **INCONCLUSIVE** until measured.

The [raw archive](measurements.tar.gz) and [checksum](archive.sha256)
retain all operator results, extracted input rows, dispatch flags,
CUDA build identity and device/host samples. After extraction, run
`python3 ../screen.py EXTRACTED/rows/manifest.json EXTRACTED` from
this directory and compare with [the replay](replay.json).
