# Fixed-mask Q4 down projection on Tesla M40

**PASS for testing magnitude thresholds on real Q4 activations.** The
lossless packed and regrouped zigzag layouts both cleared the frozen 10%
charged-latency gate at 25% and 50% active inputs on all three layers.
The native-layout indexed path missed the gate at 25% and was slower
than dense at 50%. Each charged call includes thresholding, stable
compaction, the device count, partials and reduction. These results
come from synthetic sparse vectors and three actual Qwen Q4_0 tensors.

| Active inputs | Native dense | Native charged | Lossless packed charged | Zigzag charged | Dense zigzag |
|---|---:|---:|---:|---:|---:|
| 2,752 / 11,008 | 905.5–906.3 | 838.8–843.5 | 172.4–172.7 | 147.0–147.8 | 264.8–267.2 |
| 5,504 / 11,008 | 905.7–907.3 | 1,634.4–1,651.6 | 311.7–313.2 | 242.8–244.7 | 266.0–266.6 |

Times are median synchronized wall microseconds per invocation, shown
as ranges across layers 0, 17 and 35. Every layer ran five rotating
rounds and one warm round, each with 20 repeats. The raw screen retains
per-layer MAD, paired intervals and the admission decision. No extension
or confirmation was called for. The [independent replay](replayed.json)
also separates CPU submission and CUDA-event spans.

Against dense GGUF, lossless packing reduced wall time by 80.93–80.98%
at 25% active and 65.46–65.64% at 50%. Even with every input active,
the same charged packed path took 597.0–597.8 µs against 905.9–907.7 µs
native. Layout alone explains part of the apparent sparse benefit.
Zigzag changes the weight values. Against dense execution of those
same regrouped weights it saved 44.49–44.68% at 25% active and just
8.24–8.70% at 50%. Record the dense zigzag control in any follow-up.

The 120 full-output checks include empty/full masks. Stable indices and
counts matched the host input. The native and lossless packed indexed
results were bit-identical. Maximum absolute error against each layout's
own FP64 reference was 1.92e-6 or less. The largest output difference
between regrouped and original weights was 0.25594 on these inputs.
The experiment priced neither an answer-quality change nor total FFN cost.

Each real down tensor occupied 12,681,216 bytes. Keeping an extra layout
alongside its native tensor costs another 12,681,216 bytes per layer.
All three layouts together requested 38,043,648 weight bytes; the
common benchmark workspace occupied 402,436 bytes. The standalone
process reached 140 MiB in the available 500 ms samples. Free device
memory after layer 0 was 2 MiB lower than before the first setup and
returned to that same level after each later layer. The fixed-mask run
does not establish model-level residency. The [preparation measurements](../post-pause/prepare.jsonl)
charge conversion separately.

Source `14f2a316de0adeabbfad7ac0d98be0214546dcc4`, release sm_52
probe SHA-256 `b0001b7b00dfd85b7bc027447bfc1c4d6609acfff3b962e02ea03b61a282f9c3`.
The [raw archive](measurements.tar.gz), SHA-256
`d2dca78326aa6716dd7482004390606518ff1777cb83d8fd66e013e2a9da73c6`,
contains the actual M40 process/device and host observations, build and
weight identities, raw numerical checks, timings and screen decision.
From the study worktree, extract the archive into a fresh directory and run:

```sh
d=docs/experiments/2026-11-21-sparse-q4/result
jq -es -f "$d/replay.jq" EXTRACTED/sparse-q4-01/primary.jsonl | diff - "$d/replayed.json"
target/release/examples/sparse_q4_screen EXTRACTED/sparse-q4-01/primary.jsonl | diff - EXTRACTED/sparse-q4-01/screen.json
```

Only the completed activation is available to the admitted follow-up.
Use separate Q4 calibration and held-out data, charge the full thresholded
operator and measure approximation before any runtime recommendation.
The historical Q8 threshold study found substantial perplexity penalties
at higher sparsity. Its thresholds were fitted to different weights.
[Q8 threshold and quality controls](../../2026-10-25-activation-quality.md).
