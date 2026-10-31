# Ternary screening exceeds the sparse FFN budget

The fixed Maxwell ternary screen fails both remaining overhead budgets.
All 24 captured-input cases pass numerical and ownership checks, but screening
alone takes 507.70 to 513.70 microseconds. The conservative budgets are 150.88
microseconds at 10% retained channels and 56.89 at 20%. Stop channel top-k,
rank-32 recovery and runtime integration for this implementation.

This completes the bounded channel-selection branch with a negative screening
decision. The supplied-index sparse Q4 kernels remain useful evidence that
selective execution can be fast. The tested predictor cannot pay for itself.
The result does not rule out other predictors, layouts or MoE implementations.

## Measurements

The clean release source was `327aa324e9b50125943dfaba98a2f9bc997f3c37`.
The [frozen protocol](2026-10-31-ternary-screen-protocol.md) used the first
eight distinct captured inputs at each of layers 0, 17 and 35, one row at a
time. The M40 had no other compute process. Every row warmed for at least
one GPU second, followed by five samples of ten invocations each.

| Layer | Per-row median GPU time, microseconds | Largest throughput MAD/median | Cases inside either budget |
| ---: | ---: | ---: | ---: |
| 0 | 507.88 to 512.51 | 0.193% | 0 of 8 |
| 17 | 509.51 to 513.70 | 0.403% | 0 of 8 |
| 35 | 507.70 to 512.51 | 0.100% | 0 of 8 |

All 120 samples completed. No case needed the ten-sample extension. Per-call
host submission ranges from 4.88 to 9.43 microseconds; host submission plus
completion ranges from 505.32 to 531.78 microseconds across raw samples.
Every timed invocation launches two kernels with no device allocation, copy,
stream wait or synchronization inside enqueue. Completion waits are reported
at the timing boundary.

The screen includes signed rotation, per-block top-80 coordinate selection,
all-channel ternary gate/up estimates, SwiGLU and down-norm weighting. It
excludes channel top-k, selected exact FFN execution and recovery. Even the
smallest observed median exceeds the two budgets by 356.82 and 450.81
microseconds. These compare with the earlier frozen point budgets; they are
not paired whole-model or combined-pipeline speed measurements.

## Representation and correctness

Each layer retains 12,022,784 immutable bytes: two 5,988,352-byte sketches,
the sign vector and down-column norms. Guarded rotation/coordinate scratch
uses 9,600 bytes. Guarded input/output buffers use 184,448 bytes. Peak tracked
device residency is 4,484,672,000 bytes, including the original model and its
prepared Q8 down/vocabulary weights. This run does not allocate the sparse
FFN or recovery owners. Combined-pipeline memory remains unmeasured.

Per-layer Q4 derivation, fitting, norms and hashes take 3.719 to 3.740 seconds
on the host; device upload takes 2.606 to 2.907 milliseconds. The two-bit code
representation uses 34 bytes per 128 weights at these dimensions. It remains
the explicit larger-codec adaptation described in the protocol.

Rotation matches the independent Walsh reference, selected coordinates match
the CPU ordering, and all four projection/activation/score planes match the
FP64 encoded-sketch reference. The largest recorded plane error is
0.000012251. Post-timing outputs repeat bit for bit. Input and scratch guards,
immutable hashes and per-layer allocation cleanup pass. After process exit,
the M40 has no compute process and zero allocated MiB.

These checks establish implementation correctness for the encoded screen.
They do not establish channel-ranking quality, model accuracy or sequence
quality. Those dependent experiments stop at the cost gate. The source,
failed implementation and raw measurements remain committed for inspection;
no defaults, training, teacher generation or runtime dispatch changed.

## Artifacts and reproduction

- [Raw observations](2026-10-31-ternary-screen-timing.jsonl.gz)
- [Summary](2026-10-31-ternary-screen-summary.json) and [summary query](2026-10-31-ternary-screen-summary.jq)
- [Run log](2026-10-31-ternary-screen-timing.log.gz) and [release build](2026-10-31-ternary-screen-release.log.gz)
- [CUDA correctness checkpoint](2026-10-31-ternary-cuda.md)
- [Earlier stronger-reference budgets](2026-10-31-q4-stronger-baselines.md)

Build the committed source in release mode with CUDA/cuBLAS enabled, restore
the capture manifest's environment exactly, and use a fresh output path:

```bash
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --output target/ternary-screen-timing-01.jsonl \
  --ffn-inputs /mnt/array-fastest/home/guyep/playground/2025-12-20-m40-llm/m40-llm/target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-q4-channels --ffn-ternary-screen --rows 1 --trials 5 --iterations 10
gzip -cd docs/experiments/2026-10-31-ternary-screen-timing.jsonl.gz | \
  jq -s -f docs/experiments/2026-10-31-ternary-screen-summary.jq
```

The next independent topic is the surviving multi-row FFN fusion opportunity,
with its down projection unchanged. This negative result does not close that
branch or the other research topics.
