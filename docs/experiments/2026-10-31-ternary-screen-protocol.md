# Bounded ternary screening experiment

The [stronger-reference measurements](2026-10-31-q4-stronger-baselines.md)
leave 150.88 microseconds at 10% retention and 56.89 microseconds at 20% to
retain a 5% complete-FFN improvement over qualified Q8. These are point-estimate
budgets for all screening, top-k and recovery costs together. First test the
screen alone; a screen over budget closes that retention setting for this
implementation without proceeding to recovery or runtime integration.

## Fixed representation and computation

Build the sketch from the same derived Q4 bytes, with source and plane hashes.
Use a seeded sign vector and normalized Hadamard rotations in 128-coordinate
blocks. Use one sign vector for gate, up and activation rotation at each input
width, with seed zero. No corpus fitting or teacher execution is involved.

Within each rotated weight block, fit a positive FP16 scale and ternary codes
by minimizing squared error over sorted magnitude prefixes. Test every prefix
with its rounded stored scale; ties prefer the smaller prefix. The retained
values get their original signs and all others become zero. Reject nonfinite
source values and shapes or scales outside the supported representation.
This probe also rejects rotated magnitudes above the largest finite FP16 value.

Store four ternary codes per byte, with 0 for zero, 1 for positive and 2 for
negative; code 3 is reserved. Code planes are `[input_coordinate][channel/4]`;
scales are `[input_block][channel]` in FP16. Pad only the final output-code byte
when needed. This Maxwell adaptation uses 34 bytes per 128-value weight block
at aligned dimensions, without device-side block-error metadata. It is larger
than the paper's approximately 30-byte sketch block. Charge the actual planes
and metadata; do not reuse the paper's memory percentage.

For each input, rotate once and select the 80 coordinates with largest absolute
rotated values per block. Break ties by the smaller coordinate index. Estimate
gate and up for every FFN channel, compute SwiGLU and multiply its absolute
value by the original Q4 down-column L2 norm. Retain the estimated activation
for possible recovery. This follows the low-budget screening mechanism in
[sections 5.1 and 5.2](https://arxiv.org/html/2610.37386v1#S5).

Use one fixed initial CUDA geometry: a 128-thread block for each input rotation
and coordinate selection; a 256-thread block for 32 channels, with eight
threads sharing each channel's coordinate work, for scoring. Coordinate-major
loads coalesce across output channels. Keep FP32 accumulation and ordinary
Maxwell instructions. No tuning sweep or Tensor Core path is part of this probe.

Top-k over channel scores and rank-32 recovery are excluded from this first
cost test and must never be represented as free. If the screen fits, add them
and measure the complete pipeline before any quality/runtime decision.

## Evidence and gates

Use layers 0, 17 and 35 and the first eight distinct captured input rows, each
executed as batch one. Validate the rotation against an independent Walsh
reference, coordinate order and ties against a CPU sort, and gate/up/score
values against FP64 reconstruction of the encoded sketch. Record quantization
error separately from kernel error. Preserve guards, immutable weights and
norms, input values, stream ordering, workspace reuse and allocation cleanup.

Report preparation, device payload, scratch and peak residency. Include both
GPU-event time and host submission/completion for rotation plus screening;
selection of the 80 coordinates is inside the boundary. Warm for at least one
second. Take five repeated samples per captured row, extend once to ten for
throughput MAD/median above 5%, and leave continued instability unresolved.
The absolute budget is a preliminary screen, not a new speed claim. Passing
requires subsequent paired complete-pipeline measurement against Q8.

## Commit sequence

1. `feat(research): encode rotated ternary sketches`: host representation,
   deterministic fitting, independent references and behavioral tests.
2. `feat(cuda): probe ternary channel screening`: fixed CUDA kernels, owned
   scratch/weights, invalid-input and numerical/ownership tests.
3. `feat(bench): measure ternary screening cost`: retained-source fixtures,
   bounded cost observations and reporting through existing benchmark tools.
4. `docs(perf): record ternary screening gate`: preserve raw evidence and the
   decision for each retention budget before conditional top-k/recovery work.

## Host encoding checkpoint

`evaluation::ternary_screen` implements the representation above, with hashes,
source/plane sizes, zero counts and rotated-Q4 quantization error. Its scalar
FP64 projection consumes explicitly selected coordinates. A separate dense
Walsh-matrix reference validates rotation without repeating the encoder's
butterfly implementation.

Four new CPU tests pass. They cover signed constant rows, odd channel padding,
rotation geometry, deterministic signs, tied coordinate ordering, full and
partial projections, error accounting and unsupported values/shapes. The
warning-as-error non-CUDA suite, all-target non-CUDA Clippy and formatting pass.
Logs are `2026-10-31-ternary-host-{tests,clippy}.log.gz` beside this protocol.

```bash
source scripts/dev-env.sh
CARGO_TARGET_DIR=/tmp/m40-research-1030/channel-selection/target \
  RUSTFLAGS='-D warnings' cargo test --no-default-features --locked -j 4
CARGO_TARGET_DIR=/tmp/m40-research-1030/channel-selection/target \
  cargo clippy --no-default-features --all-targets --locked -j 4 -- -D warnings
cargo fmt --all -- --check
```

This checkpoint has no CUDA timing, channel top-k, recovery or model-quality
result. The next unit implements the fixed two-kernel cost probe.

## Source-bound benchmark

The `q8_execution_bench` option `--ffn-q4-channels --ffn-ternary-screen`
requires the retained capture and report, their exact environment, batch one,
five initial samples, a clean source commit and a release binary. It conflicts
with oracle correctness-only and stronger-baseline modes. Use ten iterations
per timed sample. Each layer consumes its first eight distinct input rows.

Each sketch is fitted from the same hashed Q4-derived source as the earlier
oracle. Down-column norms use FP64 accumulation over those Q4 values, then
FP32 storage. The benchmark reports both encodings, all immutable-plane hashes,
setup/upload time, scratch/input/output bytes, retained model/prepared weights,
and peak tracked residency. The 150.88/56.89-microsecond preliminary budgets
use the rounded conservative values in the stronger-reference report.

Every row checks all output planes and coordinate selection before timing,
warms for one GPU second, then takes five samples, extending once to ten for
variation above 5%. It repeats the reference and bit-stability checks after
timing. Layer completion requires unchanged immutable hashes and full cleanup
of probe allocations. Logical code/scale payload is labeled separately from
unavailable measured DRAM traffic.

Only the two screening kernels and their submission/completion costs enter
this boundary. No combined FFN residency or performance result follows until
selection, recovery and exact sparse execution are measured together.

The implemented benchmark passes 65 CPU example tests and CUDA/server
all-target Clippy, plus formatting. Twenty existing hardware tests ignored by
the CPU example invocation are not counted as passes. Logs are retained as
`2026-10-31-ternary-bench-{tests,clippy}.log.gz`. Hardware cost observations
must come from a subsequent clean release build.
