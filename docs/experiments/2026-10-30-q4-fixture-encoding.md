# Shared Q4 fixture encoding

The October 30 research branches use the existing Q8_0 checkpoint as their
source. `evaluation::quantized_fixture::requantize_q8_to_q4` produces matched
Q4_0 bytes for dense and experimental consumers. This conversion introduces
quantization error; it is separate from lossless channel-oriented repacking.
It does not reproduce a released Q4 checkpoint or establish model quality.

The fixed encoding is `q8-f32-q4-absmax7-v1`:

1. Decode each 34-byte Q8_0 block to 32 FP32 values using its signed codes
   and FP16 scale.
2. Divide the largest absolute value by seven for a positive FP32 scale.
3. Divide values by that scale, round ties away from zero, and clamp to
   signed codes from -8 through 7. A zero block has zero scale and codes.
4. Store the scale rounded to FP16 and pair the first and second groups of
   sixteen codes in the low and high nibbles of sixteen bytes.

Quantization uses the FP32 scale before storage rounding, matching the
repository's existing Q4 fixture convention. Reject incomplete blocks,
nonfinite scales, and nonzero scales that overflow or underflow FP16.

Each report includes source/output SHA-256, payload sizes, value count,
maximum absolute error and mean squared error relative to decoded Q8 values.
Hashing and encoding are preparation work, excluded from kernel timings.
Callers must also record checkpoint identity, tensor shape and offset, and
activation provenance. This helper alone does not validate those identities.

`decode_q4_values` reconstructs Q4 bytes independently into FP64 for numerical
references. Three CPU tests cover known signed values, halfway rounding,
negative source scales, independent zero blocks, malformed lengths and
unrepresentable scales. The warning-as-error non-CUDA suite and all-target
non-CUDA Clippy pass. Raw logs are retained alongside this report.

```bash
source scripts/dev-env.sh
CARGO_TARGET_DIR=/tmp/m40-research-1030/channel-selection/target \
  RUSTFLAGS='-D warnings' cargo test --no-default-features --locked -j 4
CARGO_TARGET_DIR=/tmp/m40-research-1030/channel-selection/target \
  cargo clippy --no-default-features --all-targets --locked -j 4 -- -D warnings
```

No hardware timing or model-quality result follows from this unit. Next,
connect it to the retained FFN capture and source validation before measuring
native dense, native selected and repacked selected execution.
