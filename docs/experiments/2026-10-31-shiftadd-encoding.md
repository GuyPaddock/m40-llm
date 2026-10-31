# SA4_128 encoder and CPU reference

The experiment-only [`shiftadd`](../../src/shiftadd/mod.rs) module now fits,
validates, decodes and projects the pinned SA4_128 representation. It does
not add a GGUF type or change inference dispatch. The format remains four
sign planes and two groups of four u8 amplitudes per 256 weights, with one
FP16 block scale and a 138-byte stride.

The fitter follows the pinned greedy initialization and four alternating
least-squares refinements. It uses FP64 offline arithmetic, with precomputed
16-level codebooks, instead of upstream FP32 arithmetic. Byte identity with
the upstream encoder is not claimed. FP16 conversion and u8 amplitude
rounding use ties-to-even; final sign assignment uses the represented
amplitudes and selects the first code on equal distance. Singular systems
keep the prior amplitudes. Underflowed scales encode zero. Non-finite input,
unrepresentable finite FP16 scale, malformed payload and invalid dimensions
return errors. The upstream [MIT notice](../../src/shiftadd/LICENSE.txt) is
retained with the adapted fitter.

The decoder depends only on stored bytes. Its FP32 values are exact sums of
the four signed FP16-times-u8 amplitudes. The CPU projection reference uses
those decoded values and FP64 dot products; it deliberately materializes
weights and is not a timed inference implementation.

Seven behavioral tests pass. A manually constructed payload checks all sign
codes, both groups, adjacent blocks and output channels against an arithmetic
reference independent of the decoder. Other cases cover distinct input rows,
source immutability, deterministic fitting, zero and constant singular groups,
sixteen balanced levels, outliers, tiny and large scales, shape overflow and
non-finite rejection. Fitting sanity bounds do not establish model quality.

Validation commands, with the project environment sourced and this worktree's
`CARGO_TARGET_DIR`, pass:

```bash
RUSTFLAGS='-D warnings' cargo test --no-default-features --locked
cargo clippy --no-default-features --all-targets -- -D warnings
cargo fmt --all -- --check
```

The full non-CUDA suite passes 227 tests, including the seven new checks.
Raw logs have prefix `2026-10-31-shiftadd-encoder-` in this directory.
No GPU experiment, measured model-weight fit or performance result exists
yet. Next implement the FP32 sign-add CUDA consumer and verify it against
the decoded-weight projection reference before timing.
