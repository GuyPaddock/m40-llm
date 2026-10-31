# ShiftAdd source audit and first candidate

The first candidate is the upstream SA4_128 weight representation with a
Maxwell FP32 sign-add consumer. Its packed payload saves only 4.1667% against
Q4_0 at the same dimensions. A useful result therefore needs measured
execution or quality evidence; removing individual weight multiplications
does not by itself establish a bandwidth win.

## Pinned source

The [proposal](https://github.com/ggml-org/llama.cpp/discussions/31051) links
`ranery/llama.cpp` branch `shiftadd-cuda`. On October 31 its head is
[`93a0b5111eb71c90f302f1268f1535f83e2e1f13`](https://github.com/ranery/llama.cpp/commit/93a0b5111eb71c90f302f1268f1535f83e2e1f13).
The upstream MIT license names the ggml authors, 2023 through 2026.
[Retained source excerpts](2026-10-31-shiftadd-upstream.txt) include the full
license, format declaration, scalar decoder, fitter, CUDA vector dot and
packed-integer fallback. Preserve this notice in adapted code.

The GitHub connector cannot fetch discussions, so the proposal was read on
the public web. The pinned commit and files were read through the connector.
No upstream branch was executed or merged.

## Exact payload and decode rule

Each output row contains blocks spanning 256 input weights. A block stores
four 32-byte sign planes, eight unsigned scale codes and a two-byte FP16
block scale. Scale codes are group-major, with four codes for each of the
two 128-weight groups. The FP16 scale is at byte offset 136; block stride is
138 bytes. Sign plane bit `j % 8` in byte `p * 32 + j / 8` selects the sign
of plane `p` for input weight `j`; one means positive.

For group `g = j / 128`, the decoded weight is the sum over four planes of
`d * scale_code[g * 4 + p] * sign[p, j]`. The codes are unsigned linear
scale multipliers, not exponent indices. This representation is lossy and
is not a repack of existing Q4 or Q8 values.

Calculated payload sizes, including stored scales:

| Representation | Bytes per 256 weights | Bits per weight |
|---|---:|---:|
| SA4_128 | 138 | 4.3125 |
| Q4_0 | 144 | 4.5 |
| Q8_0 | 272 | 8.5 |

These are packed sizes, not measured DRAM traffic. SA4_128's 138-byte block
stride also requires care with unaligned word loads and memory-sector
boundaries. Both retained FFN widths, 2048 and 11008, divide by 256 exactly.

## Fitting and Maxwell arithmetic

The pinned CPU fitter initializes four amplitudes by successively fitting
the mean absolute residual, then performs four alternating least-squares
refinements. It selects among all 16 sign codes and solves a four-variable
system with pivoting. A singular system keeps the prior amplitudes. Each
block shares one FP16 scale derived from its largest fitted amplitude;
group amplitudes become rounded u8 codes, then signs are reassigned using
the representable amplitudes. This is an inexpensive weight-only fit.
It is distinct from the released checkpoints' output-aware fitting and
mixed-precision training pipeline.

The pinned CUDA dot consumes Q8_1 activations. It expands sign bits to byte
masks, computes packed integer products, and applies the activation scale
and sum correction before weight scales. `ggml_cuda_dp4a` already contains
a scalar signed-byte fallback for architectures without native DP4A.
Therefore the source does not establish an instruction-level compatibility
blocker, and it also does not establish Maxwell speed. Generated instructions
must be inspected rather than assuming that an operation named DP4A uses
that instruction on `sm_52`.

For the first local candidate, keep input activations FP32 and sum selected
positive or negative activations for each plane, applying group amplitudes
after each reduction. This isolates weight representation error from Q8_1
activation error and follows the brief's sign-add mechanism. It is an
adaptation of the representation, not a reproduction of upstream's CUDA
kernel. Retain FP32 accumulation and explicit stream ownership. No Tensor
Core, DP4A or full dequantized-weight buffer is required by this candidate.

## Bounded implementation and gates

1. Add an experiment-only SA4_128 encoder and independent scalar decoder.
   Start from the pinned weight-only fitting method. Document any arithmetic
   precision difference explicitly. Check zero, constant, mixed-sign,
   singular and outlier blocks, sign packing, scale rounding and multiple
   groups. Reject malformed sizes and non-finite values. Test numerical
   decode behavior rather than source strings.
2. Add one CUDA GEMV consumer for rows 1, 2, 4 and 8. Compare against FP64
   dot products of the decoded SA bytes with a reduction-error bound derived
   from the actual FP32 operation count. Verify guards, immutable input,
   workspace ownership, both stream directions and tracked cleanup.
3. Use the retained Qwen FFN inputs and original Q8 weights from layers
   0, 17 and 35. Fit SA4_128 and the shared Q4_0 fixture from the same
   dequantized source. Reuse isolated shared tooling commits only; do not
   import failed screening or fusion candidates. Include source, payload
   and input hashes, fit error, preparation time and total residency.
4. Benchmark all three projection shapes against native Q4_0 and Q8_0 with
   the same distinct input rows. Then measure the complete FFN if the
   projection cost supports it. Apply the common paired timing and complete
   operation gates. Save SASS/register/spill evidence; dynamic counters are
   optional only where supported and must be labeled unavailable otherwise.
5. Report captured-input projection error separately from model quality.
   A latency win can justify an opt-in quality study. A slower representation
   needs a documented memory/error tradeoff before any such study; the
   theoretical 4.1667% payload saving alone does not qualify it. No release
   checkpoint quality, generalization or whole-model speed is inferred from
   this microbenchmark. Do not add a Q4_K backend for a missing comparator.

Commit order remains encoder/reference, CUDA consumer with correctness
coverage, source-bound benchmark, then measured decision. No ShiftAdd
implementation, fit, hardware timing or model-quality result exists yet.
