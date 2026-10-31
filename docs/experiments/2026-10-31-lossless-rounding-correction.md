# Match FP32 rounding across lossless attention consumers

The [preserved release failure](2026-10-31-lossless-release-failure.md) is an
arithmetic contraction difference, not a codec reconstruction failure.
A focused replay of layer 0 at 1,024 tokens identifies the first mismatch in
`Fused`, batch one, output word 97. Actual bits are `0xbd43a5dd`; staged bits
are `0xbd43a5dc`. The fused arm's maximum absolute FP64 error is
`2.9544587487606577e-7`. Numerical tolerance passed, but the required exact
matched-output check correctly rejected the result.

The [diagnostic log](2026-10-31-lossless-real-diagnostic.log.gz) retains the
failing replay. It uses the same hash-admitted capture and synthetic queries
as the release grid. Missing artifacts fail the ignored hardware regression;
they do not produce a skipped pass.

The original executable's generated code exposes different rounding paths:

- Dense online-softmax denominator uses one FP32 fused multiply-add.
  The packed specialization uses a rounded multiply followed by an add.
- Dense accumulator updates round `weight * value` and then fuse
  `accumulator * old_weight + contribution`. For the last unrolled component,
  the packed specialization rounds `accumulator * old_weight` first and fuses
  the other product with the addition.

The [dense PTX](2026-10-31-lossless-dense-before.ptx.gz) and
[packed PTX](2026-10-31-lossless-packed-before.ptx.gz) retain both head-128
partition functions from the original release executable. Their
[dense SASS](2026-10-31-lossless-dense-before.sass.gz) and
[packed SASS](2026-10-31-lossless-packed-before.sass.gz) retain the corresponding
Maxwell instructions. These are static compiler evidence, not dynamic counts.

The correction uses explicit round-to-nearest FP32 FMA for the dot product
and denominator. For each accumulator component, it explicitly rounds the
new weighted value, then fuses the previous accumulator scaling and addition.
Both storage specializations now express the original dense operation order.
The shared merge kernel, partition count, query generation, original FP16
words, output tolerance and exact-bit gate remain unchanged.

The runner reports arm, batch, output index and both bit patterns on failure.
A focused ignored hardware regression replays the real first case across all
four batch sizes and five consumers, including storage guards and cleanup.
Full-grid release admission is still required before performance measurement.

## Qualification

The corrected real first case passes all twenty arm/batch combinations with
cuBLAS enabled and disabled. The largest matched-prototype FP64 error is
`4.334535563810249e-7`; all four prototype arms produce identical bytes.
Both configurations pass the two CUDA decoder/attention tests and fifteen
runner tests. The affected attention grid, projection, MLP and forward
checks pass. An optional model diagnostic has no input model and is not
counted as executed quality coverage; ignored tests are reported separately.
CUDA/server all-target Clippy, formatting and six warning-as-error CPU
fixture tests also pass. The initial formatting check found one trailing
blank line; formatting was corrected before the qualification run.

Raw logs and the executed qualification script are retained in the
[checksum manifest](2026-10-31-lossless-rounding.sha256). The earlier failed
implementation and its release evidence remain immutable. No benchmark,
model-quality result or runtime promotion follows from these checks.

Focused reproduction, after sourcing `scripts/dev-env.sh`, selecting the
idle M40 and exporting the two capture paths:

```bash
export LOSSLESS_CAPTURE_REPORT=target/lossless-kv-capture-01.jsonl
export LOSSLESS_KV_INPUTS=target/lossless-kv-inputs-01.bin
M40LLM_ENABLE_CUBLAS=1 cargo test --release --features cuda --locked \
  --test lossless_kv_cuda --example lossless_attention_bench \
  -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --release --features cuda --locked \
  --test cuda_transfer_order --test weight_allocation_accounting \
  -- --include-ignored --nocapture --test-threads=1
```

The explicitly ignored transfer/ownership checks were then invoked separately:
five transfer cases and one allocation replacement/cleanup case pass. Four
unrelated graph checks from the forward suite remain ignored in this focused
correction run; their earlier recorded qualification is unchanged.
