# Fixed multi-row Q4 fusion experiment

Reuse the existing single-row Q4 gate/up reduction unchanged, adding a row
coordinate to its grid and activation/output addressing. The new research
entry accepts 1, 2, 4 or 8 rows and an explicit stream. Its thread count follows
the existing K-dependent selection. There is no new weight representation,
intermediate gate/up allocation, tuning sweep or runtime dispatch change.

First validate all hidden outputs against independent FP64 native-Q4
projection/SwiGLU references and against repeated existing single-row launches.
Use distinct rows, tail channel counts, the real FFN shape, producer/consumer
stream dependencies, guards, immutable weights and allocation cleanup. Reject
invalid shapes, alignment and writable overlap before launching.

Then use the shared Q4 fixture encoder from `300e847`, transferred as `85485b8`,
and retained inputs at layers 0, 17 and 35. Compare three gate/up paths with
the same existing native down consumer: repeated fused single rows, separate
multi-row gate/up plus SwiGLU, and one fused multi-row launch. Include complete
FFN latency and projection-only diagnostics. A second complete-FFN control
uses the existing per-row down path where applicable; a candidate must beat
the strongest control rather than benefit from a slow down choice.

Freeze rows 1, 2, 4 and 8, five alternating pairs, the common 5% FFN gate and
positive paired confidence bound, and the established bounded variation and
confirmation rules. Include required cross-stream dependencies in timing.
Match encoded Q4 bytes across all paths and report source conversion error
separately. No model-quality or runtime integration follows from kernel-only
timings.

Commit sequence: `feat(cuda): probe multi-row Q4 fusion`, with numerical and
ownership tests; `feat(bench): compare multi-row Q4 fusion`, with source-bound
controls; then a measured-results commit and the protocol's stopping decision.

## CUDA validation checkpoint

The research `CudaContext::q4_fused_rows_async` entry is implemented with
explicit stream selection and shape/span validation. Runtime dispatch remains
unchanged. Two new M40 tests pass with cuBLAS enabled and disabled. All hidden
outputs match the FP64 reference within `1e-3 + 2e-5*abs(reference)` and repeated
existing single-row outputs bit for bit. Coverage includes rows 1/2/4/8 with
K64/H35 and two distinct rows at K2048/H11008, both stream directions, immutable
weights, input/output guards, repeatability and tracked allocation cleanup.
Each enqueue records one launch with no device allocation/copy/wait/sync.
Invalid shapes, pointer alignment and output overlaps fail before launch.

Regression checks pass: native Q4 projections 2, compound forward/ownership 3,
and attention grid 11. The warning-as-error non-CUDA suite, CUDA/server
all-target Clippy and formatting pass. These are numerical and ownership
results, not performance measurements. Next add the source-bound complete-FFN
comparison described above.

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/ffn-fusion/target
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --test q4_fused_rows \
  -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=0 cargo test --features cuda --test q4_fused_rows \
  -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --test gemm_mixed q4_0 \
  -- --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --test q8_compound_runtime \
  -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test attention_parity_cuda_grid -- --nocapture --test-threads=1
RUSTFLAGS='-D warnings' cargo test --no-default-features --locked -j 4
M40LLM_ENABLE_CUBLAS=1 cargo clippy --features cuda,server --all-targets \
  -j 4 -- -D warnings
```

Raw logs use `2026-10-31-q4-fusion-` with `on`, `off`, `regression`, `clippy`
and `noncuda` suffixes, followed by `.log.gz`, beside this checkpoint.
