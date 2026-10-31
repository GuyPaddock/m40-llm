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

## Source-bound timing implementation

`q8_execution_bench --ffn-q4-fusion` requires the frozen capture/report and
environment, a clean release source, `--rows 1,2,4,8 --trials 5 --iterations 10`.
The same three derived native Q4 matrices and all diagnostic activation buffers
remain resident in every arm. The four arms are:

| Arm | Gate/up/SwiGLU | Down |
| --- | --- | --- |
| `serial_down` | Existing fused single-row calls | Existing single-row calls |
| `serial_gate_batched_down` | Existing fused single-row calls | Existing block-loop batch |
| `unfused_batched` | Two block-loop projections and SwiGLU | Same block-loop batch |
| `fused_rows` | One new multi-row fused call | Same block-loop batch |

Each of twelve layer/row cases first checks all hidden and complete FFN
outputs against scalar FP64 reconstruction. New fused hidden bits must match
existing single-row fusion, and complete output bits must match the serial
gate/up control using the same down consumer. Repeat the checks after timing,
verify input guards and immutable weights, then release each layer's owners.

Warm each complete arm for at least one GPU second. Measure the candidate
against all three complete controls and against the two distinct gate/up
controls, giving 36 complete and 24 gate/up comparisons. The callback timing
code reuses the channel branch's alternating order and 5/10/20 rules, with the
existing FFN bootstrap summary. Gate/up rows are diagnostic; only complete
results against the strongest control can qualify a candidate. Required
cross-stream dependencies, GPU events and host submission/completion remain
inside their stated boundaries.

The benchmark passes 63 CPU example tests, CUDA/server all-target Clippy and
formatting. Twenty pre-existing ignored hardware tests are not counted as
passes. The first CPU reference test used exact equality for a closed form
with different FP64 association; its 2.2e-16 difference now uses a 1e-14 host
test tolerance. CUDA tolerances and bit-parity requirements are unchanged.
Initial and passing logs are preserved as `2026-10-31-q4-fusion-bench-*.log.gz`.
Hardware timings remain pending at this checkpoint.
