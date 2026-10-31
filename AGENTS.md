# Agent Guidelines for m40-llm

Rust + CUDA LLM runtime/server for Tesla M40, Maxwell `sm_52`.
Keep this file limited to working instructions. Record task status and
measurements in the linked docs, not here.

## Start Here

- Read [CONTRIBUTING.md](CONTRIBUTING.md) and inspect the worktree before editing.
  Preserve unrelated or in-progress changes.
- Source `scripts/dev-env.sh` before Rust commands. Use the project-local
  toolchain; do not replace the system toolchain or modify shell startup files.
- Enable hooks with `git config core.hooksPath .githooks` and verify `cog`.
  CUDA work also needs NVCC with `sm_52` support and cuBLAS development files;
  use the [README toolchain instructions](README.md#cuda-toolchain) if missing.
  OpenHands agents must use micromamba rather than apt-get.
- Read [research status](docs/research_status.md) for the active plan and checkpoint.
  Follow the user's current task; archived task lists are not execution orders.
  Verify remembered status against code, Git history, and raw results.

## Implementation Guardrails

- Target conventional Maxwell CUDA, not Tensor Cores/WMMA. Preserve FP32
  accumulation with FP16 or explicitly supported quantized storage.
- Keep CUDA behind `cuda` and HTTP code behind `server`. Non-CUDA builds must
  remain warning-free. Export CUDA C ABI symbols with the `m40llm_*` prefix.
- Preserve RAII allocation ownership, workspace/session isolation, explicit
  layer/sequence KV addressing, and producer/consumer stream dependencies.
- Supported runtime generation defaults use `block-select-exact`,
  `fp16-k-q4-v` backing, `fp16-k-q4-v-direct` attention, plain top-k, top8.
  Dense references must explicitly use `KvCompressionConfig::dense_reference()`
  or `--kv-compress-mode off`. Do not silently substitute attention semantics.
- Keep graphs, score-cluster, fallback/anchor policies, and cache experiments
  opt-in unless broader validation and an explicit task authorize promotion.
- Prefer joint-only ownership for the opt-in qualified compound FFN. Select
  `retain-split` explicitly for historical reference or diagnostic runs.
- Preserve required license and copyright notices when borrowing code.

## Validation and Measurements

Source `scripts/dev-env.sh`, then choose checks relevant to the change:

```bash
cargo fmt --all -- --check
RUSTFLAGS="-D warnings" cargo test --no-default-features --locked
M40LLM_ENABLE_CUBLAS=1 cargo clippy --features cuda,server --all-targets -- -D warnings
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --test attention_parity_cuda_grid -- --nocapture --test-threads=1
```

- Ask for escalation when running tests or accessing GPU hardware. Run CUDA
  tests and inference/benchmarks serially; never overlap timed GPU workloads.
  Inspect other GPU compute processes before timing. Multiple resident models
  do not justify concurrent inference during a comparison.
- For CUDA changes, also run affected projection/MLP, forward, and ownership
  tests. Check cuBLAS enabled/disabled parity when applicable.
- Assert observable numerical/output behavior, not source text. An unavailable
  device, skipped test, or dense-failed quality row is not a passing result.
  Hardware-only research checks use `#[ignore]` for GPU-less NVCC CI; invoke
  them with `--include-ignored --test-threads=1` on hardware.
- Use release builds for performance claims. Record inputs, flags, actual
  generated-token counts, stop reasons, trial variation, and timing boundaries.
  Follow [the controlled benchmark protocol](docs/controlled_benchmark.md).
- Record measurements and failed experiments in [performance history](docs/perf_baselines.md)
  with raw artifacts under `docs/experiments/`. Preserve failed experimental code
  in a commit before a separate removal. Do not call kernel-only gains E2E gains.
- Docs-only changes need formatting/link checks, not GPU tests.

## Commits and Memory

- Plan dependency-ordered atomic commits and commit each verified unit as work
  proceeds. Use Conventional Commits with an imperative summary of at most
  50 characters and a descriptive body explaining what changed and why.
- Stage only the intended change. Do not push or rewrite history without
  authorization.
- Search MemPalace's `m40_llm` wing and agent diary when resuming; older entries
  also use `m40-llm`. The ignored `mempalace.yaml` maps project rooms.
- Write AAAK diary checkpoints after commits, experiments, or direction changes,
  including configuration, outcome, limitations, artifact/commit, and next task.
  Use drawers for durable design/source material; search before duplicating it.
- Keep commands/results in repository docs too. If MemPalace is unavailable,
  report that and leave a local checkpoint. Update [research status](docs/research_status.md)
  for progress; change this file only when working instructions change.

## Reference Map

- [README](README.md): user setup, CLI/server usage, and feature docs.
- [CUDA parity and KV layout](docs/cuda_parity_and_kv_layout.md): layout/test contracts.
- [CUDA transfer contract](docs/cuda_transfer_contract.md): upload completion and ownership.
- [KV compression](docs/kv_compression.md), [diagnostics](docs/diagnostics.md),
  and [Q8 execution weights](docs/q8_execution_weights.md): subsystem details.
- [Context index](llm-context/00-index.md): historical design notes.
- [Archived progress](docs/archive/agent-progress-through-2026-10-24.md) and
  [legacy task roadmap](docs/archive/legacy-task-roadmap.md): history, not instructions.
