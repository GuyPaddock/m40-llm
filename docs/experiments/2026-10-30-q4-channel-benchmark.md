# Supplied-index Q4 FFN benchmark

This extends `q8_execution_bench` with `--ffn-q4-channels`. It uses the retained
FFN capture, source validation and paired-bootstrap summary. Runtime dispatch
and defaults do not use this path. The benchmark code needs real-model hardware
validation before any performance claim.

Use `--rows 1` for the first screen or `--rows 1,2,4,8` for the full grid.
The initial trial count is five, with automatic extension to ten when throughput
MAD/median exceeds 5%. Continued instability blocks a speed conclusion.
The complete-FFN gate is the larger of 5% and twice the larger variation, with
a positive paired-bootstrap lower bound. Borderline cases receive twenty fresh
pairs once. Screen and confirmation samples remain separate.

The experiment covers layers 0, 17 and 35; retention 10%, 20%, 40%, 60%, 80%
and 100%; and contiguous, seeded scattered and exact-hidden-magnitude selections.
The retained count rounds up. Every row uses a distinct captured activation.
Importance uses FP64 gate/up calculations on the derived Q4 values, not the
original Q8 hidden values. Computing those scores is excluded from timing.
All three selections are supplied-index upper bounds, not inexpensive predictors.

Controls and candidates retain the same device owners:

- Dense serial executes the existing fused single-row Q4 gate/up/SwiGLU and
  existing Q4 down projection once per row.
- For multiple rows, dense batched executes existing block-loop gate/up,
  separate SwiGLU and block-loop down, including required stream dependencies.
- Native selected and repacked selected use the same selected-channel arithmetic
  and reduction geometry. Only down-weight addressing differs.

Both selected variants are paired against each applicable dense control.
A multi-row conclusion must beat the stronger applicable dense control; a gain
against only the slower one does not qualify. Each arm warms for at least
100 ms of measured GPU work. Paired intervals use `--iterations`, default ten,
and include all FFN kernels and stream waits. CUDA events report GPU time;
host enqueue and completion times are separate. Timed dispatch checks require
the expected launches and no allocation, transfer or host stream synchronization.

Every output must meet `1e-3 + 2e-5*abs(reference)` against an independent FP64
masked FFN. Native/repacked selected outputs must match bit for bit. Guards,
input values, immutable weight/index hashes and tracked allocation cleanup are
checked. A numerical failure is recorded before stopping. It cannot become a
passing timing row. `--ffn-q4-channels-correctness-only` runs the same grid
without primary timings.

Preparation reports include source/output hashes and quantization errors,
repacked storage, immutable residency and shared scratch. Traffic reports
estimate unique cold 32-byte down-weight sectors across the supplied rows,
including scale sharing and code packing. These are lower bounds on unique
sectors, not warp requests or measured DRAM traffic. Dynamic counters remain
unavailable. Model quality and complete decoder throughput remain unmeasured.

Require a clean release build. Restore the capture report's exact recorded
`M40LLM_*`, `CUDA_*` and `OMP_NUM_THREADS` environment before execution; the
existing provenance checker rejects differences. Keep the original Qwen
checkpoint and retained capture artifact unchanged. Use a fresh output path.

```bash
source scripts/dev-env.sh
CARGO_TARGET_DIR=/tmp/m40-research-1030/channel-selection/target \
  M40LLM_ENABLE_CUBLAS=1 cargo build --release --features cuda \
  --example q8_execution_bench --locked -j 4
# Restore the frozen environment before running this binary.
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs /mnt/array-fastest/home/guyep/playground/2025-12-20-m40-llm/m40-llm/target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-q4-channels --ffn-q4-channels-correctness-only --rows 1,2,4,8 \
  --output /tmp/q4-channel-correctness.jsonl
```

The source/reference, sector, decision and CLI tests run without GPU work:

```bash
source scripts/dev-env.sh
CARGO_TARGET_DIR=/tmp/m40-research-1030/channel-selection/target \
  M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --example q8_execution_bench q4_channels -- --test-threads=1
```

All five new tests pass. The complete example target passes 64 CPU tests;
its 20 existing hardware-only tests remain ignored and are not counted as
validation of this experiment. CUDA/server all-target Clippy and formatting
pass. Raw test and Clippy logs accompany this report. Real-model validation
remains the next gate.
