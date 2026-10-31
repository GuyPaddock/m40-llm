# October 30 channel-selection protocol

Status: admitted for bounded exploration by the user's execution request.
No experiment has yet passed its correctness or performance gate.

Branch: `research/1030-channel-selection`. Base: `0eee578`.
Worktree: `/tmp/m40-research-1030/channel-selection`.
Build directory: `/tmp/m40-research-1030/channel-selection/target`.
Source `scripts/dev-env.sh` and set `CARGO_TARGET_DIR` to that directory.
Only the M40 is selected. Run hardware work serially after checking compute
processes. New measurements require fresh output files and a clean release build.

The source plan is `m40-llm-research-plan-2026-10-30.md` in the parent
project directory. This branch implements its topic independently from the
other eight branches. Only isolated common-tooling commits may be transferred.
Training and teacher generation remain subject to the separate drafter gates.

### 1. `research/1030-channel-selection` · P0

**Question:** Can selective Q4 execution save enough complete-FFN time to pay for screening and recovery?

1. Build Q4_0 fixtures from the existing model's dequantized Q8 weights, retaining source hashes and explicitly labeling the additional quantization. Compare against dense execution of those same Q4 bytes.
2. Implement supplied-index execution for gate/up rows and down columns. Compare conventional storage with channel-oriented storage at 10%, 20%, 40%, 60%, and 80% retention; include 100% as a correctness control.
3. Preserve original Q4 block scales and values during repacking. Account for scale reads, padding, indices, and memory sectors. Test contiguous, scattered, and captured-importance selections.
4. Validate against a dense masked reference. Supplied indices exclude selector cost and establish an execution upper bound only.
5. If that screen passes, implement the paper's weight-derived ternary screening and rank-32 recovery, initially using its 128-coordinate rotation blocks. Compare fixed top-k with thresholds on the same scores. Charge rotation, scoring, selection, recovery, and all extra weights.
6. Advance to an opt-in dense-Qwen evaluation only if the complete pipeline remains profitable. Keep MoE conclusions unvalidated under the selected model scope.

Selecting channels by first computing full gate/up projections is a diagnostic comparator, with that cost included, not a cheap predictor. The paper's findings concern routed experts. [Channel-selection paper](https://arxiv.org/html/2610.37386v1)

**Commit sequence:** shared benchmark support → `feat(cuda): probe selected Q4 channels` → `feat(cuda): add ternary channel screening` → conditional runtime evaluation.

## Common experimental contract

Reuse the existing FFN capture, paired timing, model-quality, and HTTP comparison tools. Extend their reporting utilities rather than creating competing timing protocols.

Every experiment records:

- Source revision, executable/model hashes, configuration, actual dispatch, and input provenance.
- CUDA-event execution time and host wall time, with preparation, uploads, allocation, and capture costs reported separately.
- Packed payload, metadata, scratch, retained source weights, and peak device residency.
- Logical bytes, estimated memory-sector requests, and measured counters as distinct quantities. Unavailable counters remain unavailable.
- Actual generated tokens, stop reasons, outputs, trial variation, and quality results.

Use layers 0, 17, and 35, with the existing `2048 → 11008 → 2048` FFN dimensions. Benchmark batch one first, then distinct rows at batches 2, 4, and 8. Extend existing fixture support where necessary; repeated copies of one activation are not concurrency evidence.

Use five alternating timing pairs, the repository's variation rules, and paired confidence intervals. FFN candidates need at least 5% complete-FFN improvement with a positive lower confidence bound before runtime integration. Other candidates must clear the controlled benchmark's `max(3%, 2 × variation)` threshold at their complete operation boundary. Microbenchmarks never establish whole-model gains.

Exact candidates retain existing numerical, token, task, and ownership gates. Approximate candidates receive matched-prefix logit/KL measurements, token-weighted loss/perplexity, task outcomes, and complete generations. Add a frozen long-generation extension using the existing task prompts, a 1024-generated-token cap, sufficient context capacity, and natural stopping. Report repetition and early-stop failures; do not tune against held-out results.

Follow the [controlled benchmark protocol](../controlled_benchmark.md). Historical throughput numbers provide context; each new speed claim needs a contemporaneous matched control.

## Delivery and evidence

Begin with this protocol commit. Implement the conditional units above in
dependency order, with behavioral tests in their corresponding commits.
Record raw trials and failed cases under `docs/experiments/`, summarize results
in `docs/perf_baselines.md`, and update `docs/research_status.md`.
Preserve failed implementations in a commit before any removal.
A decision report must distinguish measured results, estimates and missing
prerequisites. A negative gate closes only its dependent stages, not other topics.

For exact changes, run affected projection/MLP, forward and ownership checks,
including relevant cuBLAS on/off coverage. Run formatting, warning-free
non-CUDA tests and CUDA/server Clippy for code changes. Docs-only changes use
formatting, whitespace and local-link checks. Hardware tests run explicitly
with ignored tests included and one test thread. No skipped test is a pass.

## Initial checkpoint

- Worktree created at the agreed base; implementation and measurements pending.
- Existing Qwen model and retained fixtures are the initial inputs.
- Defaults and historical evidence are preserved.
- Next: inspect this topic's current implementation and retained results,
  freeze its smallest decisive experiment, then implement and validate it.
