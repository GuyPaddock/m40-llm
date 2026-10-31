# October 30 lossless-kv protocol

Status: admitted for bounded exploration by the user's execution request.
No experiment has yet passed its correctness or performance gate.

Branch: `research/1030-lossless-kv`. Base: `0eee578`.
Worktree: `/tmp/m40-research-1030/lossless-kv`.
Build directory: `/tmp/m40-research-1030/lossless-kv/target`.
Source `scripts/dev-env.sh` and set `CARGO_TARGET_DIR` to that directory.
Only the M40 is selected. Run hardware work serially after checking compute
processes. New measurements require fresh output files and a clean release build.

The source plan is `m40-llm-research-plan-2026-10-30.md` in the parent
project directory. This branch implements its topic independently from the
other eight branches. Only isolated common-tooling commits may be transferred.
Training and teacher generation remain subject to the separate drafter gates.

### 6. `research/1030-lossless-kv` · P2

**Question:** Does lossless compression of this runtime's FP16 KV save enough traffic to justify reconstruction?

- Begin with real FP16 KV samples and CPU compression analysis. ZipFusion's published codec targets BF16, so its compression ratio cannot be transferred directly.
- Evaluate one FP16 adaptation: blockwise shared exponent bases, packed exponent offsets, preserved sign/mantissa bits, and raw-block fallback whenever compression including metadata loses space.
- Require bit-exact reconstruction, including signed zero and special-value patterns. Preserve raw exceptional blocks.
- Stop if actual FP16 data offers no net storage benefit.
- Otherwise compare dense FP16 attention, compressed data with staged reconstruction, and compressed data reconstructed within attention. Keep token coverage identical.
- Test contexts 1K, 2K, 4K, and 8K within model and memory limits. Include compression, append, page metadata, and fallback costs.

This branch remains distinct from lossy Q4/Q8 KV attention. [ZipFusion paper](https://arxiv.org/html/2610.33719v1)

**Commits:** `feat(kv): probe lossless FP16 blocks` → conditional `feat(cuda): consume lossless KV blocks`.

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
