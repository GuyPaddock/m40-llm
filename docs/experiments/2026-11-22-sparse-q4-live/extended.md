# Expanded sparse-Q4 checkpoint quality gate

Run each thresholded arm whose initial held-out quality verdict is
`PASS`. The packed arm preserved all four dense-passing answers and
its two sampled perplexity ratios were 1.003002 and 0.996314. This
admits the next gate. Keep each layout, the three selected layers,
the original cutoff weights and the original Qwen2.5-3B Q4_0 GGUF
fixed. Run separate processes per admitted arm.

Use the 48 November B2 cases at the pinned input manifest commit
`b4d5176a64ae4134389b68a7e8bd041913d81b5a`. Read the original case order,
rendered prompts, token IDs, scoring rules and output limits from the
uncompressed manifest SHA-256
`18d0c6f69dbdfa3bf976780b5b20e14d06dfec5f32980418a9d4f308da9f1cc3`.
Verify all IDs against the Q4_0 tokenizer before loading either GPU model.
The suite uses 24 tasks in six families, each in both context lengths.
Use sequential prefill, dense FP16 KV capacity 3,072, greedy seed zero,
both models from identical Q4_0 bytes. Charge the same selected-Q4 owner
and workspace usage in each candidate. Retain the physical capacities
and actual candidate projection counts.

For each case record and flush the native generation and its frozen
score before opening the candidate result. Retain complete outputs,
individual scores, generated IDs, divergence position, repetition,
timing and stop reasons. Stop the arm on its first lost dense-passing
answer, newly repetitive output or nonfinite candidate logits. Count
dense failures separately. If an arm completes the grid, review answers
and instruction conformance against the original prompt and frozen
scorer. Require dense-passing evidence at both context lengths. Report
families with insufficient coverage separately. The initial likelihood
gate remains the recorded likelihood result.

At continuation offsets 0 and 8 where available, and at the last
sampled offset, compare the full logits on the same
native history. Retain FP32 vectors to recompute maximum and mean
absolute error, KL from the native distribution, and top-10 overlap.
These measurements remain diagnostic. A failed answer is decisive even
when the sampled greedy IDs agree. If any arm loses a dense pass, stop
research on that arm without tuning its cutoff. Complete-request speed
is a separate experiment eligible only after expanded quality passes.

This is an untimed quality comparison with two model owners in one
process. No measured latency from this run constitutes inference
speed qualification.

## Reproduction

Source `scripts/dev-env.sh`. Build `model_quality` in release mode with
`M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo build --release --locked
--features cuda --example model_quality`. Run the example's
`sparse_q4_extended::tests` before the device comparison. The original
projection guards are in the first quality experiment.

Once the build passes, run under the shared M40 lock from this worktree.

```sh
bash docs/experiments/2026-11-22-sparse-q4-live/run-quality.sh \
    QWEN_Q4_GGUF NEW_RAW_DIRECTORY MODEL_QUALITY_BINARY FINISHED_INITIAL_RUN
```

The initial run directory supplies both thresholded dispositions and
source hashes. The runner selects only the admitted arm. For each output
report, use the following from this worktree. The B2 replay module on
the consolidation branch recomputes metrics from the independent FP32
readbacks.

```sh
PYTHONPATH=../consolidation/docs/experiments/2026-11-21-kv-audit \
    python3 docs/experiments/2026-11-22-sparse-q4-live/replay-extended.py \
    ARM.jsonl RAW_DIRECTORY/inputs.json
```
