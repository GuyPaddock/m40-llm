# Quality beyond 64 old blocks

Freeze at c484536 on a separate worktree so the ongoing 48-case study and
the timing/residency replay binaries keep their source and settings.

With a 1024-token recent window and 32-token blocks the first clipped
position has length 3073. Test the 32-token top64 and top8 consumers and a
dense FP16 reference at >=3200 prompt tokens; use capacity 4096 for all.
For full mixed history use the same direct FP16-K/Q4-V consumer, top64
with 64-token blocks. Old V quantization groups are independent of attention
block size (one scale per old token and KV head). Old K and the recent ring
have the same representation. Only the selection and summary partition
changes. This full-coverage construction is valid within the tested range.
Set chronological selected-block order in every mixed arm so a shared
subset has the same accumulation order. The stock score-order grid is a
separate comparison.

Use the existing archive-line template, greedy sampling, exact Qwen GGUF,
packed dequant-SGEMM prefill and three existing tasks that passed dense at
2K: nov-crate (retrieval), nov-relay (multiple facts), nov-fallback (code).
Freeze the first archive-line count reaching 3200 valid IDs, plus a control
at the first count reaching 2048 on nov-crate. Use the original task
rubrics and output budgets. Hold task text outside the recent window.
Generate every dense reference first. Re-score at the new length; an old
dense pass cannot replace the current one.

At the 2K control both top64 configurations cover every position. Require
bit-identical full logits on matched dense-generated histories and matching
free generated IDs and scored outputs. At 3200 verify that the 32-token
partition has >64 blocks and the 64-token partition has <=64 in each layer.
Record selected budgets, complete answers/IDs, divergences, aligned full
logits, KL, top-10 overlap, allocations and teardown. Distinguish direct
mixed full versus top64 and top8, and dense versus full mixed. Abort
interpretation if the equivalence control fails. Any lost dense pass fails
the preservation gate for that configuration. This is a bounded quality
diagnostic; no runtime or speed promotion follows from it.

Two independent CPU preparations produced
`e793aaae9aa3c6d5d281a67b028ae94ae3306d574957d48200c2f6ee74f781c2`.
The checked-in compressed input contains prompts, exact token IDs and rubrics.
The [lengths](lengths.tsv) are 2051, 3210, 3203 and 3200. Each long probe
crosses the 64-block boundary with the task itself in old history.

Replay from this worktree with `scripts/dev-env.sh` sourced. Build the CPU
example with `cargo build --release --locked --no-default-features --example kv_quality`.
Use `target/release/examples/kv_quality prepare-history MODEL NOVEMBER_INPUTS NEW_INPUTS`.
Build the hardware binary with
`M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo build --release --locked --features cuda,server --example kv_quality --target-dir target/hardware`.
With exclusive access, use `bash docs/experiments/2026-11-21-kv-history/run.sh NEW_DIRECTORY`.
The runner checks allocations, retains full FP32 logits at the frozen dense
trajectory observations, and emits all five paired comparisons per task.
