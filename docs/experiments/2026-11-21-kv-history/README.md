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
