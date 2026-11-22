# Main serving baseline admission

Baseline e1403e0dbe2037cd8fc2b0af32addf97a4df3488. Before HTTP timing,
run this revision's actual loader, transformer-layer and output-head mapping
and raw prompt preparation against the local GGUFs, using the no-CUDA
build. Run models serially. Inspect all frozen Qwen inputs, and a separate
TinyLlama chat prompt. Record BOS behavior, actual IDs and the declared
context limit. A metadata or mapping failure disqualifies a full-model
timing even if an HTTP route can return text through an embedding fallback.
Successful host mapping admits a hardware smoke; it cannot prove hardware
inference or numerical correctness.

On hardware require all layers, a real sampled output and matching work
before any cross-branch speed comparison. Evaluate default cache and
explicit dense serving separately on the same supported checkpoint, context,
template and input IDs. Main allocates its model-declared capacity and has
no CLI override. Its buffered endpoint returns output text only, so collect
token IDs and counts separately. Treat unsupported model/branch pairs and
their precise failure as inventory evidence. Never substitute weights within
a paired comparison.

## Admission results

The actual no-CUDA loader at main e1403e0 mapped every layer and output head
for TinyLlama F16 (22 layers, context 2048), Qwen2.5-3B F16 (36, 32768) and
Llama-3.2-1B F16 (16, 131072). Source and exact model hashes, prompts,
tokenizer IDs and loader results are in [models.jsonl.gz](models.jsonl.gz).
Memory capacity and supported CUDA operations still need hardware checks.

The Qwen Q8_0 checkpoint failed both checks. Its first query projection is
Q8_0; main accepts F16 or Q5_1 at that projection. Its tied output embeddings
are Q8_0; the tied head requires F16. The local Q4_0 checkpoint failed on
its Q4_0 embeddings, and its dedicated Q6_K head is unsupported. **FAIL for
full-model timing on main** for these exact two checkpoints. Their November
results must be compared on a Q8/Q4-capable revision.

Main also prepends token 151643 to the preformatted Qwen inputs, giving
66/2061/8198 versus the Phase 4 fixture's 65/2060/8197. Using the same prompt
string alone would not give the same transformer work. TinyLlama's own raw
chat question yields 46 IDs on main. Require all 46 to match Phase 4 before
timing. For cross-branch comparisons an explicitly different model gets its
own cohort. The Llama-3.2 declared 131072 context also makes unbounded main
allocation a separate resource question. These host checks have no measured
speed or device-memory outcome.

Replay from this worktree. Source `scripts/dev-env.sh`, then run
`cargo build --release --locked --no-default-features --example main_model_admission`.
Run `bash docs/experiments/2026-11-21-main-baseline/replay.sh NEW_DIR PERFORMANCE_CONTEXTS LOCAL_MODELS_SHA256`.
The final two arguments identify the frozen fixtures on the consolidation branch.
