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
