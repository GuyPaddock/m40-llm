# Maxwell integration decisions

The [candidate matrix](candidates.tsv) records 38 decisions for which the
scope can already be established. Each row has the source branch, exact
commit and report, correctness and runtime tests, operator and complete
request results, memory, quality, dependencies and modules, current
defaults, risk, recommendation, tier and reopening condition. A tier
identifies the kind of integration; the recommendation states whether
it is ready on main. Open hardware measurements will add rows and refine
the dependent recommendations before the final shortlist.

**Ready on the pinned main base.** GGUF shape validation at 541cec1 has
a reproduced failing input and a passing isolated main-based repair.
It introduces no CUDA dependency.

**Correctness before performance.** The graph state repair passes the
Phase 4 M40 grid and failure paths. The separate main-based graph and Q8
eviction repairs have pinned test binaries awaiting hardware results. The verifier rollback and near-tie cases should
accompany any future verifier port. They do not warrant bringing the
verifier into main by themselves. [Archived CUDA evidence](../2026-11-21-evidence-audit/README.md).

**Do not broaden aggressive KV selection.** Top-8 lost three of ten
dense-passing long answers. Top-64 retained all ten where every historical
block fit, and both arms retained all 19 short passes. At 3,203 prompt
tokens, the relay task had 69 eligible historical blocks. Top-64 again
matched the dense answer, while top-8 changed the Boolean from true to
false. This repeats one task at a longer context and leaves the original
ten-case denominator intact. [Relay generations](../2026-11-21-kv-audit/history-relay-generations.json).

The profile already defaults to top-8 on
supported paths. Changing that setting would be a separate compatibility
decision. At capacity 8,448, e100c54 measured a 1.01% peak saving for mixed KV.
Main already consumes this storage directly. In November, all 144 matched operator comparisons reproduced the October
benefit. The live HTTP comparison and the lossy historical-block policy
have separate qualification gates.

**When compound FFN is enabled, keep joint-only ownership.** The paired
HTTP control retained identical gate/up weights in two layouts. Eliminating
the redundant split owners saved 1,647 MiB at the fresh loader peak and
matched seven full-logit rows. The HTTP interval lay within −0.16% to
+0.32%. This comparison qualifies a memory choice inside compound FFN.
[Ownership report](../2026-11-22-q8-ownership/README.md).

**Start Q8 work on main with the missing layer and output-head support.**
The strongest measured prepared-weight subset uses only down and
vocabulary weights. It improved the short dense HTTP request by 11.73%
against native Q8 and cost 1,192,935,424 extra device bytes. Its quiet-host
repeat and single-model residency evidence still matter for a port. The
earlier all-family setting lost 64.21% on HTTP; preparing QKV and output
alongside down and vocabulary lost 6.71% with native MLP. Both losing
settings matched the control's 512 generated IDs.
[Native and selective prepared comparison](../2026-11-21-native-split/README.md).
[Historical family controls](../2026-10-25-q8-prepared-http.md).

Packed GGUF QKV/output, compound FFN, and the packet/argmax portfolio
improved HTTP throughput by 4.02%, 3.20%, and 4.01%, respectively, on
their own historical controls. Packed GGUF loads added no weight or
scratch allocation. The portfolio control already owned the same 37
packets as its candidate. A deployment would still pay for them. Keep
these three comparisons separate when budgeting a port. Each cleared
its frozen 3% gate and fell below the general November 5% threshold.
[Packed-load comparison](../2026-10-25-maxwell-packed-projection-incremental-http.md).
[Compound comparison](../2026-10-27-maxwell-compound-runtime.md).
[Portfolio comparison](../2026-10-29-portfolio-runtime.md).

On the frozen 65-prompt-token, 512-output-token workload, the portfolio's
52.548523 HTTP tok/s leaves 93.92 ms per request to reach 53.06, or
625.45 ms to reach the 56.153149 target. The latter is 1.30 times the
October official Ollama result. The paired November gate remains 5%.
[Historical request budgets](../2026-11-22-http-targets/README.md).

**Excluded work.** The original channel selector, ShiftAdd, additional
multi-row gate/up fusion, lossless KV compression and DualPace all missed
their October gates. The later causal-selector and general adaptive
drafting studies also missed their gates. Activation preparation still
requires a compatible rotated model, and the learned draft requires a
qualified checkpoint. Keep the supplied-index Q4
executor as a control for sparse-layout work. Retain GSQ as a checkpoint
preparation question, and preserve the terminal-prefill dependency trace
for workloads with a larger measured opportunity. [Closure register](../2026-11-21-closed-directions/README.md).

The [dependency-ordered rebuild](../2026-11-21-main-dependencies/README.md)
makes native Q8 checkpoint admission distinct from optional prepared
weights, GPU greedy sampling, compound FFN and speculation. Tier 2
requires qualified model and request performance with acceptable resource
cost. The five historical projection-profile cohorts supply per-layer
timing context for that decision.

Research commits are local. To retrieve any linked GitHub report before
its branch is published, use `git show FULL_SHA:docs/experiments/PATH`
from this repository. Each row's full SHA and path remain valid when its
worktree is removed.
