# Maxwell integration decisions

The [candidate matrix](candidates.tsv) records 17 decisions for which the
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
Phase 4 M40 grid and failure paths. Main's independent port has its own
hardware qualification. The verifier rollback and near-tie cases should
accompany any future verifier port. They do not warrant bringing the
verifier into main by themselves. [Archived CUDA evidence](../2026-11-21-evidence-audit/README.md).

**Do not broaden aggressive KV selection.** Two measured top-8 outcomes
lose a correct dense answer. The profile already defaults to top-8 on
supported paths. Changing that setting would be a separate compatibility
decision. Both a less aggressive policy and the actual peak-VRAM/speed
comparison are under investigation. The data do not support a single
quality ranking across every model or workload.

**Excluded work.** Seven October hypotheses failed their gates, one
activation-preparation operator has no compatible local model, and the
learned draft lacks a qualified checkpoint. Keep the supplied-index Q4
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
