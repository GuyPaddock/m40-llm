# B3 prepared Q8 economics

On 48c6333, short dense native and prepared split down/vocabulary now have a matched November reference (consolidation cac9d80). Freeze a separate comparison of compound joint-only and the packet/argmax portfolio using the existing joint_only phases, full-web prompt/512 outputs and original quality checks. Bind stage samples to the matching request and charge profiling perturbation with the existing dataflow off/on pairs. Include load, repacking, duplicate storage, prefill and batch limitations. Compare with the applicable October separate-projection, tiled and later prepared controls, keeping each baseline distinct. A single-row admission says nothing about 2–16-row verification.

| Family | Selected dispatch to verify | Dependency question |
|---|---|---|
| QKV, output | Existing packed-warp real-model projections and shared normalized input | Does any activation preparation still repeat for the same input? Count actual dispatch/launches. |
| Gate/up/SwiGLU | Joint-only compound pair with shared scales and fused SwiGLU | Count materialized representations and intermediates; R4 already has a shared reader. |
| Down | Prepared per-family split 4/256 versus opt-in packets | Determine uncovered-row fallback and whether its owner can be dropped. Reference direct verifier still issues four independent down calls after its shared gate/up. |
| Vocabulary/argmax | Prepared vocabulary versus packet head and partitioned reduction | Charge scratch, logits, extra launches and their whole-request effect. |

Fixed admission: reconstruct per-family resource/latency accounting using measured runtime data and exact owner lifetimes. Prototype at most one removal of a demonstrated duplicate computation/dispatch cost when it offers enough removable work for the 5% whole-request gate, preserves exact output and adds no resident weight copy. A nonduplicated mandatory projection, a storage copy needed by another supported row shape, or a change that misses the frozen gate closes the proposal. Preserve previously positive subthreshold opt-ins with their actual costs. Do not sum GEMV gains. Evaluate C1 as a separate small-N verification hypothesis.

## Historical per-layer replay

The [replay](2026-11-21-q8-economics/history.sh) reads the original
release M40 HTTP records and event traces from five October cohorts.
For every cohort the two profiled/unprofiled token traces matched all
512 sampled IDs. The five measured pairs per cohort generated the same
512-token continuation from the same 65-ID Qwen Q8_0 prompt. This gives
75 measured token samples across steps 0/255/511. Each contains 400
spans, including 11 operations on each of 36 layers. The four token-level
operations include vocabulary and greedy selection. The source revisions,
binary and raw-data hashes, and any historical working-tree changes are
in [the cohort index](2026-11-21-q8-economics/historical-stages/cohorts.jsonl).

| October runtime configuration | QKV | Output projection | Gate/up/SwiGLU | Down | Vocabulary | Argmax | Profile off/on HTTP tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Packed FFN, native attention projections | 2.496 | 1.741 | 10.044 | 4.751 | 1.563 | 0.258 | 41.441 / 39.869 |
| Split FFN, native attention projections | 2.522 | 1.773 | 8.047 | 4.794 | 1.563 | 0.259 | 45.044 / 44.600 |
| Split FFN, warp attention projections | 1.882 | 1.356 | 8.031 | 4.750 | 1.565 | 0.259 | 47.470 / 46.991 |
| Split FFN, packed-warp attention projections | 1.388 | 1.009 | 8.045 | 4.771 | 1.563 | 0.259 | 49.402 / 48.866 |
| Compound FFN, packed-warp attention projections | 1.392 | 1.016 | 7.421 | 4.761 | 1.564 | 0.259 | 50.878 / 50.379 |

Stage entries are median CUDA-event milliseconds at generated step 255.
For a per-layer operation, the total is taken across all 36 layers of
one token before calculating the five-request median. Within the compound
cohort, the median gate/up/SwiGLU span at each individual layer ranged
from 0.2053 to 0.2069 ms. Every sampled middle token had one fused gate/up
launch per layer. Its separate host synchronization median was 13.710 ms,
overlapping GPU work within the 19.795 ms median instrumented token wall.

The [layer and operation files](2026-11-21-q8-economics/historical-stages/)
retain medians at all three positions, including CPU wrapper times,
synchronization and copy durations. Changing the layer of one span,
changing a token position, or removing one session's middle token
caused the replay to reject the cohort. Reproduce from this worktree
with `bash docs/experiments/2026-11-21-q8-economics/history.sh NEW_DIRECTORY`.
The profile switch is the only paired difference *within* each row.
Runs across revisions cannot isolate component speedups. The measured
instrumentation cost and host waits preclude interpreting event spans as
standalone kernel service or the savings from a proposed patch.

These results locate the largest remaining projection costs in gate/up
and down. A large mandatory projection alone does not establish redundant
work. The B3 joint-only study separately measures actual ownership,
additional launches, exact output and whole requests on the pinned model.

## Does single-row decode prepare the same activation twice?

**FAIL for a redundant preparation launch in the measured compound path.**
At generated position 255 the stage counters give 36 attention norms, 36
fused QKV launches, 36 FFN norms, 36 fused gate/up/SwiGLU launches and
36 down projections. There are 36 layers. The two norms read different
inputs. QKV consumes one 8,192-byte normalized row. The later FFN norm
reads the result of the attention residual and writes the single
8,192-byte row used by both gate and up. Gate/up writes a 44,032-byte
SwiGLU output. Down consumes that entire row. The two 44,032-byte gate
and up scratch slots serve fallback paths and receive no writes here.
The counters, source arguments in [forward.rs](../../src/infer/forward.rs)
and the [buffer lifetime analysis](2026-10-28-dataflow-findings.md) agree.

Independent output blocks reread the normalized input, which can reside
in cache. The available traces count launches and record event spans.
They do not count physical DRAM reads. Cross-block reuse of FP32 inputs
would need a new ownership or execution scheme. The profile supplies no
duplicate producer to remove from the measured single-row path. Keep the
existing fused execution and leave the weight-reuse hypothesis for
multi-token verification to C1.

The [separate-plane study](2026-10-24-q8-separate-projections.md) measured
isolated Q, K, V, gate and up against individual GGUF projections. Those
ratios use a different baseline from the fused decode calls above.
The [input-tiled study](2026-10-24-q8-input-tiled.md) exhausted its twelve
frozen mappings. Neither mapping displaced the existing separate-plane
single-token down/vocabulary path. Keep the vector layout for admitted
row shapes and the losing coalesced mapping at `bb4d30e` for provenance.
