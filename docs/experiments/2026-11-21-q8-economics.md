# B3 prepared Q8 economics

On 48c6333, short dense native and prepared split down/vocabulary now have a matched November reference (consolidation cac9d80). Freeze a separate comparison of compound joint-only and the packet/argmax portfolio using the existing joint_only phases, full-web prompt/512 outputs and original quality checks. Bind stage samples to the matching request and charge profiling perturbation with the existing dataflow off/on pairs. Include load, repacking, duplicate storage, prefill and batch limitations. Compare with the applicable October separate-projection, tiled and later prepared controls, keeping each baseline distinct. A single-row admission says nothing about 2–16-row verification.

| Family | Selected dispatch to verify | Dependency question |
|---|---|---|
| QKV, output | Existing packed-warp real-model projections and shared normalized input | Does any activation preparation still repeat for the same input? Count actual dispatch/launches. |
| Gate/up/SwiGLU | Joint-only compound pair with shared scales and fused SwiGLU | Count materialized representations and intermediates; R4 already has a shared reader. |
| Down | Prepared per-family split 4/256 versus opt-in packets | Determine uncovered-row fallback and whether its owner can be dropped. Reference direct verifier still issues four independent down calls after its shared gate/up. |
| Vocabulary/argmax | Prepared vocabulary versus packet head and partitioned reduction | Charge scratch, logits, extra launches and their whole-request effect. |

Fixed admission: reconstruct per-family resource/latency accounting using measured runtime data and exact owner lifetimes. Prototype at most one removal of a demonstrated duplicate computation/dispatch cost when it offers enough removable work for the 5% whole-request gate, preserves exact output and adds no resident weight copy. A nonduplicated mandatory projection, a storage copy needed by another supported row shape, or a change that misses the frozen gate closes the proposal. Preserve previously positive subthreshold opt-ins with their actual costs. Do not sum GEMV gains. Evaluate C1 as a separate small-N verification hypothesis.
