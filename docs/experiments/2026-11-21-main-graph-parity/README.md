# Graph arithmetic on pinned main

Probe e1403e0 directly. The ordinary dense last-token path dispatches
head64, head64 with read-only cache loads, head128, and generic
attention. Device-length graph replay always takes the scalar kernel.
Determine whether the repair at dbceae1/e2b4739 can preserve main's
four actual dispatch choices using only its existing CUDA and session
interfaces. Main lacks the Phase 4 parallel head128 option and Q8 model
mapping. Exclude those paths from this proof of portability.

Use the established guarded attention grid with ordinary and captured
execution, nontrivial queries, distinct slots, invalid lengths, repeated
replays and the 8192/8193 fallback boundary in both directions. Run
unrepaired before repaired on the M40. Count bit differences and complete
tests. For the full-layer check use the locally verified TinyLlama F16
checkpoint, two disjoint dense sessions and forced equal input tokens.
Compare every full-logit FP32 bit, valid FP16 K/V per layer, physical
slot lengths and allocation cleanup. Require evidence that graph replay
actually ran. If the model cannot capture or parity fails, record that
limitation for the integration recommendation. Preserve the graph-off
default and run fixed checks with cuBLAS enabled and disabled.

The Phase 4 study covers cancellation and failed launch injection in its
two-layer Q8 fixture. Reuse those results only for that branch. This
main-based check is confined to correctness and dependencies. Graph
capture/replay and whole-request timing remain in the B1 performance
cohorts. No throughput result can be derived from this test.
