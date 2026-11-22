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

## Sources and executable provenance

05da6a0 adapts the October operator tests and imports the synchronized
physical length readback. 1ca369f includes the TinyLlama full-session
test with main's u32 layer identifier. 63ea66a ports the two C launch
entry points onto a common dispatcher. The five kernel preimages match
e2b4739's parent byte-for-byte; the corrected scalar and three specialized
kernel bodies match e2b4739, and the redundant scalar device-length
kernel is removed. Only main's supported launch paths are present.

Compile with `M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo test --release --locked --features cuda --lib --test attention_graph_dispatch --no-run --message-format=json`.
At 05da6a0 save the unrepaired operator test (grid); at 1ca369f save
the unrepaired library test (session). At 63ea66a save both tests, then
rebuild and save again with cuBLAS disabled. The [build targets](build-targets.txt)
and [hash manifest](binaries.sha256) identify the six executables.
Under the common exclusive lock execute [the driver](run.sh) with the
binary directory, new output directory and the pinned TinyLlama F16 file.
The full-session check uses the first eight IDs of the verified A5 prompt.
Each case gets a separate process so an anticipated negative-control
failure cannot poison subsequent tests.

The [first M40 run](first-run/README.md) passed the two short repaired
grids and exposed two setup problems. The follow-up keeps synchronized
cache seeding and validates model mapping after layer slots are allocated
in both control and candidate. Its binaries use a fresh output directory
and a separate hash list passed as the driver's fourth argument. The
original binary manifest remains the reproduction recipe for the
original run.
