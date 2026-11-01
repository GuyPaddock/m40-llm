# Fixed Rank-256 Proposal Cost Protocol

Execute only after the ILP4 decision. Use the approved paper architecture,
existing operations and synthetic deterministic FP32 coefficients, not a
trained checkpoint. No datasets, downloads, teacher features or training.

Load the original Qwen Q8_0 with qualified compound, joint-only ownership and
dense context1024 on the M40, retaining model/KV residency. No portfolio or
approximate consumer is installed. The target is not invoked as a teacher.
Synthetic hidden input cannot establish acceptance or context-class quality.

One fixed schedule on the decode stream:

1. Copy an already-resident 2048-float feature into owned scratch.
2. Existing SGEMV computes three independent affine maps, represented by one
   2048-to-768 matrix. Add three width256 biases.
3. Existing SwiGLU with an immutable ones vector computes SiLU.
4. Existing SGEMM computes three rank256-to-full151936 vocabulary rows using
   the shared FP32 vocabulary matrix. Do not assume measured weight reuse.
5. Existing original argmax computes three token IDs, synchronize the decode
   stream and copy 12 bytes to the host.

All parameters/scratch/events are allocated once. Report parameter creation,
upload, target-load time and bytes separately. Charge feature snapshot,
launches, synchronization, readback and host validation in per-decision wall
time. No per-decision allocation or upload. A real target feature is assumed
available on the same stream; live extraction and target cache interference
remain unmeasured, not assigned a universal zero cost.

Validate all 768 hidden values and 455808 full-vocabulary logits against
independent FP64 arithmetic using the existing 1e-3+2e-5 absolute/relative
tolerance. Validate exact greedy IDs, immutable weights/source input and
tracked cleanup. Do not call synthetic token IDs meaningful predictions.

After at least one second of warmup, collect five trials of 200 complete
decisions in release mode. Require isolated M40, P0, SM1113/memory3004 clocks
before and after each trial. Preserve all 1000 samples, medians and p95/p99.
Only afterward collect five event-instrumented stage diagnostics. Their extra
events/synchronizations are not primary timing evidence. No simultaneous GPU
work, builds, MemPalace embeddings or A4000 use during timing.

The fixed admission remains median <=3.884ms. Report tails and variation. Use
the existing measured verifier recurrence for acceptance/cost projections;
do not call these measured HTTP throughput. A failed cost gate ends this
fixed candidate, without alternate dimensions, kernels or argmax rescue.
A passing gate requires a user checkpoint before a frozen dataset manifest,
teacher generation or small training pilot. No automatic training follows.
The probe has a four-GPU-hour cap, included in the 72-hour total ceiling.
