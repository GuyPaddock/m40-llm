# Compound FFN Experiment: Scope Decision Required

## Status

**Approved scoped exception, 2026-10-27; measured gate passes.** See the
[four-arm confirmation](2026-10-27-maxwell-compound-ffn-screen.md). The user
explicitly approved exactly one fixed joint-packing plus shared-scale consumer,
the contemporaneous four-arm comparison and all existing gates. This overrides
only the no-combination restriction for this experiment. It does not authorize
geometry, arithmetic, reduction, threshold or search changes. Report compound
benefit over both joint-only and qualified split control. Runtime integration
remains conditional on the complete-FFN gate; failure closes the experiment.

The existing
[shared-scale contract](2026-10-26-maxwell-shared-scales-ffn-hypothesis.md)
explicitly forbids combining rejected experiments. That restriction remains
in force for all other combinations. Neither individual candidate is reopened
for runtime use, and no acceptance gate is changed.

Preserve the qualified 49.146 HTTP tok/s configuration. The original 53.06
and 56.037 HTTP tok/s targets remain unmet. The original source assessment
changed no kernel or hardware setting; its later implementation remains
benchmark-only without generation/HTTP qualification.

## Existing Evidence And Remaining Work

[Joint packing](2026-10-25-maxwell-joint-ffn-screen.md) improves complete
one-row FFN 4.66-4.74% in fresh twenty-pair confirmation, below the 5% gate.
It pairs gate/up payload words and original FP16 scales. Compiled code
confirms three global loads per four-value task rather than five.

[Shared-scale reuse](2026-10-26-maxwell-shared-scales-ffn-screen.md)
improves complete FFN 1.99-2.19%, also below the gate. It converts each
original scale once into shared FP32 storage instead of once per task.
Both experiments preserve checked original gate/up, SwiGLU and down bits.
Neither has generation or HTTP qualification. **Do not add their gains**
or treat their different historical trials as compound measurements.

The source mechanisms partially overlap in scale loads/address work, but
joint packing still repeats both FP16 conversions for every task. A possible
compound consumer would retain joint payload records and publish paired
scales once into the existing shared-scale arrays before accumulation.

At Qwen K=2048, one CTA handles 512 four-value tasks and 64 scale pairs.
The following are source-level load/conversion uses across threads, not
newly measured instructions, cache transactions or DRAM traffic:

| Per output-column CTA | Split control | Joint only | Shared scales only | Hypothetical compound |
| --- | ---: | ---: | ---: | ---: |
| Global load expressions, including float4 activation | 2560 | 1536 | 1664 | 1088 |
| FP16-to-FP32 scale conversions | 1024 | 1024 | 128 | 128 |
| Shared FP32 scale reads | 0 | 0 | 1024 | 1024 |
| Dynamic shared bytes | 1024 | 1024 | 1536 | 1536 |
| Block publication barriers | 1 | 1 | 2 | 2 |

The compound would therefore target repeated scale loads/conversions still
present after joint packing, while keeping paired payload/address issue.
It adds shared reads/writes and a barrier relative to joint-only. Compiler
register use, spills, scheduling and actual gain are unknown; this is not
proof of independent or additive savings.

Source inspected at `f55cea3766a72e356ff31a2aac802c7af0dbd4a8`:
[joint consumer](../../cuda/q8_ffn.cuh#L634),
[shared-scale consumer](../../cuda/q8_ffn.cuh#L200), and
[typed joint representation](../../src/infer/execution_weights/ffn_pair.rs#L20).

## Memory And Qualification Scope

No new weight representation is needed for the proposed consumer. One
joint pair retains 47,906,816 bytes, exactly the split pair size. A fair
four-arm benchmark would keep both resident for every arm, 95,813,632 bytes,
below the existing 128 MiB per-case preparation bound. Dynamic shared grows
512 bytes versus control/joint-only; there is no proposed new global scale
table, activation quantization, per-token allocation, copy or synchronization.
Original source weights and prepared down/vocabulary remain immutable.

The historical joint medians would need another **0.84-1.08 us per layer**
to cross the 5% complete-FFN gate under their respective controls. This is
only the arithmetic distance to a threshold, not a predicted compound gain
or permission to promote a marginal result. HTTP improvement can still fail
even if a new microbenchmark passes. The measured current FFN scope is
large enough to consider a whole-FFN test, but captured GPU-stage shares
must not be relabeled as exclusive HTTP latency fractions.

If an exception is approved, use **one fixed consumer** with the original
128-thread/output geometry, exact FP32 scaling/FMA association and reduction.
Compare four arms contemporaneously: split control, joint-only, shared-only,
compound. Keep genuine rows 1/4 and layers 0/17/35, frozen M02 inputs,
both weight owners resident and unchanged prepared down/vocabulary.

Require behavioral/bitwise/CPU-reference, bounds, source-identity, producer
ordering, ownership/reuse and cleanup tests with CUDA/cuBLAS on/off;
non-CUDA warnings, Clippy and formatting; and unchanged original compiled
control words. Inspect the new consumer's emitted work/resources without
register, geometry, cache or unroll rescue. Five initial alternating pairs,
fresh prescribed variation/borderline confirmation, no pooling.

The compound must independently clear >=5% complete one-row FFN with
positive paired bounds at all three layers. Only then consider immutable
opt-in runtime integration and actual512/frozen-quality/HTTP qualification.
The HTTP gate remains >=max(3%, twice observed variation) with positive
paired bounds. Defaults and original final targets stay unchanged. If the
compound fails, preserve code/results and close it without another variant.

## Approved Atomic Sequence

The user approved this sequence on 2026-10-27:

1. `docs(research): authorize fixed compound study`: record the user's narrow
   exception and fixed compound contract. No code or hardware changes.
2. Add one benchmark-only consumer, four-arm comparison and behavioral tests;
   commit verified code before timing. No generation selector yet.
3. Record clean release results and the gate decision. Preserve any losing
   implementation before a separate removal.
4. Only after qualification, plan runtime integration, actual-model quality
   and controlled HTTP delivery as separate verified commits.

After this checkpoint, follow the independently requested
[exact multi-token verification study](2026-10-27-maxwell-exact-verification-plan.md).
It does not relax this experiment's gate or replace either original target.
