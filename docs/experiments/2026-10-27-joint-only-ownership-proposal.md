# Joint-Only Prepared Gate/Up Ownership

## Decision Requested

Review a storage-only, opt-in construction change. Do not remove split
owners from the [immutable compound reference](2026-10-27-compound-reference.md).
Its original 109 split tensors plus 36 joint pairs remain the control.
No throughput gain, default change or kernel change is assumed.

## Current Dependencies

The qualified compound consumer reads joint gate/up weights. Separate
prepared gate/up allocations remain resident but are redundant for that
consumer. Original uploaded GGUF gate/up tensors are not redundant: bounded
dequant-SGEMM prefill still reads their source bytes.

Three dependencies prevent simply dropping the split buffers:

- [Cache construction](../../src/infer/execution_weights/cache.rs) derives
  pair identities by looking up resident split gate/up entries.
- [Pair ownership](../../src/infer/execution_weights/ffn_pairs.rs) receives
  split allocation bytes and validates the combined budget before upload.
- [Model validation](../../src/infer/q8_ffn.rs) requires split FFN planes
  before checking the compound pair, and diagnostics can borrow split views.

The historical quality and HTTP audit programs explicitly require 109
resident split identities and 2,917,580,800 split bytes. Those audit contracts
cannot qualify a different ownership scheme. Keep them unchanged and retain
their outputs as reference evidence.

## Proposed Construction Contract

Propose `M40LLM_Q8_FFN_OWNERSHIP=retain-split|joint-only`, represented by a
typed model-local ownership choice captured at load time. Default remains
`retain-split`. `joint-only` requires `joint-shared-scales-128`, the existing
split down/vocabulary configuration and all current model/dtype/shape guards.
Reject incompatible consumers instead of silently restoring missing owners,
changing projection backend or falling back to another FFN.

Build a logical source registry from validated GGUF `Q8SourceIdentity`
records before device preparation. Retain model hash, tensor name, byte
offset, dtype, shape and ordered gate/up role for every layer. Original
source pointers remain model-local lookup keys, not evidence of ownership.
Never dereference a pointer merely to validate its identity.

For joint-only, plan 36 joint pairs and 37 split down/vocabulary tensors
directly from that registry. Validate all source bounds, distinct roles,
aliasing, shapes, checked byte sums and the combined budget before uploads.
Do not allocate the 72 redundant split gate/up copies even temporarily.
Keep pair packing/upload one layer at a time, with one temporary host pair
and existing RAII partial-failure cleanup.

Preserve all uploaded original GGUF weights, including gate/up sources and
tied output embeddings. Preserve down/vocabulary packing, launch selection
and consumer instruction words. Prefix SGEMM must still resolve original
GGUF tensors; it must not depend on a missing split gate/up representation.

The compound validator resolves logical gate/up identity plus the joint
owner under joint-only. Retained-split validation keeps its existing split
requirements. A request for an absent separate gate/up diagnostic view must
fail clearly; it must not create a new allocation or return a dangling view.
No eviction, mutation or ownership switch is allowed after model publication.

## Memory And Audit Contract

For this Qwen2.5-3B Q8_0 model, expected requested device payloads are:

| Component | Retained-split compound | Joint-only compound |
| --- | ---: | ---: |
| Original uploaded GGUF | 3,279,519,744 B | 3,279,519,744 B |
| Split gate/up, 72 tensors | 1,724,645,376 B | 0 B |
| Split down/vocabulary, 37 tensors | 1,192,935,424 B | 1,192,935,424 B |
| Joint gate/up, 36 pairs | 1,724,645,376 B | 1,724,645,376 B |
| Combined prepared | 4,642,226,176 B | 2,917,580,800 B |
| Source plus all prepared owners | 7,921,745,920 B | 6,197,100,544 B |

The expected reduction is exactly 1,724,645,376 device bytes. KV, reusable
attention scratch, the 90,177,536-byte prefix temporary and driver/library
storage are separate. This is not a prediction of standalone peak VRAM or
setup RSS. Measure allocation high-water marks and host temporary capacity
as well as final resident bytes. Keep the existing 6144 MiB budget.

Create a distinct candidate report/audit schema. Report logical source
coverage separately from resident split identities, joint pair identities
and bytes. The 109 logical source tensors still participate, but only 37
have split allocations under joint-only. Do not relabel joint pairs as
split tensors to satisfy an old count or silently change the meaning of
an existing resident-byte field.

The new independent audit must verify every expected layer pair and all
down/vocabulary identities exactly once, ordered gate/up source roles,
model/source hashes, byte spans, no aliasing and aggregate allocation sums.
Bind actual dispatch to joint owners and verify no split gate/up allocation
is constructed or borrowed. Explicitly distinguish local logical coverage
from physical residency and from the original source upload.

Reject deliberately corrupted records: wrong ownership choice, missing or
duplicate pair, swapped gate/up, changed tensor offset/model hash, missing
down/head identity, understated bytes and incomplete cleanup. No relaxation
or reuse of frozen 109-resident-owner audits is permitted.

## Validation And Measurement

Keep the no-old-cache experiment disabled. Compare rebuilt retained-split
compound against joint-only compound with explicit dense KV and every
other qualified runtime setting fixed. Use a newly named variant and raw
artifact set; neither side rewrites the immutable reference.

Require bit-identical joint payloads, full-vocabulary logits at the frozen
steps, all actual 512 generated IDs and all twelve frozen task outputs.
Preserve the seven dense-valid passes and five reference-inconclusive rows.
Keep all 36 layers, source tensor routing, prompt IDs and prefix execution.
Require identical compound/down/head/projection instruction words and
ordinary enqueue/copy/synchronization counts. No new CUDA kernel is needed.

Ownership tests must cover two simultaneously live models, model-local
selection, source-view lifetime, invalid/missing/reversed pairs, unsupported
rows, one-byte-short budget and injected upload failures during both split
down/head and joint preparation. Verify allocations return to the initial
baseline after each failed construction and after model/session teardown.
Check allocation-free warm reuse and unchanged stream/context ownership.

Run affected ownership, FFN/projection, prefix, full-forward and attention
parity checks with cuBLAS on/off, plus non-CUDA warning checks, Clippy and
formatting. Request escalation and run hardware validation serially.

After correctness passes, use release builds and the unchanged controlled
HTTP protocol for paired retained-split/joint-only measurements. Report
prefill/decode/HTTP, actual token counts/stop reasons, preparation time,
final allocation, setup peaks and all retained duplicates. Measure cold
setup separately from warm HTTP. Avoid simultaneous inference or profiling.

The memory hypothesis passes only when new ownership/quality evidence
confirms the exact resident reduction without omitted source/scratch costs.
A neutral timing result is a storage result, not a speed improvement.
Any speed claim still needs the existing meaningful-gain/confidence gates;
do not change those gates to qualify a memory-only change. Stop for review
on a meaningful paired regression, changed output/dispatch, resource leak,
identity failure or unexpected peak/budget requirement.

Default promotion requires a later explicit decision. Keep this experiment
independent of compressed-cache routing so its allocation and performance
effects can be attributed to ownership alone.
