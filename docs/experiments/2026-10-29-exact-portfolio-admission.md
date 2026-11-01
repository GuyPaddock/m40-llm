# Exact Portfolio Admission

## Inventory

Inspected current dispatch and preserved reports at parent9b02fb3. The following
are historical microbenchmarks, not contemporary compound HTTP improvements.
Historical rejection decisions remain unchanged under the new portfolio-only
authorization. No measurement has run in this checkpoint.

| Candidate / preserved code | Original control and numerical contract | Historical evidence | Current disposition |
| --- | --- | --- | --- |
| Down warp packet, 966ead2 | Prepared four-column down; original FP32 task/FMA/tree and exact output bits | One-row projection +7.520/+7.525/+7.608%; paired95% intervals [7.177,9.514], [6.388,7.888], [7.004,7.783]%; complete FFN only +2.46-2.62% | Eligible positive, not in runtime dispatch |
| Vocabulary warp packet, 966ead2 | Prepared vocabulary, exact output bits; normalized FFN surrogates rather than real final hidden | +9.280%, interval[9.238,9.349]% | Eligible, requires real vocabulary inputs and whole-model checks |
| Partitioned argmax, 125d3a3; runner61a48d9 | One-block argmax; exact indices including lowest-index ties | 15.43-16.28x; speedup intervals14.69-16.54/15.89-16.62/14.56-16.35x on three finite synthetic distributions | Eligible, not model-logit or HTTP proof |
| Fixed-K FFN, deef80b | Older split-f32x4-128; same arithmetic and bits | Complete FFN+1.24-1.32%, intervals[1.05,1.40]/[1.19,1.41]/[.84,1.25]% | Exclude from this portfolio: different consumer; transfer needs compound-specific compiled-work audit |
| Exact RMSNorm, f8c00bf | Original sum tree | Four of six one-row intervals cross zero | Exclude inconclusive result |
| Native-warp FFN, dd78a1a | Split control; changed32-partial association | Complete FFN regresses4.72-4.77% | Exclude regression, also not exact-bit contract |

Warp packets keep original bytes in a different lossless layout. The current
joint consumer changes gate/up only; down, vocabulary and argmax are not
subsumed by it. Current source still exposes packet owners and primitive APIs
without wiring these consumers into ordinary generation. Fixed-K source still
differs from compound, but source difference is NOT proof of incremental work.
Do not credit its historical speedup or restore the older split backend.

## Compatibility And Resource Matrix

| Pair | Independent cost | Interaction requiring qualification |
| --- | --- | --- |
| Down + vocabulary | Different tensors and stages | Same packet implementation and additional resident/cache pressure |
| Down + argmax | Projection versus final sampling | Propagated logits must stay identical; no timing percentage addition |
| Vocabulary + argmax | Full logits production versus reduction | Preserve vocabulary ordering, ties, dependency and full-logit diagnostics |
| Any + fixed-K | Would touch a different FFN consumer | Not admitted; avoid a fourth candidate or compound rewrite |

Fixed packet geometry remains128/256 as historically assigned per family,
four outputs/CTA, original eight-value thread tasks and reduction. Registers40,
no reported local spills, dynamic shared4096/2048 bytes. All36 down packets
add862,322,688 bytes; vocabulary adds330,612,736. Total candidate weight storage
is1,192,935,424 bytes. Retain original37 prepared owners and36 joint pairs in
all arms initially; prepared total becomes4,110,516,224 bytes, below6144MiB.
Original GGUF sources remain for prefill. Budget exact host/setup high-water
separately; historical inverse/upload timing is not a fresh setup measurement.
Argmax adds384 bytes of partial payload per exclusive session, allocated once.
Never share mutable partials between live sessions or allocate in enqueue.

## Non-Double-Counted Scope

Use the latest dataflow middle sample: down4.773ms, vocabulary1.564ms and
argmax0.260ms per token. For historical speed ratio s, the conditional saving
is512*stage_ms*(1-1/s), not stage_ms times a throughput percentage.
Reference latency is10056.879ms; a3% gain needs292.919ms removed.

| Component/set | Conditional saved ms/request |
| --- | ---: |
| Down, minimum historical point gain | 170.919 |
| Argmax, minimum historical point speedup | 124.492 |
| Vocabulary | 67.999 |
| Down + argmax | 295.411 |
| Down + vocabulary + argmax | 363.409 |

The two-member set has only2.492ms overhead margin, insufficient evidence of
a credible integration margin. Select the three-member set, conditionally
3.749% at zero added cost, with70.490ms margin before dropping below3%.
Using the least favorable historical lower bound per stage gives roughly
338ms, still only a sensitivity calculation, NOT a combined confidence bound.
No DRAM/occupancy claim or linear transfer of historical gains is justified.
If a contemporary screen loses that scope, stop rather than adding fixed-K.

## Fixed Implementation And Measurement Arms

Admit a research-only load-time configuration with five arms: qualified
compound control, down-only, vocabulary-only, argmax-only, all-three. Pin
joint-only, dense attention, model, prefill and all other settings. Keep original
consumer code and compiled controls unchanged. Original and candidate buffers
must coexist equally across timing arms; report final and setup residency.
No default change, additional geometry or arithmetic variation.

First check packing inverse/source keys, input/output guards, alignment,
alias rejection, streams, reuse, failure cleanup and primitive bits. Include
real normalized vocabulary vectors and real logits, ties, signed zeros and
documented nonfinite behavior, not just old surrogates. Check no timed
allocation/copy/sync and unchanged prefill. Original full-logit diagnostics
must use the selected vocabulary consumer rather than bypassing it.

Rebase complete FFN on compound at layers0/17/35; measure all operations
before generation. Only continue with positive applicable paired intervals
and supported combined scope. Then require actual512 IDs/counts/stops, all
frozen outputs, sampled full-vocabulary logits, valid KV, compiled controls,
ownership/teardown and affected regression tests. Whole-model failure stops.

Run contemporary control-versus-each-standalone and control-versus-combined
buffered HTTP pairs using existing variation/confirmation rules. Report all
four comparisons; only combined must exceed max(3%,2*variation) with positive
paired bounds. Do not pool trials or treat component gains as additive.
No fresh Ollama run: use43.194730 with clear historical labeling.

## Sources And Status

- [Packet report](2026-10-26-maxwell-warp-packet-screen.md) and [raw-derived intervals](2026-10-26-maxwell-warp-packet-audit.json).
- [Argmax report](2026-10-26-maxwell-decode-reductions-screen.md) and [confirmation audit](2026-10-26-maxwell-decode-reductions-audit.json).
- [Fixed-K report](2026-10-25-maxwell-fixed-k-ffn-screen.md).
- [Current profile](2026-10-28-dataflow-findings.md).

Paper admission complete. Runtime adapters, correctness qualification and
paired measurements are pending. No portfolio HTTP gain is claimed.
