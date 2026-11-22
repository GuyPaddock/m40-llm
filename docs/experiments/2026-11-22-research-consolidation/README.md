# Maxwell research consolidation

Research status as of November 22, 2026. The baseline and conditional
studies listed below still have outstanding measurements. This summary
records completed evidence and the current rebuild recommendation.

## Integration shortlist

| Status | Recommendation | Evidence and boundary |
|---|---|---|
| Integrate first | Checked GGUF tensor products and the narrow Q8 history-scale synchronization repair, as two independent changes on current main | The checked tensor repair catches an actual malformed shape. The Phase 4 Q8 control exhibited the synchronization defect. Both fixed main CUDA modes passed the guarded incremental cache case and ten attention cases. The unfixed main guard also passed, so the race itself remains unreproduced on main. |
| Carry when the main fixture qualifies | The narrow CUDA Graph attention-dispatch repair | The Phase 4 fix restores exact ordinary/replay logits and FP16 KV. The repaired main short grids pass in both CUDA builds. The corrected main boundary and real-model fixtures are still queued. Repaired Qwen and TinyLlama Graph serving missed the complete-request gain gate. |
| Retain existing functionality | Direct single-sequence historical FP16-K/Q4-V attention | The matched TinyLlama head64 workload gained 8.775% HTTP throughput and cut sampled process peak by 16 MiB with the same output IDs. Pinned main already has the consumer. Block selection and staged Qwen head128 have separate limitations. |
| Study after the foundation | Native Q8 model support, prepared down and vocabulary dispatch, shared four-row exact verification | Selective preparation gained 11.776% on the short Qwen HTTP workload with 1,192,935,424 additional requested bytes. Shared down weights reduced complete four-row verifier latency 5.876% with the same scratch allocation and prepared weight owners. Pinned main needs layer, output-head and verifier foundations. |
| Keep as research controls | Host selected-Q4 packing and device execution, staging and residency harnesses, graph diagnostics, both exact verification controls | These isolate cost or correctness. Their presence alone gives no reason to change inference dispatch. |
| Exclude or defer | Original and follow-up FFN predictors, ShiftAdd, further generic gate/up fusion, tested lossless KV compression, DualPace, general adaptive drafting, rotation and Soft-MTP pending their documented prerequisites | The closure register retains each original comparison and a specific reopening condition. |

The [candidate matrix](../2026-11-21-integration/candidates.tsv) carries
full hashes, source modules, flags, evidence and allocation scope for
each row. The [dependency-ordered rebuild](../2026-11-21-main-dependencies/README.md)
lists the independent first two commits and the separate conditions for
graph, Q8 model support, experimental prepared weights and verification.
The [closure register](../2026-11-21-closed-directions/README.md) is the
record for negative and prerequisite-limited studies.

## Complete-request and quality findings

- **Graphs.** Qwen native-head128 Graph HTTP throughput gained 0.361%
  across five pairs. TinyLlama asynchronous F16 gained 0.261%, with a
  paired interval of −0.041% to +0.366%. The equivalent F16 graph and
  ordinary arms returned all 128 IDs exactly. The original synchronous
  F16 path stopped at capture. Keep replay opt-in and track the dispatch
  correctness fix on its tested paths.
  [Qwen graph](https://github.com/GuyPaddock/m40-llm/blob/379002a62a19e12ed7b8e0da2777b3025e9393b5/docs/experiments/2026-11-21-graph-http/results/qwen.md) ·
  [TinyLlama graph](https://github.com/GuyPaddock/m40-llm/blob/a43a9a5199d81eae82cdd5798c2a049775c52026/docs/experiments/2026-11-21-graph-http/results/tiny.md).
- **Mixed KV.** The independent archive replay recovered 348 aligned
  comparisons from 522 complete FP32 vectors. All-history mixed KV
  preserved 19 of 19 dense-passing short tasks and 10 of 10 long tasks.
  Top-8 lost three long answers. Nineteen dense failures are counted
  separately. The mixed representation used 47 MiB less sampled peak
  process memory at capacity 8,448 and 3 MiB more at 3,072. Mixed
  post-release residency plateaued at a 1 MiB offset in three
  same-process lifetimes. Scope the direct consumer win to TinyLlama
  head64. Four further dense-passing answers held at top-64 beyond
  the original grid, including prompts with more than 64 eligible
  blocks. Top-8 lost one of the four.
  [Quality replay](https://github.com/GuyPaddock/m40-llm/blob/d6cabf1cd0e0bd97df030eb47fbf96f613bb0722/docs/experiments/2026-11-21-kv-quality-result/full-replay/README.md) ·
  [Boundary vectors](https://github.com/GuyPaddock/m40-llm/blob/efaf5e8d84b74c84dad9ff0de3693fb8bd031cec/docs/experiments/2026-11-21-kv-history/result/README.md) ·
  [Direct HTTP comparison](../2026-11-21-serving-baseline/tiny-kv-consumer/result/README.md) ·
  [Reuse](https://github.com/GuyPaddock/m40-llm/blob/27fc182d9e92d2fbe509550e7aab1cf5ee385850/docs/experiments/2026-11-22-kv-cache-reuse.md).
- **Short Qwen baseline.** Explicit dense KV gained 7.744% against
  default KV for 65 prompt tokens and 512 outputs, with identical
  IDs and 37 MiB lower sampled peak. Its median internal first sample
  took 8.521 s versus 3.830 s default. Both requests fit inside the
  compressed cache's recent window. [Matched default and dense baseline](../2026-11-21-serving-baseline/qwen-short/README.md).
- **Prepared Q8.** Preparing only down and vocabulary layouts had the
  strongest measured short HTTP result against native Q8. Preparing all
  families lost 64.21% on that historical request. When compound FFN is
  already enabled, releasing duplicate split owners saved 1,647 MiB
  in fresh isolated processes and matched measured logits. The paired
  request interval was −0.16% to +0.32%. Combined packet and argmax
  dispatch improved 3.842% against an already prepared joint-FFN
  control, below the 5% November gate. It reached 52.469 HTTP tok/s
  against the historical 56.153149 target.
  [Selective preparation](../2026-11-21-native-split/README.md) ·
  [Compound ownership](../2026-11-22-q8-ownership/README.md) ·
  [Paired portfolio retest](https://github.com/GuyPaddock/m40-llm/blob/7ff9d33f48c831c4218b5027d6a1cbf01d6c0ad3/docs/experiments/2026-11-21-q8-economics/portfolio/README.md).
- **Four-row verification.** Across 20 fresh rounds, median transaction
  latency fell from 56.417 to 53.102 ms against the fastest exact control.
  All 511 full-logit row digests and 10,616,832 valid FP16 KV words
  matched. The corresponding resident Qwen request fell 5.52%.
  Sixty forced rejections preserved accepted state and matched the
  ordinary correction. Live copy workload qualification follows
  separately. [Complete target and rejection results](https://github.com/GuyPaddock/m40-llm/blob/837e0708edda8ca4d74aabcf6436e18dfffad6bc/docs/experiments/2026-11-21-small-n/target-result/README.md).

## November follow-ups

| Direction | Disposition and evidence |
|---|---|
| Active-row compressed KV | Close the idle-slot hypothesis. Up to 4,096 tokens in an idle slot had no measurable effect on another head128 request in the controlled test. [Slot-isolation result](https://github.com/GuyPaddock/m40-llm/blob/377bec468d51844a917648ab3700baa0ad8b4258/docs/experiments/2026-11-21-active-row-result/README.md). |
| SpQt-style sparse Q4 | The fixed-mask hardware screen warranted one real-activation study. Calibration and held-out captures preserved dense output IDs. The operator test charges device thresholding and compaction. Keep the original causal channel predictors closed. [Captured Q4 inputs](https://github.com/GuyPaddock/m40-llm/blob/159edb4218f5a4a2090ca0e082f60bf61b603179/docs/experiments/2026-11-22-sparse-q4-live.md). |
| GSQ checkpoints | Defer to model preparation until supported layer mappings and kernels can execute the Q2_K and Q3_K_M tensor mixes. The GGUF type enums alone establish format recognition. [GSQ feasibility](https://github.com/GuyPaddock/m40-llm/blob/59aaf765e2d21078f41ff0e2f873df7ce029cc96/docs/experiments/2026-11-21-gsq-feasibility.md). |
| Terminal prefill pruning | Keep the dependency trace. The valid KV-only saving estimated from the measured terminal operations was at most 0.092% of a complete request, below the admission gate. [Dependency and timing screen](https://github.com/GuyPaddock/m40-llm/blob/8965fd3f5f5ee2c021e928917fbfab67d7bdf3a3/docs/experiments/2026-11-21-output-pruning/README.md). |
| Prefill fairness | A 2,060-token prompt stalled ongoing decode by 344–351 seconds in the matched dense HTTP cohort. This qualified one 16-position scheduling trial with exact state, cancellation and fairness checks. [Measured stall](https://github.com/GuyPaddock/m40-llm/blob/a37d0bc8b9d9537f626b696868857f88dd80ada2/docs/experiments/2026-11-21-prefill-fairness/result/README.md). |

## Evidence index

- [October branch and artifact inventory](../2026-11-21-inventory/README.md).
- [Frozen hardware, workloads and November source provenance](../2026-11-21-protocol/README.md).
- [Serving methodology, retained traces and reproducibility](../2026-11-21-serving-baseline/README.md).
- [Independent correctness and source audit](../2026-11-21-evidence-audit/README.md).
- [Build compatibility matrix](../2026-11-21-build-compatibility/README.md).
- [Complete candidate decisions and current dispositions](../2026-11-21-integration/README.md).

The locally committed topic SHAs and paths are usable through
`git show SHA:PATH`. GPU timing used the actual Tesla M40 and a
shared host lock. The Windows VM was running during the current
suite. The raw captures include the host process observations available
for each run. A clean-host repeat would be a separate cohort.
