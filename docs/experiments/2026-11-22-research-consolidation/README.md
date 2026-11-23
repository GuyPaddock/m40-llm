# Maxwell research consolidation

Research status as of November 22, 2026. The baseline and conditional
studies listed below still have outstanding measurements. This summary
records completed evidence and the current rebuild recommendation.

## Integration shortlist

| Status | Recommendation | Evidence and boundary |
|---|---|---|
| Integrate first | Checked GGUF tensor products and the narrow Q8 history-scale synchronization repair, as two independent changes on current main | The checked tensor repair catches an actual malformed shape. The Phase 4 Q8 control exhibited the synchronization defect. Both fixed main CUDA modes passed the guarded incremental cache case and ten attention cases. The unfixed main guard also passed, so the race itself remains unreproduced on main. |
| Integrate first, within tested scope | The narrow CUDA Graph attention-dispatch repair | Main passed both attention grids with cuBLAS on and off. The cuBLAS F16 session used eight actual replays and matched all logits and valid KV. The off build used ordinary fallback after F16 capture failed. Keep replay opt-in. [Main correctness result](https://github.com/GuyPaddock/m40-llm/blob/beb1fafd746bab4d8d88f9f9f692b5da6d4e1337/docs/experiments/2026-11-21-main-graph-parity/result/README.md). |
| Retain existing functionality | Direct single-sequence historical FP16-K/Q4-V attention | The matched TinyLlama head64 workload gained 8.775% HTTP throughput and cut sampled process peak by 16 MiB with the same output IDs. Pinned main already has the consumer. Block selection and staged Qwen head128 have separate limitations. |
| Study after the foundation | Native Q8 model support, prepared down and vocabulary dispatch, shared four-row exact verification | Selective preparation gained 11.776% on the short Qwen HTTP workload with 1,192,935,424 additional requested bytes. Shared down weights reduced complete four-row verifier latency 5.876% with the same scratch allocation and prepared weight owners. Pinned main needs layer, output-head and verifier foundations. |
| Keep as research controls | Host selected-Q4 packing and device execution, staging and residency harnesses, graph diagnostics, both exact verification controls | These isolate cost or correctness. Their presence alone gives no reason to change inference dispatch. |
| Exclude or defer | Original and follow-up FFN predictors, ShiftAdd, further generic gate/up fusion, tested lossless KV compression, DualPace, general adaptive drafting, rotation and Soft-MTP pending their documented prerequisites | The closure register retains each original comparison and a specific reopening condition. |

The [candidate matrix](../2026-11-21-integration/candidates.tsv) carries
full hashes, source modules, flags, evidence and allocation scope for
each row. The [dependency-ordered rebuild](../2026-11-21-main-dependencies/README.md)
lists the independent first three commits and the separate conditions for
Q8 model support, experimental prepared weights and verification.
The [closure register](../2026-11-21-closed-directions/README.md) is the
record for negative and prerequisite-limited studies.

## Integration tiers

| Tier | Scope on the pinned `main` |
|---|---|
| 1. Correctness and foundation | Review the checked GGUF tensor products, the Q8-history scale barriers with their guard, and the CUDA Graph dispatch repair with its parity fixtures as three atomic commits. The graph change keeps the existing opt-in dispatch and fallback. |
| 2. Production-qualified performance | Keep the existing direct FP16-K/Q4-V history consumer. Its measured head64 win qualifies that workload and requires no new port. The current evidence supports zero new changes to automatic performance dispatch. |
| 3. Opt-in experiments | Selective down/vocabulary preparation, shared four-row verification and fixed copy drafting need the native Q8 and exact-transaction foundations. Dense prefill chunking warrants a separate opt-in evaluation after a narrow server port. Keep top-64 history selection and graphs within their measured scope. |
| 4. Preserve or exclude | Preserve the selected-Q4 execution controls, rejected Q8 layouts and negative study evidence. The closure register defines when to revisit failed predictors, lossless KV, ShiftAdd, DualPace, generic adaptive drafting and the other closed directions. |

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
- **Medium Qwen baseline.** At 2,060 prompt IDs and 128 outputs, the
  first sampled ID differed between default top-8 and explicit dense
  KV. Each mode reproduced its own output in all five timing trials.
  Buffered medians were 489.292 and 373.309 s, with peaks of 3,332 and
  3,320 MiB respectively. Both remain separate workload baselines.
  [Medium CLI, streaming and buffered metrics](../2026-11-21-serving-baseline/qwen-medium/README.md).
- **Long Qwen feasibility.** Default mixed KV returned 128 matching CLI
  and buffered IDs after an 8,197-token prompt. Buffered and streaming
  requests finished in about 43 minutes. The dense CLI produced zero
  sampled IDs at the one-hour request limit. This leaves a default
  diagnostic and a dense feasibility failure, with no paired speed
  verdict. In the separate M40 operator diagnostic, the ordinary dense
  head128 launcher switched kernels at 8,193 and the median event span
  rose from 7.921 ms to 9,045.630 ms with FP64 tolerance preserved. The
  measured cliff identifies the long-context dispatch to investigate.
  [Long scope and retained stop](../2026-11-21-serving-baseline/qwen-long/README.md) ·
  [Head128 operator boundary](https://github.com/GuyPaddock/m40-llm/blob/141ad984b3bcac6f3da738f9850c3f0d3bedc3ff/docs/experiments/2026-11-22-long-context-boundary/result/README.md).
- **Prepared Q8.** Preparing only down and vocabulary layouts had the
  strongest measured short HTTP result against native Q8. Preparing all
  families lost 64.21% on that historical request. When compound FFN is
  already enabled, releasing duplicate split owners saved 1,647 MiB
  in fresh isolated loader and normal-server processes and matched
  measured logits. The joint-only ownership timing interval was −0.16%
  to +0.32%. The separate selective preparation study added 1,139 MiB
  of sampled process peak and 11.1 seconds to launch-to-health.
  Combined packet and argmax
  dispatch improved 3.842% against an already prepared joint-FFN
  control, below the 5% November gate. It reached 52.469 HTTP tok/s
  against the historical 56.153149 target.
  [Selective preparation](../2026-11-21-native-split/README.md) ·
  [Compound ownership](../2026-11-22-q8-ownership/README.md) ·
  [Isolated server residency](../2026-11-22-q8-ownership/server-peaks/README.md) ·
  [Paired portfolio retest](https://github.com/GuyPaddock/m40-llm/blob/7ff9d33f48c831c4218b5027d6a1cbf01d6c0ad3/docs/experiments/2026-11-21-q8-economics/portfolio/README.md).
- **Four-row verification.** Across 20 fresh rounds, median transaction
  latency fell from 56.417 to 53.102 ms against the fastest exact control.
  All 511 full-logit row digests and 10,616,832 valid FP16 KV words
  matched. The corresponding resident Qwen request fell 5.52%.
  Sixty forced rejections preserved accepted state and matched the
  ordinary correction. [Complete target and rejection results](https://github.com/GuyPaddock/m40-llm/blob/837e0708edda8ca4d74aabcf6436e18dfffad6bc/docs/experiments/2026-11-21-small-n/target-result/README.md).
- **Fixed live copy drafting.** Shared Q8 down weights gained 17.416%
  and 8.934% resident CLI throughput against ordinary decode in the
  two copy cases. The gains against the existing multi-row copy arm
  were 2.551% and 1.308%. All 21 correctness cases matched, including
  13 natural rejections per candidate. The specific copy path warrants
  continued opt-in work after the clean-base verifier qualifies.
  [Live copy evidence](https://github.com/GuyPaddock/m40-llm/blob/e1fcebb8f7bf274465f2d4493e0423ba0268c1ff/docs/experiments/2026-11-22-copy-verifier/result/README.md).

## November follow-ups

The one-client serving controls preserved all 64 IDs in each cache
mode and recorded zero packed decode forwards. Enabling batching
changed throughput +0.349% with default KV and −0.295% with dense KV.
Both intervals span zero. Sampled peak rose 176 MiB and 108 MiB,
respectively. The measured aggregate batching gains apply to multiple
simultaneous short requests.
[Default one-client control](https://github.com/GuyPaddock/m40-llm/blob/5891b68cb9f9c6662ad1b3c575ccb10e306a7cc1/docs/experiments/2026-11-21-serving-concurrency/default-one/README.md) ·
[Dense one-client control](https://github.com/GuyPaddock/m40-llm/blob/e047b51036ad3c9b3222571842ed93498ca5050b/docs/experiments/2026-11-21-serving-concurrency/dense-one/README.md).

| Direction | Disposition and evidence |
|---|---|
| Active-row compressed KV | Close the idle-slot hypothesis. Up to 4,096 tokens in an idle slot had no measurable effect on another head128 request in the controlled test. [Slot-isolation result](https://github.com/GuyPaddock/m40-llm/blob/377bec468d51844a917648ab3700baa0ad8b4258/docs/experiments/2026-11-21-active-row-result/README.md). |
| SpQt-style sparse Q4 | Packed saved 41.63–42.95% against full packed at the operator level with thresholding and compaction. Its expanded gate preserved 26 of 26 dense-passing answers in 48 paired cases. Long code and arithmetic had no dense passes. The isolated resident request comparison is separate. Zigzag saved 42.89–47.19% against full zigzag, then failed its perplexity gate with a 4.348% increase. The original causal channel predictors remain closed. [Operator result](https://github.com/GuyPaddock/m40-llm/blob/be7bf4bbfa90eb468b2af53f5c8732e8d1a05c04/docs/experiments/2026-11-22-sparse-q4-live/operator-result/README.md) · [Four-arm quality result](https://github.com/GuyPaddock/m40-llm/blob/d64a2edc93ee60c5f148874b51b6e2eac3174ce5/docs/experiments/2026-11-22-sparse-q4-live/quality-result/README.md) · [Expanded packed quality](https://github.com/GuyPaddock/m40-llm/blob/2269a0aa1f6fb3f61e4834a3d55cc2ddd82eeb25/docs/experiments/2026-11-22-sparse-q4-live/extended-result/README.md). |
| GSQ checkpoints | Defer to model preparation until supported layer mappings and kernels can execute the Q2_K and Q3_K_M tensor mixes. The GGUF type enums alone establish format recognition. [GSQ feasibility](https://github.com/GuyPaddock/m40-llm/blob/59aaf765e2d21078f41ff0e2f873df7ce029cc96/docs/experiments/2026-11-21-gsq-feasibility.md). |
| Terminal prefill pruning | Keep the dependency trace. The valid KV-only saving estimated from the measured terminal operations was at most 0.092% of a complete request, below the admission gate. [Dependency and timing screen](https://github.com/GuyPaddock/m40-llm/blob/8965fd3f5f5ee2c021e928917fbfab67d7bdf3a3/docs/experiments/2026-11-21-output-pruning/README.md). |
| Prefill fairness | The 16-position dense policy passed its frozen five-pair gate with 800 exact IDs. The median worst gap fell from 347.470 to 3.377 s. The short client's P95 gap rose from 0.379 to 3.318 s and the medium client's first sample arrived 4.632% later. Tracing-off makespan changed −0.287%, with a 3,564 MiB sampled peak in both arms. Keep it opt-in for the tested workload. [Completed fairness comparison](https://github.com/GuyPaddock/m40-llm/blob/3e362c7a7c14883cdbe7f45cff5e321b279d7afa/docs/experiments/2026-11-21-prefill-fairness/chunking-result/README.md) · [Chunking guard evidence](https://github.com/GuyPaddock/m40-llm/blob/c0cdbcb373b80248a551d87e6a942330cd5c7161/docs/experiments/2026-11-21-prefill-fairness/chunking-guards/README.md) · [Stopped first diagnostic](https://github.com/GuyPaddock/m40-llm/blob/f2333456ec479e4bd5fb68ed8a4bf7090bf341d7/docs/experiments/2026-11-21-prefill-fairness/chunking-attempt-01/README.md). |
| Two-client batching | Packed scheduling increased aggregate throughput by 59.495% with dense KV and 21.557% with default KV in separate short-prompt cohorts. Each retained matchable output IDs and increased P95 token gaps. Peak costs were 134 MiB and 189 MiB, respectively. [Dense](https://github.com/GuyPaddock/m40-llm/blob/dbc20a66eeb5fb3977063a83b65e876f744236e2/docs/experiments/2026-11-21-serving-concurrency/dense-two/README.md) · [Default KV](https://github.com/GuyPaddock/m40-llm/blob/c708a507a3d28a8bdbb1f634fb4ed45d055239f6/docs/experiments/2026-11-21-serving-concurrency/default-two/README.md). |
| Four-client batching | Aggregate throughput rose 49.435% with default KV and 94.733% with dense KV in separate short-prompt cohorts. Each matched 256 IDs and recorded 64 real width-four forwards. Dense reduced the latest first sample from 59.195 to 14.265 s. Dense P95 token gaps reached 327 ms and peak cost rose 161 MiB. Default P95 reached 332 ms and cost rose 216 MiB. [Default four-client result](https://github.com/GuyPaddock/m40-llm/blob/8d2c839d9fc0922d06eaae3ed3bbf8b76508e9fa/docs/experiments/2026-11-21-serving-concurrency/default-four/README.md) · [Dense four-client result](https://github.com/GuyPaddock/m40-llm/blob/3d10ccaa44be00fa7fd3d66ea7b8b836836ad45e/docs/experiments/2026-11-21-serving-concurrency/dense-four/README.md). |

## Qualification gates

Each verdict applies to the comparison in its row. See the individual
reports for the raw observations, trial variability and the frozen stop.

| Phase | Verdict and qualification scope |
|---|---|
| A1. October inventory | **PASS.** Replay rebuilt 13 topics, 793 changed paths and 48 implementation commits. The inventory assigns an updated disposition to each topic. |
| A2. Baseline | **PASS** for matched short Qwen requests, separately reproducible medium arms, common-model TinyLlama controls and complete one-, two- and four-client cohorts. **FAIL** for 8,197-token dense Qwen feasibility at the existing one-hour limit. **INCONCLUSIVE** for equivalent-work default-versus-dense comparison at medium and long lengths. |
| B1. Graphs | **PASS** for exact attention arithmetic on both builds and actual cuBLAS F16 replay. **PASS** for expected ordinary fallback and cleanup without cuBLAS, where real F16 graph replay remains unsupported. **FAIL** for full-model speed promotion and fallback-node pruning admission. |
| B2. Quantized KV | **PASS** for the existing matched head64 direct consumer and scope-limited dense-passing retention under all-history or top-64 settings. **FAIL** for top-8 quality promotion. **FAIL** for a substantial memory-saving claim in the tested allocations. The head128 staged-versus-direct server comparison remains **INCONCLUSIVE** because the staging configuration is unavailable. |
| B3. Q8 projection economics | **PASS** for the 5% speed gate on selective down/vocabulary preparation and for removing duplicate split owners where compound FFN is already enabled. **FAIL** for the additional-family configurations and for the confirmed portfolio at the November 5% gate. Budget selective preparation against actual peak and cold start. |
| C1. Small-N target verification | **PASS** for exact complete target and resident request costs at four rows on the frozen Qwen control. Retain the width sweep as evidence for this selection. |
| C2. Batch and rollback correctness | **PASS** for the measured exact target paths, rejection/repair, zero-scale and near-tie cases. Keep these tests at the transaction boundary of any future target port. |
| C3. Active-row compressed KV | **FAIL** at admission. The idle slot added no detectable work to the active sequence up to the measured 4,096 idle tokens. |
| C4. Sparse Q4 | **PASS** for bounded packed and zigzag operator costs and the expanded packed quality gate. **FAIL** for thresholded zigzag's first perplexity gate. Packed entered a separate isolated request-cost comparison. |
| C5. GSQ | **FAIL** at runtime admission for the screened GGUF tensor mixes. Model preparation remains conditional on format-specific runtime support. |
| D1. Copy speculation | **PASS** for both frozen live copy workloads versus ordinary decode and the correctness requirements. The incremental gains over the existing copy verifier missed the general 5% threshold. |
| D2. Output-dependent pruning | **FAIL** at the bounded complete-request opportunity gate. The valid terminal dependency case accounted for at most 0.092%. |
| D3. Prefill fairness | **PASS** for the frozen worst-gap gate in the two-request dense workload, including the medium TTFT allowance, exact IDs, and sampled peak. Higher normal inter-token latency limits the opt-in workload. |
| D4. GGUF compatibility | **PASS** for the independently tested checked tensor and stride products. Native Q8 model mapping is separate foundation work on `main`. |
| E. Closed directions | **PASS** for preservation and review. The closure register records each experiment's performance, quality or prerequisite failure and what new evidence could reopen it. |
| F. Regression compatibility | **PASS** for the source-isolated build gates and the actual, directory-specific M40 tests linked in the reports. **INCONCLUSIVE** for the initial shared-target compatibility logs, retained separately. |

## Evidence index

| Package deliverable | Record |
|---|---|
| Research inventory and exact branch provenance | [October inventory](../2026-11-21-inventory/README.md) and the [source audit](../2026-11-21-evidence-audit/README.md). |
| Frozen workload baseline and benchmark method | [Hardware and November sources](../2026-11-21-protocol/README.md) and [serving baselines](../2026-11-21-serving-baseline/README.md). |
| Per-experiment reports | Report links beside each finding above and the [candidate matrix](../2026-11-21-integration/candidates.tsv). |
| Raw records and reproduction commands | Archive manifests, executable hashes and run commands linked from the individual reports and [replay index](../2026-11-21-evidence-audit/README.md). |
| Independent correctness and evidence audits | [Evidence audit](../2026-11-21-evidence-audit/README.md) and [source-isolated compatibility checks](../2026-11-21-build-compatibility/README.md). |
| Consolidated candidate comparison | [Disposition matrix](../2026-11-21-integration/candidates.tsv), with source, gates, memory, quality, flags and risk. |
| Closed experiments and reopening conditions | [Closure register](../2026-11-21-closed-directions/README.md). |
| Enhancement shortlist | The recommendations and tiers above. |
| Dependency-ordered rebuild from `main` | [Atomic commit sequence, dependency graph and qualification commands](../2026-11-21-main-dependencies/README.md). |
| Executive summary | Complete-request findings and November dispositions above. |

The locally committed topic SHAs and paths are usable through
`git show SHA:PATH`. GPU timing used the actual Tesla M40 and a
shared host lock. The Windows VM was running during the current
suite. The raw captures include the host process observations available
for each run. A clean-host repeat would be a separate cohort.
