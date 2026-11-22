# Maxwell integration decisions

The [candidate matrix](candidates.tsv) records 48 decisions for which the
scope can already be established. Each row has the source branch, exact
commit and report, correctness and runtime tests, operator and complete
request results, memory, quality, dependencies and modules, current
defaults, risk, recommendation, tier and reopening condition. A tier
identifies the kind of integration; the recommendation states whether
it is ready on main. Open hardware measurements will add rows and refine
the dependent recommendations before the final shortlist.

**Ready on the pinned main base.** GGUF tensor bounds, Q8 history-scale
synchronization and the narrow CUDA Graph attention repair have
independent patch and correctness evidence. Their ordered application
passed in a temporary Git index.
[Initial three rebuild commits](../2026-11-21-main-dependencies/README.md#independent-fixes).

**Correctness before performance.** The repaired main builds passed
96 short and 56 boundary attention comparisons each. The cuBLAS F16
session matched 256,000 logits and 405,504 valid FP16 cache words in
eight actual replays. The unfixed control reproduced state differences.
The off build used ordinary fallback after its F16 Q projection could
not be captured. Retain the opt-in flag and test both behaviors.
[Main graph qualification](https://github.com/GuyPaddock/m40-llm/blob/beb1fafd746bab4d8d88f9f9f692b5da6d4e1337/docs/experiments/2026-11-21-main-graph-parity/result/README.md).

TinyLlama returned identical 128 sampled IDs using the asynchronous F16
decode stream in both arms. Five alternating requests measured a 0.261%
throughput gain, with a paired interval of −0.041% to +0.366%. The
original synchronous F16 path failed graph capture before timing.
Qwen's 0.361% gain on native head128 also missed the 5% gate. Both
cohorts support keeping replay opt-in.
[TinyLlama graph results](https://github.com/GuyPaddock/m40-llm/blob/a43a9a5199d81eae82cdd5798c2a049775c52026/docs/experiments/2026-11-21-graph-http/results/tiny.md).

On main, the Q8 eviction repair passed the guarded fixture and all ten
attention cases with cuBLAS enabled and disabled. The unfixed eviction
control passed, leaving direct reproduction of the race on main
inconclusive. The Phase 4 failure witness and the shared-scratch write
order support a narrow Tier 1 port. The bulk Q8 builder is compile-covered.
[Main-based M40 checks](https://github.com/GuyPaddock/m40-llm/blob/a383c40d381b5374211069c9ab5de1e64517d0df/docs/experiments/2026-11-21-main-kv-scale/result/README.md).

The verifier rollback and near-tie cases should
accompany any future verifier port. They do not warrant bringing the
verifier into main by themselves. [Archived CUDA evidence](../2026-11-21-evidence-audit/README.md).

Shared prepared down weights lowered complete four-row target latency
by 5.876% against the fastest exact control across 20 confirmation
rounds. All 511 full-logit row digests and the 10,616,832 valid FP16
KV words matched. The three-arm request used the same device
allocation in each arm and independently exercised 60 rejection
and correction cases. A main port still depends on Q8 model support,
prepared weights and the exact verifier stack.
[Complete target result](https://github.com/GuyPaddock/m40-llm/blob/837e0708edda8ca4d74aabcf6436e18dfffad6bc/docs/experiments/2026-11-21-small-n/target-result/README.md).

The fixed prompt-copy source passed both live copy workloads against
ordinary decoding and against the existing multi-row verifier. The
shared verifier improved resident throughput by 17.416% and 8.934%
over ordinary decoding, and by 2.551% and 1.308% over the existing
copy verifier. All 21 correctness cases retained exact logits and
valid KV, including natural rejections. Keep this opt-in after the
verified target port. [Live copy results](https://github.com/GuyPaddock/m40-llm/blob/e1fcebb8f7bf274465f2d4493e0423ba0268c1ff/docs/experiments/2026-11-22-copy-verifier/result/README.md).

The graph fallback node adds 1.569–3.019 µs synchronized wall time
per head128 attention replay at the two sampled lengths. The implied
36-layer cost is about 0.109 ms at the longer of the two tested lengths.
That misses the frozen threshold for a pruning prototype.
[Measured node counts and paired timings](https://github.com/GuyPaddock/m40-llm/blob/6bb6a1bb70ee5feac13f2d7b98a639825b40943e/docs/experiments/2026-11-21-graph-node-cost/result/README.md).

**Do not broaden aggressive KV selection.** Top-8 lost three of ten
dense-passing long answers. Top-64 retained all ten where every historical
block fit, and both arms retained all 19 short passes. At 3,203 prompt
tokens, the relay task had 69 eligible historical blocks. Top-64 again
matched the dense answer, while top-8 changed the Boolean from true to
false. This repeats one task at a longer context and leaves the original
ten-case denominator intact. [Relay generations](../2026-11-21-kv-audit/history-relay-generations.json).

The completed boundary cohort retained all four dense-passing answers
with top-64 and three with top-8. It covered three distinct tasks, one
repeated at two lengths. The 3,200-token code case passed in every arm.
Across 60 matched comparisons, the maximum KL against full mixed history
was 0.00010917 for top-64 and 0.32391938 for top-8. Dense versus full
mixed history reached 0.009902. Every arm agreed on the greedy ID at
all 12 sampled positions, yet top-8 lost the relay answer between samples.
Retain free generation as a separate gate. The replay from the archive
checks all 48 original FP32 readbacks and 60 comparisons.
[Complete boundary cohort](https://github.com/GuyPaddock/m40-llm/blob/efaf5e8d84b74c84dad9ff0de3693fb8bd031cec/docs/experiments/2026-11-21-kv-history/result/README.md).

The profile already defaults to top-8 on
supported paths. Changing that setting would be a separate compatibility
decision. At capacity 8,448, e100c54 measured a 1.01% peak saving for mixed KV.
The allocator formula puts the requested-byte crossover at capacity
3,403 for the measured 36-layer, one-sequence geometry. The fixed recent
ring and the replicated summary capacity account for the short-context
penalty. [Allocation arithmetic and measured peaks](https://github.com/GuyPaddock/m40-llm/blob/623fd33aa425350124b4c07953d3a65c4c265d08/docs/experiments/2026-11-21-kv-residency/README.md).

In one long-lived process per arm, mixed KV held a flat 1 MiB
post-release offset after each of three full cache lifetimes. Dense
returned to baseline each time. This narrows the memory result to a
first-use cost at capacity 3,072. Sustained serving residency remains
a separate measurement.
[Repeated cache lifetimes](https://github.com/GuyPaddock/m40-llm/blob/27fc182d9e92d2fbe509550e7aab1cf5ee385850/docs/experiments/2026-11-22-kv-cache-reuse.md).
Main already consumes this storage directly. All 144 matched operator
groups reproduced the October benefit. On Phase 4, five matched
TinyLlama F16 head64 HTTP pairs gained 8.775% throughput
(8.575–9.089%) and reduced sampled process peak from 6,224 to
6,208 MiB. This qualifies the existing single-sequence consumer for
that workload. The staged runtime is unavailable on Qwen head128.
The lossy block-selection policy has its own qualification gate.
[Matched historical-KV result](../2026-11-21-serving-baseline/tiny-kv-consumer/result/README.md).

With the short Qwen request inside the recent window, explicit dense
KV finished 7.744% more buffered output tokens per second and used
37 MiB less sampled process memory. The first internal sample came
later with dense. Keep this as a complete-request reference for the
same 65 prompt and 512 generated IDs. Historical contexts have their
own baseline. [Short Qwen controls](../2026-11-21-serving-baseline/qwen-short/README.md).

**When compound FFN is enabled, keep joint-only ownership.** The paired
HTTP control retained identical gate/up weights in two layouts. Eliminating
the redundant split owners saved 1,647 MiB at the fresh loader peak and
matched seven full-logit rows. The HTTP interval lay within −0.16% to
+0.32%. This comparison qualifies a memory choice inside compound FFN.
[Ownership report](../2026-11-22-q8-ownership/README.md).

**Start Q8 work on main with the missing layer and output-head support.**
The strongest measured prepared-weight subset uses only down and
vocabulary weights. It improved the short dense HTTP request by 11.73%
against native Q8 and cost 1,192,935,424 extra device bytes. After the upgrade pause, five fresh pairs reproduced an 11.776% gain
(11.579–11.828%) with all 512 IDs unchanged. A fresh-server diagnostic
measured a 1,139 MiB peak increase and an 11.1 s increase in
launch-to-health. The
earlier all-family setting lost 64.21% on HTTP; preparing QKV and output
alongside down and vocabulary lost 6.71% with native MLP. Both losing
settings matched the control's 512 generated IDs.
[Native and selective prepared comparison](../2026-11-21-native-split/README.md).
[Historical family controls](../2026-10-25-q8-prepared-http.md).
[Isolated process residency](../2026-11-22-q8-ownership/server-peaks/README.md).

Packed GGUF QKV/output, compound FFN, and the packet/argmax portfolio
improved HTTP throughput by 4.02%, 3.20%, and 4.01%, respectively, on
their own historical controls. Packed GGUF loads added no weight or
scratch allocation. The portfolio control already owned the same 37
packets as its candidate. A deployment would still pay for them. Keep
these three comparisons separate when budgeting a port. Each cleared
its frozen 3% gate and fell below the general November 5% threshold.
[Packed-load comparison](../2026-10-25-maxwell-packed-projection-incremental-http.md).
[Compound comparison](../2026-10-27-maxwell-compound-runtime.md).
[Portfolio comparison](../2026-10-29-portfolio-runtime.md).

The November joint-FFN portfolio retest used 120 paired HTTP requests and
exact sampled IDs, logit and KV checks. The combined dispatch gained
3.842% across the twenty fresh pairs (3.750–3.996%). It cleared the
original 3% gate and fell short of the 5% November gate. Peak process
residency was 14,395 MiB with both servers present. All five variants
already paid for the same 1,192,935,424 bytes of packets.
[November paired portfolio result](https://github.com/GuyPaddock/m40-llm/blob/7ff9d33f48c831c4218b5027d6a1cbf01d6c0ad3/docs/experiments/2026-11-21-q8-economics/portfolio/README.md).

On the frozen 65-prompt-token, 512-output-token workload, the portfolio's
52.548523 HTTP tok/s leaves 93.92 ms per request to reach 53.06, or
625.45 ms to reach the 56.153149 target. The latter is 1.30 times the
October official Ollama result. The paired November gate remains 5%.
[Historical request budgets](../2026-11-22-http-targets/README.md).

**Bounded serving experiment.** In all five paired trials, a
2,060-token prefill stalled an already decoding request by 344–351 s
on the opt-in dense buffered scheduler. Both clients matched their
isolated IDs. The five tracing-off pairs matched the instrumented work.
This admits a 16-position chunking test with cache and output parity
and paired latency gates. [Completed serving baseline](https://github.com/GuyPaddock/m40-llm/blob/a37d0bc8b9d9537f626b696868857f88dd80ada2/docs/experiments/2026-11-21-prefill-fairness/result/README.md).

Two simultaneous short requests gained 21.56% aggregate throughput in
the opt-in compressed buffered scheduler. Both clients returned the same
IDs and the batched process peaked 189 MiB higher. The first client's
TTFT and both clients' median token gaps rose, while the second client
started sooner. The prompts fit in the recent ring. The measured gain
qualifies this concurrency cohort and leaves historical-block serving
and general defaults open.
[Two-client result](https://github.com/GuyPaddock/m40-llm/blob/c708a507a3d28a8bdbb1f634fb4ed45d055239f6/docs/experiments/2026-11-21-serving-concurrency/default-two/README.md).

For one medium Qwen request, the original top-8 KV and explicit dense
KV each reproduced their own continuation. Their first sampled IDs
differed with a 2,060-token input. The five trial ranges were
489.120–490.156 s default and 372.221–373.511 s dense. Keep both as
independent baselines for later ports and leave cache defaults outside
the proposed review series. [Medium serving result](../2026-11-21-serving-baseline/qwen-medium/README.md).

The matched dense two-client cohort gained 59.495% aggregate
throughput. Both clients reached their first sample sooner and
returned the same 128 IDs. P95 token gaps rose from about
135–136 ms to 226 ms and sampled process peak rose by 134 MiB.
This scopes the batching win to aggregate throughput for two equal
short prompts. [Dense two-client result](https://github.com/GuyPaddock/m40-llm/blob/dbc20a66eeb5fb3977063a83b65e876f744236e2/docs/experiments/2026-11-21-serving-concurrency/dense-two/README.md).

Four clients with default KV gained 49.435% aggregate throughput.
The batched diagnostic brought the last client's first sample from
42.999 to 14.343 s while P95 token gaps increased from about 150 to
332 ms. The sampled process peak rose by 216 MiB. This configuration
still uses recent-only KV for every request.
[Default four-client result](https://github.com/GuyPaddock/m40-llm/blob/8d2c839d9fc0922d06eaae3ed3bbf8b76508e9fa/docs/experiments/2026-11-21-serving-concurrency/default-four/README.md).

The four-client dense cohort gained 94.733% aggregate throughput
with the same generated IDs. The latest first sample fell from
59.195 to 14.265 s. Per-request P95 gaps increased from about 135
to 327 ms. The sampled process peak rose 161 MiB. Its complete
comparison supports the existing opt-in short-prompt configuration
and supplies a useful clean-base regression workload.
[Dense four-client result](https://github.com/GuyPaddock/m40-llm/blob/3d10ccaa44be00fa7fd3d66ea7b8b836836ad45e/docs/experiments/2026-11-21-serving-concurrency/dense-four/README.md).

**Excluded work.** The original channel selector, ShiftAdd, additional
multi-row gate/up fusion, lossless KV compression and DualPace all missed
their October gates. The later causal-selector and general adaptive
drafting studies also missed their gates. Activation preparation still
requires a compatible rotated model, and the learned draft requires a
qualified checkpoint. Keep the supplied-index Q4
executor as a control for sparse-layout work. Retain GSQ as a checkpoint
preparation question, and preserve the terminal-prefill dependency trace
for workloads with a larger measured opportunity. [Closure register](../2026-11-21-closed-directions/README.md).

The separate real-activation Q4 study passed the charged operator gate
in both layouts. The packed 50% cutoff saved 41.63–42.95% against full
packed. The zigzag 25% cutoff saved 42.89–47.19% against full zigzag.
Both include device selection and compaction, with three measured
layers on the same checkpoint. Each thresholded arm preserved four
dense-passing answers. Zigzag thresholding raised the first held-out
passage's perplexity 4.348%, which stopped that arm at its 3% gate.
Only packed advances to the broader quality grid. Both arms add
38,043,648 requested device bytes for weights and 350,212 for scratch.
[Held-out operator result](https://github.com/GuyPaddock/m40-llm/blob/be7bf4bbfa90eb468b2af53f5c8732e8d1a05c04/docs/experiments/2026-11-22-sparse-q4-live/operator-result/README.md) ·
[Four-arm quality result](https://github.com/GuyPaddock/m40-llm/blob/d64a2edc93ee60c5f148874b51b6e2eac3174ce5/docs/experiments/2026-11-22-sparse-q4-live/quality-result/README.md).

The [dependency-ordered rebuild](../2026-11-21-main-dependencies/README.md)
makes native Q8 checkpoint admission distinct from optional prepared
weights, GPU greedy sampling, compound FFN and speculation. Tier 2
requires qualified model and request performance with acceptable resource
cost. The five historical projection-profile cohorts supply per-layer
timing context for that decision.

Origin has the November research branches. Each row pins a full source
SHA and path. From a checkout with the research objects, the same files
are available with `git show FULL_SHA:docs/experiments/PATH`.
