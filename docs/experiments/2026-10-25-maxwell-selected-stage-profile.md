# Current Selected-Runtime Stage Profile

The retained packed FP32/128 plus SGEMM runtime remains FFN dominated.
This separately measured profile supports the plan's bounded native-FFN
follow-up; it is not a new throughput improvement or target claim.

Release source `b7c23aa`, comparison executable SHA-256
`730c7057455bbeb3ae70c90cb5d326c6248ccd8476d0476d63d276ed385a5c12`.
Original Qwen2.5-3B Q8_0 weights, prepared down/vocabulary, dense context
1024, canonical 65 ChatML IDs and natural greedy 512-token generation stay
fixed. The explicit profile mode loads the selected configuration, not the
older native control. Matching completed quality evidence is required.

## Profiling Perturbation and Parity

Five alternating off/on pairs measure **41.441017 / 39.869478 HTTP tok/s**,
or **-3.792231%** with profiling. These numbers compare instrumentation,
not two inference optimizations, and do not replace unprofiled authoritative
timings. Do not subtract an estimated profiling cost to claim a higher rate.
Event-pool setup is about 0.84 ms in the first session; it alone does not
explain the total perturbation. Host event queries/report handling and GPU
probe gaps are included in the full measured request cost.

Both actual off/on replays preserve all 512 IDs and match completed quality.
All ten primary requests keep the same output. The final profile parity
record confirms this; no runtime default or kernel changes occur here.

## GPU Stage Samples

Seven profiled sessions each emit steps 0, 255 and 511: one warmup, five
primary requests and one untimed actual-ID replay, in serial request order.
The derived audit excludes the first and last sessions. Each table value
is the median of five per-token stage sums across all 36 layers; shares
are medians of each sample's fraction of its summed GPU stage time.
Global lm-head timings occur once per token and are not multiplied by 36.

| Stage | Step 0 ms | Step 255 ms | Step 511 ms | Step 255 share |
| --- | ---: | ---: | ---: | ---: |
| Fused gate/up/SwiGLU | 10.0354 | 10.0441 | 10.0466 | 42.94% |
| MLP down | 4.7611 | 4.7507 | 4.7660 | 20.32% |
| QKV projection | 2.5167 | 2.4960 | 2.5063 | 10.67% |
| Attention output projection | 1.7324 | 1.7412 | 1.7382 | 7.44% |
| Vocabulary projection | 1.5635 | 1.5634 | 1.5642 | 6.68% |
| Attention | 0.6097 | 0.7807 | 0.9772 | 3.34% |
| All summed GPU stages | 23.2377 | 23.3922 | 23.6348 | 100% |

These are event spans in the instrumented path, not measured DRAM traffic,
occupancy or a perfect decomposition of the unprofiled token. The event pool
is reused after the first sampled token. Absolute positions are 64, 319
and 575. Attention grows with context but remains far below gate/up cost.
This replaces the older stage-share assumption for selecting the next work.

## Next Bounded Experiment

Test one 128-thread FP32-only gate/up variant. Keep packed loads, task
assignment, warp reduction, prepared down/vocabulary and runtime defaults
fixed; compute each eight-value unscaled partial dot before applying that
Q8 block's scale. This reduces repeated inner-loop scale multiplication
without Q8_1 activation rounding, but changes FP32 association. It is not
assumed bit-identical or faster.

First require full CPU-reference output checks and guarded distinct rows,
then use existing real layer 0/17/35 inputs and complete-FFN paired timing
against the retained packed FP32/128 control. Do not expand the geometry
search if this bounded experiment loses. Only a passing micro gate can
advance to model integration, actual 512-ID equality, frozen quality and
incremental E2E qualification. Preserve negative code/results before removal.

## Evidence

- [Off/on HTTP rows and actual token traces](2026-10-25-maxwell-selected-stage-http.jsonl)
- [Raw CUDA event samples](2026-10-25-maxwell-selected-stage-events.jsonl)
- [Derived stage medians](2026-10-25-maxwell-selected-stage-audit.json)
- [Audit program](2026-10-25-maxwell-selected-stage-audit.jq)
- [Comparison output](2026-10-25-maxwell-selected-stage-profile.log.gz)
- [Isolated Ollama oracle log](2026-10-25-maxwell-selected-stage-ollama.log.gz)
- [Checksums](2026-10-25-maxwell-selected-stage-profile.sha256)

The official isolated Ollama server supplies only one untimed oracle, then
remains idle. No build/test, MemPalace embedding or other GPU work overlaps
measurement. Its parent PID 1083946 is stopped afterward and the GPU process
list is empty. Fifteen comparison-tool tests, all-target CUDA/server Clippy,
non-CUDA example checks and formatting pass before the diagnostic. Neither
original throughput target is achieved; the exact-engine goal remains active.
