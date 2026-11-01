# Qualified Split-Plane Runtime Stage Profile

The [HTTP-qualified split-plane runtime](2026-10-25-maxwell-split-ollama-http.md)
remains FFN dominated. This profile selects the next bounded FP32 experiment;
it does not establish a new throughput improvement or either original goal.

Clean revision `05e52760640cfee057e81d14dfc02a4430a5e69e`, release comparison
SHA-256 `9540358f1eb1eee6c4658125f27387694125f8cf670e377b54bb3ee1c85a0e2e`.
The same Qwen2.5-3B Q8_0 GGUF, canonical 65 prompt IDs, greedy 512-token
continuation, dense KV/context 1024, split-f32x4-128, prepared
gate-up/down/vocabulary and SGEMM prefix remain. The passing quality report
is explicit. Official Ollama 0.40.11 supplies an untimed oracle only.

## Overhead And Parity

Five alternating off/on pairs measure **45.044261 / 44.600112 HTTP tok/s**,
**0.986028% slower** with profiling. Do not subtract this perturbation to
claim an inferred unprofiled rate. All ten measured outputs and both actual
512-ID replays match completed quality. The profiler remains diagnostic.

The existing tool uses `--profile-q8-runtime`,
`--compare-q8-runtime split-f32x4-128`,
`--q8-runtime-prefill dequant-sgemm`,
`--q8-runtime-control qualified-packed128-sgemm`, the matching quality report,
and a new `M40LLM_STAGE_PROFILE_REPORT` path. Runtime defaults do not change.

## Sampled GPU Stages

Seven profiled sessions emit steps 0/255/511. Exclude the first warmup and
last actual-ID replay, leaving five primary sessions. Values below are
medians of each stage's per-token sum across all 36 layers. Shares are
medians of per-sample fractions, not ratios of independently rounded medians.
The vocabulary projection occurs once per token, not once per layer.

| Stage | Step 0 ms | Step 255 ms | Step 511 ms | Middle share |
| --- | ---: | ---: | ---: | ---: |
| Fused gate/up/SwiGLU | 8.0598 | 8.0473 | 8.0444 | 37.43% |
| Prepared down | 4.7855 | 4.7937 | 4.7846 | 22.31% |
| QKV projection | 2.5223 | 2.5221 | 2.5230 | 11.72% |
| Output projection | 1.7600 | 1.7734 | 1.7783 | 8.25% |
| Vocabulary projection | 1.5630 | 1.5633 | 1.5640 | 7.28% |
| Attention | 0.6156 | 0.7793 | 0.9856 | 3.62% |
| All summed stages | 21.3571 | 21.5035 | 21.6933 | 100% |

Each event row contains 400 spans, with complete layer coverage and absolute
positions 64/319/575. Events are instrumented spans, not a perfect breakdown
of unprofiled latency, measured DRAM traffic or occupancy. Legacy CUPTI
counters remain unsupported; no bandwidth claim is inferred from this table.

## Next Bounded Experiment

Keep the qualified one-column split-plane kernel as control. Test exactly
one two-column-per-block candidate with 128 threads and the same four-value
FP32 tasks, original scales and reduction tree. Reuse each activation vector
across two independent gate/up outputs. For H=11008 this halves the static
block count from 11008 to 5504, but doubles published reduction storage from
1024 to 2048 bytes and may increase register pressure. These are design
properties, not measured gains. Inspect generated sm_52 registers/spills.

This is a test of the newly qualified split layout, not a geometry search to
rescue the old native four-column experiment. Do not try more tile sizes if
it loses. Require CPU references, guarded odd hidden tails, distinct rows,
capture/ordinary agreement and unchanged prepared planes. Compare directly
with the current split kernel's bits and complete-FFN latency on layers
0/17/35, genuine rows 1/4, five alternating pairs, unchanged prepared down.
Only a >=5% complete-FFN gain with positive paired interval permits runtime
integration, actual 512-ID/frozen quality and incremental HTTP qualification.

Atomic order: this profile checkpoint; one CUDA/benchmark/test commit for the
fixed candidate; a measured screening decision; conditional runtime and HTTP
work only if it passes. Defaults and both original unmet targets stay intact.

## Evidence And Execution

- [Off/on requests and token traces](2026-10-25-maxwell-split-stage-http.jsonl)
- [Raw stage events](2026-10-25-maxwell-split-stage-events.jsonl)
- [Stage medians](2026-10-25-maxwell-split-stage-audit.json), using the unchanged [stage audit](2026-10-25-maxwell-selected-stage-audit.jq)
- [HTTP/coverage audit](2026-10-25-maxwell-split-stage-http-audit.json) and [program](2026-10-25-maxwell-split-stage-http-audit.jq)
- [Run log](2026-10-25-maxwell-split-stage-profile.log.gz), [oracle log](2026-10-25-maxwell-split-stage-ollama.log.gz), [checksums](2026-10-25-maxwell-split-stage-profile.sha256)

Process 1271892 executes alone except the idle oracle runner. All twenty
primary boundaries report 1113/3004 MHz clocks; no build, tests or MemPalace
embedding overlaps timing. Owned server PID 1271590 is stopped after the
profile; both process sessions exit successfully and the GPU ends empty.
