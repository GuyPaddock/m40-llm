# Packed-Prefill Quality After Prefix Ownership Repair

Both independent packed-prefix backends pass the fresh corrected-control
quality gate. This replaces the old exact-to-unsafe-control failure as the
current correctness checkpoint; it does not rewrite historical evidence or
establish a new HTTP speedup.

## Fixed Configuration

Release report revision `6ea8801`, executable SHA-256
`83cecd04d01ded613c1e4fc97df6065570ab0b80cc1c38a414f5e8c3ddb7fc01`.
The binary's runtime source matches the preceding corrected-control
[FP32/128 checkpoint](2026-10-25-maxwell-completed-packed128-quality.md).
The source model, frozen M02 environment, corpus/rubrics, both splits,
prepared down/vocabulary and dense context 1024 remain fixed. Both models
use native FFN. Only prompt processing changes: sequential control versus
a 64-token packed prefix followed by the normal final prompt token.

The reports record `prefix-decode-completed-v1`. Runs are serial, with an
empty GPU before the first and no Ollama inference, build, test or MemPalace
embedding overlap. Uncompiled comparison-tool edits during quality affect
neither the running binary nor library/CUDA source. These single-run
intervals are diagnostic, not paired throughput measurements.

## Results

| Candidate | Actual browser IDs | Frozen task outputs | Dense-valid regressions | Sampled logit max diff | Minimum top-10 overlap |
| --- | --- | --- | --- | ---: | ---: |
| Native packed prefix | All 512 equal | All 12 equal | 0 / 7 | 0.010689 | 10 |
| Dequant SGEMM prefix | All 512 equal | All 12 equal | 0 / 7 | 0.013475 | 10 |

Both browser branches generate 512 tokens and stop naturally at the length
bound, with no divergence. Five reference-failed tasks remain inconclusive
for candidate quality. Forced-prefix comparisons use the same actual
reference history at seven steps. The generic full-vocabulary head is labeled
separately from the production GPU-greedy token path. No numeric tolerance
or rubric is relaxed. Nonzero projection/cache-rounding drift is measured,
not called bitwise parity.

Prefill-specific task/browser generation exercises the actual packed path.
The runner intentionally does not claim sequential passage loss as validation
of packed prefill; teacher-forced loss remains available for the independent
sequential FFN candidate.

| Single diagnostic browser run | Control prompt | Candidate prompt | Control decode | Candidate decode | Control total | Candidate total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native packed | 1590.81 ms | 3208.55 ms | 12731.72 ms | 12715.14 ms | 14322.59 ms | 15923.73 ms |
| Dequant SGEMM | 1584.68 ms | 478.68 ms | 12749.16 ms | 12730.01 ms | 14333.88 ms | 13208.74 ms |

Native packed remains a likely timing loser; SGEMM advances to controlled
HTTP pairs. Do not turn these single diagnostic timings into percentage
performance claims. Generated-token decode is held native in both cases.

## Dispatch and Memory

Prepared down/vocabulary retain 1,192,935,424 bytes/model; dense KV retains
37,748,736. Native packed reports 237,013 launches, 41 allocations/frees,
1097 stream synchronizations and 324 stream waits. SGEMM has the same launch,
sync and wait counts, plus 252 cuBLAS calls and 42 allocations/frees. Its
90,177,536-byte temporary capacity includes all seven conversions/SGEMMs
in all 36 layers, inside the prompt interval; the workspace is released
before resident post-request accounting. No full FP32 cache or vocabulary
materialization is introduced. Both reports return tracked bytes from zero
to zero.

## Evidence and Next Step

- [Native packed report and actual IDs](2026-10-25-maxwell-completed-native-packed-quality.jsonl)
- [Native packed log](2026-10-25-maxwell-completed-native-packed-quality.log.gz)
- [SGEMM report and actual IDs](2026-10-25-maxwell-completed-dequant-sgemm-quality.jsonl)
- [SGEMM log](2026-10-25-maxwell-completed-dequant-sgemm-quality.log.gz)
- [Checksums](2026-10-25-maxwell-completed-prefill-quality.sha256)

Re-time the independently quality-valid candidates against the corrected
control, then validate and time a combination only if the independent gains
qualify. Fresh selected-runtime Ollama comparison remains separate. Neither
original speed target is achieved, and no runtime default changes.
