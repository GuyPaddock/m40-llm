# Input-Channel Redundancy Has Insufficient Scope

## Decision

Stop this fixed conditional-symbol model without an encoder, decoder or
CUDA implementation. Conditioning Q8 payload-byte distributions on their
known input-channel coordinate reduces entropy slightly, but the optimistic
stored-byte savings are only **0.41-1.88%** at the six measured tensors.
Every tensor misses the **10%** feasibility gate.

Even removing all codebook/index costs leaves only **4.07-5.54%** ideal
savings with original scales. Shrinking metadata cannot make this model
reach the gate. Alignment, integral code lengths, decoder tables, lookup
work and random-access costs are still omitted, not assumed solved.

This is full-file information/size evidence, **not measured compression,
memory traffic, FFN latency or HTTP throughput**. It is not a universal
lossless-compression bound; models exploiting other correlations are outside
this fixed question. Do not turn this result into another conditioning,
codebook, chunk-size or predictor search.

The qualified **49.146 HTTP tok/s** configuration remains unchanged versus
the previous **43.106 official Ollama** result. Both original targets,
**53.06** and **56.037 HTTP tok/s**, remain unmet. No hardware-ceiling,
quality/default or runtime promotion follows.

## Actual Source Counts And Optimistic Sizes

Each tensor has K=2048, N=11008, **23,953,408 original bytes**, comprising
22,544,384 signed payload bytes and 1,409,024 literal FP16 scale bytes.
All six actual tensors are reread from the original Qwen2.5-3B Q8_0 file
and checked against frozen model/tensor SHA and absolute GGUF byte ranges.

The [fixed contract](2026-10-26-maxwell-channel-entropy-hypothesis.md)
counts each input channel across all output columns. Every channel has
11,008 symbols and 256 possible raw byte values. The pooled distribution
exactly matches the earlier independently validated source scan. Signed
weight interpretation does not change byte entropy, and scale bits never
enter payload histograms.

| Layer | Tensor | Pooled bits/byte | Conditional bits/byte | Optimistic stored bytes | Saving with metadata | Ideal saving without metadata |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Gate | 7.65094 | 7.52922 | 23,503,274 | 1.88% | 5.54% |
| 0 | Up | 7.64898 | 7.53845 | 23,529,279 | 1.77% | 5.43% |
| 17 | Gate | 7.66887 | 7.64544 | 23,830,788 | 0.51% | 4.17% |
| 17 | Up | 7.66253 | 7.63740 | 23,808,131 | 0.61% | 4.27% |
| 35 | Gate | 7.68322 | 7.65055 | 23,845,182 | 0.45% | 4.11% |
| 35 | Up | 7.67837 | 7.65408 | 23,855,124 | 0.41% | 4.07% |

Per tensor, the fixed size accounting is:

```text
ideal_payload_bytes = ceil(sum(channel_entropy * 11008) / 8)
literal_scale_bytes = 1,409,024
code_length_bytes   = 2048 * 256             = 524,288
index_bytes         = 2048 * ceil(11008/256) * 4 = 352,256
optimistic_stored   = ideal_payload_bytes + all three costs above
```

The missing costs can only worsen this estimate within the fixed model.
No actual code lengths, encoded stream, reversible format or execution
consumer is constructed. No execution-copy owner is replaced; original
GGUF remains needed by prefill. These are not resident-VRAM savings.
The previously closed global-byte models remain unchanged.

## Provenance And Validation

Source checkpoint `7600c7a24b3492e74b7fb4ed24f3a178ac0cd1d3`; original
model SHA-256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
CPU scanner SHA-256
`93f519f29258d372f3c63cb40574069c5fc18edba75a77b0b098ea92297144e5`.
All scanner/orchestrator/recount sources and their hashes are archived.
No Rust, CUDA, inference, benchmark selector or dependency changes occur.

The scanner passes **15 behavioral checks**, including constant-by-channel
data with nonzero pooled entropy, uniform all-byte distributions, literal
scale exclusion/accounting, multiple input blocks, ragged index chunks,
malformed tensors and deterministic repeated results. A separate AWK check
verifies the exact selected byte positions and malformed-row rejection.

The actual scan records every histogram for all **12,288 input channels**
across six tensors. All pooled histograms reproduce the earlier counts.
An independent `od`/AWK pass recounts channels **0,31,32,2047** at every
tensor, covering first/last positions and the input-block boundary. This
is a sampled mapping check, not a claim that all channel mappings were
independently reread. The JSON audit recomputes all channel entropies,
pooled conservation and complete byte accounting from the raw histograms.

Twelve corruption controls reject source/hash errors, scale/index/codebook
omissions, altered counts, a channel swap that preserves pooled counts,
false entropy/savings, a decoder claim, CUDA-library presence and resource
overflow. The scanner processes each tensor serially with a 64 MiB raw
buffer cap and 512 MiB process limit. Observed peak RSS is at most
**106,180 KiB**. Scanner wall times are 8.39-11.64 seconds per tensor;
those are analysis costs, not inference benchmarks. Process maps contain
no loaded CUDA libraries; no CUDA context, upload or GPU setting is used.

An initial attempt computes one tensor, then its independent AWK recount
fails because `index` is a reserved function name. Session 55444 exits one;
there is no completed matrix. Its code and partial artifacts are preserved
separately. The corrected recount uses a fresh output directory, passes
its behavioral check and reruns **all six tensors**. Session 38169 exits
zero; no partial result is pooled into this decision. No failed kernel or
timing experiment is implied by that orchestration error.

## Artifacts And Reproduction

The hardware-free audit replays the archived completed scan:

```bash
bash docs/experiments/2026-10-26-maxwell-channel-entropy-audit.sh
bash docs/experiments/2026-10-26-maxwell-channel-entropy-audit.sh --negative-controls
```

- [Completed raw histograms, scripts, source hashes and checks](2026-10-26-maxwell-channel-entropy-data.tar.gz)
- [Interrupted first attempt, excluded from results](2026-10-26-maxwell-channel-entropy-interrupted.tar.gz)
- [Audit](2026-10-26-maxwell-channel-entropy-audit.jq), [derived sizes](2026-10-26-maxwell-channel-entropy-audit.json), [corruption-control log](2026-10-26-maxwell-channel-entropy-controls.log.gz)
- [Checksums](2026-10-26-maxwell-channel-entropy.sha256)

Repeating the source scan needs the explicit original Ollama GGUF blob,
fresh output paths and archived CPU scripts. It performs no cache-tree
search, model download or Hugging Face access. No CUDA regression suite is
claimed for this CPU-only study; no CUDA/runtime code changed.

Next needs a genuinely different sufficient avoided-work hypothesis, not
a decoder whose optimistic byte budget already misses feasibility. Existing
complete-FFN, actual512/frozen-quality and HTTP gates remain intact.
