# QKV/Output Warp Runtime Versus Fresh Ollama

## Decision

The selected split-plane FP32 FFN, SGEMM-prefix and QKV/output warp runtime
measures **47.378948 versus 43.096583 buffered HTTP E2E tok/s**,
**9.936669% faster** than controlled official Ollama 0.40.11. Five
alternating pairs give a paired 95% gain interval of
**[9.878961%, 10.665805%]**. MAD/median is 0.014293% M40-LLM and 0.114540%
Ollama. The original 3%/variation gate passes without trial extension or
borderline confirmation.

Both original goals remain **unmet**: **53.06 HTTP tok/s** and
**56.025558 HTTP tok/s**, the latter being 1.30x this fresh Ollama median.
This retains the opt-in exact-output runtime, not a global default. The
[incremental result](2026-10-25-maxwell-projection-incremental-http.md)
separately demonstrates +5.279% against the current native-projection
split-FFN/SGEMM control. Do not combine percentages from different runs.

## Configuration And Provenance

Clean source `c7b2f242ae72c8a333bc0c9e365c69330da4c053`, release comparison
executable SHA-256
`48db57beb05b4738d0ceb54fed3572d743403f534a7aabe244e93f569f5a2960`.
Official Ollama executable SHA-256
`b0a4becfaf872e3ad5d2e9051244e0dfe55409f03d6e2cd9bd9c34400e201b4c`;
its pinned llama.cpp revision is `b11845`, resolved to
`9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e`. The official binary, not an
instrumented attribution build, supplies the comparison timings.

Both engines use the same Qwen2.5-3B Q8_0 GGUF, canonical 65-token ChatML
prompt, context 1024, greedy sampling, natural stopping and 512 actual
generated tokens. This workload explicitly uses **dense FP16 KV**, not the
project's default compressed KV. M40 selects `split-f32x4-128`,
`dequant-sgemm`, `warp-256` and prepared `gate-up,down,vocabulary`.
The packed 64-token prefix includes all temporary weight conversions; the
last prompt token takes the normal decode path. Source weights, attention
semantics, down/vocabulary arithmetic and sampling remain fixed. There is
no Q8_1 activation quantization, pruning, sparsity, graph replay or scheduler
change in this result.

Source `scripts/dev-env.sh` and export the frozen environment from the
[quality reproduction](2026-10-25-maxwell-projection-runtime-quality.md#configuration-and-reproduction).
Use the [isolated official server setup](../controlled_benchmark.md#ollama-setup)
and the release comparison executable with:

```text
--compare-q8-runtime split-f32x4-128
--q8-runtime-prefill dequant-sgemm
--q8-runtime-control qualified-split-sgemm
--q8-runtime-attention warp-256
--q8-runtime-quality-report docs/experiments/2026-10-25-maxwell-projection-runtime-quality.jsonl
--time-ollama
```

Supply `--model`, the owned `--ollama-pid` / `--ollama-log` and a new output
path. The named quality control does not become a timed third participant.
The manifest records the complete environment, executable hashes and actual
loaded projection backend. The model comes from the local Ollama blob,
not `~/.cache/huggingface`; tokenizer metadata comes from that GGUF.

## Timing, Correctness And Memory

| Independent medians | M40-LLM selected runtime | Official Ollama |
| --- | ---: | ---: |
| Buffered HTTP | 10,806.487 ms | 11,880.292 ms |
| Prompt interval | 494.506 ms | 229.761 ms |
| Generated-token decode interval | 10,263.248 ms | 11,644.989 ms |

Phase boundaries differ between engines and independent medians need not
sum to HTTP. Complete client-observed HTTP is authoritative. Every measured
request evaluates all 65 prompt tokens with zero reuse; Ollama's parsed
cache lines were also checked against each original runner-log byte range.
Each engine's output is stable internally; cross-engine output equality
is not assumed. M40's actual 512-ID replay matches the completed
[quality report](2026-10-25-maxwell-projection-runtime-quality.md), whose
twelve task outputs and seven sampled full-vocabulary logits match control.
Seven dense-valid tasks have zero regressions; five reference failures
remain inconclusive. The benchmark-only loader correction below does not
change the quality executable's already-correct construction path.

All 109 prepared tensor identities and payload capacities are unchanged:
**3,279,519,744 source bytes + 2,917,580,800 prepared bytes =
6,197,100,544 bytes** before KV/workspace. Dense KV is **37,748,736 bytes**.
The reusable **90,177,536-byte** SGEMM dequantization buffer is released
after prefill; no full FP32 model cache or quantized activation buffer is
retained. These are requested payload bytes, not standalone peak VRAM.

Every primary M40 request reports 237,013 kernel launches, 252 SGEMMs,
42 allocations/frees, 1097 synchronizations and 324 waits. QKV warp and
output warp each execute **18,432 launches**. Complete per-operation
counters equal completed quality, not just the expected backend label.
Allocation counts cover admission/session work, not per-token preparation.

All twenty primary GPU boundaries show P0, 1113 MHz SM / 3004 MHz memory,
with only benchmark PID 1522077 and official runner PID 1494146 resident.
Requests run serially, with the other engine idle. No build, test or
MemPalace work overlaps timing. The benchmark exits successfully. Owned
official server PID 1492825 and its runner are stopped afterward; the GPU
compute-process list is empty on the cleanup check.

## Rejected Attempt And Correction

The [first fresh attempt](2026-10-25-maxwell-projection-ollama-loader-rejected.md)
printed the requested warp configuration but loaded native projections.
Its approximately 45.064 tok/s result is **not** candidate evidence. The
run was stopped and preserved separately, not overwritten or averaged in.

`d1f6231` routes primary loading through the same typed runtime constructor
as quality and incremental candidates, checks the loaded backend before
timing, and checks actual QKV/output dispatch afterward. `c7b2f24` appends
the extra manifest field outside the large JSON macro to fix release-build
recursion; it changes no inference math. Tests pass: comparison example
19, model-quality example 32 with one external compact-model case excluded,
CUDA/server all-target Clippy, non-CUDA example 11, and the corrected release
build. No timing data came from the intervening failed build.

The earlier quality and incremental runs already used the correct candidate
factory. Their evidence remains unchanged. This corrected fresh run has a
different executable hash and therefore its own frozen auditor; the older
checksum-bound audit program is deliberately not rewritten.

## Evidence And Next Work

- [Raw paired requests, dispatch, token traces and qualification](2026-10-25-maxwell-projection-ollama-http.jsonl)
- [Frozen audit program](2026-10-25-maxwell-projection-ollama-http-audit.jq) and [derived result](2026-10-25-maxwell-projection-ollama-http-audit.json)
- [Comparison log](2026-10-25-maxwell-projection-ollama-http.log.gz), [official server log](2026-10-25-maxwell-projection-ollama.log.gz) and [checksums](2026-10-25-maxwell-projection-ollama-http.sha256)

Run the auditor with `jq -s --arg mode ollama`,
`--arg revision c7b2f242ae72c8a333bc0c9e365c69330da4c053`,
`--slurpfile quality` pointing to completed quality, `-f` pointing to this
frozen auditor, and the raw JSONL path. It binds source/binaries, frozen
environment, preparation, actual backend/launches, tokens, memory, cache
reset, paired medians/variation, clocks and processes. Negative controls
with native backend, missing warp launches, changed IDs or altered KV bytes
all fail. The earlier incremental checksums still pass unchanged.

Re-profile this newly selected combination before choosing another bounded
Maxwell kernel hypothesis. Earlier stage shares describe the old native
projections, not this runtime. Do not revive failed FFN geometry/layout
experiments or substitute numerical/model-changing research for either
original unmet E2E goal.
