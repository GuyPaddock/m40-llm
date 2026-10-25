# Split-Plane Runtime Versus Fresh Ollama

After [incremental HTTP qualification](2026-10-25-maxwell-split-incremental-http.md),
the exact-output split-plane FP32 FFN plus SGEMM runtime measures
**45.075177 vs 43.119374 HTTP E2E tok/s**, **4.535786% faster** than the
fresh controlled official Ollama 0.40.11 median. The paired 95% gain interval
is **[4.360502%, 4.637279%]**.

This closes the measured Ollama gap for this workload. It does **not** finish
the original goal: neither **53.06 tok/s** nor **56.055187 tok/s** for
1.30x this fresh Ollama median is achieved. Runtime defaults stay unchanged.

## Configuration And Qualification

Clean source `f93bb89767cb8d47b926903975ece5211d06d6cb`, release comparison
executable SHA-256
`9540358f1eb1eee6c4658125f27387694125f8cf670e377b54bb3ee1c85a0e2e`.
Official Ollama executable SHA-256
`b0a4becfaf872e3ad5d2e9051244e0dfe55409f03d6e2cd9bd9c34400e201b4c`;
its pinned backend/source provenance remains the earlier M01 checkpoint.
No diagnostic/instrumented Ollama build supplies authoritative throughput.

Both engines use the same original Qwen2.5-3B Q8_0 GGUF, canonical 65 ChatML
IDs, context 1024, greedy sampling, natural length stopping and 512 actual
generated tokens. Dense FP16 KV is explicit. M40-LLM selects
`split-f32x4-128`, dequant-SGEMM prefix and prepared `gate-up,down,vocabulary`.
The 64-token prefix includes every conversion; the final prompt token uses
normal decode. Neither model pruning, Q8_1 activation quantization in
M40-LLM, approximate attention, graphs nor scheduling changes participate.

Use the same arguments as incremental qualification, adding `--time-ollama`.
`--q8-runtime-control qualified-packed128-sgemm` still identifies the quality
report's reference; it does not make that reference a timed participant here.

Five alternating pairs pass the unchanged 3%/variation gate. MAD/median is
0.043042% Ollama and 0.054005% M40-LLM. No trial extension is needed.
Every measured M40 output and actual 512-ID replay matches completed quality.
Each engine's measured outputs are stable internally; cross-engine output
equality is not assumed. Every primary request validates a full 65-token
prompt evaluation with zero cache reuse, including Ollama runner-log evidence.

| Independent medians | M40-LLM split/SGEMM | Official Ollama |
| --- | ---: | ---: |
| Complete buffered HTTP | 11,358.80 ms | 11,874.01 ms |
| Prompt interval | 500.86 ms | 230.62 ms |
| Generated-token decode interval | 10,817.91 ms | 11,638.29 ms |

Prompt/decode timing boundaries differ between engines and phase medians
need not sum to HTTP. Only client-observed complete buffered HTTP is the
primary performance comparison. Qualification covers this model/workload
and the frozen quality corpus, not every model, prompt or context length.

## Memory And Execution

The M40 runtime holds **2,917,580,800 bytes** of prepared weights plus
3,279,519,744 bytes of original weights: **6,197,100,544 bytes** before
KV/workspace. Dense K/V capacity remains **37,748,736 bytes**. The temporary
90,177,536-byte SGEMM buffer is released before warm decode. No quantized
activation scratch or full FP32 model cache is added. These are requested
payload bytes, not standalone peak VRAM.

All five primary M40 requests preserve 237,013 launches, 252 SGEMMs,
42 allocations/frees, 1097 synchronizations and 324 waits. All twenty
primary boundaries observe P0, 1113/3004 MHz clocks. Requests are awaited
serially; per-request Ollama runner-log regions are nonoverlapping. No
build, test or MemPalace embedding overlaps timing. The preceding incremental
process exits before this run starts. Owned official server PID 1246178 and
its runner are stopped afterward; the final GPU process list is empty.

The first selected-runtime attempt failed before model construction with a
missing gate/up preparation error and collected zero timing samples.
Correction `f93bb89` derives primary preparation from the selected runtime,
with pair/Ollama/profile configuration regression checks. The incremental
measurement used the correctly configured packed128 primary and is unaffected.
The [failed preflight log](2026-10-25-maxwell-split-ollama-preflight-failed.log.gz)
is retained; its empty report is not substituted into this five-pair result.

## Evidence And Next Work

- [Raw requests, verified cache resets, actual replay and target status](2026-10-25-maxwell-split-ollama-http.jsonl)
- [Derived audit](2026-10-25-maxwell-split-ollama-http-audit.json)
- [Reproducible audit program](2026-10-25-maxwell-split-http-audit.jq)
- [Comparison log](2026-10-25-maxwell-split-ollama-http.log.gz)
- [Owned official Ollama log](2026-10-25-maxwell-split-ollama.log.gz)
- [Checksums](2026-10-25-maxwell-split-http.sha256)

Retain this explicit exact configuration experimentally. Re-profile the
selected runtime before choosing another bounded Maxwell kernel hypothesis;
the older stage shares are not measurements of the newly optimized FFN.
Do not substitute the numerical, sparse, pruning or approximate-attention
tracks for either original unmet E2E objective.
