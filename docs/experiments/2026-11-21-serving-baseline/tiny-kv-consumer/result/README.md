# Direct historical KV improves complete TinyLlama requests

**PASS for the frozen head64 TinyLlama workload.** The direct FP16-K/Q4-V
consumer delivered 8.775% higher buffered HTTP throughput than staging,
with a paired 95% interval of 8.575–9.089%. Five alternating warmed pairs
used the same 1,390 prompt IDs and returned the same 32 generated IDs.
Each arm used the same mix of recent and historical positions, a recent
window of 1,024, and a 64-block budget covering all historical blocks.
Each timed response matched its instrumented companion.

| Measure | Staged | Direct |
|---|---:|---:|
| Median buffered request wall, s | 116.642 | 107.232 |
| Median buffered output tokens/s | 0.2743 | 0.2984 |
| Median admission to first sample, s | 112.409 | 103.506 |
| Median prompt tokens/s through first sample | 12.366 | 13.429 |
| Median steady decode tokens/s | 7.412 | 8.339 |
| Median server user plus system CPU, s | 55.34 | 52.93 |
| Sampled peak process VRAM, MiB | 6,224 | 6,208 |
| Diagnostic median/P95 inter-token gap, ms | 135.028 / 136.485 | 120.082 / 121.594 |
| Cold CLI wall, s | 123.93 | 114.12 |
| Streaming first nonterminal record, s | 113.238 | 104.670 |

Within-arm MAD/median was 0.114% staged and 0.128% direct. The sampled
VRAM peak fell 16 MiB, or 0.26%, under the 10% memory-only promotion
criterion. The separate diagnostic verified the intended attention
consumer at all 31,262 layer/position rows per arm. The staged session's
17,039,488 requested workspace bytes explain the scale of the sampled
VRAM delta. [Dispatch and memory accounting](../diagnostic/README.md).

The end-to-end comparison includes the normal costs of prefill, cache
append and token generation. The gains are for this model and workload.
Phase 4 cannot use the staged mixed consumer for the Qwen head128 runtime.
Long-context quality under restricted history and batched serving are
separate gates. Keep the existing single-sequence direct consumer on
main and retain this comparison as its matched workload evidence.

Portage was stopped in all 220 host samples within timed requests. The
VM remained running. Maximum sampled ten-second CPU pressure was 4.56%
staged and 4.25% direct. The final GPU snapshot reported zero MiB and
zero compute processes. CPU, streaming and instrumented token clocks
use their own stated endpoints.

The [archive](measurements.tar.gz) includes the raw CLI and HTTP
records, both complete profiled requests, executable and model hashes,
GPU/process timelines, and host load. Verify [the checksum](archive.sha256),
then extract it and run these from the serving-baseline directory.

```bash
bash replay.sh EXTRACTED_DIRECTORY | diff - tiny-kv-consumer/result/replay.json
bash verify-pair-statistics.sh EXTRACTED_DIRECTORY RESEARCH_HTTP_PAIRS
python3 kv-trace.py EXTRACTED_DIRECTORY | diff - tiny-kv-consumer/result/dispatch.json
```
