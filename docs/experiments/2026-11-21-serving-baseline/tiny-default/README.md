# TinyLlama default-KV baseline across revisions

**PASS for matched stable outputs. FAIL for the 5% throughput
threshold.** On the same TinyLlama F16 checkpoint, 46 prompt IDs and
128 generated IDs matched across pinned main and Phase 4. Both used
their existing default cache configuration. The five-pair screen
showed a 5.541% Phase 4 gain, close enough to trigger a separate
20-pair confirmation. Confirmation measured **4.028%** higher
Phase 4 buffered throughput, with a paired 95% interval of
3.182–5.096%. Within-arm MAD/median was 0.216% main and 0.938%
Phase 4. All timed outputs matched the corresponding trace.

| Confirmation measure | Pinned main | Phase 4 |
|---|---:|---:|
| Median buffered HTTP wall, s | 5.252 | 5.049 |
| Median output tokens/s | 24.372 | 25.354 |
| Median server user + system CPU, s | 5.25 | 3.38 |
| Median server launch through health, s | 6.752 | 6.754 |
| Peak sampled process VRAM, MiB | 6,208 | 6,216 |
| Diagnostic token gap median/P95, ms | unavailable | 39.709 / 42.441 |
| Cold CLI wall, s | 12.42 | 12.16 |
| Warm streaming wall, s | 5.274 | 6.347 |
| Warm streaming first nonterminal record, s | 1.402 | 1.784 |

The legacy main endpoint lacks per-sample clocks. Its output count
comes from the exact companion IDs, whose response text matched
every primary request. Phase 4's buffered median admission to first
sample was 145 ms. The separately logged companion took 231 ms.
Streaming record arrivals include transport and can contain empty
text. Their two measurements remain distinct from internal TTFT.

This comparison changes whole revisions. The separate
[explicit dense baseline](../tiny-dense/README.md) measured Phase 4
19.423% below main on its own configuration. These cohorts cannot
identify a particular kernel or establish a default change. Carry
both TinyLlama workloads into the combined clean-base regression
check.

The [raw archive](measurements.tar.gz) includes both phases and
the complete provenance. Verify [its checksum](archive.sha256)
and rerun the [cohort replay](../replay.sh) and
[statistics verifier](../verify-pair-statistics.sh) to reproduce
[the receipt](replay.json).
