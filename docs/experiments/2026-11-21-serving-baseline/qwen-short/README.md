# Short Qwen default and dense KV baseline

**PASS for the frozen short complete-request comparison.** On the
same Phase 4 server and Qwen2.5-3B Q8_0 weights, explicit dense KV
delivered 7.744% higher buffered HTTP throughput than the default
cache. The five warmed alternating pairs gave a paired 95% interval
of 7.632–7.832%, over the 5% gate. The input remained 65 tokens with
512 greedy outputs. Both CLI and server traces returned identical
IDs and all timed responses matched their companion. Within-arm
MAD/median was 0.103% default and 0.025% dense.

| Measure | Default mixed cache | Dense FP16 |
|---|---:|---:|
| Median buffered HTTP wall, s | 87.970 | 81.647 |
| Median buffered output tokens/s | 5.820 | 6.271 |
| Median admission through first sample, s | 3.830 | 8.521 |
| Median prompt tokens/s through first sample | 16.972 | 7.628 |
| Median decode tokens/s | 6.078 | 6.993 |
| Median server user + system CPU time, s | 34.23 | 31.82 |
| Sampled process peak, MiB | 3,312 | 3,275 |
| Diagnostic median/P95 token gap, ms | 165.16 / 188.55 | 143.18 / 152.65 |
| Cold CLI complete process wall, s | 94.76 | 88.23 |
| Warm streaming complete wall, s | 93.73 | 81.88 |
| Warm streaming first nonterminal record, s | 10.08 | 9.71 |

The buffered timing arms omit token tracing. Token gaps come from
separate instrumented requests, 511 intervals per arm. The streamed
first record includes server transport and can carry empty text.
Both caches were set to capacity 1,024. The complete request fits
inside the mixed cache's recent ring. Its reserved historical
backing still counted toward the 61,747,488-byte allocation, compared
with 37,748,736 bytes of dense K/V payload. The actual sampled
process-peak difference was 37 MiB, including all model and request
allocations.

These are whole-request baselines on the normal dispatch with the
existing explicit dense option. Dense reaches its first internal
sample later and finishes the fixed 512-token continuation sooner.
Medium and long contexts exercise actual historical cache use and
have their own qualification cohorts.

The [raw archive](measurements.tar.gz) includes the complete client
records, token traces, model and executable hashes, SM 5.2 build,
host logs and 250-ms GPU observations. Use the
[paired replay](../archive-cohort.sh) and
[frozen workload](../../2026-11-21-protocol/performance-contexts.json)
to reproduce [the receipt](replay.json).
[Archive checksum](archive.sha256).
