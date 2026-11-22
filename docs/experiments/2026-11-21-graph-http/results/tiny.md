# TinyLlama F16 CUDA Graph replay

**PASS for exact sampled tokens on the asynchronous F16 decode stream.
FAIL for the 5% whole-request gain.** Five warmed, alternating M40 pairs
returned the same 128 IDs for the same 46 prompt tokens. Graphs improved
median buffered HTTP throughput by 0.261%, with a paired 95% interval
of −0.041% to +0.366%. The original F16 control path failed capture
in the Q projection with FFI status −3. That trial stopped before timing.
Both arms in the completed comparison used the same asynchronous
implementation and the same dense cache configuration.

| Median or process peak | Ordinary | Graph |
|---|---:|---:|
| HTTP tokens/s | 39.341 | 39.443 |
| Complete HTTP wall, s | 3.253639 | 3.245183 |
| Admission through first token, ms | 1,103.05 | 1,114.51 |
| Steady decode interval, ms | 16.904 | 16.735 |
| Server user plus system CPU, s | 3.26 | 3.27 |
| Isolated companion sampled peak, MiB | 2,239 | 2,239 |
| Timed throughput MAD/median | 0.104% | 0.059% |

The 32-row diagnostic recorded 330 nodes, 2.397 ms capture,
1.862 ms instantiation, 26.96 µs median CPU graph launch, and 13.61 ms
median synchronized CUDA-event replay span. This includes a full
captured forward. Per-kernel service time and resident warp count
were unavailable. All 67 device observations in the paired windows
showed the SM at 1,113 MHz and 60–62 °C, with both servers resident
and one timed request in flight. Idle process attribution and the
independent one-server traces remain in the raw capture.

Keep graphs opt-in. Carry the attention-dispatch repair on the
paths with equivalent ordinary/graph state. A port must show the
necessary F16 capture support on its actual base. This measurement
uses Phase 4 source `379002a62a19e12ed7b8e0da2777b3025e9393b5`,
with release executable SHA-256
`4cd3c2d98fc0c50562c52afb08eafb4cb6744f58c3ec0f59b48e2e87a5c68eb3`.
The pinned TinyLlama F16 weight and runner hashes, both capture
attempts, warmups, five pairs, and diagnostic are in the
[raw archive](tiny-http.tar.gz). [Independent endpoint summary](tiny-summary.json).

The archive SHA-256 is
`73aeac6da46cfe4ce6f1a90488c06fcd48061f4e8bd802dc86a23cbd361e4254`.
Extract the archive and run `python3 ../audit.py graph-tiny-decode-stream-01`
from this results directory with the extracted path in place of the final
argument. Run the release `research_http_pairs` example recorded in the
preflight on the extracted `timings.jsonl` to reproduce `screen-5.json`.
Run the hardware comparison through [the capture runner](../run.sh)
with cohort `tiny-decode-stream` under the shared GPU lock. The original
`tiny` cohort reproduces the capture precondition failure.
