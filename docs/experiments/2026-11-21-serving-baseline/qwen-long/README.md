# Long Qwen serving diagnostics and dense feasibility

**PASS for the completed default mixed-KV diagnostics. FAIL for dense
request feasibility at the existing one-hour ceiling. INCONCLUSIVE for
paired performance.** The original Qwen2.5-3B Q8_0 checkpoint received
8,197 raw ChatML prompt IDs, capacity 8,448 and a 128-token greedy
generation budget. Both modes used the same release `sm_52` executable,
greedy seed zero, prompt bytes and model bytes.

| Default mixed-KV diagnostic | Observation |
|---|---:|
| CLI process wall time | 2,607.44 s |
| Buffered request wall time | 2,601.471 s |
| Admission to first sampled output | 2,549.565 s |
| Prompt / subsequent decode throughput | 3.215 / 2.450 tok/s |
| Median / P95 sampled inter-token gap | 409.000 / 411.986 ms |
| Streaming warmup / measured wall time | 2,599.498 / 2,602.566 s |
| Measured streaming first nonterminal record | 2,553.889 s |
| Maximum sampled default server process peak | 3,488 MiB |

The CLI and buffered diagnostic matched all 128 sampled IDs. Buffered,
streaming warmup and measured streaming responses matched text exactly.
Each stream returned 128 output records. The default cache recorded
260,514,576 requested bytes against a dense equivalent of 311,427,072
bytes. Its configuration used top-8 historical selection with direct
FP16-K/Q4-V consumption and the existing FP16 recent window.

The dense diagnostic emitted **zero sampled tokens** after 3,602.569
seconds at the stop snapshot. Its sampled process peak was 3,536 MiB.
The M40 still reported 100% utilization at the snapshot. The watcher
verified the PID, process start time, executable and dense mode before
sending SIGTERM to that benchmark process. The process time record ended
at 3,606.93 seconds and the queue recorded status 143. The HTTP harness
already uses a 3,600-second request limit. This triggered the
[frozen feasibility rule](../qwen-long-boundary.md) and left zero completed
timing-off pairs. The default numbers are standalone diagnostics. A
separate two-length operator check measures the known dense dispatch
boundary.

The host replay observed QEMU averages of 18.06–18.40 logical CPUs
across the default observations and 18.90 during the dense observation.
The sampled maximum ten-second CPU pressure during the dense diagnostic
was 4.76%, and I/O pressure reached 1.45%. The raw records retain both
the VM-on host observations and the limits of their temporal coverage.

The run preflight used checkout `a632e7aeae322b6cbc3439b047b3a0d9f90fa454`.
The prompt SHA-256 is
`8d2a709195f168460cc8406b2865cba001b31fe98708daac2c2db635fa44ff45`.
The model SHA-256 is
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
The executable SHA-256 is
`a6462eb0b64326c47eb465a9bed257700794438f0a32b01296f2f454a6554036`.
Verify the [archive checksum](archive.sha256) and extract
[the original records](measurements.tar.gz). Then run the partial-cohort
replay from its parent study directory.

```bash
bash replay-long-cutoff.sh EXTRACTED/serving-qwen-long-01 \
    EXTRACTED/serving-qwen-long-cutoff-01 qwen-long/queue-exits.txt
```

The [receipt](replay.json) verifies the completed default IDs and
responses, the cut-off output checksums, the process status and sampled
peak memory. The [host receipt](host-replay.json) uses the watchdog's
stop snapshot as the dense observation's end boundary.
