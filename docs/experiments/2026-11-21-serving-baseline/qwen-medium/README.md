# Medium Qwen serving baseline on the M40

**PASS for repeatable single-request baselines in both cache modes.
FAIL for cross-mode greedy ID equivalence.** Every request processed the
same 2,060 prompt IDs and generated 128 IDs to the length limit. The
default and dense continuations diverged at the first sampled token,
IDs 4498 and 95456 respectively. Within each arm, the CLI and buffered
diagnostic produced the same 128 IDs. All ten timing-off responses
matched the corresponding diagnostic output and token accounting.

| Metric | Default mixed KV | Explicit dense FP16 KV |
|---|---:|---:|
| Median buffered HTTP wall time | 489.292 s | 373.309 s |
| Five-trial wall time range | 489.120–490.156 s | 372.221–373.511 s |
| Median buffered throughput | 0.2616 tok/s | 0.3429 tok/s |
| Median prompt throughput | 4.535 tok/s | 5.940 tok/s |
| Median decode throughput | 3.621 tok/s | 4.807 tok/s |
| Median admission-to-first-sample time | 454.221 s | 346.775 s |
| Diagnostic median / P95 inter-token latency | 276.335 / 279.489 ms | 208.878 / 212.743 ms |
| Streaming first nonterminal record | 456.525 s | 348.280 s |
| Streaming complete request | 489.414 s | 373.134 s |
| CLI whole process wall time | 495.59 s | 380.57 s |
| Median server CPU time per timed request | 148.87 s | 130.63 s |
| Median server launch-to-health | 6.485 s | 6.495 s |
| Maximum sampled timed server process peak | 3,332 MiB | 3,320 MiB |

The alternating order and fresh-process warmups establish variability
within each cache mode. The divergent outputs leave two independent
baselines, with no cross-mode speed gate or confidence interval. This
prompt alone supplies no answer-quality verdict or default recommendation.
Per-layer and kernel costs need separate profiling. Buffered throughput
includes the full prompt, generation, HTTP handling and client boundary.
The streaming time records the first complete nonterminal response
record. The diagnostic inter-token figures use server sample clocks.

Both arms ran the same release `sm_52` executable, original Qwen2.5-3B
Q8_0 bytes, raw ChatML prompt, greedy seed zero and capacity 2,304.
The prompt SHA-256 is
`8902b52db89221e622a347f310e89c8b7429a20484bec6ee24e0f509fb52d14a`.
The model SHA-256 is
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
The executable SHA-256 is
`a6462eb0b64326c47eb465a9bed257700794438f0a32b01296f2f454a6554036`.
The default trace recorded block-select-exact top-8 with a recent
FP16 window and direct FP16-K/Q4-V historical attention. It reported
95,926,032 actual cache bytes and an 84,934,656-byte dense equivalent.
The independently sampled process peak differed by 12 MiB.

QEMU used 18.24–18.92 logical CPUs on average over observed intervals
within the ten timing trials. The maximum sampled ten-second host CPU
pressure reached 20.67% on one dense trial. The raw monitor retains all
samples and the [host replay](host-replay.json) records coverage. The two
GPU-hidden compilation windows were inside streaming warmups. Their
commands and exits are retained with the separate packed-Q4 build.

The run used checkout `84232fea53c1e4856bd5f346d868828807f1ae29` and
the [frozen serving runner](../run.sh) with cohort `qwen-medium`. The
[archive](measurements.tar.gz) and [SHA-256 receipt](archive.sha256)
retain exact inputs, flags, model and executable identities, request
responses, output IDs, host samples, process peaks, and device state.
Replay from the consolidation worktree with the extracted directory.

```bash
study=docs/experiments/2026-11-21-serving-baseline
bash "$study/replay.sh" EXTRACTED_DIRECTORY
bash "$study/verify-pair-statistics.sh" EXTRACTED_DIRECTORY \
    target/release/examples/research_http_pairs
```

The [receipt](replay.json) regenerates all trial rows, CPU and GPU
summaries, CLI and streaming observations, and the unequal-work gate.
