# Q8 packet and argmax portfolio on Maxwell

**PASS for the original 3% combined-portfolio gate. FAIL for the
November 5% gate.** Twenty fresh confirmation pairs improved the
combined configuration from 50.528 to 52.469 buffered HTTP tokens/s.
The gain was 3.842%, with a paired 95% interval of 3.750–3.996%.
The five-pair screen was within one percentage point of the original
threshold, which triggered the independent confirmation. The best
configuration still fell below 53.06 tok/s and the 56.153149 tok/s
target derived from the October Ollama result.

| Compared with the same joint-FFN control | Final pairs | Control tok/s | Candidate tok/s | Gain and 95% paired interval |
|---|---:|---:|---:|---:|
| Packet down projection | 20 | 50.467 | 51.610 | 2.264% [2.091%, 2.369%] |
| Packet vocabulary projection | 5 | 50.551 | 50.945 | 0.778% [0.421%, 1.094%] |
| Partitioned argmax | 5 | 50.594 | 51.180 | 1.157% [0.873%, 1.275%] |
| Combined | 20 | 50.528 | 52.469 | 3.842% [3.750%, 3.996%] |

The two 20-pair series each had a separate five-pair screen. All
120 timed requests matched the reference text, sample count, stop
and that arm's dispatch. The preflight and post-HTTP generation
checks matched all 512 generated IDs on the shared 65-token prompt.
In the independent quality phase every arm matched seven full
151,936-value logit digests, all 10,616,832 valid FP16 KV words and
the seven passing real-model cases. Five reference failures remain
inconclusive. MAD/median for final measured throughput ranged from
0.011% to 0.147%.

The control and every candidate already owned 1,192,935,424 bytes
of down and vocabulary packets. Full prepared storage was
4,110,516,224 bytes and the measured single-model tracked total
was 7,390,035,968 bytes in each arm. Preparing the packets took
6.845–7.390 s during the quality phase and reached 661,225,472
temporary host payload bytes. One benchmark process hosted both
servers for each comparison, with one request in flight. The
500-ms NVML sampler observed a peak of 14,395 MiB during that
process. The single-model tracked total and the two-server process
peak have different scopes. This trial qualifies dispatch within
an already prepared configuration. A deployment starting without
the packet owners pays the full allocation and preparation cost.

Keep the combined portfolio as an opt-in follow-up for a rebuilt
Q8 foundation. The same model with only down and vocabulary
prepared beat its own native reference, which is a different
comparison. Individual packet or argmax gains are measured
separately and cannot be added to the combined result.
[Selective prepared versus native](https://github.com/GuyPaddock/m40-llm/blob/e25b2ed41a022870fcebad776e352310f6f221d4/docs/experiments/2026-11-21-native-split/README.md).

## Reproduce

The release runner, source and all measurements came from
`4f6e70a2d9d76d5460c9e53d5b822adf630d3d5c`. Run
`bash docs/experiments/2026-11-21-q8-economics/run.sh MODEL RAW portfolio`
in a clean worktree with the release `joint_only` example, after
sourcing `scripts/dev-env.sh` and acquiring the shared M40 lock.
The [raw archive](measurements.tar.gz) contains the five-arm
quality and HTTP records, model and executable digests, flags,
sampled residency, host load and timestamps. [Archive SHA-256](archive.sha256).

Extract the archive and run the
[independent portfolio replay](https://github.com/GuyPaddock/m40-llm/blob/60bf48abefd1493d709694f69632709029683f47/docs/experiments/2026-11-22-q8-ownership/portfolio-replay.sh)
with the extracted directory and release `research_http_pairs` as
arguments. The statistics executable SHA-256 is
`7d0b0038af46a468998d81ef606ca1754ddca75598c18be5b744d906cbe0a148`.
The outcome must equal [the committed replay](replay.json).
