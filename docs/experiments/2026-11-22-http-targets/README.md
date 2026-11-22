# Historical HTTP controls and remaining request budgets

The 20-pair Qwen portfolio confirmation reached 52.548523 buffered HTTP
tokens/s. Its paired control reached 50.523256, a 4.0086% gain. At that
candidate rate, the original 53.06 target would require a further 93.92 ms
saved across a complete 512-token request. Reaching 1.30 times the
October 27 official Ollama median would require 625.45 ms. These are
request budgets. The operator screens measure individual components and
cannot establish how much request time a port saves.

| M40 configuration | Measured HTTP tok/s | Additional throughput for 53.06 | Request saving for 53.06 | Additional throughput for 56.153149 | Request saving for 56.153149 |
|---|---:|---:|---:|---:|---:|
| October 27 compound FFN | 50.910426 | 4.2223% | 407.43 ms | 10.2979% | 938.96 ms |
| October 29 packet and argmax portfolio | 52.548523 | 0.9733% | 93.92 ms | 6.8596% | 625.45 ms |

On October 27, five alternating pairs gave 50.910426 for M40-LLM and
43.194730 for official Ollama 0.40.11, a 17.8626% throughput gain. Both
evaluated all 65 prompt tokens and generated 512 tokens on the same Q8_0
GGUF and M40. Output hashes were stable within each runtime and differed
across runtimes. The October 29 confirmation used one output across both
arms. It had its own contemporary control. Its 21.6549% margin over the
older official Ollama rate is a cross-date comparison.

The [independent replay](replay.jq) recovers each median from the original
client wall intervals. It checks source identity, sample counts, alternating
order, prompt length, generated count, stop reason and each reported rate.
The [result](summary.json) includes the two official-cohort output hashes.
For each target it converts the remaining throughput gap to milliseconds
for a whole request by subtracting the target wall time from the measured
wall time. The frozen historical targets remain distinct from the November
5% end-to-end promotion gate.

```bash
d=docs/experiments/2026-11-22-http-targets
jq -ne --slurpfile official docs/experiments/2026-10-27-maxwell-compound-runtime-ollama-http.jsonl \
    --slurpfile portfolio docs/experiments/2026-10-29-portfolio-runtime-http.jsonl \
    -f "$d/replay.jq" | diff - "$d/summary.json"
```

The October 27 runs and qualification are retained at
[00e94a2](https://github.com/GuyPaddock/m40-llm/blob/00e94a2d0255d028fa919666717bdcd090748e5e/docs/experiments/2026-10-27-maxwell-compound-runtime.md).
The October 29 confirmation and ownership restrictions are retained at
[a75e97b](https://github.com/GuyPaddock/m40-llm/blob/a75e97bb825b17e2288df1e1a380779281d00483/docs/experiments/2026-10-29-portfolio-runtime.md).
