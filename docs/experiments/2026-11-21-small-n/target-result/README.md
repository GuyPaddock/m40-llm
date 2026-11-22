# Shared prepared down weights pass four-row target qualification

**PASS at the complete verification endpoint.** Across 20 fresh M40
confirmation rounds the shared-weight schedule lowered complete
four-row latency **5.876%** against the faster correct multi-row split
control (paired 95% interval **5.803–5.979%**). Against four independent
single-row down calls the saving was **8.941%** (8.856–9.046%).
Both exceed the frozen 5% latency gate. The five initial rounds came
in within one percentage point of that gate against the stronger
control, which triggered the separate confirmation series.

| Confirmation median | Serial down | Existing multi-row | Shared weights |
|---|---:|---:|---:|
| Complete four-row begin through commit, ms | 58.316 | 56.417 | 53.102 |
| Resident request (65 inputs, 512 outputs), ms | 7,858.164 | 7,613.660 | 7,193.426 |
| Down launches per request | 18,396 | 4,680 | 4,680 |
| Session device scratch, bytes | 1,521,188 | 1,521,188 | 1,521,188 |

Each request verified 127 four-row batches and finished with three
ordinary single-row steps. The resident interval includes setup, packed
prefill, target, tails and teardown. It fell 5.52% from the faster
control. Both multi-row schedules issued the same down launch count.
The shared-weight path needed no separate weight owner. Each measured
decode retained 6,198,896,172 tracked device bytes across model, cache
and workspace. The single measurement process reached a sampled
6,162 MiB peak over startup and all schedules. Every timed target used
zero device allocations, frees or cuBLAS calls. Both multi-row variants
kept the same copy and synchronization counts as serial.

Before timing, each schedule passed its own complete correctness
capture. They matched the same 512 generated IDs, all 511 full-logit
row digests and the final 10,616,832 valid FP16 KV words. The equal
payload hashes, layer lengths, and zero tracked bytes after teardown
were checked again from the raw archive. The compiled-resource
comparison and both three-test guard suites are retained separately.
[Resources and operator scope](../README.md) ·
[CUDA guard checks](../guards/README.md).

## Rejection and correction

Five rounds per schedule and prefix exercised actual rejected draft
suffixes and the ordinary correction. Every correction matched the
frozen trace. The total time from submission through correction
is in milliseconds.

| Accepted proposals | Serial down | Existing multi-row | Shared weights |
|---:|---:|---:|---:|
| 0 | 76.529 | 74.614 | 71.147 |
| 1 | 76.523 | 74.577 | 71.223 |
| 2 | 76.590 | 74.613 | 71.185 |
| 3 | 76.497 | 74.526 | 71.170 |

The correction time stays around 18.7 ms across arms. At immediate
rejection the shared schedule spends 71.147 ms for one committed
token. At three accepted proposals that falls to 17.793 ms per
committed token. These are controlled target transactions. Real
draft generation and acceptance are separate costs. The perfect-draft
request and rejection results admit a bounded copy-only experiment
with the existing causal drafter. The timing gate covers this specific
verifier geometry and a single resident model.

The [archive](measurements.tar.gz) retains the exact source and
executable hashes, all three correctness captures, both timing
populations, host and process samples, and all 60 rejection trials.
Verify with `sha256sum -c archive.sha256`. The
[latency gate](gate-replay.json), [request accounting](target-replay.json)
and [rejection accounting](rejection-replay.json) reproduce after
extraction using the consolidation audit at
`38b34e5a2a8a41ed220023b4296cf9587bb63d21`.
Run the commands from the small-N worktree with `raw` set to the
extracted directory.

```bash
audit=../consolidation/docs/experiments/2026-11-22-verifier-rejections
q=docs/experiments/2026-10-27-maxwell-compound-runtime-quality.jsonl
jq -es --slurpfile quality "$q" --slurpfile correct <(cat "$raw"/correctness/*.jsonl) \
    -f "$audit/target.jq" "$raw/measurement.jsonl"
jq -es --slurpfile quality "$q" -f "$audit/replay.jq" "$raw/measurement.jsonl"
bash "$audit/replay-target-gate.sh" "$raw/measurement.jsonl" \
    ../consolidation/target/release/examples/research_http_pairs
```
