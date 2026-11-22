# Does prefill interrupt active HTTP decode?

Base a1b2470 on the November consolidation branch. Profile stock Phase 4
scheduling before admitting any scheduler changes. Portage was paused by
the operator; use the common host/GPU monitors. Load one real Qwen2.5-3B
Q8_0 server. Use explicit dense cache, four slots, 2304 tokens per slot
and native Q8. Use the default batch tick behavior with
M40LLM_SERVER_BATCH_DECODE=1 and M40LLM_SERVER_BATCH_PREFILL=1. This is
an opt-in serving configuration. The production serialized endpoint has
no active decode/prefill overlap in its generation lock.

Use frozen 65-token browser and 2060-token archive prompts from the common
manifest, original template and seed zero/top-1. Short generation is capped
at 128 for this concurrency study, medium at 32; record real output IDs
in a separate diagnostic. Warm up each request. First run the short prompt
in isolation and freeze the arrival delay at its observed fourth sampled
token time plus 25 ms. Admit a mixed trial only when actual short decode
has started and has not ended when the other request arrives. Keep invalid
overlap attempts as evidence. The two-client treatment is the short request
plus medium prefill. The control has a second short request at the same
offset, holding client count and arrival constant. Also run short/medium
alone and four short clients together to measure 1/2/4-client behavior.

Run five alternating treatment/control pairs. Supply both the primary
instrumentation-off means of observing client request wall and a separate
matched sampled-clock cohort for ITL. Clock-aligned diagnostics capture
per-token times relative to request admission and the corresponding host
wall-clock origin; this avoids treating client queue delay as active decode.
Report client TTFT (including queues where clocks observed), per-request
median/P95 and maximum inter-token gap, maximum inter-request completion
delay, aggregate sampled tokens/s and whole-process peak. Compare output
text, actual generated counts, stop reasons and diagnostic IDs per request.
Different prompts are a workload contrast, not an exact transformation.

Predeclare a scheduling opportunity when at least four of five valid mixed
pairs have a short-decode gap overlapping the new prefill that exceeds both
twice the corresponding short/short gap and 100 ms extra. Report P95
separately since one stall may affect fewer than 5% of tokens. A qualifying
stall admits a bounded scheduling prototype; it does not qualify any policy.
If the arrived request only waited in a queue, count it as waiting rather
than as a stall in active compute. Do not add a second CPU decoder.

The runner uses the A5 HTTP client and hardware/host monitors from a1b2470
in the adjacent consolidation worktree. Build the research binary with
`M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo build --release --locked --features cuda,server --bin m40-llm`.
Build the analyzer with
`cargo build --release --locked --no-default-features --example research_fairness_report --target-dir target/cpu`.
Run `bash docs/experiments/2026-11-21-prefill-fairness/run.sh NEW_DIRECTORY`
under the serial M40 lock. The output includes completed responses and all
token samples in the diagnostic run. A sampled time is anchored to the host
clock. Transport/queue delay, first sample, median/P95/maximum spacing and
submission-to-completion use distinct boundaries. A host-clock inversion
greater than 1 ms invalidates correlation. Read `summary.json` and the raw
GPU/process timelines together. A failure or timeout cannot establish absence
of starvation.

For an independent replay of one completed diagnostic pair, run the
following from this worktree with `d` set to its raw directory. Replace
both zeroes in the filenames with the desired pair index.

```bash
jq -es -f docs/experiments/2026-11-21-prefill-fairness/pair.jq \
    "$d"/diagnostic-{single-short,single-medium,0-control,0-treatment}.json
```

The replay checks both requests against their isolated text, stop reason
and complete sampled IDs. It identifies the two consecutive short-request
samples bounding the longest gap that overlaps the arriving request's
admission-to-first-sample interval. Both indices are zero-based. Only the
intersection of those two intervals counts toward `overlapping_ms`.
The second request's wait before admission is recorded separately. The
five-pair decision also requires matching work in the instrumentation-off
series.
