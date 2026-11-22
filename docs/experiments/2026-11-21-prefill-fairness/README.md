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
