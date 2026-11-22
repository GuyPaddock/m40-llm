# Whole-request baseline and serving workload

Use the three [frozen Qwen requests](../2026-11-21-protocol/performance-contexts.json)
and matched checksum from the common protocol. Run single-server buffered
and separate streaming cohorts. The raw prompt and sampled length cap
remain identical within each comparison. Record absolute wall-clock anchors,
monotonic client start/end, complete buffered responses and per-record NDJSON
arrival offsets. Request timing starts at client submission. A completed
stream must terminate with exactly one done record. Report client record
arrival separately from the internal token sampling clock.

Run short, medium and long Phase 4 Q8 workloads with its normal default KV
and explicit dense KV on the same executable. Apply the protocol's
1024/2304/8448 per-workload capacities to both arms. Use native configuration
selection from an environment cleared of research settings. Keep all model
weights, CPU submission and KV state in the normal CLI/server paths.
Alternating warm pairs compare complete requests. If the greedy output or
stop differs, report throughput and quality for each arm independently and
do not rank it as an equivalent-work improvement. Add an independent replay
with token logging/optional sample timing enabled for TTFT and median/P95
inter-token latency. Measure process peak and free/used VRAM across load and
requests, and retain peak sampling cadence.

For cross-branch performance use a checkpoint that both loaders map in full.
TinyLlama F16 first uses its own frozen raw chat question and 128-token cap.
Use actual model capacity in both arms. Keep default-KV runs separate from
matched explicit dense. Verify full forward and greedy ID traces before
drawing a speed conclusion. Main has no response metrics; its matched
untimed ID trace supplies actual sampled count only when all timed responses
match that trace. A missing metric stays missing.

For supported multi-request comparisons use the existing server scheduler
controls and 1/2/4 buffered clients. Collect separate request timelines for
the active short decode and arriving larger prefill. Measure whether a long
prefill overlaps the short request's actual decode window. Compare P95,
makespan, throughput, per-request waiting and starvation with the same
requests in isolation. Any scheduling prototype requires a repeatable delay
during real baseline overlap.

The CPU admission probe dd81f05 on research/1121-main-baseline excluded
Qwen Q8_0 and Q4_0 from main. The two revisions agreed on the complete 46
TinyLlama IDs and stop IDs. [Phase 4 TinyLlama input](tiny-input-ids.json).
The server runner verifies that identity again before launching either arm.

After building the release native server, build the two client/provenance
examples with `cargo build --locked --release --no-default-features --example research_http_load --example research_serving_ids`.
The retained Phase 4 server includes the optional token timing change at
3f8db89; primary requests clear it. The diagnostic buffered runs enable
the sample clock and token-ID logging. The separately warmed stream uses
ordinary tracing-off execution. Run each cohort from this worktree under
exclusive M40 ownership using `bash docs/experiments/2026-11-21-serving-baseline/run.sh COHORT NEW_DIRECTORY`.
Cohorts are tiny-dense, tiny-default, qwen-short, qwen-medium and qwen-long.
Release binaries and source identities are checked against their original
builds. Sampled process VRAM is recorded every 250 ms; load-to-health,
request intervals and CLI process measurements have separate boundaries.
Five alternating pairs each use a fresh process and a full warmup per arm to
avoid jointly resident models changing backend or allocation decisions.

## Local Qwen Q4_0 secondary baseline

The Phase 4 CPU loader mapped all 36 standard layers at width 2048 and FFN
width 11008 on the local Q4_0 GGUF, SHA-256
670d82d0fbee6289b661eb323612c65bd2e97ec8c65cdbd9131aa44d30e9816a.
Its dedicated Q6_K output.weight has 151936 vocabulary rows. The
[admission record](qwen-q4-admission.json) retains the 65 raw prompt IDs
and both stop IDs. The primary Q8 request has the same prompt IDs.
Main e1403e0 could not map this model (admission dd81f05).

Run the existing protocol with cohort `qwen-q4-short`. Freeze a 32-token
cap, natural EOS, greedy sampling, context 1024 and identical weights in
both arms. Compare Phase 4 default KV against explicit dense through the
CLI, streaming and buffered HTTP. Apply the same parity, five paired trials,
variability and stopping rules as the primary short comparison. Reject
fallback-only execution at the full-layer log check. A missing GPU path
ends this cohort at the first failed validation. This is a short-context
secondary baseline. It supplies neither a historical-block quality result
nor a matched speed comparison between different weight quantizations.

## Direct versus staged historical KV in a complete request

Run cohort `tiny-kv-consumer` with the Phase 4 release server for both
arms. Qwen2.5-3B uses 128-wide heads and its mixed staged runtime is
unsupported. TinyLlama F16 uses 64-wide heads and supports both consumers.
The fixed input at `research/1121-quantized-kv` commit `470f091`
contains 64 archive lines inside the TinyLlama chat template. The tokenizer
produced 1,390 prompt IDs. The generation cap is 32. With a 1,024-token
recent window, 32-token blocks and top-64, all historical blocks fit in
both arms. Each uses the same FP16 K and Q4 V backing at capacity 2,048.

Arm 0 explicitly requests staged attention and its session workspace.
Arm 1 uses the existing direct consumer. Check the diagnostic HTTP
operation names, complete CLI/HTTP IDs, stop and prompt accounting before
ranking equivalent requests. Only the diagnostic HTTP run enables per-op
profiling. The ordinary stream and alternating buffered trials include
their respective full prefill, append and decode costs. Retain differences
in prefill mode and residency. The normal 5% request gate and confirmation
rules apply. A result on TinyLlama cannot establish an equivalent Qwen
comparison or attribute the entire wall-time difference to one kernel.
