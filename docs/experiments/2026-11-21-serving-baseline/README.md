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
