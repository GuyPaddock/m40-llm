# Intrusive November Q8 stage profiling

**FAIL for ordinary-request stage attribution.** On the Qwen Q8_0
joint-only compound configuration, enabling stage collection increased
median HTTP latency by 4.913%. The frozen perturbation limit is 3%.
Five pairs were stable enough to stop. Profiling throughput was
48.351 tok/s against 50.726 tok/s with profiling off, a change of
−4.68% (paired bootstrap 95% interval −5.36% to −4.23%).
Within-arm MAD/median was 0.478% on and 0.133% off. The run exited
with the specified attribution failure. No further sweep is warranted.

The bitwise companion passed before timing. The [raw archive](profile.tar.gz)
retains all ten timed rows, the comparator, event bounds, all 24 stage
records, preflight, host/GPU telemetry, and failure log. These stage
durations apply to the instrumented path. Uninstrumented operator
contributions in November remain inconclusive. The earlier five
[historical cohorts](../../2026-11-21-q8-economics.md) have their own
perturbation controls.

Run source c4807e1ff3a49603a37fb1a20fd274eb775f107e with the frozen
quality report identified in the binding. Inspect the endpoint offline:

```sh
out=$(mktemp -d)
tar -xzf profile.tar.gz -C "$out"
jq -s 'map(select(.kind=="dataflow_http_trial")) | group_by(.arm) |
    map(map(.row.http_ms)|sort|.[2]) | (.[1]/.[0]-1)*100' "$out/profile/dataflow.jsonl"
jq 'select(.kind=="dataflow_http_summary")' "$out/profile/dataflow.jsonl"
```

The ownership/load and uninstrumented HTTP phases completed before
this failure. The separate packet/argmax portfolio requires its own
quality prerequisite and uninstrumented timings.
