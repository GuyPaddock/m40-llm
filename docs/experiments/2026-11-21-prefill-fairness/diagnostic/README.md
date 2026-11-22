# Sampled token gaps during concurrent prefill

On the Tesla M40, a 2,060-token prompt that arrived during active decode
coincided with a 344.171–350.970 s gap in the ongoing request. The control,
an arriving 65-token prompt, coincided with an 8.585–8.749 s gap. All ten
second requests arrived after the first had produced three or four tokens.
The treatment spent 344.041–350.839 s of the gap between the second
request's admission and its first token sample. Waiting for admission took
4.9–129.0 ms. Both requests in every trial matched their isolated controls
on generated IDs, output text and stop reason.

| Pair | Short/short gap, s | Short/medium gap, s | Short/medium overlap, s |
|---:|---:|---:|---:|
| 0 | 8.637 | 346.577 | 346.449 |
| 1 | 8.749 | 344.171 | 344.041 |
| 2 | 8.585 | 346.553 | 346.421 |
| 3 | 8.602 | 345.053 | 344.922 |
| 4 | 8.739 | 350.970 | 350.839 |

The five diagnostic pairs clear the predeclared gap threshold. The full
admission decision also requires the instrumentation-off workload checks.
The tested server used dense KV, four sequence slots and the opt-in
buffered scheduler. The host monitor sampled a stopped emerge in all 420
observations during the ten requests. The Windows VM remained active.
These are task-level observations, with no change to scheduler policy.

The [response archive](responses.tar.gz) retains the source and executable
hashes, exact requests, warm-server calibration, wall clocks, raw response
IDs and trial snapshots. [SHA-256](archive.sha256) · [Pair replay output](pairs.json)
· [Per-request host summary](host.json). The host and GPU timelines are in
the full study's raw directory pending the primary series.

To replay, extract the response archive into a fresh directory and set
`d` to that directory. Run from the directory containing this report:

```bash
sha256sum -c archive.sha256
for i in 0 1 2 3 4; do
    jq -ce -s -f ../pair.jq "$d"/diagnostic-single-{short,medium}.json \
        "$d"/diagnostic-"$i"-{control,treatment}.json
done | jq -s 'to_entries|map({pair:.key}+.value)' | diff - pairs.json
```
