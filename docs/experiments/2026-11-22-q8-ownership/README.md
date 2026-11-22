# Qwen compound FFN weight ownership

**PASS for dropping duplicate gate/up weights when the opt-in compound
FFN is already active.** Both runs used the same quantized model, compound
kernel and KV configuration. Joint-only ownership skipped 144 device
allocations and 1,724,645,376 bytes of device uploads. The median HTTP
throughput change was +0.10% with a 95% paired bootstrap interval of
−0.16% to +0.32%. The memory result applies to the choice of owner within
the compound path. Costs relative to native Q8 and main use different
controls.

| Measure | Retain split and joint | Joint only |
|---|---:|---:|
| Tracked device bytes after load | 7,921,745,920 | 6,197,100,544 |
| Combined prepared weight bytes | 4,642,226,176 | 2,917,580,800 |
| Fresh loader process peak at 500 ms cadence, MiB | 7,671 | 6,024 |
| Load elapsed, one observation, s | 24.50 | 23.70 |
| Joint preparation, one observation, s | 1.367 | 1.349 |
| Host high-water mark, KiB | 6,999,556 | 6,999,636 |
| Device allocations at load | 291 | 147 |
| Median HTTP throughput, tokens/s | 50.795 | 50.847 |

The fresh loader process used 1,647 MiB less at its sampled peak
(21.47%). Load times are single observations. Both HTTP arms resided
in one process during timing, which prevents a per-arm inference peak
comparison. Runtime fallback for row shapes outside the one-row path
continues to require the original GGUF backing.

The hardware runner compared 36 joint payload hashes, checked all 36
layer mappings and matched all 512 generated IDs on the 65-token prompt.
At seven specified positions the SHA-256 of the complete 151,936-logit
row matched. Twelve quality cases had identical generated IDs, with
seven reference passes and five inconclusive reference failures.
The two arms made identical ordinary dispatches in the generation
companion. Every timed response reproduced its output, stop and length.
After unloading the runner reported zero Rust-tracked device bytes.
Only five alternating pairs were needed. Within-arm MAD/median was
0.075% and 0.040%.

The minimum port would require native Q8 admission, prepared plane
ownership and a qualified compound one-row FFN on main. Carry the
joint-only selection with that path if those predecessors qualify.
Retain `retain-split` as an explicit diagnostic control. The prepared
Q8 portfolio and the benefit of adopting compound FFN have separate
qualification gates.

The [raw archive](measurements.tar.gz) holds the four completed phases,
hardware telemetry and their shared load timeline through the final
HTTP observation. The [summary](summary.json) includes binary and model
hashes and the launch environment. Source `c4807e1ff3a49603a37fb1a20fd274eb775f107e`
on research/1121-q8-projections supplies the frozen `run.sh` in
`docs/experiments/2026-11-21-q8-economics/`. Replay from this directory:

```sh
out=$(mktemp -d)
tar -xzf measurements.tar.gz -C "$out"
bash replay.sh "$out/ownership" | diff - summary.json
```

## Single-owner HTTP residency check

The paired timing harness holds both owners at once. Freeze an independent
memory observation using the normal server binary and one process per
arm. Each process handles the original prompt twice, first cold and then
warm. Require the expected owner in startup logs, 36-layer packed prefill,
and both sequences of 512 generated IDs against the frozen reference.
Record NVML by PID every 250 ms across model load and both requests.
Report full-process peak and each request window separately. The
single-sequence memory gate is 10% lower observed process peak, with
the exact-output and HTTP non-regression controls above. These two
logged requests supply memory attribution only.

Run `bash docs/experiments/2026-11-22-q8-ownership/server-peak.sh NEW_DIRECTORY`
from the consolidation worktree with the project environment and shared
M40 lock. The experiment uses the same dense KV, 1,024 capacity and
compound gate/up kernel in both arms.

**PASS for isolated peak residency.** The
[normal-server study](server-peaks/README.md) reproduced a 1,647 MiB
reduction from 7,807 MiB to 6,160 MiB with exact sampled IDs.
The same study records a separate 1,139 MiB increase for selective
prepared Q8 against native Q8.
