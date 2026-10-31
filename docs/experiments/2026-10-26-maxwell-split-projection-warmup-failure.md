# Packed Output Control Exceeds The Old Warmup Budget

Source `34a242770f0851f206b36b44f8a4831b114275e7`, release executable SHA-256
`fd27330265709ae50bcb2c2ef58d5487f5fd8feb5b96c823371f7e77bcce8f79`.
The first split-output attempt exits 1 with `projection bounded warmup
exhausted`, after layer-0/one-request bit/CPU/guard validation but before
any primary timing pairs. GPU compute is empty after process exit.

The pre-existing projection warmup permits 256 batches of 128 iterations.
To accumulate its requested 1000 GPU milliseconds, each call must cost at
least 30.518 microseconds. The qualified packed output control is faster
than that in earlier measured screens, so the warmup budget itself is
insufficient. This is not a candidate kernel correctness/performance result.

Increase untimed iterations per batch to 256 while retaining the maximum
256 batches and requested 1000/100-ms control/candidate durations. Reject
invalid measurements and still fail if the bounded budget cannot reach
the target. Primary iterations, trial count, CUDA kernels, launch geometry,
inputs and promotion gates do not change. CPU regression tests simulate
a 27-us control that cannot complete the old warmup and verify bounded
failure and invalid-measurement behavior.

Preserve the initial complete QKV run too. Its one-request candidate medians
are 0.86-1.59% slower and every stage gate fails. It is not pooled with the
fresh common-source screens after this harness correction. No failed primary
output pairs are discarded because none were collected. Both processes
are terminal before the correction or any rerun; neither is restarted
because of an observation timeout.

- [Incomplete output records](2026-10-26-maxwell-split-projection-incomplete-output.jsonl)
- [Output failure log](2026-10-26-maxwell-split-projection-incomplete-output.log.gz)
- [Initial complete QKV records](2026-10-26-maxwell-split-projection-initial-qkv.jsonl)
- [Initial QKV log](2026-10-26-maxwell-split-projection-initial-qkv.log.gz)

The shared warmup correction is a separate commit, preserving the original
benchmark implementation and this failed attempt. It is not a CUDA tuning
rescue and does not change the exact-output target or runtime defaults.
