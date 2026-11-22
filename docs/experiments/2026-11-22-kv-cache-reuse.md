# Does mixed KV accumulate device memory across cache lifetimes?

At e100c54, fresh Qwen processes released all dense cache bytes. Mixed
caches retained 1 MiB after historical positions had been used, at both
3,072 and 8,448 positions. The first allocation alone cannot establish
whether a long-lived server accumulates memory with successive requests.

Freeze two fresh processes, one dense and one top-64 mixed. Each keeps
the same model, CUDA context, deterministic device input, and execution
flags alive for three allocate/fill/release cycles at capacity 3,072.
Use the same release executable, default stream synchronization and
per-layer length checks as the original cohort. Record CUDA used bytes,
Rust-tracked bytes and process-attributed NVML at the same boundaries.

PASS for bounded reuse if the second and third post-release observations
equal the first, and both arms return to their respective first-release
levels after each full fill. Report the first-cycle offset separately.
FAIL if post-release CUDA bytes increase on both subsequent cycles while
the same process owns the GPU. Any other pattern is INCONCLUSIVE.
Stop after these two processes. A plateau over three cycles qualifies
only this controlled allocation sequence. It does not establish the
allocator's internal cause or a long-running HTTP server's peak.

## Result

**PASS for bounded reuse.** Dense returned to 4,377 MiB and
4,595,122,176 CUDA-used bytes after every cycle. Mixed returned to
4,378 MiB and 4,596,170,752 bytes after every cycle. The mixed
first-cycle difference was 1 MiB with zero increase on cycles two
and three. Each full mixed cache reached 4,488 MiB against
4,485 MiB dense. The tracked weight counter stayed at 4,472,456,192
bytes. This result removes evidence of accumulation in the controlled
sequence. The first-cycle offset and sustained HTTP residency still
need independent attribution.

The [replay](2026-11-22-kv-cache-reuse/replay.json) was verified
against the extracted [raw archive](2026-11-22-kv-cache-reuse/measurements.tar.gz).
The [checksum](2026-11-22-kv-cache-reuse/archive.sha256) and preflight
pin the archive, executable, model and complete environment. At source
`3371054610b09293d59d09bd66235cc001ded5d1`, use release CUDA
`kv_quality residency --model MODEL --arm dense --capacity 3072 --cycles 3 --output RAW/dense.jsonl`
and repeat with `--arm top64 --output RAW/top64.jsonl` in a fresh
process, under the same M40 lock. Use the environment in the preflight.
From the worktree, reproduce the replay with
`jq -n --slurpfile dense RAW/dense.jsonl --slurpfile top64 RAW/top64.jsonl -f docs/experiments/2026-11-22-kv-cache-reuse/replay.jq`.
