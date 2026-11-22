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
