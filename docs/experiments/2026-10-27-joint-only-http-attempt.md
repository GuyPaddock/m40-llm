# HTTP Request Metrics Correction

The first HTTP attempt at clean source `2d45353` stopped during the first
excluded warmup with `missing generation metrics`. It recorded no primary
timed pairs. The request omitted the server's `include_metrics` opt-in;
this is a qualification-runner error, not an ownership or performance result.

Retain the [incomplete attempt](2026-10-27-joint-only-missing-metrics.jsonl)
and the preceding [completed quality report](2026-10-27-joint-only-pre-http-quality.jsonl)
with its [passing independent audit](2026-10-27-joint-only-pre-http-quality-audit.json).
The executable hash in each raw manifest identifies this attempt separately
from later measurements. Do not pool or overwrite it.

The correction uses the server's typed `GenerateRequest`, requests metrics
explicitly, and checks the serialized request through a real HTTP router in
the keep-alive ownership regression. Match the original qualified request's
prompt, greedy sampling and 512-token limit. No runtime, ownership, numerical,
memory or statistical gate changes. Rebuild, rerun complete quality for the
new executable, then run paired HTTP only if that audit passes.
