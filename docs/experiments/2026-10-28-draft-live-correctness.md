# Live Prompt-Copy Correctness Admission

Implementation `8936b71` connects the unchanged causal prompt-copy source to
the existing fully direct four-row target. It is an explicit research example,
not a generation/server default or a general prose accelerator. The
[offline checkpoint](2026-10-28-draft-checkpoint.md) justified this conditional
implementation only for the predeclared verbatim-copy workload.

The [executable-bound preflight](2026-10-28-draft-live-correctness.jsonl)
passes all 21 cases: browser512, twelve previously frozen tasks and all eight
fresh held-out traces. Every actual ID, count and stop reason matches the
qualified serial reference. Every committed-input full-vocabulary vector
passes the unchanged `1e-3 + 2e-5*abs(reference)` requirement and happens to
be bit-identical. Every valid dense KV word matches bit-identically; layer
lengths match and rejected suffix words are checked zero before correction.
Tracked final teardown is zero, prepared owner statistics are unchanged, and
target cuBLAS calls/device allocations/frees are zero.

The live controller receives only prompt, output budget, stop IDs and actual
target predictions. It never uses reference EOS to suppress a legal proposal.
The source trains only on accepted target inputs; rejected guesses are not
committed. A target-produced stop suppresses unnecessary correction, without
skipping target verification. Frozen futures are used only for post-evaluation
diagnostics and final output comparison. This also exercises ordinary misses,
partial acceptance, correction and stops on non-copy tasks; identity agreement
is not a claim that all model answers are semantically correct.

Nine causal/cost CPU tests and sixteen CUDA-build CPU/statistical tests pass.
CUDA/server Clippy and formatting pass. No CUDA kernel or runtime library
consumer changes. Immutable reference artifacts are not modified.

Timing now requires this exact preflight executable hash and complete unique
case/ID/stop/logit evidence. Warm each arm, then run five alternating resident
pairs for the two unchanged copy cases, with automatic variation/borderline
extension rules. No future fitting, fallback policy or geometry rescue is
permitted. These 52/68-token naturally stopped copy requests are not the
512-token HTTP/Ollama workload and cannot qualify either original HTTP goal.
