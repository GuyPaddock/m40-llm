# Packed FP32/128 HTTP Qualification

The first M08 runtime timing comparison qualifies the opt-in, typed
`Q8FfnConfig::packed_f32(128)` candidate against the original prepared
down/vocabulary control. It does not change runtime defaults, qualify packed
prefill, or establish either original Ollama speed target.

## Controlled Configuration

Source `c79e919`, with runtime comparison tooling from `c854579`. Release
executable SHA-256:
`e07292f27c7da0d97c3e8aff1ac95fbfc5c376a421ddbb0aeeea1fbf7031d283`.
The model, environment, canonical 65 prompt IDs, prepared weights and dense
KV/context-1024 configuration are identical to the
[six-case quality checkpoint](2026-10-25-maxwell-runtime-quality.md).

Only gate/up/SwiGLU changes to the screened packed/warp FP32 kernel with 128
threads. Prepared down/vocabulary retain their original native-order control.
Both use sequential prompt processing. Because sequential prompt tokens also
use single-row forward, the FFN candidate affects that interval too; this is
not a packed-prefill comparison. No activation quantization, sparsity,
approximate attention, graph or scheduler change participates.

The isolated official Ollama 0.40.11 server supplies one untimed one-token
request plus tokenizer/offload checks before M40 timings. Its resident
`llama-server` is idle during M40 requests. Two resident GPU processes do not
mean two simultaneous inference workloads. The preserved Ollama log contains
exactly one `/api/generate` request. No builds, tests or MemPalace embeddings
overlap the measurements. The isolated server was stopped afterward, and an
empty GPU process list was verified before subsequent regression tests.

## Results

Five alternating control/candidate pairs, all 512 actual generated tokens,
natural `length` stop and identical output:

| Median | Prepared control, native FFN | Packed FP32/128 |
| --- | --- | --- |
| Buffered HTTP E2E tok/s | 35.597905 | 38.493055 |
| HTTP total | 14382.87 ms | 13301.10 ms |
| Prompt/setup interval | 1579.81 ms | 1469.18 ms |
| Generated-token decode interval | 12759.81 ms | 11796.60 ms |

Phase and total rows are independently computed medians, not additive timing
accounting. Throughput uses actual generated tokens divided by complete
client-observed buffered HTTP time.

The paired protocol reports **+8.132920%**, with a 95% bootstrap gain interval
of **[7.938410%, 8.274587%]**, 10,000 resamples and seed zero. Control/candidate
MAD divided by median are 0.085061% / 0.062801%. The 3% promotion threshold
dominates twice observed variation. Neither variation extension nor a fresh
borderline confirmation is required by the existing protocol for this clear
screen result. This does not imply five observations suffice for marginal
gains.

Every primary request reports the same 250,369 kernel launches, five
request-level allocations and five frees. The candidate records 20,736
`ffn_q8_f32_packed_warp_128` launches: 576 processed tokens times 36 layers.
Neither model retains Q8_1 activation scratch or a prefill temporary.
Prepared weights remain 1,192,935,424 bytes/model and dense KV 37,748,736.
Twenty before/after boundaries observe SM/memory clocks 1113/3004 MHz;
temperature ranges from 48 to 71 C. These samples are not an occupancy or
DRAM-bandwidth measurement.

The existing three HTTP arithmetic/OK/config smokes pass for both models.
The completed frozen quality report is checked against source model,
configuration, both splits, corpus and benchmark prompt before timing.
Finally, untimed production replays verify all 512 actual generated IDs equal,
with zero different IDs and no divergence. The recorded promotion decision
passes timing, completed quality and actual-ID gates.

Raw `model_variant="prepared"` denotes the FFN candidate in this reused tool;
both models already have prepared down/vocabulary. The explicit runtime
configuration and per-model fields disambiguate these historical labels.

## Evidence and Validation

- [Raw HTTP report and actual token traces](2026-10-25-maxwell-runtime-packed128-http.jsonl)
- [Derived serial-interval, workload, counter and gate audit](2026-10-25-maxwell-runtime-packed128-http-audit.json)
- [Comparison process log](2026-10-25-maxwell-runtime-packed128-http.log.gz)
- [Isolated Ollama log](2026-10-25-maxwell-runtime-packed128-ollama.log.gz)
- [cuBLAS-on regression log](2026-10-25-maxwell-runtime-parity-on.log.gz)
- [cuBLAS-off regression log](2026-10-25-maxwell-runtime-parity-off.log.gz)
- [Artifact SHA-256 checksums](2026-10-25-maxwell-runtime-packed128-http.sha256)

After timing and isolated-server cleanup, selected CUDA attention, projection,
MLP, forward, runtime/prefill and allocation suites complete with cuBLAS on/off.
The on run reports 51 successes and the off run 45. The legacy optional
real-model forward diagnostic is not supplied a model here; it is not cited
as real-model coverage. The six explicit-model quality runs above provide
that evidence. Runtime-runner tests, warning-as-error non-CUDA checks,
CUDA/server all-target Clippy and formatting pass. The unrelated optional
compact-export test still requires separately supplied artifacts and is not
part of this qualification. Release examples are rebuilt with cuBLAS enabled
after the off regression run.

## Remaining Work

This qualifies a benchmark candidate, not a new runtime default. The existing
environment spelling `packed-f32` still selects its original 256-thread
geometry; request 128 explicitly through the typed configuration/comparison
tool. Complete the remaining bounded comparisons and M09 attribution/fresh
Ollama qualification before selecting a retained configuration. The two
failed prefill gates remain recorded; investigate their common execution
semantics before claiming a combined exact-output gain. Neither 53.06 HTTP
tok/s nor 1.30x a fresh controlled Ollama median is achieved by this result.
