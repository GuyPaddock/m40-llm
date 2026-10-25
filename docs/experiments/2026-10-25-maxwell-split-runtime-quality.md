# Split-Plane FFN Runtime Quality

The [qualifying microbenchmark](2026-10-25-maxwell-split-ffn-screen.md)
now has a real-model quality checkpoint. Runtime integration `05607ab` and
comparison tooling `f6078fe` preserve the current qualified packed128/SGEMM
control, not the older native/sequential baseline. No defaults change.

## Source And Fixed Workload

Clean tracked source revision `cf8eba0621903fb5945948f5125b9ad6c0427318`,
release `model_quality` SHA-256
`65c8fca18f62c0859fa61144d6f54a8c7a195eabaddc5e238c9e5fec63f248fd`.
The manifest's only untracked path is the JSONL report being written, not a
source change. CUDA flags and the frozen M02 environment are unchanged.

Both arms use original Qwen2.5-3B Q8_0 weights, the same 65 ChatML browser
IDs, greedy generation with natural length stop, explicit dense KV/context
1024, and dequant-SGEMM packed-prefix prefill. The control selects typed
packed FP32/128. The candidate selects split-plane FP32/four-value/128 and
adds gate/up preparation to the unchanged down/vocabulary cache. No Q8_1,
model modification, approximate attention, graphs or scheduling changes.

After restoring the environment from the first frozen capture record:

```bash
source scripts/dev-env.sh
target/release/examples/model_quality \
  --reference-model "$MODEL" --model "$MODEL" \
  --variant-id maxwell-split-f32x4-vs-qualified \
  --q8-runtime-candidate split-f32x4-128 \
  --q8-runtime-prefill dequant-sgemm \
  --q8-runtime-control qualified-packed128-sgemm \
  --context-tokens 1024 --split all --output /tmp/new-split-quality.jsonl
```

## Outcome

- All **512 actual generated IDs**, output and natural length stop match.
  The current control also matches the prior qualified combined report's
  actual continuation; it was not replaced with a new easier reference.
- All **twelve task outputs** match. **Seven dense-valid tasks, zero
  regressions**; five reference-failed tasks remain inconclusive.
- Identical-prefix full-logit probes cover steps 0/1/8/32/128/256/511, with
  each arm using its actual configured packed-prefix admission. Maximum
  sampled absolute drift is **0.004531**; largest mean drift is **0.000424**.
  Top-10 overlap is 10 throughout. This is not bitwise arithmetic parity.
- Both browser arms perform **252 conversions and 252 SGEMMs**, using the
  same **90,177,536-byte** temporary weight capacity. Their FFN counters
  record **18,432** launches of packed128 or split-f32x4-128 respectively.
  All aggregate enqueue counters match, including allocations, copies and
  synchronization. New gate/up caching has not diverted SGEMM prefill.
- Each model's prepared inventory remains unchanged during evaluation.
  Tracked requested device storage returns from zero to zero after cleanup.

Packed-prefix quality does not claim sequential passage loss as validation
of packed prefill. Actual continuations, tasks and forced-prefix logits
exercise the packed path. This experiment neither measures a new held-out
perplexity value nor relaxes any quality gate.

## Ownership And Memory

| Requested device payload | Qualified control | Split candidate |
| --- | ---: | ---: |
| Uploaded original weights | 3,279,519,744 B | 3,279,519,744 B |
| Prepared down/vocabulary | 1,192,935,424 B | 1,192,935,424 B |
| Additional prepared gate/up | 0 B | 1,724,645,376 B |
| Source plus all prepared weights | 4,472,455,168 B | 6,197,100,544 B |
| Dense KV K/V capacity | 37,748,736 B | 37,748,736 B |
| FP32 materialization cache | 0 B | 0 B |
| Quantized activation scratch | 0 B | 0 B |

The candidate's **109** prepared tensors occupy **2,917,580,800 bytes**;
the control has 37. These are now actual requested/uploaded cache bytes,
not merely the prior all-layer estimate. They exclude driver/library,
allocation granularity, workspace and native-KV metadata. The Rust tracked
total during setup covers both resident models; it is not standalone peak
VRAM. Temporary packed host copies are released after each tensor upload.

Single-run cache hash/preparation time is **10,135.87 / 10,513.31 ms** for
control/candidate. These are setup observations, not paired loading gains.
The 128 MiB prefill budget and reusable temporary capacity are unchanged;
the temporary weight storage is released before warm decode.

## Timing Limits And Validation

The one untimed resident generation reports **498.21 / 484.72 ms** prompt,
**11,874.56 / 10,792.37 ms** decode and **12,417.72 / 11,323.22 ms** wall
for control/candidate. Do not turn these diagnostic intervals into an HTTP
speed claim. The next gate is alternating controlled HTTP pairs against
packed128/SGEMM, then fresh official Ollama qualification if they pass.

The M40 started and ended with no compute processes. Process 1237682 ran
alone; no Ollama, build, test or MemPalace embedding overlapped generation.
Its authoritative session exited successfully. Runtime tests cover
head64/head128, biased split-half-RoPE prefill, exact prefix KV, numerical
final/warm logits, actual SGEMM dispatch, immutable selection, cache reuse,
foreign/aliased weights and allocation cleanup. cuBLAS-on/off affected CUDA
tests, warning-as-error non-CUDA tests, Clippy and formatting pass.

Expanded example checks exposed a stale diagnostic assertion from `a26748f`
after published input-reuse repair `2ca0de1`. The deliberately unfenced manual
probe cannot require equality with the corrected session. Separate test-only
commit `cf8eba0` retains its observed drift in reports and still requires
drained/per-prefix-head equality at the original tolerance. All 31
self-contained model-quality tests and 17 comparison tests pass; the compact
external-model test was excluded without its required three artifact paths.

Neither original HTTP goal is established here. The prior qualified fresh
measurement remains 41.331 vs 42.946 tok/s; targets **53.06** and **55.829**
remain unmet until new controlled evidence proves otherwise.

## Evidence

- [Raw generations, tasks, logits, cache inventories and counters](2026-10-25-maxwell-split-runtime-quality.jsonl)
- [Reproducible quality audit](2026-10-25-maxwell-split-runtime-quality-audit.jq)
- [Audit result](2026-10-25-maxwell-split-runtime-quality-audit.json)
- [Run log](2026-10-25-maxwell-split-runtime-quality.log.gz)
- [Checksums](2026-10-25-maxwell-split-runtime-quality.sha256)
