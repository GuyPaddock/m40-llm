# Compound FFN Qualifies As An Opt-In Runtime

## Decision

The fixed joint-packing/shared-scale consumer passes actual-512, frozen
quality, ownership/resource checks and the unchanged incremental HTTP gate.
Twenty fresh pairs measure **50.798730 versus 49.221402 HTTP tok/s**, a
**3.2046% gain**, paired 95% interval **3.1565-3.3032%**. Keep the original
split configuration as the qualified control; no runtime defaults change.

A separate five-pair official comparison measures **50.910426 versus
43.194730 HTTP tok/s**, **+17.8626%**, interval **17.6394-18.0302%**.
Both original targets remain unmet: **53.06** and **56.153149 HTTP tok/s**,
the latter 1.30 times the newly measured official median. This is not a
hardware-ceiling claim or permission to expand the kernel search.

The [four-arm complete-FFN screen](2026-10-27-maxwell-compound-ffn-screen.md)
remains the attribution experiment. Compound gains **5.43-5.55% total over
split** and **0.58-0.67% incremental over joint-only** at one-row layers 0/17/35.
Those are contemporaneous GPU-event results, not additive gains or an HTTP
comparison with joint-only. No joint-only generation configuration was added.

## Fixed Configuration

Runtime implementation `e5a06ac` exposes model-local opt-in
`M40LLM_Q8_FFN_KERNEL=joint-shared-scales-128` and
`Q8FfnConfig::joint_shared_scales()`. It calls the one previously authorized
consumer from `c629471`, with original FP32 activation arithmetic, reduction
order, 128 threads and one-output geometry. No CUDA source changes accompany
runtime integration. All 137 prior compiled entries retain identical words.

Keep Qwen2.5-3B Q8_0 model hash
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`,
all 36 layers, full dense FP16 KV/context 1024, canonical 65 ChatML prompt IDs,
greedy generation and natural length stopping at 512 actual tokens. Bounded
dequant-SGEMM prefix, packed-warp-256 QKV/output, original prepared
down/vocabulary and all other frozen flags stay fixed. No activation
quantization, compressed KV, graphs, model modification or scheduler change.

The shared benchmark loader rejects changed comparison axes, unsupported
devices/dtypes/row counts, K>4096, missing pairs and insufficient combined
budget. There is no native fallback. Default FFN remains native, and the
project's compressed-KV generation default is unrelated and unchanged.

## Quality And Performance

Both current arms and the previously qualified packed-projection checkpoint
produce identical 512 actual IDs and output. Full 151,936-element logits
at steps 0/1/8/32/128/256/511 are bit-identical, top-10 overlap 10. All twelve
frozen task outputs and scores match; seven dense-valid tasks pass without
regression, five remain reference-inconclusive. No task or tolerance changed.

Each request launches the selected FFN exactly 18,432 times, once per decoded
layer. The only changed operation is split versus compound gate/up/SwiGLU.
All other dispatch and aggregate enqueue counts match the completed quality
report, including 252 prefix cuBLAS calls. Timed requests also match quality
output; separate untimed replays verify actual token IDs.

The five-pair screen gives +3.1775%, within one percentage point of the 3%
gate. Therefore use twenty **fresh** confirmation pairs without pooling.
Confirmation MAD/median is 0.0557% control and 0.0616% candidate. The prescribed
`max(3%, twice variation)` threshold remains 3%, with positive paired bounds.
The informational `needs_fresh_confirmation` remains true near the threshold;
the existing protocol stops after exactly twenty fresh pairs, not repeated
attempts until a more favorable result appears.

Independent phase medians, milliseconds:

| Comparison | Runtime | HTTP | Reported prefill | Reported decode |
| --- | --- | ---: | ---: | ---: |
| Fresh 20 incremental | Split control | 10401.979 | 505.147 | 9854.019 |
| Fresh 20 incremental | Compound | 10078.992 | 498.052 | 9533.150 |
| Fresh 5 official | Compound | 10056.879 | 493.262 | 9523.231 |
| Fresh 5 official | Ollama 0.40.11 | 11853.298 | 229.170 | 11619.536 |

Do not sum separately reported phase medians or equate the engines' internal
timing boundaries. Buffered client HTTP is the promotion measurement. The
official run is separate from the split comparison, not pooled with it.

Official Ollama is the pinned prebuilt 0.40.11, SHA-256
`b0a4becfaf872e3ad5d2e9051244e0dfe55409f03d6e2cd9bd9c34400e201b4c`,
with verified CUDA 12 sm_52 backend. RAM prompt caching is disabled and actual
runner log regions prove zero prompt reuse and 65 evaluated IDs for every
accepted timed request. Inference is serial, including the untimed oracle;
two resident compute processes do not mean concurrent benchmarks.

## Ownership, Resources And Setup

| Requested storage | Bytes |
| --- | ---: |
| Original uploaded GGUF weights | 3,279,519,744 |
| Original 109 split prepared tensors, unchanged | 2,917,580,800 |
| Additional 36 joint gate/up pairs | 1,724,645,376 |
| Combined prepared allocation | 4,642,226,176 |
| Existing 6144 MiB preparation budget | 6,442,450,944 |
| Candidate weights including every duplicate owner | 7,921,745,920 |
| Explicit dense KV per model | 37,748,736 |
| Temporary prefix conversion capacity | 90,177,536 |
| Largest temporary host joint pair | 47,906,816 |

Original source and all 109 split identities remain resident. They are not
replaced or hidden in the joint accounting. Model load validates every pair
and combined budget before any candidate upload, packs/uploads one pair at
a time and drops its host copy. Cache-owned RAII releases partial failures
and complete models. One-row decode borrows those stable owners with no new
allocation, upload or stream synchronization.

The quality run records 10.316 s original split preparation, 10.281 s candidate
split preparation and 1.646 s additional joint preparation. Model hashing
accounts for 9.161/9.117 s of the respective split preparation intervals.
Joint time includes 1.268 s packing and 0.274 s upload. These are cold costs,
excluded from warm HTTP. Original setup host payload reporting and the
separate 47,906,816-byte temporary joint report are both preserved; neither
is a measured standalone peak RSS/VRAM claim.

The paired HTTP process retains both models, 14,118,846,464 tracked device
weight bytes total, with native KV separately accounted. Candidate cold
loading takes 17.897 s; combined process peak host RSS is 10,341,556 KiB.
The GPU snapshots also include the idle official model and driver/library
storage. Do not label the combined process as candidate-only resident or
peak memory. All tracked bytes return 0 to 0 in quality, and the GPU is empty
after both HTTP processes exit.

At K=2048 the existing compound consumer remains at 32 registers, 1536 dynamic
shared bytes and two block barriers, no reported stack/local storage.
These are static resources, not measured occupancy, stalls or DRAM traffic.
All 120 measured GPU boundaries across the two HTTP comparisons show the
same owned process pair per run, P0, 1113 MHz SM/3004 MHz memory. No build,
test, MemPalace operation or other GPU workload overlaps primary timing.

## Validation And Reproduction

Runtime tests cover head64 and biased/split-half-RoPE head128 full-forward
bit equality, unchanged split owners, warm allocation-free reuse, model-local
selection, reversed source/unsupported-row rejection, one-byte-short budget
and partial upload cleanup. New runtime 2, primitive 2 and private ownership 2
tests pass with cuBLAS on and off. Attention 11, forward 20, projection 2 and
MLP 1 pass; optional graph 4 and old opt-in FFN 8 tests remain ignored in that
regression invocation. The existing model-quality hardware prefix diagnostic
explicitly exercises the compound separately and passes.

Example CPU suites pass 47 tests, with 9 hardware tests ignored in that
invocation. Non-CUDA warning-as-error suite passes 218 tests and the non-CUDA
example check passes 19. CUDA/server all-target Clippy, formatting and diff
checks pass. Independent audits validate source/cache identities, original
compiled words, output, dispatch, memory, pairs, clocks and process isolation.
All 21 deliberately corrupted records are rejected. JSON round-tripping
changes one cold preparation value by 1.82e-12 ms; the audit uses the existing
1e-9 metadata round-trip tolerance, never a changed numerical/performance gate.

Clean measured source is `63240c52b64e06870922c8b6c0cbae8b0fe2e6f4`.
Release `model_quality` SHA-256
`9a659afbdfbf02249588b9669b42f14e18cc86266ef24227b4f05ad0de1738d3`;
`compare_ollama` SHA-256
`eb6dc69f94a2e67ecd62bc07a02a12e47c26067236ac6f7df40e2c4f78b5b86f`.
NVCC 12.8.93, `-O3`, sm_52 SASS/compute_52 PTX, no fast math.

Build both examples in release with `--features cuda,server`. Source dev-env,
restore `.environment` from the prior packed-projection quality report and
use the same model path for reference and candidate. Quality arguments:

```text
--q8-runtime-candidate joint-shared-scales-128
--q8-runtime-control qualified-packed-projection-sgemm
--q8-runtime-prefill dequant-sgemm
--q8-runtime-attention packed-warp-256
--context-tokens 1024 --split all
```

For `compare_ollama`, replace the candidate flag with `--compare-q8-runtime`,
provide the completed new `--q8-runtime-quality-report`, owned Ollama PID/log
and a new output path. Incremental comparison omits `--time-ollama`; official
comparison includes it. The archived orchestration records complete commands.

Artifacts: [quality](2026-10-27-maxwell-compound-runtime-quality.jsonl),
[quality audit](2026-10-27-maxwell-compound-runtime-quality-audit.json),
[incremental HTTP](2026-10-27-maxwell-compound-runtime-http.jsonl),
[HTTP audit](2026-10-27-maxwell-compound-runtime-http-audit.json),
[official HTTP](2026-10-27-maxwell-compound-runtime-ollama-http.jsonl),
[official audit](2026-10-27-maxwell-compound-runtime-ollama-http-audit.json),
[quality audit program](2026-10-27-maxwell-compound-runtime-quality-audit.jq),
[HTTP audit program](2026-10-27-maxwell-compound-runtime-http-audit.jq),
[instruction audit](2026-10-27-maxwell-compound-runtime-sass-audit.json),
[corruption controls](2026-10-27-maxwell-compound-runtime-negative-controls.sh),
[orchestration/logs](2026-10-27-maxwell-compound-runtime-orchestration.tar.gz),
and [checksums](2026-10-27-maxwell-compound-runtime.sha256).

The independent [perfect-drafter verifier](2026-10-27-maxwell-exact-verification-screen.md)
remains closed, no EAGLE integration. Any next optimization still needs a
new avoided-work hypothesis, bounded scope and existing gates. This result
does not reopen losing kernels or authorize a compound parameter search.
