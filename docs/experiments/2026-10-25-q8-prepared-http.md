# Prepared Q8 HTTP Gate: 2026-10-25

## Decision

No prepared candidate passes the combined release timing and actual-token
parity gate. Keep preparation off and retain the selected C07 native-Q8 plus
split8/256 exact-attention benchmark configuration. Global runtime/KV defaults
do not change. This closes the bounded C10A HTTP screen, not the overall goal.

| Prepared families | Native HTTP tok/s | Prepared HTTP tok/s | Gain | 95% paired-bootstrap gain interval | Actual 512 IDs equal | Decision |
| --- | ---: | ---: | ---: | --- | --- | --- |
| All | 31.946 | 11.433 | -64.212% | [-64.270%, -64.185%] | Yes | Reject timing |
| QKV, output, down, vocabulary; native MLP | 31.907 | 29.765 | -6.714% | [-6.730%, -6.521%] | Yes | Reject timing |
| Down, vocabulary only | 31.847 | 35.837 | +12.529% | [12.430%, 12.796%] | No | Timing lead; reject parity |

Each row has its own five contemporaneous native/prepared pairs in alternating
order, following one full warmup per model. The last subset was bounded in
advance after the first two negative results: it isolates the strongest C09
down/vocabulary candidates while retaining native fused QKV/MLP and native
output projection. No new kernel geometry or layout search was opened.

Control/prepared MAD divided by median, in percent, is 0.0165/0.0461 for all,
0.0435/0.0173 for native MLP, and 0.0660/0.1594 for down/vocabulary. No row
requires the protocol's variation extension or borderline confirmation. The
timing threshold is 3%, with 10,000 paired bootstrap samples and seed zero.
Additional trials are not used to turn a negative result into a positive one.

The historical controlled Ollama median is still 43.035 tok/s. These runs use
Ollama only as an untimed oracle, not as the timed control. Neither the 53.06
tok/s objective nor the binding 1.30x-Ollama objective of 55.9453 tok/s is met.
Even the parity-failed timing lead is below both. Do not present it as the new
qualified best result or an exact-output speedup.

## Workload and Isolation

- Same Qwen2.5-3B-Instruct Q8_0 file in both m40 models and Ollama. SHA-256 is
  `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
- The browser-loading user question, canonical Qwen system message, raw
  ChatML assistant prefix, and all 65 prompt IDs match the tokenizer oracle.
  Rendered prompt SHA-256 is
  `3005a73b43160b265f25b4545853c3fdab0ab75014d315c701ec6bef62648f9a`.
- Every warmup and screen request generates 512 IDs with a natural length
  stop, greedy top_k=1, seed zero, fresh dense FP16 KV, and context 1024.
  No stop suppression or continuation padding is used. Dense KV is 37,748,736
  bytes per model, plus the separately accounted attention workspace.
- The primary interval is HTTP send through receipt of the complete buffered
  body. Response parsing, boundary GPU queries, output hashing, setup, smokes,
  and actual-ID companion runs are outside that interval.
- Both m40 models may be resident in one process. Requests are awaited
  serially. The minimum gap between recorded request intervals is 87/88/88 ms
  for the three comparisons. No GPU tests, builds, or other inference overlap
  timed requests. The only other compute PID is the idle isolated Ollama
  runner; its oracle work completes before m40 measurement starts.
- Tesla M40 24GB, UUID `GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`, sm_52,
  24 SMs, driver 580.159.04. Boundary snapshots record clocks, temperature,
  power, utilization, and processes; clocks were not locked. Screen boundaries
  show 1113 MHz SM / 3004 MHz memory clocks and 47-76 C temperatures.
- Release Rust 1.95.0, CUDA/NVCC 12.8.93, CUDA `-O3` with sm_52 SASS and
  compute_52 PTX. Full manifests record executable hashes, host compiler,
  environment, source revision, and clean initial worktrees.

The all-family run uses `26bc810`. `e74ed87` corrects only quality-smoke prompt
formatting and is the revision for the other two runs. All use the same model,
runtime kernels, backend flags, and measured browser prompt. Runtime loading
and preparation happen before requests; no shared dispatch flag changes
between control and prepared calls.

## Timing Breakdown

Independent medians in seconds; internal component medians need not sum to
the median internal total or HTTP wall time.

| Candidate | Variant | Prefill | Decode | Internal total | HTTP total |
| --- | --- | ---: | ---: | ---: | ---: |
| All | Native | 1.680 | 14.302 | 15.981 | 16.027 |
| All | Prepared | 5.017 | 39.721 | 44.738 | 44.784 |
| Native MLP | Native | 1.687 | 14.322 | 16.002 | 16.047 |
| Native MLP | Prepared | 1.908 | 15.244 | 17.153 | 17.202 |
| Down/vocabulary | Native | 1.677 | 14.345 | 16.029 | 16.077 |
| Down/vocabulary | Prepared | 1.582 | 12.662 | 14.233 | 14.287 |

Keeping native fused MLP avoids most of the all-prepared regression. This is
whole-path evidence, not an isolated measurement assigning every lost
millisecond to one kernel. The down/vocabulary subset establishes a timing
lead worth preserving, but its differing continuation blocks promotion.

## Quality and Token Divergence

Each model's output is stable across its own warmup and five measured calls.
Untimed companion generations reproduce those outputs and save actual prompt
and generated IDs, not IDs inferred by re-tokenizing decoded text.

All-family and native-MLP continuations match all 512 native IDs. The
down/vocabulary candidate first differs at zero-based generated step **310**,
the 311th generated token: native ID **1008**, candidate ID **5815**. There are
199 position-wise ID mismatches after the branching continuation; this is not
an edit distance or evidence of 199 independent numerical errors.

The native phrase is `HTML file and any other necessary resources.`; the
candidate writes `HTML file and any associated resources (CSS, JavaScript,
images, etc.).`. Both remain coherent, but coherence is not exact-token parity.
Lossless Q8 repacking and existing numerical tests do not establish the cause
of this long-generation branch. Do not label it a quantization change, a
proven kernel bug, or harmless rounding without a matched-prefix diagnostic.
No further subset tuning or changed quality threshold is used here.

The first run's user-only Auto smokes pass arithmetic but fail exact-OK and
config lookup on **both** native and prepared models. Those two cases are
retained as inconclusive, not prepared regressions. `e74ed87` reuses the C07
canonical system-message/raw template for the other two runs. Native and
prepared models then all return `4`, `OK`, and `37`, three dense-valid passes
per configuration. The smoke correction does not change the timed browser
input. These short passes do not override down/vocabulary's trace mismatch.

## Setup and Memory

One cold observation per candidate, not a repeated setup benchmark. Source
uploaded weight payload is 3,279,519,744 bytes per model in every run.

| Prepared families | Extra prepared bytes | Candidate source + prepared bytes | Candidate cold load, s | Hash, s | Preparation including hash, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 3,278,553,088 | 6,558,072,832 | 17.551 | 9.482 | 11.204 |
| Native MLP | 1,553,907,712 | 4,833,427,456 | 17.884 | 10.015 | 10.926 |
| Down/vocabulary | 1,192,935,424 | 4,472,455,168 | 17.476 | 9.829 | 10.543 |

Hash time is included in preparation time; do not add those columns. Cold
loading also includes GGUF loading, native upload, and KV allocation. Setup
records include exact per-tensor identity/layout/bytes and the 6144 MiB extra
payload budget. Keeping native source and prepared weights consumes both.

Post-warmup tracked device payload for **both m40 models combined** is
9,838,141,456 / 8,113,496,080 / 7,752,523,792 bytes. Combined process host
peak RSS is 10,277,096 / 10,276,024 / 10,276,592 KiB. These are not standalone
candidate memory costs. The cache's single-model host setup payload estimate
is 6,895,608,736 bytes and excludes allocator overhead.

Boundary nvidia-smi usage includes contexts, allocator granularity, both m40
models, and idle Ollama. It is not measured peak VRAM. No actual DRAM traffic,
cache-hit, or occupancy counters are available from the earlier CUPTI preflight.
All benchmark/oracle processes were stopped after these measurements; a final
device query reports 0 MiB / 0% utilization.

## Reproduce and Inspect

Use [the complete controlled command](../controlled_benchmark.md) with the
selected C07 attention flags: parallel, partitions=8, threads=256. Keep every
listed Q8 fusion/greedy/stream/backend assignment, dense KV/context 1024,
and diagnostics off. Supply `--compare-q8-execution split` and one of:

```text
--q8-families all
--q8-families qkv,output,down,vocabulary
--q8-families down,vocabulary
```

Use separate new report paths and await each complete process before starting
the next. The input-tiled `mixed` mode was not tested here: dense sequential
prefill and one-token decode do not exercise its rows>=64 tiled dispatch.
This result does not establish a full-model multi-row/verifier speedup.

Read summary, parity, and promotion records directly:

```bash
jq 'select(.kind == "q8_execution_summary" or
           .kind == "q8_token_comparison" or
           .kind == "q8_promotion_decision")' \
  docs/experiments/2026-10-25-q8-*-http.jsonl
```

Raw evidence:

- All-family [JSONL](2026-10-25-q8-prepared-full-http.jsonl) and
  [runner log](2026-10-25-q8-prepared-full-http.txt).
- Native-MLP [JSONL](2026-10-25-q8-native-mlp-http.jsonl) and
  [runner log](2026-10-25-q8-native-mlp-http.txt).
- Down/vocabulary [JSONL](2026-10-25-q8-down-vocab-http.jsonl) and
  [runner log](2026-10-25-q8-down-vocab-http.txt).
- Shared isolated [Ollama backend/oracle log](2026-10-25-q8-prepared-ollama.txt).

Only trailing horizontal whitespace was removed from the copied Ollama log
for repository whitespace checks. JSONL and m40 runner logs are unmodified.

## Checks and Next Task

The family-selector unit passes CUDA numerical/dispatch/ownership tests,
including retained native fused MLP, prepared QKV/MLP/greedy tests, full-layer
forward and attention-grid tests, and cuBLAS-off family/grid/wrappers. Non-CUDA
warning-as-error tests and CUDA/server/non-CUDA Clippy lanes pass. The HTTP
runner adds passing CLI/protocol tests, 8 with CUDA/server and 7 non-CUDA,
plus warning-free Clippy and formatting checks. Both runner revisions remain
in history; raw results are not rewritten to use the later smoke template.

Next is C11's keep/remove decision and exact-kernel checkpoint, preserving
these implementations/results before any removal. Keep the C07 configuration
as the qualified control. The formal C11A offline n-gram feasibility study
follows C11; do not extend this round into another layout/geometry search.
