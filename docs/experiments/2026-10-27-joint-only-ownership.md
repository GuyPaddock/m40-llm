# Joint-Only Ownership Passes The Storage Gate

## Decision

Accept the opt-in storage change. It saves exactly **1,724,645,376 resident
device bytes**, without allocating the 72 redundant split gate/up tensors.
Original uploaded GGUF weights remain intact for packed-prefix SGEMM.
Joint payloads, sampled full-vocabulary logits, all actual 512 generated IDs,
stop reason and all twelve frozen task outputs are unchanged.

Five alternating release HTTP pairs give **50.862963 retained-split versus
50.893176 joint-only tok/s**. The difference is +0.0594%, with a paired 95%
interval of -0.0588% to +0.3253%. This excludes a meaningful regression under
the existing 3% threshold; it is not a demonstrated throughput gain. Neither
variation extension nor fresh confirmation was required. A storage result
does not require a 3% speedup.

The [immutable reference](2026-10-27-compound-reference.md) remains
50.910426 HTTP tok/s versus official Ollama 43.194730. Its 56.153149 target,
evidence, hashes, binaries and settings are unchanged. This comparison is
not progress toward that speed target. Default ownership is still
`retain-split`; runtime KV defaults are unchanged too.

The independently approved [no-old adapter](2026-10-27-no-old-screen.md)
failed its exact valid-cache gate and was removed in `f76d541`, after code
preservation in `db04697`. Both approved experiments are now closed. No new
kernel search, compatibility inference, downloads or default promotion follows.

## Source And Configuration

Implementation is `e83e172`. Final quality and HTTP use clean source
`3e7caeef90434e79a9fa394b306a6d19fe103306`, release `cuda,server`, executable
SHA-256 `3b9cb5785bc44c3cb0feca88a64f5869f8e22f6dd3c6a6c8499463277520f628`.
The model is the frozen Qwen2.5-3B Q8_0 blob with SHA-256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.

Keep the original compound 128-thread consumer, FP32 arithmetic/reduction,
prepared down/vocabulary, packed-warp-256 projections and bounded
dequant-SGEMM prefix. All 137 compiled CUDA function instruction words match
the frozen binary. NVCC 12.8.93, sm_52/compute_52, `-O3`, no fast math and
all frozen environment values remain unchanged. There is no CUDA code change.

The runner installs preparation explicitly: split `gate-up,down,vocabulary`,
6144 MiB, `joint-shared-scales-128`, dense KV context 1024. The raw frozen
environment's execution selector remains `off`; it does not describe the
loader's effective preparation by itself. Only ownership differs between arms.

For normal opt-in loading, the new setting is
`M40LLM_Q8_FFN_OWNERSHIP=joint-only`, alongside the existing compound consumer
and split preparation of `gate-up,down,vocabulary`. Incompatible consumers,
families, shapes and budgets fail rather than allocating missing split copies
or changing kernels. Keep all other qualified settings when reproducing
this measurement; the ownership flag alone is not a performance preset.

## Ownership And Memory

| Component | Retained-split | Joint-only |
| --- | ---: | ---: |
| Logical validated source tensors | 109 | 109 |
| Resident split tensors | 109 | 37 |
| Resident joint pairs | 36 | 36 |
| Original uploaded GGUF bytes | 3,279,519,744 | 3,279,519,744 |
| Split execution bytes | 2,917,580,800 | 1,192,935,424 |
| Joint execution bytes | 1,724,645,376 | 1,724,645,376 |
| All prepared bytes | 4,642,226,176 | 2,917,580,800 |
| GGUF plus every prepared owner | 7,921,745,920 | 6,197,100,544 |
| Device allocation calls during loading | 291 | 147 |
| Dense KV component, separate | 37,748,736 | 37,748,736 |
| Prefix temporary capacity, separate | 90,177,536 | 90,177,536 |

The 144 skipped allocations are the scale/value planes of 72 split tensors.
Joint pairs are built directly from validated logical sources; those split
copies never exist, even temporarily. Every retained down/head identity is
unchanged. The reduction is about 1.606 GiB, not a reduction of the original
GGUF upload. It does not include driver/library allocations.

Solo cold loads at clean `9c4a027`, executable
`5a37540144a34b2f68c859529a2777741b4b120f55dbc21d50cae41e57841d85`, recorded:

| Cold observation | Retained-split | Joint-only |
| --- | ---: | ---: |
| Load wall time | 17,834.655 ms | 17,297.025 ms |
| Split preparation | 10,244.574 ms | 9,691.351 ms |
| Joint preparation | 1,570.527 ms | 1,598.145 ms |
| Tracked setup device high-water | 7,921,745,920 B | 6,197,100,544 B |
| Process VmHWM | 6,999,336 kB | 6,999,296 kB |
| Largest temporary host joint pair | 47,906,816 B | 47,906,816 B |
| Retained host joint-pair payload | 0 B | 0 B |

These are single, allocation-logged observations from separate serial
processes, not a paired cold-speed claim. Later runner changes affect HTTP
requests and shutdown only, not loading or ownership. Logs verify the
tracked high-water values, which equal final tracked weight residency.
`CudaContext` tracking excludes the separately allocated KV component and
CUDA/library storage; do not call these full-process peak VRAM measurements.
The existing setup-host payload estimate is 6,895,608,736 B in both arms,
not measured RSS. Keeping original sources means host setup memory is
essentially unchanged. Both load processes exit successfully after cleanup.

## Correctness And Performance

The new [quality audit](2026-10-27-joint-only-quality-audit.jq) separates
logical sources from physical split/joint owners. It does not reuse or
weaken the immutable 109-resident-tensor audit.

- All 36 actual joint payload hashes match, including ordered gate/up roles.
- All 512 actual greedy IDs, 65 prompt IDs, output and length stop match.
- Full 151,936-vocabulary logits at steps 0/1/8/32/128/256/511 are bit-identical.
- All twelve frozen outputs/scores match: seven dense-valid passes and five
  unchanged reference-inconclusive rows, not twelve newly passing tasks.
- Ordinary launches, allocation/free counts, copies, waits, synchronizations,
  prepared routing and packed-prefix behavior match the frozen warm request.
- Both ownership models are simultaneously live but execute serially; all
  tracked allocations return to zero after session/model/server teardown.
- Ownership tests cover invalid combinations, one-byte-short budget,
  allocation-free warm reuse and injected partial-upload failure cleanup.

The GPU is isolated for timed requests at P0, 1113 MHz SM and 3004 MHz memory.
Each server has an excluded full 512-token warmup. HTTP timing spans request
start through the complete buffered response, before parsing. No builds,
tests, profiling or MemPalace work overlaps these measurements. Every timed
request preserves output, 65/512 counts, length stop and ordinary dispatch.
Post-timing greedy replays verify all 512 actual IDs independently on both models.

| Five-pair median | Retained-split | Joint-only |
| --- | ---: | ---: |
| HTTP wall time | 10,066.264 ms | 10,060.288 ms |
| Reported prefill | 488.868 ms | 485.956 ms |
| Reported decode | 9,531.999 ms | 9,528.324 ms |
| HTTP tok/s | 50.862963 | 50.893176 |
| MAD / median | 0.06094% | 0.02724% |

Phase medians are independently aggregated; their sum need not equal HTTP
wall time. Timing uses the unchanged paired bootstrap, 10,000 resamples and
seed 0. The positive-speed promotion flag is false, as expected for a neutral
storage change; the separate storage non-regression gate passes. Do not
pool this comparison with prior compound/Ollama pairs.

## Evidence And Reproduction

- [Final quality](2026-10-27-joint-only-final-quality.jsonl),
  [quality audit](2026-10-27-joint-only-final-quality-audit.json), and
  [unchanged CUDA entries](2026-10-27-joint-only-final-sass-audit.json).
- [HTTP trials](2026-10-27-joint-only-http.jsonl) and
  [independent HTTP audit](2026-10-27-joint-only-http-audit.json).
- [Control load](2026-10-27-joint-only-load-control.jsonl),
  [candidate load](2026-10-27-joint-only-load-candidate.jsonl), and
  [derived measurements](2026-10-27-joint-only-measurement-summary.json).
- [28 corruption controls](2026-10-27-joint-only-audit-checks.jsonl) and
  [their executable checks](2026-10-27-joint-only-audit-checks.sh).
- [Raw logs and assembly](2026-10-27-joint-only-orchestration.tar.gz) and
  [artifact checksums](2026-10-27-joint-only.sha256).

Preserve the [cold-counter attempt](2026-10-27-joint-only-cold-attempt.md)
and [missing-metrics warmup](2026-10-27-joint-only-http-attempt.md) separately.
They are qualification-tool errors, not ownership or performance results.
Neither is pooled with the completed run. The first restored the exact
qualified warmup; the second explicitly requested HTTP metrics. No runtime
or acceptance gate changed to fix them.

The archive includes all validation logs: attention grid 11; prepared
projection 4; compound 3, including joint-only head64/head128; packed-prefix 5;
private execution/ownership 27; compound 3 with cuBLAS disabled; full-forward 2;
non-CUDA warning-as-error suite; all-target CUDA/server Clippy; and the
real-HTTP request/shutdown test. Early toy-forward failures are retained:
tests needed explicit dense references for head4 and completion before
reading asynchronous outputs. Fixes `e3450cf` and `86ee034` change tests only.

Reproduce quality and HTTP using clean measured source, a release build and
the exact manifest environment. Use new output paths; reports never overwrite.

```bash
source scripts/dev-env.sh
while IFS=$'\t' read -r key value; do export "$key=$value"; done < <(
  jq -r '.configuration.environment | to_entries[] | [.key,.value] | @tsv' \
    docs/experiments/2026-10-27-compound-reference.json
)
cargo build --release --features cuda,server --example joint_only
target/release/examples/joint_only --phase quality --model "$model" \
  --output /tmp/new-joint-quality.jsonl
target/release/examples/joint_only --phase http --model "$model" \
  --quality-report /tmp/new-joint-quality.jsonl --output /tmp/new-joint-http.jsonl
bash docs/experiments/2026-10-27-joint-only-audit-checks.sh \
  /tmp/new-joint-quality.jsonl /tmp/new-joint-http.jsonl
```

Run with approved hardware access and no competing workload. Cold observations
use `M40LLM_ALLOC_LOG=1 /usr/bin/time -v` with `--phase load`, separately for
`--ownership retain-split` and `--ownership joint-only`.

The measured executable and clean source archive are retained read-only in
ignored `local-artifacts/qualified-references/joint-only-3e7caee/`. The archive
SHA-256 is `5408c3b99a1421b74d0c06f6cbf69e5a4bf242a2c50f900657b61c4cf2dfd076`.
The earlier load executable is separately retained under
`joint-only-evidence-9c4a027/`. Third-party notices remain with the sources.
All 34 immutable-reference artifact/model/binary/archive checksums still pass.
Local copies are not an off-machine backup.

Next requires a new authorization. Keep the metadata-only compatibility
matrix pending actual artifacts and separate testing approval. Any dense-speed
kernel still needs an independently supported structural hypothesis and fixed
stop criteria; do not reopen failed FFN experiments or the rejected verifier.
