# Qualified Packed Runtime Stage Profile

The [freshly qualified packed QKV/output runtime](2026-10-25-maxwell-packed-projection-ollama-http.md)
remains FFN dominated. This profile measures the remaining work, not a new
optimization. Exact HTTP status stays **49.146 versus 43.106 tok/s**;
both original objectives, 53.06 and 1.30x fresh Ollama, remain unmet.

Clean source `be149f69495856a2b7d5e352d2c0b8b21ed06555`, release executable
SHA-256 `4b7f15aa9361a29e42472e7c0194c92863b050ed1c3ab436f2f68514f4ff9cf7`.
This is unchanged from completed quality and fresh official timing; later
changes added only diagnostic tests/docs. Frozen model bytes/environment,
65 prompt IDs, 512 natural greedy tokens, dense FP16 KV/context 1024,
packed-warp-256 QKV/output, split-f32x4-128 FFN, dequant-SGEMM prefix and
all 109 prepared identities remain fixed. Ollama is an untimed oracle only.

## Instrumentation And Stages

Five alternating profiling-off/on pairs measure **49.402298 / 48.865582
HTTP tok/s**, a **1.086420% instrumentation throughput reduction**. All ten
outputs and both actual 512-ID replays match completed quality. Counts,
copies, waits, synchronization and allocation are unchanged. This does not
supersede authoritative unprofiled timing or permit subtracting overhead
to claim an inferred optimization.

Seven profiled sessions emit 21 rows at steps 0/255/511. Excluding warmup
and final actual-ID replay leaves five primary sessions and fifteen samples.
Each row has 400 spans, complete 36-layer core coverage, one vocabulary
projection and absolute token positions 64/319/575. Table values are medians
of per-sample stage totals, with median per-sample shares.

| Stage | Step 0 ms | Step 255 ms | Step 511 ms | Middle share |
| --- | ---: | ---: | ---: | ---: |
| Gate/up/SwiGLU | 8.0487 | 8.0449 | 8.0500 | 41.11% |
| Prepared down | 4.7655 | 4.7708 | 4.7630 | 24.37% |
| Vocabulary | 1.5652 | 1.5627 | 1.5626 | 7.98% |
| Packed QKV | 1.4046 | 1.3883 | 1.3910 | 7.09% |
| Packed output | 1.0134 | 1.0090 | 1.0079 | 5.16% |
| Attention | 0.6357 | 0.7786 | 0.9700 | 3.98% |
| All summed stages | 19.4513 | 19.5706 | 19.7602 | 100% |

QKV/output are smaller than in the previous scalar-load profile, while
FFN dominates more of the remaining work. These are separate diagnostics,
not paired per-stage promotion evidence. Event spans do not measure DRAM
bandwidth or occupancy, nor directly equal unprofiled kernel latency.
The first sampled token includes setup outside summed GPU work.

## Compiled Follow-Up And Reproduction

Prepared down/vocabulary still use the qualified shared-tree projection.
Current source and SASS show eight scalar bounds-checked activation loads
per eight-value task; prepared weights already use packed 64-bit loads.
The archived full control SASS is byte-identical to the earlier reduction
experiment's control, SHA-256
`333013085bd1e550afc35c6bf329183b675be4679e71bea69cdf994ea6673fa1`.
Resource use remains 40 registers and zero reported stack/local storage.
This does not attribute the measured stage cost to those loads.

The [next fixed hypothesis](2026-10-26-maxwell-prepared-activation-hypothesis.md)
tests **activation vector loads only**, retaining the original shared
reduction, already-packed weights, geometry and FP32 association. Do not
combine the below-gate prepared-warp finish or revive rejected FFN variants.
Complete-FFN, stage feasibility, actual-model quality and HTTP gates remain
mandatory before runtime recommendation.

Reproduce using the frozen environment and exact arguments from the
[packed-runtime qualification](2026-10-25-maxwell-packed-projection-ollama-http.md#fixed-setup-and-reproduction),
replacing `--time-ollama` with `--profile-q8-runtime`. Set
`M40LLM_STAGE_PROFILE_REPORT` to a new event JSONL path; default selected
steps are 0/255/511. The quality report and both runtime/control selectors
stay fixed. No CUDA/kernel source was changed for this measurement.

## Evidence And Ownership

- [Off/on HTTP requests and token traces](2026-10-26-maxwell-packed-stage-http.jsonl)
- [Raw CUDA-event spans](2026-10-26-maxwell-packed-stage-events.jsonl)
- [Stage medians](2026-10-26-maxwell-packed-stage-audit.json), using the unchanged [aggregator](2026-10-25-maxwell-selected-stage-audit.jq)
- [HTTP and coverage audit](2026-10-26-maxwell-packed-stage-http-audit.jq) and [result](2026-10-26-maxwell-packed-stage-http-audit.json)
- [Control SASS](2026-10-26-maxwell-packed-stage-prepared-control.sass.gz) and [resources](2026-10-26-maxwell-packed-stage-prepared-resources.txt)
- [Run log](2026-10-26-maxwell-packed-stage-profile.log.gz), [oracle log](2026-10-26-maxwell-packed-stage-ollama.log.gz) and [checksums](2026-10-26-maxwell-packed-stage-profile.sha256)

All twenty primary boundaries show P0 and 1113/3004 MHz, only benchmark
PID 2372464 and idle owned oracle runner 2373330. No inference workloads,
builds, tests or MemPalace operations overlap timing. Session 68902 exits
zero after actual-ID verification. Owned oracle server 2370706/session
64656 is stopped; the final GPU compute-process list is empty. Offline
negative checks reject missing packed dispatch and per-layer event coverage.
