# Maxwell Execution Plan Evidence Audit

## Scope And Current Decision

This audits the workspace's
[active plan](../../../M40_LLM_MAXWELL_Q8_EXECUTION_PLAN.md) against committed
reports and current comparison code at `61f5e10`. It does not declare the
objective complete. The current exact runtime is qualified at **49.146449
HTTP tok/s**, versus fresh official Ollama **43.105570**. Both required
thresholds, **53.06** and **56.037241 HTTP tok/s**, remain unmet.

The execution guide still called the historical 35.724705 down/head-only
configuration current. Correct that guide without rewriting dated reports.
The 41.331 M09 checkpoint is historical too. Current evidence pins the
split-f32x4 FFN, packed-warp-256 QKV/output, original prepared down/head,
dequant-SGEMM prefix, dense KV and all 109 prepared identities.

## Requirement Ledger

| Plan requirement | Authoritative evidence | Assessment |
| --- | --- | --- |
| Pin official Ollama/backend/source/patches; distinguish source from executed paths | [M01 identity and launch report](2026-10-25-maxwell-q8-dispatch.md), per-file hashes and observer artifacts linked there | Verified for the pinned benchmark version; not a claim about newer releases |
| Compare arithmetic, layout, launch geometry, scratch and synchronization | M01 source-to-runtime table, [current runtime profile](2026-10-26-maxwell-packed-stage-profile.md), [qualified runtime setup](2026-10-25-maxwell-packed-projection-ollama-http.md) | Historical and current configurations must stay separately labeled |
| Quantization, MMVQ and prefill cuBLAS dispatch observed without contaminating official timing | M01 launch records and completeness audit | Verified; whole-process launch counts are not per-token durations |
| Inspect generated sm_52 instructions/resources and retain licenses | M01 extracted PTX/offline assembly/license; subsequent experiment SASS/resources | Verified within stated JIT limitations; no achieved-occupancy or measured-DRAM claim |
| Shared real FFN tensor/input benchmark, layers 0/17/35, genuine rows 1/4 | [M02 warm baseline](2026-10-25-maxwell-ffn-baseline-warm.md), frozen fixture/capture identities | Verified; no four repetitions relabeled as matrix rows |
| Bounded FP32 packed-load/reduction ablation | [M03 screen](2026-10-25-maxwell-fp32-ffn-screen.md) | Completed; later qualified changes have separate gates |
| Q8_1 block contract, reuse and included quantization cost | [M04 numerical screen](2026-10-25-maxwell-q81-ffn-screen.md), Q8_1 primitive/owner tests | Completed primitive screen; full-model evidence needs the correction below |
| Quantize, gate/up/SiLU, isolated down and complete FFN measured separately | M02-M04 raw stage records and audits | Verified; isolated down does not authorize replacing the prepared control |
| Independent sequential/native-packed/dequant-SGEMM prefill | [Corrected independent HTTP](2026-10-25-maxwell-completed-http.md), [prefill quality](2026-10-25-maxwell-completed-prefill-quality.md) | SGEMM qualifies; native packed loses on timing |
| Bounded reusable dequantization storage, all per-prompt conversions included | Prefill quality/HTTP dispatch and memory records | 90,177,536-byte capacity, below 128 MiB; no all-model FP32 cache |
| Immutable model choices, typed ownership, no silent fallback or per-token workspace allocation | [Runtime contract](2026-10-25-maxwell-ffn-runtime.md), affected owner/forward tests and current runtime config code | Verified by linked implementation/test checkpoints; diagnostics remain explicit |
| FP32 actual512 equality, frozen tasks and reference-prefix full logits | [Current packed-projection quality](2026-10-25-maxwell-packed-projection-runtime-quality.md) | Verified: actual512 IDs and all twelve outputs match; seven dense-valid rows, five inconclusive; sampled logits have zero difference |
| Q8_1 reconstruction, full-model drift, divergence, task quality and held-out loss/PPL | [Historical M08 numerical report](2026-10-25-maxwell-runtime-quality.md) | Primitive evidence remains valid; full-generation/control conclusions predate the prefix-completion repair and need refreshing |
| Independent decode/prefill qualification before combined timing | Corrected independent reports, [combined quality](2026-10-25-maxwell-completed-combined-quality.md), subsequent incremental reports | Verified for the retained exact path |
| Actual model/prompt IDs, greedy natural stopping, fresh KV, release HTTP timing | [Fresh official comparison](2026-10-25-maxwell-packed-projection-ollama-http.md), raw records and audit | Verified for the qualified checkpoint, not rerun by this audit |
| Five pairs, fixed gain/variation/CI gates and fresh borderline confirmation | Fresh official comparison and separate twenty-pair packed-projection confirmation linked there | Verified; no pooling or microbenchmark-only promotion |
| CUDA on/off, relevant parity/ownership/forward/prefill, non-CUDA warnings, Clippy/fmt | Implementation reports, latest [role-parallel validation](2026-10-26-maxwell-role-parallel-ffn-screen.md) | Relevant recorded checks pass; ignored hardware tests remain identified as ignored |
| Atomic history, preserve failed code, raw evidence and diaries | Git commits, experiment artifacts, perf history and MemPalace checkpoints | Ongoing discipline, not an optimization or completed speed objective |
| Final exact-output targets | Fresh official paired HTTP data | Incomplete; numerical/approximate/scheduler changes cannot replace these targets |

## Bounded Numerical Evidence Repair

The original Q8_1 reports carry no `prefix-decode-completed-v1` contract.
The later ownership investigation changes the actual sequential continuation,
so the historical numerical divergence step must not be presented as current.
Keep old reports unchanged with their source identity and limitations.

Refresh only the two already-tested Q8_1 geometries, 128 and 256 threads,
using the existing `model_quality` tool and frozen corpus. Both reference
and candidate use original Qwen Q8_0 bytes, dense context 1024, native
sequential prefill with completed prefix work, unchanged native QKV/output
and the same 37 prepared down/vocabulary identities. The control is the
explicit corrected `native-sequential` configuration, not the current best
packed runtime. Changing both FFN and projection/prefill strategies would
confound the numerical comparison.

Record actual production browser IDs/output and natural stop counts,
first divergence, forced-reference-prefix full-logit drift, frozen task
scores and token-weighted calibration/held-out loss/PPL. Verify reference
IDs against the corrected control in the completed packed128 quality report.
Require actual Q8_1/quantizer dispatch, immutable prepared identities,
scratch bytes and final cleanup. Do not infer activation reconstruction
quality from model task scores; retain M04's explicit primitive measures.

Run the existing Q8_1 parity/ownership tests and relevant runner tests before
clean release evaluation. Request escalation, inspect competing GPU jobs,
run geometries serially, and avoid all builds/tests/other inference or diary
calls during GPU evaluation. The output paths must be new. These are
quality diagnostics, not paired timing experiments; do not report speedups
from their resident durations or run official Ollama throughput unnecessarily.

Do not tune kernels, prompts, rubrics, stop conditions or tolerances if the
results change. Dense failures remain inconclusive. A task regression blocks
recommendation; even a clean task result cannot make a Q8_1 candidate eligible
for the original exact-output objective. No runtime/default/policy promotion
or speculative/sparse/approximate-attention substitution follows.

## Atomic Follow-Up

1. `docs(research): audit Maxwell plan evidence`: ledger, corrected guide and
   fixed scope before evaluation. No CUDA/inference behavior change.
2. `docs(perf): refresh Q8_1 quality checkpoint`: clean release reports for
   the two fixed geometries, independent artifact audits, raw outputs/logs,
   numerical conclusions and updated status; depends on 1 and relevant tests.
3. Continue the exact execution objective from independent remaining-cost
   evidence. Do not rescue the closed ownership/cache/layout experiments or
   treat this evidence repair as satisfying either unmet speed target.
