# Prepared Q8 Same-Prefix Branch

## Finding

The archived down/vocabulary candidate's token-310 branch reproduces on clean
`6ff0f63`. It follows the hidden vector, not the choice of vocabulary head.
Both heads choose native token 1008 from native hidden state and candidate
token 5815 from prepared hidden state. Forward drift crosses a narrow
top-two margin. This does not establish harmless rounding or a kernel bug,
and it does not qualify the historical +12.53% timing lead for promotion.

The later [native-order correction and fresh HTTP screen](2026-10-25-q8-native-order-http.md)
remove this branch while preserving the original acceptance rules. The
pre-fix measurements below remain historical evidence, not current status.

## Identity and Scope

The release revision is `6ff0f630439e1cd431ff45bbdd232763e885012d`, executable
SHA-256 `bf3e9bb01cc9f97656beff72e7a3c5840cdea5cc463b4c788f69713e21c44557`.
Qwen2.5-3B Q8_0 source bytes are unchanged, SHA-256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
Rust 1.95.0, NVCC 12.8.93, g++ 14.3.1 and the selected C11 flags remain:
CUDA `-O3`, sm_52/compute_52, no fast math, native Q8 decode stream, fused
QKV/MLP, shared-activation native vocabulary argmax, head128 split8/256.

The historical [HTTP source report](2026-10-25-q8-down-vocab-http.jsonl),
SHA-256 `c7e1f14b0137197be9eb296888299813a7d124357611cb6a4fce2e59a3171259`,
contains both verified actual traces. Model, raw prompt/IDs, context 1024,
backend/family settings and base environment are checked before forward.
Explicit preparation `off` is normalized against the older omitted-off
environment entry; no other execution mismatch is accepted.

Two resident models execute serially on the idle Tesla M40, with no other
inference, test, build or MemPalace work during the probe. The candidate
prepares only down/vocabulary using the existing split layout. Dense KV,
36 active layers, full context and original absolute positions remain.
This is an untimed diagnostic, not another HTTP trial or a geometry search.

## Matched History and Head Intervention

Replay the recorded 65-token prompt followed by native committed IDs through
generated step 316. Both models consume exactly the same prefix; predictions
are observed but never fed back. Future target IDs are used only for scoring.
All 317 native predictions match the archived native trace. Prepared
predictions match their historical trace through its first branch. Beyond
that point, the historical prepared continuation has another prefix and is
not used as an expected answer.

Twelve sampled steps are 0/128/256 and 308-316. For each, read the completed
hidden vectors and recompute a two-by-two matrix of vocabulary heads, without
new forward work or KV mutation. Hidden bits and session positions remain
unchanged. Recomputed own-head greedy predictions match their original step.

The native production greedy API uses a specialized fused head. Full generic
logits use another reduction path, so their statistics are explicitly named
generic. In this run, all 48 sampled head interventions agree between actual
greedy and generic top token; this observation is not assumed universally.

At generated step 310, absolute prefix length 375:

| Completed Hidden Vector | Native Actual Head | Prepared Actual Head | Both Generic Top Tokens |
| --- | ---: | ---: | ---: |
| Native forward | 1008 | 1008 | 1008 |
| Prepared forward | 5815 | 5815 | 5815 |

| Generic Statistic | Native Own Path | Prepared Own Path |
| --- | ---: | ---: |
| Expected native token 1008 rank | 1 | 2 |
| Token 1008 logit | 25.423492432 | 25.422340393 |
| Top token | 1008 | 5815 |
| Top logit | 25.423492432 | 25.422803879 |
| Top-two margin | 0.000270844 | 0.000463486 |

The expected token's signed lead changes from +0.000270844 to -0.000463486.
For the same native hidden vector, changing the head gives only 3.815e-6
maximum / 3.568e-7 mean absolute generic-logit difference. Changing the
hidden vector under the native head gives 0.004250050 maximum / 0.000473203
mean difference. Hidden-vector maximum / mean difference is
0.012268066 / 0.000970519 over 2048 values. The two-by-two intervention,
not merely the larger error number, identifies forward state as sufficient
to change this greedy choice.

All sampled whole-path top-ten sets overlap 10/10. On forced native history,
step 310 is the only actual native/prepared disagreement among 317 steps;
311-316 agree again. The archived free-generation continuation's 199 differing
positions are therefore not 199 independent same-prefix numerical failures.
The first branch changes subsequent text and input history.

## Next Numerical Check, Not Promotion

Only prepared down differs inside forward; vocabulary is downstream of the
captured hidden vector. This narrows investigation to down projection and its
layerwise propagation. It does not prove the precise instruction-level cause.
The source shows both native coltile4/split4 and separate-plane consumers
using eight-value tasks and a block reduction, but native uses 256 threads
while the model cache currently selects 128 for separate planes.

The next bounded check is pointwise native-order parity using the **existing**
4-column/256-thread separate launch on identical inputs. Compare FP32 bits
against the selected native decode kernel at Qwen down dimensions. Do not
add a layout/geometry sweep, weaken the trace gate, suppress stopping or
promote the candidate from coherence alone. Any dispatch correction still
needs fresh same-prefix and full-workload validation.

### Existing-Launch Pointwise Check

The added CUDA regression now checks the native coltile4/split4 kernel against
both existing split-plane launches on identical non-binary activations and
synthetic signed Q8 payloads. These are Qwen-shaped correctness inputs, not
new measurements of real-model speed or quality. Rust tests use the dev
profile; CUDA retains its normal `-O3` build.

| K / N | 128-Thread Different FP32 Outputs | Maximum Absolute Difference | 256-Thread Different FP32 Outputs |
| --- | ---: | ---: | ---: |
| 2048 / 33 | 30 / 33 | 1.90735e-6 | 0 / 33 |
| 11008 / 2048 | 1893 / 2048 | 2.47955e-5 | 0 / 2048 |

The existing 4-column/256-thread separate kernel matches every native output
bit at both shapes, including the output-column tail. Payload preservation
alone did not imply bitwise native arithmetic under the old 128-thread launch.
Both separate-projection tests pass with cuBLAS on/off, as does CUDA/server
all-targets Clippy. See [on](2026-10-25-q8-native-order-parity-on.txt) and
[off](2026-10-25-q8-native-order-parity-off.txt) raw logs and the regression in
`tests/q8_separate_projection.rs`. Model-cache dispatch still uses 128 until
a separate correction is implemented and validated; no speed claim follows.

## Setup, Memory and Validation

Candidate preparation takes 9697.245 ms within 39388.540 ms setup through
identity checking. These single observations are not a cold-load comparison.
Each model uploads 3,279,519,744 bytes; the candidate adds 1,192,935,424
prepared bytes. Native dense KV is 37,748,736 bytes/model and is separate
from the Rust tracker. Diagnostic head/session buffers are RAII-owned and
released before final accounting. Tracked bytes return 0 -> 0; no peak-VRAM
or DRAM-traffic measurement is inferred.

All 21 runner tests pass, including the live compact-Qwen smoke and new
provenance, native/native zero-drift, reset, unchanged input and alias-rejection
checks. Baseline parity passes 40 cases with cuBLAS on and 16 off; the focused
three diagnostic tests pass off as well. Both warning-as-error CI lanes,
non-CUDA/CUDA-server all-targets Clippy, fmt and whitespace checks pass.
An explicit hardware request with the device hidden fails, not skips.

## Artifacts

- [333 raw JSONL records](2026-10-25-q8-prepared-prefix-drift.jsonl), SHA-256
  `be43c5969616765b810e945d9beba3b4e7ad48d10162f18851d1f845dd1d2527`.
- [Derived audit and branch record](2026-10-25-q8-prepared-prefix-audit.json).
- [Release build](2026-10-25-q8-prepared-prefix-drift-build.txt) and
  [probe log](2026-10-25-q8-prepared-prefix-drift-run.txt).
- [Runner validation](2026-10-25-q8-prefix-runner.txt),
  [parity on](2026-10-25-q8-prefix-parity-on.txt),
  [parity off](2026-10-25-q8-prefix-parity-off.txt),
  [focused tests on](2026-10-25-q8-prefix-unit-on.txt),
  [focused tests off](2026-10-25-q8-prefix-unit-off.txt),
  [explicit no-device failure](2026-10-25-q8-prefix-no-device.txt).

Use the [shared runner command](../q8_execution_weights.md#matched-prefix-branch-diagnostic)
with the original model and a new output path. No Ollama/server or download
is needed. Global runtime/defaults and both unmet exact-throughput objectives
remain unchanged.
