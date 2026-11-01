# Fixed Four-Row Shared Reader Passes The Operator Gate

## Decision And Scope

The single [approved consumer](2026-10-28-four-row-operator-plan.md) passes
the unchanged >=5% complete-FFN throughput gate at all three frozen layers,
against both contemporaneous controls, with positive paired confidence
bounds. No extra geometry, scale format or parameter variant was tried.

This is an **operator result**, not a measured full verifier, HTTP speedup,
drafter qualification or runtime promotion. Continue only with the authorized
measured-cost feasibility update. The immutable 50.910426 HTTP tok/s compound
reference, preferred joint-only ownership and all runtime defaults remain
unchanged. Neither exact-dense HTTP target is established by these results.

## Contemporaneous Results

Release CUDA-event medians in milliseconds for a complete four-row FFN:

| Frozen layer | Four one-row compound calls + four one-row down calls | Independent-row compound + four one-row down calls | Shared-read candidate + four one-row down calls | Gain vs serial | Gain vs independent |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.334438 | 1.314035 | 0.918218 | +45.329% | +43.107% |
| 17 | 1.336550 | 1.315536 | 0.919930 | +45.288% | +43.004% |
| 35 | 1.334976 | 1.315370 | 0.919283 | +45.219% | +43.086% |

These percentages are throughput gains, not latency reductions. Complete
latency falls by about 31.1-31.2% against serial and 30.1% against independent.

Paired 95% bootstrap throughput-gain intervals, 10,000 resamples, seed zero:

| Layer | Versus serial | Versus independent |
| ---: | ---: | ---: |
| 0 | +45.138% to +45.488% | +43.007% to +43.190% |
| 17 | +45.170% to +45.455% | +42.860% to +43.094% |
| 35 | +44.932% to +45.305% | +42.818% to +43.175% |

Each layer has five contemporaneous trials. Arm order rotates/reverses;
stage order alternates. Every timed stage repeats ten times. Each arm warms
with at least 1000 ms of complete-FFN GPU work before primary trials. Maximum
throughput MAD/median across the three arms is 0.0365%, 0.0336% and 0.0578%
at layers 0/17/35. No variation extension or borderline confirmation condition
is met. Do not collect extra trials until favorable; the prescribed screen
is complete.

Isolated gate/up/SwiGLU and original down medians:

| Layer | Serial gate/up | Independent gate/up | Shared-read gate/up | Original down, serial / independent / candidate |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.812483 | 0.792032 | 0.395238 | 0.523216 / 0.523242 / 0.523053 |
| 17 | 0.812102 | 0.792058 | 0.395821 | 0.524310 / 0.524282 / 0.523942 |
| 35 | 0.812163 | 0.791946 | 0.395571 | 0.523418 / 0.523136 / 0.523526 |

The candidate's measured gate/up time is 48.65-48.74% of four qualified
one-row gate/up calls. Down execution is unchanged, not a new batched-down
kernel. Separate stage medians need not sum exactly to complete FFN.
Raw host enqueue/completion times cover all ten repeats; raw GPU times are
normalized per four-row operator. Do not compare those differently normalized
fields without accounting for the repeat count.

All gate calls precede the four down calls in each arm. This is not four
autoregressive target steps, nor an experiment in changing row dependencies.
Historical four-row FFN results using generic multi-row down have a different
control contract and are not the denominator here.

## Correctness, Resources And Ownership

The [separately committed correctness checkpoint](2026-10-28-four-row-correctness.md)
passes before timing. Primary observations repeat the same exact checks:
gate/up/SwiGLU/completed-down bits match across all arms at layers 0/17/35;
ordinary/capture execution matches, CPU errors satisfy the original tolerance,
and guards, immutable inputs and actual joint/down payload checks pass.
No sampled logits, whole-generation equality or twelve-task claim is inferred
from an isolated FFN result.

Exactly one new kernel uses four rows, D=2048, H=11008, one 128-thread CTA
per output column, 36 registers/thread, 4608 dynamic shared bytes, no reported
stack/local storage and no additional global weight representation. All 137
original compiled controls retain identical instruction words, including
compound gate/up and original prepared down. There is no geometry rescue.

Every stage's enqueue counters show only expected kernel launches. Complete
FFN uses eight launches for serial and five for each four-row arm. All arms
issue exactly four original `gemm_f32xq8_separate_f32` down calls, with the
qualified four-column/256-thread single-row configuration. No timed enqueue
allocates, frees, copies, uploads, synchronizes, waits between streams or calls
cuBLAS. CUDA-event completion occurs after the repeated stage and outside the
enqueue-counter snapshot.

Uploaded original sources occupy 3,279,519,744 bytes; the unchanged 37 prepared
down/vocabulary tensors occupy 1,192,935,424 bytes. A benchmark-local joint pair
adds 47,906,816 bytes in every arm, with no redundant split gate/up planes.
Guarded scratch is 594,240 bytes. Tracked residency is 4,520,956,224 bytes;
each layer returns to 4,472,455,168, and model teardown returns to zero. These
are tracked allocations, not total driver VRAM, host RSS or runtime-all-layer
residency. The production joint-only preference is not changed.

The GPU is isolated at all 30 primary trial boundaries, with only the owned
benchmark PID, P0, 1113 MHz SM and 3004 MHz memory. No build, test, other
inference or MemPalace work overlaps primary measurement. There are 135
primary stage intervals across 15 trials, plus nine separately reported
warmups. Raw timestamps retain serial interval order.

## Artifacts And Reproduction

- [Primary raw JSONL](2026-10-28-four-row-screen.jsonl) and
  [audit](2026-10-28-four-row-screen-audit.json).
- [Correctness observations](2026-10-28-four-row-correctness.jsonl) and
  [compiled-control/resource audit](2026-10-28-four-row-compiled-audit.json).
- [Negative controls](2026-10-28-four-row-negative-controls.jsonl), exercised
  by the [corruption runner](2026-10-28-four-row-negative-controls.sh).
- [Build/test/stderr/SASS/resource archive](2026-10-28-four-row-validation.tar.gz).

Clean measured source is `de0481dbdf8345a110eb9e36082c8a860e33b727`; implementation
is `cac4c29`. The unchanged release executable SHA-256 is
`a19efd16c9a46790776729dc4bc83e4713b6dd1a535569a7f36a1a994a8cebf1`.
NVCC 12.8.93 uses the existing `-O3`, sm_52/compute_52 flags without fast math.
The original model hash is
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.

Source archive SHA-256 is
`292fbf0f5401cb3212e59f08b0ff589b2fd2035e1ac64f233122a02942d45d53`.
Read-only local source and executable copies are retained under
`local-artifacts/qualified-references/four-row-de0481d/`. This is an
operator-only checkpoint, not a qualified runtime. The immutable reference and its
measured binaries are not modified.

```bash
source scripts/dev-env.sh
mapfile -t frozen_env < <(jq -r '
  select(.kind=="ffn_capture_manifest") |
  .execution_configuration.environment | to_entries[] |
  "\(.key)=\(.value)"' docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl)
env "${frozen_env[@]}" target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report target/research-fixtures/m02-ffn-capture.jsonl \
  --ffn-four-row-reuse --rows 4 --iterations 10 --trials 5 \
  --output /tmp/new-r4-screen.jsonl
```

Use the preserved source/build settings and an isolated M40; this command is
not authorization for additional searches or a runtime-default change. The
separate feasibility assessment must still account for the rest of the model,
remaining materialization, attention, logits, KV transactions and drafting.
