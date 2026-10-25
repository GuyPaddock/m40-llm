# Paired FFN Benchmark Inputs

M02 in the [Maxwell Q8 investigation](maxwell_q8_execution.md) needs actual
normalized gate/up inputs and completed SwiGLU inputs to down from the same
forward pass. Activation-sparsity calibration stores only the latter and
forbids the qualified prepared down/vocabulary control. Do not enable a
sparsity transform to obtain these observations.

## Artifact Contract

`evaluation::ffn_capture` provides a CPU-only collector and checked JSON
artifact. It has no inference policy, threshold fitting or CUDA dependency.
The shared benchmark consumer remains a separate M02 step.

- A plan binds model, corpus and execution SHA-256 identities, original layer
  count, selected original IDs, dimensions and ordered calibration task IDs.
  The real Qwen checkpoint selects layers 0, 17 and 35, with dimensions 2048
  and 11008. Tests may use smaller complete Q8_0 blocks.
- Gate, up and down sources retain GGUF names, relative tensor byte offsets,
  Q8_0 type, logical shapes and hashes of the native tensor bytes. Aliased
  source ranges, mismatched shapes and renamed original layer IDs are invalid.
- Each row pairs normalized FP32 input to gate/up with completed FP32 SwiGLU
  input to down, plus task ID and absolute token position. The vectors are
  row-major and unmodified. They are not repeated synthetic rows.
- Capture excludes prompt inputs. If the prompt has `P` tokens and generation
  actually emits `G` tokens, eligible generated-input positions start at `P`
  and contain `min(G - 1, row_cap)` rows. The last emitted token has not yet
  been processed. The row cap is 1 to 32 per task.
- Every selected layer must supply the same contiguous positions for every
  frozen calibration task. Incomplete, duplicate, out-of-order and nonfinite
  pairs fail. A rejected pair appends neither vector; a failed task completion
  leaves the active task available for correction.
- Reserve all FP32 observation storage before capture under an explicit byte
  budget. Report used and reserved FP32 bytes separately. These counts exclude
  metadata, serialization, allocator overhead and model-observer read scratch.
  They are not process RSS or GPU memory measurements.
- Export requires all tasks complete and at least one observed row. JSON is
  bounded to 64 MiB and validates positions, sources, finite values and memory
  counts. Prompt and actual generated token IDs remain in the artifact.

The parser establishes internal consistency, not trusted provenance. Consumers
must call `plan.validate_corpus` with the actual frozen corpus/hash, check the
original model and tensor hashes and execution identity, and verify the row
selection against their benchmark configuration. No held-out or browser input
may supply these calibration observations.

## Model Observer

`LoadedModel::begin_ffn_capture` installs model-owned host observations before
KV or forward workspace creation. It permits prepared down/vocabulary weights,
but forbids concurrent activation transforms, layer influence and layer masks.
It requires sm_52, decode-stream Q8 projections, sequence zero, explicit dense
KV and ordinary sequential forward. Packed prefill, batched forward and graph
entrypoints reject capture before device work. Configure this only for untimed
diagnostics, never an authoritative throughput trial.

Call `begin_ffn_capture_task` with the next frozen task ID and encoded prompt
IDs. Immediately before down projection, the observer drains the decode stream
once and reads normalized input plus completed SwiGLU for selected original
layers at eligible positions. Inactive capture and unselected layers add no
GPU copies, launches or synchronization. `finish_ffn_capture_task` supplies the
actual generated IDs; `finish_ffn_capture` validates all tasks before removing
the observer and exporting the artifact. Incomplete export preserves the state.

Setup reserves two reusable host read buffers in addition to the observation
vectors, under the same host budget. Their capacity is
`4 * (d_model + hidden_dim)` bytes. The artifact's observation counts exclude
these temporary read buffers, which are freed on export/model drop. Capture
owns no device allocation and never writes model buffers.

## Real-Model Capture Command

The existing release `model_quality` example accepts `--capture-ffn-inputs`.
It uses the qualified control through an explicit model-construction argument;
the evaluator's ambient `M40LLM_Q8_EXECUTION_WEIGHTS=off` reference guard stays
unchanged. All six frozen calibration tasks run once with inactive capture and
again with capture active. Actual prompt IDs, every generated ID, output and
stop reason must match. Each selected layer records at most four generated
input rows per task. Both the artifact and its JSONL companion are required
provenance for subsequent benchmark use.

```bash
source scripts/dev-env.sh
while IFS=$'\t' read -r key value; do
  export "$key=$value"
done < <(head -n1 docs/experiments/2026-10-25-ranked-layer-one.jsonl |
  jq -r '.configuration.environment|to_entries[]|[.key,.value]|@tsv')
MODEL="$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
cargo run --release --features cuda --example model_quality -- \
  --reference-model "$MODEL" --model "$MODEL" \
  --variant-id maxwell-ffn-inputs --split calibration \
  --capture-ffn-inputs new-ffn-inputs.json --output new-ffn-capture.jsonl
```

Use new destinations and request hardware access. This is untimed diagnostic
work, not an E2E or FFN latency measurement. Do not run it beside a timed trial.
Context is fixed to 1024 and model shape to Qwen2.5-3B for this M02 entrypoint.
Held-out/all splits, other transforms/capture modes, timing trials and aliased
or pre-existing artifact/report destinations are rejected. The companion
records original model/tokenizer identities, actual immutable preparation,
execution configuration/hash, source revision/build, raw task results, host
observation/read-scratch bytes and final tracked allocation cleanup.

## Current Validation

Eight CPU tests cover paired values and distinct rows, original IDs and source
shapes, signed-zero/subnormal JSON round-trip, calibration-only eligibility,
row bounds, invalid-pair atomicity, incomplete capture, short generation,
budget/overflow/alias failures, external artifact corruption and corpus binding.
Two explicit M40 observer tests cover fused/unfused gate/up, all/selected-layer
capture, prepared down/vocabulary, exact logits and KV-byte parity, no added
kernel/allocation activity, the documented D2H/sync deltas, paired-input replay
and an independent CPU projection/SwiGLU reference. They also cover invalid
setup, incomplete export, unsupported paths and device-allocation cleanup.

The warning-as-error non-CUDA test lane and CUDA/server all-targets Clippy
pass. M40 attention/projection/MLP parity passes with cuBLAS enabled/disabled;
the cuBLAS-enabled lane also passes activation calibration, layer influence,
full forward/prefill and allocation ownership regressions. Explicitly invoking
the observer hardware test with `CUDA_VISIBLE_DEVICES=''` fails with a context
creation error, as required; GPU-less CI does not establish observer parity.

Real-model capture and warm complete-FFN baselines are now recorded in the
[investigation checkpoint](maxwell_q8_execution.md). Neither establishes a
candidate performance improvement by itself.

## Complete FFN Baseline

The existing `q8_execution_bench` accepts `--ffn-inputs` and
`--ffn-capture-report`. This mode verifies the artifact, completed task histories,
frozen corpus, execution hash/environment and original model/tensor bytes. It
rejects repeated first-four rows, mixed layout/sparsity switches, unsupported
shapes and any silent single-row fused-kernel fallback. The original projection
benchmark CLI remains available.

```bash
# Reuse the qualified environment from the capture command above.
mkdir -p target/research-fixtures
gzip -dc docs/experiments/2026-10-25-maxwell-ffn-inputs.json.gz > \
  target/research-fixtures/m02-ffn-inputs.json
cargo run --release --features cuda --example q8_execution_bench -- \
  --model "$MODEL" --output new-ffn-baseline.jsonl \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --rows 1,4 --trials 5 --iterations 10
```

The immutable model cache retains qualified down/vocabulary preparation. Only
FFN primitives execute: no KV, attention, embedding or vocabulary projection is
timed. Whole-model weights and the vocabulary cache remain resident and are
reported; these microbenchmarks are not full-model peak-memory measurements.

Single-row gate/up/SwiGLU uses the native fused FP32-activation kernel; down
uses prepared four-column/256-thread reduction on decode. Four-row gate/up
uses real multi-row Q8 projections on prefill, then waits for decode SwiGLU;
down waits back onto prefill and uses prepared four-column/128-thread dispatch.
The next iteration also waits before reusing gate/up buffers. Each of the four
rows has a distinct captured input; multi-row results are not autoregressive
decode timings.

Measure gate/up/SwiGLU, down on frozen captured SwiGLU, and complete FFN with
freshly computed SwiGLU separately. Events cover each causal multi-stream path;
host enqueue/completion time and enqueue counters accompany GPU elapsed time.
Stage order alternates across baseline trials. This is not yet an A/B speedup
screen; the later bounded candidates require their own alternating pairs.
After CPU validation, each shape warms through at least 1000 ms of measured
complete-FFN work, in 32-iteration batches capped at 256. Warmup records are
separate from primary trials; exhaustion fails rather than extending until a
desired latency appears. GPU boundary clocks remain reported and no system
clock settings are changed. The [initial three-iteration run](experiments/2026-10-25-maxwell-ffn-baseline.md)
is preserved because its clock ramp limits cross-case comparisons.
Quantization cost is explicitly zero because the FP32 control has no activation
quantization operation. Allocation, upload, CPU references, readbacks and the
single completion wait lie outside the repeated enqueue region. Any enqueue
allocation/copy/synchronization rejects the row.

Guarded, reusable RAII buffers protect both inputs and outputs. Validation
compares every gate/up/SwiGLU and down output with scalar Q8 dequantization/FP64
CPU references, plus actual capture parity; single-row SwiGLU must match frozen
bits exactly. Changed decode-stream inputs exercise cross-stream ordering in
hardware tests. Whole-model final logits and generated-token quality remain
separate runtime acceptance requirements.

M02 delivery is split into the tested benchmark extension, then a clean-release
measurement/evidence commit. Do not promote a kernel or change runtime defaults
from this baseline alone.

## Bounded FP32 FFN Screen

Add `--ffn-f32-ablation` to the complete-FFN command to measure all five M03
variants. Preserve the frozen capture environment exactly:

```bash
source scripts/dev-env.sh
while IFS=$'\t' read -r key value; do
  export "$key=$value"
done < <(head -n1 docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl |
  jq -r '.execution_configuration.environment|to_entries[]|[.key,.value]|@tsv')
MODEL="$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
cargo run --release --features cuda --example q8_execution_bench -- \
  --model "$MODEL" --output new-ffn-f32-screen.jsonl \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-f32-ablation --rows 1,4 --trials 5 --iterations 10
```

The same pipeline allocations serve both arms. For each layer/shape, native
and all candidates undergo numerical validation before timing. Initial
256-thread fused variants additionally match the fused row-control bits;
single-row variants match native complete output bits. The 128-thread variant
retains CPU/capture tolerance rather than claiming unchanged arithmetic order.
Each variant gets bounded 1000 ms native and 100 ms candidate GPU warmup,
followed by at least five contemporaneous pairs. Arm order and stage order
alternate. Record all three stages, stream dependencies, enqueue counters,
clock/process samples, scratch capacity and zero activation-quantization cost.
No allocations or host transfers occur inside timed enqueue.

The initial complete-FFN screen needs at least 5% one-row throughput gain and
a positive lower paired-bootstrap confidence bound. Reports retain all raw
pairs, 10,000 resamples with seed zero, numerical results and failed variants.
This is not the E2E promotion gate or authorization to enable a runtime path.
Four-row fused results include an algorithm/fusion change versus the native
two-projection/SwiGLU implementation, so report them independently rather than
attributing that entire gain to load/reduction changes. Prepared down and
vocabulary remain fixed; whole-model logits/512-token equality remain pending.

## Bounded FP32 Scale-Factoring Follow-Up

Use `--ffn-f32-factored-scales` instead of `--ffn-f32-ablation` in the command
above to test one further 128-thread candidate. Its control is the qualified
packed/warp128 kernel, not the old native implementation. Both arms retain
FP32 activations, the same packed weight reads, split4 assignment, warp
reduction and prepared down/vocabulary paths. The candidate accumulates eight
activation/int8 products before applying their common Q8 scale. This changes
FP32 association but does not quantize activations or modify model weights.

The original five-variant M03 screen is unchanged. The follow-up tests layers
0/17/35 at genuine rows 1/4, five alternating pairs, ten iterations per stage,
and the same complete-FFN micro gate. It adds no geometry search. Candidate
gate/up capture must match CPU references and ordinary-kernel SwiGLU bits;
SwiGLU and completed down outputs retain the existing numerical tolerance.
Capture stores are validation-only, never part of timed work. JSONL names the
actual timing control and emits `control_median_ms` for this follow-up.

Only a passing micro screen permits model-local runtime evaluation. Actual
512-token equality, frozen-task quality and incremental HTTP gains versus
the selected packed128/SGEMM runtime remain required before retaining it.

## Bounded Adjacent-Pair Load Follow-Up

Use `--ffn-f32-coalesced-pairs` instead of `--ffn-f32-ablation` for one
128-thread lane-ownership experiment. Neighboring lanes own adjacent pairs of
Q8 weights and FP32 activations. A lane loads one aligned `float2` activation
and one alignment-safe `uint16_t` payload from each weight matrix; it retains
per-weight FP32 scaling. Shared publication, final-warp reduction, fusion and
prepared down/vocabulary are unchanged. Summation ownership changes, so the
existing tolerance and actual 512-token runtime gates still apply.

Its timing control is qualified packed/warp128, not old native or the failed
scale-factor candidate. Only this one geometry is tested at the original
layers 0/17/35, rows 1/4 and five alternating pairs. The manifest includes a
static per-warp sector-request hypothesis. It assumes the scalar/float2 loads
and must be checked against generated instructions. It does not measure
cache misses or DRAM traffic; [legacy CUPTI counters are unavailable on the
current stack](experiments/2026-10-25-maxwell-counter-capability.md).

No activation quantization, weight repacking, runtime default or scheduling
change is part of this screen. Whole-model runtime evaluation remains
conditional on at least 5% complete one-row FFN gain with a positive paired
lower confidence bound. Do not add geometry configurations to rescue a loss.

The Rust validation entrypoint is now `ffn_q8_f32_capture_async` with an
explicit variant. It replaces the factored-only name rather than adding a
compatibility alias; timed enqueue and the existing C ABI remain separate.

## Audit Packed128-Control Follow-Ups

`scripts/audit_ffn_followup.jq` checks six-shape FP32 screens against explicit
source, candidate, newly created report path and sole GPU PID. It verifies
native/control/candidate validations, alternating pairs, stage order,
allocation/copy/sync counts, observed clocks, non-overlapping intervals,
reported medians and the unchanged micro gate. It does not establish E2E
qualification. For example, re-audit the preserved scale-factor result:

```bash
jq -s \
  --arg source 498699a8b7131cb766c08776e5799099c9b59807 \
  --arg candidate packed-warp128-factored-scales \
  --arg report docs/experiments/2026-10-25-maxwell-factored-ffn-screen.jsonl \
  --arg pid 1114652 -f scripts/audit_ffn_followup.jq \
  docs/experiments/2026-10-25-maxwell-factored-ffn-screen.jsonl
```

The new audit leaves historical report-specific scripts and hash manifests
unchanged. Changed control, timed allocation and fabricated median data must
fail rather than producing a successful summary.
