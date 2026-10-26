# Q8 Execution-Weight Experiments

The API in `infer::execution_weights` owns lossless execution representations,
separate from `DeviceTensorView`, uploaded GGUF weights, and the model's FP32
materialization cache. C08-C10 established ownership and isolated measurements;
C10A.1 adds opt-in model-owned preparation and standard projection dispatch.
Preparation remains disabled by default. The historical
[release HTTP gate](experiments/2026-10-25-q8-prepared-http.md) rejects all-family
and native-MLP timing regressions. Down/vocabulary-only preparation initially
failed token parity at step 310. Matching native down reduction order now
removes that branch and passes the unchanged full-workload gate: 35.725 versus
31.962 HTTP tok/s, +11.771%, with all 512 actual IDs equal. See
[the corrected confirmation and costs](experiments/2026-10-25-q8-native-order-http.md).
This qualifies an opt-in benchmark configuration, not a global default or
achievement of either original throughput target.

## Matched-Prefix Branch Diagnostic

The historical down/vocabulary branch has a bounded, untimed follow-up in
the existing `model_quality` runner. It consumes the original completed HTTP
report, not re-tokenized output, and validates model/prompt/config identities.
The candidate is fixed to split down/vocabulary preparation; no kernel/layout
or family search is added. Global preparation remains off, with immutable
candidate configuration passed explicitly at load.

The native recorded continuation is forced into both models through six
steps after the recorded first branch. Every actual greedy prediction is
recorded; hidden/head probes sample the branch neighborhood plus early
checkpoints. A prepared historical prediction is comparable only through its
first branch, because later recorded prepared tokens used different prefixes.

Each probe runs native and prepared vocabulary heads on both completed hidden
vectors, without repeating forward or changing KV. Actual native greedy uses
its specialized fused head; generic full logits use another reduction path.
The report names both, rather than calling generic argmax the production
prediction. Logit differences/margins are generic-projection diagnostics;
they are not observed logits of the fused argmax kernel.

Use the unchanged C11 native-Q8 environment, including explicit
`M40LLM_Q8_EXECUTION_WEIGHTS=off`, then:

```bash
source scripts/dev-env.sh
cargo build --release --features cuda --example model_quality
target/release/examples/model_quality \
  --reference-model "$model" --model "$model" \
  --q8-prefix-drift docs/experiments/2026-10-25-q8-down-vocab-http.jsonl \
  --variant-id prepared-q8-prefix-drift --context-tokens 1024 \
  --output /tmp/new-prepared-q8-prefix-drift.jsonl
```

No Ollama process, server, calibration corpus, new generation or timing trial
is needed. The source report must be complete; changed source/prompt hashes,
context, execution flags or trace counts are errors. Report files are new,
flushed per row and never overwrite prior evidence. Reproduction or failure
to reproduce the historical branch stays explicit in the summary.

Atomic sequence: `feat(eval): probe prepared Q8 prefix drift` adds the shared
loader's explicit execution configuration, evidence checks, probe and controls.
Then `docs(perf): record Q8 prefix drift` records a clean release run. This
diagnostic neither relaxes the parity gate nor qualifies an optimization;
any correction or new performance claim needs its own evidence.

C09 now adds a lossless separate-scale/value repacker, bounded CUDA consumer,
and real-tensor release microbenchmark. See the
[measured results](experiments/2026-10-24-q8-separate-projections.md).
C10 compared lossless input-tiled storage with two bounded CUDA mappings.
See the [tiled experiment](experiments/2026-10-24-q8-input-tiled.md). The vector
mapping remains a per-family research candidate. The slower coalesced-lane
mapping and its API/CLI switches are removed after preservation in `bb4d30e`.
Those microbenchmarks alone do not qualify a runtime configuration.

## Opt-In Runtime Preparation

Set these before loading a model, not during generation:

```bash
M40LLM_Q8_EXECUTION_WEIGHTS=split
M40LLM_Q8_EXECUTION_BUDGET_MB=6144
```

`M40LLM_Q8_EXECUTION_FAMILIES` defaults to `all`. A comma-separated subset of
`qkv,output,gate-up,down,vocabulary` prepares only those groups, for example
`qkv,output,down,vocabulary` retains native fused MLP while using prepared
QKV/output/down/vocabulary weights. Excluded groups consume no prepared memory
and use their existing native paths, including fused kernels and greedy head
selection. Family selection is captured at model loading, never toggled between
active requests; explicit constructors use `Q8ExecutionConfig.families`.
Both the budget estimate and actual cache stats reflect this subset. Invalid,
empty, or duplicate family names fail; use mode `off` to disable all preparation.

`off` is the default and skips hashing, packing, and prepared allocations.
`split` prepares separate original FP16 scales and signed Q8 values for each
recognized Q/K/V/O, gate/up/down, and vocabulary projection. Tied output
embeddings are prepared once under their chosen tensor identity. The measured
single-token launch uses four columns/128 threads, except K/V use one column
and down uses the existing four-column/256-thread launch to match native
coltile4/split4 FP32 accumulation order. Multi-row down remains unchanged.
`mixed` adds width16 input-tiled Q/O/down copies for rows >=64; smaller rows
still use split planes. It does not globally replace the layout for every role.

`LoadedModel::from_gguf` reads this configuration once;
`from_gguf_with_q8_execution` accepts an explicit `Q8ExecutionConfig` so two
models can use different immutable caches in the same process. There is no
mutable global prepared-weight registry or on-demand decode repack.

Preparation currently requires an sm_52 CUDA device, uploaded source weights,
and at least one recognized rank-two Q8_0 projection with K divisible by 32.
Invalid modes, malformed eligible tensors, aliased source offsets, insufficient
budget, skipped upload, and allocation/upload errors fail clearly rather than
silently bypassing the requested cache. Non-Q8/unprepared projections retain
their existing implementation. The cache budget limits extra prepared payload,
not total process VRAM; the default 6144 MiB does not reserve that amount.

The cache remains separate from native GGUF storage. `split` adds another copy
of each prepared tensor's payload; `mixed` adds selected tiled copies too.
Model SHA-256 is calculated over the supplied original GGUF bytes with `ring`.
Keys retain that hash, tensor name, source offset, dtype, logical shape, and
execution layout. A model-local native-pointer lookup finds those immutable
owners; pointers are not used as cross-model cache identities.

`q8_execution_stats()` exposes estimated/actual prepared bytes, budget,
source uploaded bytes, total source+prepared device payload, selected families, per-tensor/layout
keys and bytes, vocabulary-prepared status, and hash/preparation milliseconds.
`single_row_down_split_launch` reports effective columns/threads when down
is actually prepared; it is null for off or caches without down weights.
`peak_setup_host_payload_bytes`
estimates the original file + retained tensor-data blob + largest temporary
packed tensor. These payload estimates exclude allocator/page overhead,
metadata, CUDA contexts, workspace, and KV. Source upload remains part of cold
model loading, outside the separately reported preparation/hash time.

Standard sync/async projection APIs consume the cache without per-call
allocation, upload, or packing. Enqueue explicitly uses the existing producer
stream policy: multi-row work uses prefill; single-row uses decode only when
`M40LLM_Q8_DECODE_STREAM=decode`, otherwise prefill. The sync API waits on that
actual producer; the async API does not synchronize.

`M40LLM_FUSED_Q8_QKV=1` now lets split/mixed caches supply a single fused
Q/K/V launch. Query tiles use four columns, K/V tiles one, all with 128 threads
and the original optional FP32 biases. The launch preserves the existing Q8
producer-stream policy and reuses the split projection's dot/reduction code.
All three prepared tensors must be present and shape-compatible; missing cache
entries retain native fused QKV, while invalid cached shapes fail before launch.
Multi-row calls still use the standard projection path.

With native FFN selection, `M40LLM_FUSED_Q8_MLP_SWIGLU=1` retains the native
fused gate/up/SwiGLU kernel even when gate/up weights are prepared. The slower prepared fused MLP
was removed after preservation in `e2dbc61` and release measurements in
`2e2bb5c`. Model-level shape and pointer checks remain before enqueue. Prepared
gate/up weights are still available to ordinary projection APIs and multi-row
work; they do not replace the retained native fused decode kernel implicitly.
The separately screened `M40LLM_Q8_FFN_KERNEL=split-f32x4-128` is an explicit,
fixed-geometry exception requiring split gate/up preparation. See
[load-time FFN selection](maxwell_q8_execution.md#load-time-configuration).

When vocabulary weights are prepared, GPU greedy selection uses the prepared
projection into existing logits scratch followed by the existing device argmax.
This also applies when `M40LLM_Q8_LM_HEAD_ARGMAX=1`: that flag no longer bypasses
the cache to read native GGUF weights. Off/unprepared vocabulary still uses the
native specialized path when requested. Output normalization, producer-stream
dependencies, first-index tie handling, and the single-u32 D2H result remain
unchanged. No full logits readback or extra device scratch is added.

Use of the cache remains experimental. The qualified HTTP result applies
specifically to `split`, families `down,vocabulary`, native-order down and
the documented C11 flags. It does not qualify all-family preparation, another
model/workload or a frozen model-quality baseline.

## Unfused Gate/Up Probe

The independent fixed unfused gate/up experiment uses
`q8_execution_bench --ffn-f32-unfused` with the same frozen FFN inputs.
It keeps the qualified one-column split control and prepared down/head,
uses two fixed four-column/128-thread projections on the existing planes,
then applies the control-order SwiGLU expression. Gate/up scratch is reused,
not allocated during enqueue. All three candidate launches and intermediate
traffic are included; complete FFN includes the unchanged fourth down launch.
`CudaContext::ffn_q8_split_unfused_f32_async` requires explicit caller-owned
gate/up buffers and does not select a generation policy. See the
[bounded hypothesis](experiments/2026-10-25-maxwell-unfused-ffn-hypothesis.md).

The independent `q8_execution_bench --ffn-f32-weight-cg` comparison changes
only weight/scale load hints in the qualified fused split-plane kernel.
It keeps read-only activations, original FP32 association, one-column /
four-value / 128-thread geometry and the same prepared down/vocabulary.
`CudaContext::ffn_q8_split_weight_cg_f32_async` uses the same buffer contract
as the qualified split API, including optional gate/up capture. This is a
benchmark-only cache experiment, not a runtime or default selector. Require
distinct sm_52 instructions and the original complete-FFN / quality / E2E
gates described in its [bounded hypothesis](experiments/2026-10-25-maxwell-cache-ffn-hypothesis.md).

`q8_execution_bench --ffn-f32-byte-permute` independently tests an exact
signed-byte conversion using byte permutation and representable FP32 bias
subtraction. It retains read-only loads, all scale/FMA association and the
qualified fused geometry, planes and down/vocabulary. The explicit
`CudaContext::ffn_q8_split_byte_permute_f32_async` API has the same buffer
contract as the qualified split path and is benchmark-only. No activation
quantization or runtime selector is added. See the [conversion proof and
fixed gates](experiments/2026-10-25-maxwell-byte-ffn-hypothesis.md).

Independent raw-projection screens use `q8_execution_bench
--qkv-f32-packed` or `--output-f32-packed` with frozen FFN inputs/report.
The control is the qualified scalar-load warp kernel; the candidate uses
four uint16 weight loads and two float4 activations per eight-value task,
with the original signed conversion, FP32 association, reduction and bias.
Geometry and raw Q8_0 storage stay fixed; no additional preparation is
retained. The explicit CUDA APIs require sixteen-byte activation alignment
but only two-byte weight alignment. They do not select runtime behavior.
See the [fixed stage/quality/HTTP gates](experiments/2026-10-25-maxwell-packed-projection-hypothesis.md).

The model-local runtime selector also accepts
`M40LLM_Q8_ATTN_PROJECTIONS=packed-warp-256`. It captures the choice once
and uses packed QKV/output only for single-row decode. The qualified
`warp-256` control and default `native` remain available; multi-row prefill
is unchanged. Both explicit variants require uploaded sm_52 raw Q8_0
QKV/output weights and validated model-local spans, reject conflicting
prepared families, and allocate no extra weight/scratch storage. Packed
inputs additionally require sixteen-byte alignment. This opt-in runtime
path still requires full-model quality and HTTP qualification; primitive
microbenchmarks do not promote it or change defaults.

The quality and HTTP tools select this candidate with
`--q8-runtime-attention packed-warp-256` and require the explicit control
`--q8-runtime-control qualified-warp-sgemm`. Both use
`--q8-runtime-prefill dequant-sgemm` and split-f32x4-128 FFN, with identical
prepared weights. The shared typed loader verifies the actual model backend;
quality and timed runs also require one selected QKV/output launch per
forwarded token and layer. A backend label without those dispatch counts
does not qualify a comparison. The earlier `warp-256` experiment retains
its own `qualified-split-sgemm` control.

## Prepared Projection Reduction Probe

`DeviceQ8ExecutionWeight::enqueue_separate_warp_async` is a benchmark-only
alternative on existing `SeparateScalesValues` planes. It keeps four columns
and the control's 128/256-thread task assignment, FP32 accumulation and
reduction tree, replacing repeated block barriers with a shared publication
and final-warp finish. Callers explicitly supply the stream and retain all
input/output/weight spans until completion. No allocation, packing, copy or
synchronization occurs inside enqueue. Wrong layout, column count or pointer
metadata fails rather than selecting another implementation. Runtime
projection dispatch and all generation defaults remain unchanged.

The [qualified runtime profile](experiments/2026-10-25-maxwell-projection-stage-profile.md)
defines the bounded experiment and promotion gates. Primitive parity alone
does not qualify it for model generation or establish an E2E speedup.

Use `q8_execution_bench --prepared-f32-warp` with the existing explicit
model, frozen `--ffn-inputs` / `--ffn-capture-report`, release build and
`--rows 1,4`. It compares complete FFN plus isolated down on layers 0/17/35,
holding split-f32x4-128 gate/up fixed, then independently compares vocabulary
projection. Down uses 256 threads for one row and 128 for four; vocabulary
uses 128. Both arms borrow the same prepared model-owned weights, and the
only extra preparation is the shared per-layer gate/up pair already used by
the FFN benchmark. Vocabulary inputs are frozen normalized-FFN surrogates,
not captured output-normalized hidden states. Actual operation counts,
planes, guards and allocation-free enqueue are checked. Timings do not
qualify runtime or change default dispatch; the original complete-FFN,
quality and HTTP gates still apply.

## Source and Identity

`Q8SourceIdentity` includes the caller-supplied 32-byte model SHA-256, tensor
name, original byte offset, Q8_0 dtype, and logical `[K,N]` shape. It does not
contain a transient device pointer. `Q8ExecutionKey` adds the execution layout
and tile width, so different models, tensors, shapes, offsets, or layouts do
not alias. Callers compute/verify real model hashes; synthetic fixtures use
explicit synthetic identities rather than claiming a file hash was measured.

`Q8SourceView` borrows exactly one tensor from a **tensor-data-relative weights
blob**, not the complete GGUF file. Checked offset, element, block, and byte
arithmetic rejects overflow/out-of-range views before allocation. Source bytes
are immutable and never replaced by an execution representation.

For real GGUF tensors, `from_gguf_tensor` requires rank two, Q8_0, nonzero
dimensions, and complete 32-element input blocks. Synthetic ragged-K tests
must explicitly use `from_column_blocks`; each output column then stores
`ceil(K/32)` complete 34-byte blocks, including all inactive tail bytes.
Do not size that synthetic format as `ceil(N*K/32) * 34`, which can undercount
per-column padding and mix adjacent columns.

The original source offset is identity/bounds metadata. A copied execution
buffer starts at its own allocation base; do **not** add that source offset
again when invoking a projection.

## Layout and Memory Contracts

Each Q8_0 block has the original two FP16 scale bytes and 32 signed-value
bytes. C08 copies existing column blocks verbatim. C09 implements separate
planes, and C10 implements the input-tiled representation:

| Layout | Owned Planes | Ordering |
| --- | --- | --- |
| `ColumnBlocks` | One block plane | Original `[column][input_block][34 bytes]` |
| `SeparateScalesValues` | FP16 scale bytes; signed-value bytes | Both column/block major; values retain all 32 lanes |
| `InputTiled { columns }` | One record plane | `[input_block][output_tile]`, then scale header and `[quant_index][output_column]` values |

Input-tile widths must be powers of two from 1 through 32. Each record places
`columns * 2` scale bytes before a 16-byte-aligned value region, followed by
`columns * 32` value bytes. Output tiles pad to the chosen width. The allocation
plan exposes tile count, value offset, record bytes, per-plane bytes, original
source bytes, and total/padding bytes; all size arithmetic is checked.

For `K=35,N=3`, original padded column blocks occupy 204 bytes, split planes
12+192 bytes, and four-column tiled records 288 bytes. For a Qwen gate/up
matrix `K=2048,N=11008`, original and eight-column-tiled payloads both require
23,953,408 bytes. These are format calculations, not measured memory traffic.
`resident_bytes` reports requested device payload bytes, not allocator/page
granularity, CUDA context overhead, or total process VRAM.

`HostQ8ExecutionWeight::tile_inputs` copies all original payload bytes and scale
bits into those records, zeroing only alignment and padded output columns.
`Q8TiledLaunch::new(columns, threads)` permits widths4/8/16 and threads128/256
for the retained vector mapping. Storage widths1/2/32 remain valid descriptors
and host packing choices but
are not CUDA launch choices in this experiment. The requested launch width
must match the owning execution layout. `project_tiled_async` enqueues on the
decode stream after validating dimensions, buffer sizes, and non-aliasing.

## Ownership and Use

`HostQ8ExecutionWeight::new` verifies plane counts/sizes against its key before
GPU allocation. It does not validate a plane's encoded values; C09/C10 packers
must prove lossless scale/value mappings. C09's `separate_scales_values` copies
all scale/payload bytes unchanged, and `from_tensor_bytes` accepts an already
sliced tensor while preserving its original source offset as identity.
`DeviceQ8ExecutionWeight::upload`
synchronously uploads planes into immediately RAII-owned `DeviceBuffer`s.
Partial allocation or upload failures release both prior and uncommitted
buffers. The device owner outlives the source/host planes, and its borrowed
view distinguishes original, split, and tiled buffers by enum variant.

```rust
use m40_llm::infer::execution_weights::{
    DeviceQ8ExecutionView, DeviceQ8ExecutionWeight, HostQ8ExecutionWeight,
    Q8SourceIdentity, Q8SourceView,
};

// model_sha256, tensor, weights_blob, and ctx are supplied by the benchmark.
let identity = Q8SourceIdentity::from_gguf_tensor(model_sha256, tensor)?;
let source = Q8SourceView::new(&identity, weights_blob)?;
let host = HostQ8ExecutionWeight::copy_column_blocks(source)?;
let device = DeviceQ8ExecutionWeight::upload(&ctx, &host)?;
match device.view() {
    DeviceQ8ExecutionView::ColumnBlocks { blocks, .. } => {
        // Use blocks.as_ptr() with the existing unsafe Q8 projection API.
        // Keep device alive until the consuming stream completes.
    }
    _ => unreachable!(),
}
```

The standalone owner API above is also used by the opt-in runtime cache.
It does not reuse FP32 materialization buffers. Preparation time and duplicate
resident payload must be included in any later runtime report.

## Validation and Next Work

C10A.1 runtime tests cover all eight projection roles at rows 1/4/64 against
native GPU and CPU references, per-family split/tiled selection, both producer
streams, sync/async behavior, distinct model hashes/lifetimes, unmodified source
bytes, and budget/unsupported/skipped-upload failures. A later-tensor allocation
failure releases prior prepared planes while preserving uploaded source data.
Invalid aliased/out-of-bounds/shape plans never reach the upload boundary.
The `81948df` prerequisite corrects native-weight accounting on final context
drop; the native buffer was already physically freed before that correction.

The real Qwen2.5-3B Q8_0 split-cache canary answers `4`, captures finite prompt
logits, and records 5,313 prepared projection launches. It prepares 253 tensor
layouts: 3,278,553,088 extra bytes alongside 3,279,519,744 source-weight bytes,
for 6,558,072,832 weight payload bytes before KV/workspace. Hash plus preparation
takes 30.512 s in this **debug** correctness run, not a release performance
claim. See the [raw canary log](experiments/2026-10-24-q8-prepared-split-canary.txt).
CUDA layout/attention/forward/projection/MLP checks pass, including cuBLAS-off
grid/wrappers; non-CUDA warning-as-error tests and both Clippy feature lanes pass.

Prepared fused QKV tests cover equal/GQA-shaped/tail widths, no/partial/all
biases, split and mixed caches, both producer streams, CPU/native GPU parity,
and output guards. Each invocation remains one launch with no internal sync,
allocation, free, H2D, or D2H. Invalid cached dimensions, multi-row requests,
and a disabled fusion flag do not launch this consumer.

The real Qwen2.5-3B split-cache canary with prepared QKV and native fused MLP
also answers `4`, with finite prompt logits and **756 prepared fused-QKV
launches** over 21 forward tokens and 36 layers. The canary asserts that
requested prepared fusion actually executes. This debug correctness run is
not a controlled timing comparison; see its [raw log](experiments/2026-10-24-q8-prepared-qkv-canary.txt).
It uses `M40LLM_Q8_EXECUTION_WEIGHTS=split`, decode-stream Q8 scheduling,
`M40LLM_FUSED_Q8_QKV=1`, `M40LLM_FUSED_Q8_MLP_SWIGLU=1`, and the native
Q8 decode/MLP split-qblock settings of 4. The KV cache is explicitly dense,
the context bound 256, and the Rust test profile is unoptimized.

Prepared MLP tests use distinct gate/up payloads and cover hidden widths
1/31/65/128, positive/negative input amplitudes, both streams, both cache modes,
CPU/native parity, output guards, and rejected invalid/non-decode calls.
Synthetic tensor payloads are now distinct by tensor so accidental operand
swaps cannot pass QKV/MLP tests merely because fixture weights were identical.

The full prepared-QKV/MLP Qwen canary answers `4`, with 756 launches of each
prepared fused consumer and finite prompt logits. See the
[raw MLP canary](experiments/2026-10-24-q8-prepared-mlp-canary.txt). Its debug
generation total is 15.637 s versus 10.698 s in the earlier native-MLP canary.
These are unpaired debug correctness runs, not a release promotion comparison,
but the increase is a reason to keep native fused MLP as an alternative in
C10A.3. Do not infer that preparation improves every fused consumer. The later
release gate must measure per-family choices without another broad tuning round.

Prepared greedy tests compare native/CPU/full-logits references, zero-row ties,
both producer streams, split/mixed caches, and both native shared-activation
argmax variants. The hot selection performs no device allocation/free/H2D and
copies exactly one u32 D2H. The real-model canary additionally compares device
greedy with full logits using the loaded model's vocabulary and output norm.
The Qwen probe agrees on token 140525 for a deterministic synthetic hidden row,
asserts exactly four D2H bytes, then separately generates the arithmetic answer
`4` with both prepared fused consumers active. See the
[raw greedy canary](experiments/2026-10-24-q8-prepared-greedy-canary.txt).
The reference token is a numerical probe, not an answer to the arithmetic prompt.

C08 validation passes eight CPU tests for semantic identities, source bounds,
real/synthetic shapes, plan overflow/alignment, byte copies, and failing
allocation/upload boundaries using a working RAII fake. A real CUDA test
forces the second plane's allocation to fail after uploading the first, then
checks one allocation/free and restoration of tracked bytes. The prerequisite
`0750d85` fixes a pre-existing double-lock in allocator error reporting so
that failure can return instead of hanging.

Two CUDA integration tests round-trip every layout's planes, drop host storage,
check tracked cleanup, and run a real `M=2,N=3,K=35` projection from an owned
column-block view with a nonzero original source offset. Non-CUDA warning-as-error
tests, CUDA/server and non-CUDA/server Clippy, and formatting pass. No model
inference timing is claimed for this ownership-only change.

Run through the project environment with hardware access for CUDA tests:

```bash
source scripts/dev-env.sh
RUSTFLAGS='-D warnings' cargo test --no-default-features --locked
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --lib \
  infer::execution_weights -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test q8_execution_weights --test cuda_allocation_errors \
  -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test q8_runtime_weights -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test q8_prepared_qkv -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test q8_native_mlp_cache -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test q8_prepared_greedy -- --include-ignored --nocapture --test-threads=1
```

The repeated HTTP gate is now complete. Full preparation reduces throughput
by 64.21%, retaining native MLP reduces it by 6.71%, and down/vocabulary-only
increases it by 12.53% but fails actual-ID parity. See the linked release report
before selecting an experimental family subset. C09/C10 isolated results do not qualify a runtime
configuration; keep setup, duplicate weights, and memory scope explicit.
C11's [keep/remove checkpoint](experiments/2026-10-25-exact-kernel-checkpoint.md)
is complete, including the prepared MLP and coalesced-lane removals. Next is
C11A's offline n-gram study, not another layout search.
The selected exact-attention
benchmark configuration remains split8/256 from C07; global runtime/KV defaults
are unchanged, and neither unchanged-model throughput target has been reached.
