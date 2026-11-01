# CUDA Graph attention dispatch result

The bounded dispatch investigation passes its correctness gates at `dbceae1`
on `research/1031-graph-dispatch`, based on feature commit `b9333e7`. The fix
is suitable for carrying forward as an opt-in graph correctness repair. It
does not justify promoting graph execution or claiming a decoding speedup.

## Attribution and implementation

The original exact-cache failure reproduces on `b9333e7`: 93 unequal FP16 cache
comparisons in the native full-forward case. The independent short attention
negative control finds 30,909 unequal FP32 output elements in 144 comparisons,
while cache/input/guard/allocation assertions pass. The generic odd-head path
has no reported mismatch. See [negative controls](2026-10-31-graph-dispatch-baseline.md).
The protocol is [frozen separately](2026-10-31-graph-dispatch-plan.md).

The candidate shares the host launcher and existing arithmetic kernel bodies
between ordinary and device-length attention. It preserves head64 normal/ldg,
head128 native/shared/warp32, and generic attention. Each kernel reads the live
length before addressing data or shared scratch. Ordinary calls pass no device
length and retain their length-specific launch resources.

Captured specialized launches reserve shared memory for `min(cache capacity,
8192)`. For larger caches, a second scalar node is captured. The specialized
node writes only at lengths 1..8192; the scalar node writes only at valid lengths
8193..capacity. Both reject zero/out-of-range live lengths without writing.
Only one node writes output for a valid replay. This avoids a device-to-host
length read or a graph update while preserving the ordinary dispatch boundary.
It may cost occupancy or an extra inactive node; performance is unmeasured.

The opt-in parallel/partitioned backend is dispatched in Rust before the CUDA
launcher, uses a separate workspace, and cannot consume device lengths. Graph
session creation and direct device-length attention now reject that combination
explicitly. Ordinary parallel attention remains available. Backend/environment
choices are captured; changing them requires a new graph.

## Validation

All four unchanged original native-Q8 cases pass in both debug and release: 128 token
comparisons across replay, capture cancellation, first-launch rejection and
later-launch rejection. Every case reports zero FP16 cache differences and zero
maximum logit error. Callbacks, sequence lengths, cache reuse and tracked
allocation cleanup pass. The new parallel session guard passes too.

The release direct grid passes 228 exact FP32 comparisons: 144 at cache capacity
129 and 84 at capacity 8193. The latter replays lengths 3/8191/8192/8193/33,
then zero and capacity-plus-one, through each captured executable. Both grids
cover two physical slots and all five specialized choices plus generic head7.
Every initialized FP16 cache value matches its input after the grid; query and
device-length values, output guards, unused-slot length and allocation cleanup
pass. Unsupported parallel capture is rejected without output/cache changes.

CUDA/server all-target Clippy and all 220 warning-free non-CUDA tests pass.
The affected release command reports 55 passes, but one optional real-Qwen
prefill diagnostic returns early without its model-path environment variable.
Count **54 executed checks**, not 55 hardware passes. The two explicit parallel
attention checks and native-FFN full-forward/workspace check also pass.
With cuBLAS disabled, all five state/compatibility checks pass, the short grid
passes all 144 exact comparisons, and 14 attention/forward/MLP checks pass.
No unavailable-device skip appears in those logs. Both cuBLAS configurations
report zero cache and logit differences in all four native state cases.

Formatting and whitespace checks pass. The final GPU snapshot contains no
compute process. All measurement/test processes have exited. This is a local
research branch; it has not been merged into the feature branch or pushed.
The feature worktree remains clean at `b9333e7`; a read-only `git apply --check`
of the source/test patch succeeds there. The nine original October 30 branch
heads remain unchanged. Integration must preserve the feature's separate V3
status and performance entries.

The first direct debug grid was deliberately terminated with SIGTERM while its
boundary job remained live. Inspection found `DEBUG=true`, which makes the
installed `cc` crate add NVCC `-G`. Its partial run is cancelled, not a pass or
a demonstrated kernel failure. The full identical grid then ran in release
with `OPT_LEVEL=3`, `DEBUG=false`; no cases or equality gates were removed.

## Limits

These are correctness checks on the Tesla M40 24GB, UUID
`GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`, driver 580.159.04. They are not
performance measurements, including when compiled in release mode.
The full-forward fixture is two layers, width 256, FFN 512, two query heads,
one KV head, head128 and vocabulary 320. It is not a real-model quality study.
Failure injection covers capture cancellation and pre-submission rejection,
not asynchronous device faults. Tracked allocations exclude CUDA-internal graph
allocation accounting. No release timing, graph setup amortization, real-model
throughput, concurrent server or buffered HTTP result is established here.

Graphs remain opt-in. Default block selection, FP16-K/Q4-V backing/direct
attention, top8 selection and experimental-Q8 graph restrictions remain intact.
No result promotes graphs or parallel graph support to supported defaults.

The original nine branches remain experimental records. This result closes the
attention-dispatch parity follow-up for the tested native graph paths. Real-model
quantized-KV quality and hardware-adaptive drafting remain separate pending
items in the user's broader objective. The fixed causal-selector study remains
closed with its negative result on `research/1031-selector-followup`.

## Reproduction and evidence

Source the project environment, select the M40 UUID above, verify that no other
compute process is running, and use a disk-backed target directory. The full
release commands are:

```bash
M40LLM_ENABLE_CUBLAS=1 cargo test --release --features cuda \
  --test attention_graph_dispatch -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --release --features cuda --lib \
  decode_session::graph_tests -- --include-ignored --nocapture --test-threads=1
bash docs/experiments/2026-10-31-graph-dispatch-regressions.sh
```

The [regression script](2026-10-31-graph-dispatch-regressions.sh) preserves the
executed commands; [exit records](2026-10-31-graph-dispatch-regression-exits.tsv)
show all seven commands completed successfully. The script writes its named
logs, so use a new checkout/output destination when preserving an earlier run.
The [source manifest](2026-10-31-graph-dispatch-source.sha256) matches `dbceae1`.
The [artifact manifest](2026-10-31-graph-dispatch-artifacts.sha256) covers the
retained raw logs and reproduction inputs. Logs with `kernel-release`,
`state-release`, `state-cublas-off` and `short-cublas-off` in their names contain
the exact comparison and cleanup summaries. `state-fixed` is the completed
debug state run; `debug-cancelled` is explicitly incomplete.
