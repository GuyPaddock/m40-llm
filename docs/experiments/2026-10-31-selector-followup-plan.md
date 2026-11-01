# Carry forward verified code and test cheap channel signals

## Scope and starting evidence

The October 31 user follow-up treats the nine `research/1030-*` branches as
independent experimental records. Their published heads remain unchanged.
Work here starts from feature HEAD `a7479bad2a53418cf9aa4f984625ae03473911d9`
on `research/1031-selector-followup`. No runtime default changes follow.
The separate V3 corpus-admission work remains stopped for review. This work
does not authorize corpus acquisition, teacher generation, training, learned
drafter evaluation, or a GPU switch.

At the post-study audit, the main feature branch has independently advanced
to `b9333e7` with four V3 rank/review commits. Its Rust/CUDA sources, build
configuration and tests are unchanged from `a7479ba`; a read-only
`git apply --check` accepts this follow-up's complete source patch against
that newer HEAD. The main worktree remains clean. Future integration must
preserve its newer V3 research-status and performance-history entries;
do not replace those documents with this branch's older V3 snapshot.

The supplied-index Q4 experiment passed twelve complete-FFN comparisons
against full Q4 and qualified Q8 at 10%/20% retention, using one captured
input at each of layers 0/17/35. This is an execution upper bound with
unqualified approximation error. Its remaining 5%-gain budgets are about
150.88 microseconds at 10% and 56.89 at 20% in the conservative cases.
Ternary screening alone took 507.70–513.70 microseconds on 24 inputs, before
channel selection or recovery. Do not repeat that failed screen unchanged.

The October 26 channel-entropy study concerns static Q8 weight-byte
compression, not temporal predictability of token-dependent FFN channels.
Its failed compression models remain closed.

## Dependency and API audit

| Item | Feature-branch state | Carry-forward decision |
| --- | --- | --- |
| Direct quantized KV | Existing Q8 and FP16-K/Q4-V direct consumers and runtime dispatch | Preserve existing implementation; no duplicate backend or promotion |
| Q8 scale synchronization | Both head64 quantizers read K's shared maximum and reuse scratch for V without a separating barrier | Extract the two barrier hunks from `1e445a5`; add independent regressions suitable for this branch |
| Q4 host packing | Missing | Carry the lossless layout and CPU tests from `2705ef2` |
| Q4 selected execution | Missing | Carry kernels, FFI, stubs, build dependency, RAII owner and hardware tests from `1aaaddc`, after host layout |
| Q4 fixture and timing tools | Separate benchmark dependencies | Retain on original branch unless needed for a concrete follow-up consumer |
| Head128 staged KV | Research comparison path added on quantized-KV branch | Preserve there; it is not a missing direct runtime consumer |

The selected-Q4 owner validates shapes, byte counts and sorted, unique,
in-range per-row indices before allocation. It owns uploads and scratch;
the unsafe enqueue contract requires live disjoint FP32 spans and explicit
producer/consumer ordering. Both down layouts remain resident for paired
diagnostics. Selection is immutable after upload. This is suitable for
isolated research, not dynamic runtime routing. A later runtime design
must separate immutable weight ownership from per-session selection and
scratch, avoid unnecessary duplicate layouts, and define asynchronous index
updates. Do not conceal per-token uploads or owner construction in timings.

All CUDA arithmetic remains FP32 on `sm_52`. The Q4 layout preserves Q4_0
codes and scales, but converting Q8 model weights into Q4 is lossy. Carrying
the layout does not establish model quality or apply a Q4 model conversion.

## Cheap-selector study

Freeze the diagnostic protocol before collecting new observations. Use the
existing original Qwen2.5-3B Q8_0 file and already retained diagnostic
calibration prompts. Record their exact hashes, configuration and IDs.
Do not recruit held-out quality examples for tuning or admit new sources.

1. Observe complete naturally stopped or length-limited generations,
   recording actual token IDs, counts and stop reasons. Existing captures
   retain only four generated-input rows per task and cannot answer this
   question. Verify observer-on/off token identity. Explicitly account for
   the prompt-terminal state that predicts the first generated token and
   the last generated token, which is not consumed after stopping.
2. Collect normalized FFN inputs and completed SwiGLU vectors at fixed
   representative layers, including adjacent predecessors when testing
   previous-layer signals. Channel indices from different layers have no
   shared meaning; use previous-layer features to predict the target layer,
   not raw cross-layer index overlap.
3. Use deterministic exact top-k rankings at 10%/20% retention, with a fixed
   channel-ID tie break. Report hidden magnitude and down-column-weighted
   contribution diagnostics separately. Q8-derived rankings are not exact
   Q4 oracles; any Q4 execution experiment must identify this difference.
4. Compare previous-token selections, a causal history-frequency baseline,
   and a fixed inexpensive low-dimensional representation of activations.
   Distinguish previous-token **oracle** reuse from deployable reuse: an
   approximate sparse step does not reveal the full next oracle selection.
   Evaluate explicit dense-refresh intervals and recursive prediction
   without silently refreshing from full hidden vectors at every token.
5. Split reference and evaluation by whole prompt/generation, never random
   token rows. Use only prior observations within a generation. Fixed
   projection seeds and dimensions must be selected before seeing outcomes.
   No target-row hidden state or future-token information may enter a
   prediction. Any trained selector is a separate proposed experiment.
6. Report top-k recall, retained weighted contribution, masked FFN-output
   error and worst per-task/layer results, including sequence position.
   Include random and static-frequency controls. High set overlap alone
   does not establish tolerable logits or generation quality.
7. Account for scoring, sorting/top-k, feature preparation, index transfer,
   refresh work, scratch and resident memory. Offline predictability is not
   an M40 speedup. Only a promising causal signal proceeds to measured
   selector-plus-FFN timing and complete-generation approximation checks.

If the fixed signals fail, report the negative result without a new large
GEMV, representation search or training-time change. If they succeed,
report a candidate with explicit remaining cost/quality gates; do not promote
it based on teacher-forced or oracle diagnostics.

## Other investigations retained in scope

- **CUDA Graph attention parity:** compare ordinary and captured dispatch
  for identical dense-reference settings. Localize the first differing KV
  write before timing. Keep graphs opt-in; earlier greedy/logit agreement
  did not clear the exact KV gate.
- **Real-model quantized KV quality:** freeze complete-generation comparisons
  against explicit dense reference, separating selection approximation from
  quantization and direct-versus-staged execution. Prior synthetic-query
  attention timings with captured K/V establish no generation quality.
- **Hardware-adaptive drafting:** price candidate draft lengths using actual
  proposal and target-verification costs, including fallback and acceptance.
  Keep Soft-MTP and rank-256 drafter incompatibility explicit. No training
  or learned evaluation follows from the old estimated acceptance threshold.

ShiftAdd, multi-row FFN fusion, rotated activation preparation, lossless KV
and CPU/GPU scheduling remain experimental records with no production
dispatch. Preserve their negative or conditional measurements and code.

## Dependency-ordered delivery and validation

1. `docs(research): plan selector follow-up`: this audit and protocol outline.
2. `fix(cuda): synchronize Q8 scale reuse`: independent pre-existing fix,
   behavioral regression, affected attention checks with cuBLAS on/off.
3. `feat(infer): retain Q4 channel packing`: lossless host layout and CPU tests.
4. `feat(cuda): retain selected Q4 execution`: isolated CUDA owner/kernels,
   CPU warning checks, selected-FFN, projection, forward and ownership tests.
5. Freeze the capture/analysis protocol, implement bounded observers and
   causal diagnostics as separate coherent commits, then record actual
   generation evidence and the selector decision.
6. Investigate the remaining three questions with their own explicit
   correctness/cost/quality gates; keep unrelated results separately reviewable.

Use project-local tools and hooks. Tests and GPU access require escalation;
GPU execution is serial on M40 UUID
`GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`. Inspect competing processes first.
Preserve raw results in `docs/experiments/` and performance interpretations in
`docs/perf_baselines.md`. Record skipped or unavailable checks as such. New
commits remain local until separately authorized for publication.
