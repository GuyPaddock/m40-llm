# No-Old Compressed Attention Stops At Cache Equality

## Gate Decision

Reject the single adapter. The full-forward sequential Qwen head128 fixture
changes one valid FP16 K word and one V word in layer 1 at prefix length 2.
Greedy IDs still match and logits remain well inside the existing tolerance,
but the required bit-identical cache-state gate fails. Do not time, promote
or rescue this candidate. No tolerance, geometry or cache algorithm changed.

The rejected implementation and tests are preserved in `db04697`, after
layout prerequisite `d05a6c2`. Remove the adapter separately before starting
the independently approved joint-only ownership experiment. The
[immutable reference](2026-10-27-compound-reference.md), its measured binaries,
50.910426 HTTP tok/s and 56.153149 target remain unchanged.

## Correctness Evidence

| Check | Result |
| --- | --- |
| Physical recent-ring layout | 88 cases pass |
| Primitive attention / read-only state / rollover | 140 cases pass |
| Full-forward valid cache equality | fails at N=2 |
| N=2 greedy token | control 34, candidate 34 |
| N=2 max / mean logit error | 5.58793545e-9 / 1.55159796e-9 |
| N=2 layer 0 recent K/V | identical |
| N=2 layer 1 recent K/V | one word differs in each |

The [layout record](2026-10-27-no-old-layout.jsonl) verifies four physical
slots, adjacent and split-half RoPE, reset/reuse, absolute recent positions,
and published valid lengths. Both sequential append and dense-prefix recent
reconstruction match actual dense FP16 words. W=17, block=4, capacity=35;
checkpoints include W-1/W/W+1, subsequent block boundaries and 2W-1/2W/2W+1.
This proves recent-buffer layout, not reconstruction of exact-old backing.

The [primitive record](2026-10-27-no-old-primitive.jsonl) checks every valid
length 1..35 across four slots. Borrowed recent attention is bit-identical
to the existing dense eight-partition/256-thread path for identical stored
inputs. Its largest difference from generic compressed attention is
4.17232513e-7, below the unchanged 1e-3 assertion. All snapshots remain
unchanged by attention. At N=18, the first incomplete old block, the adapter
is ineligible immediately. All 72 post-window cases use the original direct
path and match it bit-for-bit.

The [full-forward rejection](2026-10-27-no-old-full-forward.jsonl) uses the
existing two-layer Q8 Qwen fixture: hidden256, FFN512, head128 GQA, real
Q/K/V biases and split-half RoPE. Both models retain prepared split owners
and joint compound pairs, raw packed projections, original compressed KV
and sequential prefill. Only the no-old adapter differs. At N=1, full logits
and cache are identical. At N=2, both greedy IDs match, but downstream FP16
rounding produces distinct layer-1 cache words.

This confirms the proposal's reduction-order risk. Read-only attention on
identical inputs does not imply identical future KV after a full forward.
Intermediate attention/logit bit equality is diagnostic, but valid cache
equality is binding. Do not reinterpret the tiny logit difference as a pass.

Stop at that first failure. Packed full-forward construction, the real-model
W=1024 continuation, actual512/frozen quality, further resource/compiled-word
audits and resident/HTTP comparisons were not run. They are not passing or
skipped qualifications. The hardware primitive already demonstrates the
rollover guard, but it does not qualify the failed full-forward candidate.

## Reproduction And Preservation

Build through `scripts/dev-env.sh` with release Rust, CUDA and cuBLAS enabled,
on the isolated Tesla M40. Hardware tests are ignored for GPU-less CI and
explicit invocation treats unavailable hardware as an error. Example:

```bash
source scripts/dev-env.sh
CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852 \
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
M40LLM_NO_OLD_REPORT=/tmp/new-no-old-full-forward.jsonl \
cargo test --release --features cuda --test compressed_no_old_attention \
  test_no_old_full_forward_cache_gate -- \
  --include-ignored --nocapture --test-threads=1
```

Use implementation `db04697` to reproduce the rejected gate; it exits 101
after writing the failure record. The tests ran against that implementation
before its preservation commit, with the implementation diff present. The
release test executable SHA-256 is
`4b73f87091b65353e82d77bd23f59830d7629004d267f9e2d4afc09646aa67e6`.
No clean-source HTTP or performance claim is made from those runs.

The [archived logs](2026-10-27-no-old-orchestration.tar.gz) retain primitive,
full-forward and Clippy output. Primitive/full-forward hardware execution
took 0.25/0.18 seconds; these are test durations, not inference baselines.
CUDA/server all-target Clippy, formatting and diff checks pass. The GPU had
no compute process before the serial checks. No timing comparison occurred.

Next is only the separately authorized joint-only storage experiment after
this rejection is committed and the runtime adapter is removed. Do not reopen
this kernel, change defaults or combine the experiments.
