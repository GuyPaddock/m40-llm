# Exact Portfolio: Contemporary Operator Checkpoint

## Decision

Continue to the already-approved fixed five-arm runtime experiment, not a
new kernel search. Actual-input correctness and applicable single-row
operator intervals are positive. The conditional saving budget is only
321.030 ms per 512-token request, versus 292.919 ms needed for 3%. The
28.112 ms margin is narrow. This is a reason to measure the combination,
not evidence that it will pass HTTP qualification.

The immutable 50.910426 HTTP tok/s compound reference, joint-only preferred
ownership, defaults and acceptance gates are unchanged. There is no new
HTTP throughput result or exact-dense milestone claim. No Ollama rerun,
A4000 switch, training, approximate kernel or model download occurred.

## Correctness And Controls

Implementation commits are `d66d279` for actual-logit validation and
`7b50493` for separate operator timing. The FFN rebase is `da3994d`.
The Qwen model hash remains
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
Release builds use the existing CUDA `-O3`, `sm_52`/`compute_52`, no-fast-math
settings. No CUDA source or production dispatch changed.

The actual-logit runner uses the frozen compound environment, joint-only,
dense KV, SGEMM packed-prefix prefill and all 512 frozen greedy IDs after
the 65-token prompt. At steps 0, 1, 8, 32, 128, 256 and 511 it reruns the
qualified output norm into guarded scratch, then compares packet vocabulary
with the qualified prepared vocabulary. All seven 151,936-entry vectors
are bit-identical. Original and partitioned argmax agree on both control
and packet logits. This replaces the old surrogate-only vocabulary evidence;
it does not yet exercise runtime-selected packet generation.

Every timed arm checks its output before another arm overwrites scratch.
The runner checks finite inputs, guards, immutable prepared planes, stable
residency and zero timed allocations, frees, copies, synchronizations or
cuBLAS calls. Partitioned argmax uses 384 bytes of reusable partial scratch.
Both complete runs return tracked device bytes to zero.

The full research-runner suite passes 9/9, including the M40 actual-logit
fixture and rejection of an intentionally wrong expected token. The fixture
initially lacked output norm; a nonuniform FP32 norm was added to that test,
without weakening the probe. Changed-example CUDA/server Clippy and format
checks pass. The non-CUDA example builds with warnings denied; it contains
no non-CUDA test cases. Broader runtime quality and ownership tests remain
required when the five runtime arms are implemented.
The independent evidence audit passes and rejects four deliberately corrupt
inputs: missing completion, failed logit equality, timed allocation and a
negative paired confidence bound.

## Measurements

Each case has five contemporaneous alternating pairs on the isolated M40.
Warmup and correctness copies are outside primary CUDA-event timing.
FFN uses ten repetitions per arm; actual vocabulary/argmax uses 100.
Raw records retain GPU process/clock observations, host completion intervals,
counters, hashes and per-trial values. No timed workloads overlapped.

| Single-row FFN layer | Compound control ms | Packet-down ms | Complete FFN gain | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.333891 | 0.324688 | 2.834% | 2.173-2.983% |
| 17 | 0.334186 | 0.324854 | 2.872% | 2.647-3.086% |
| 35 | 0.333808 | 0.324659 | 2.818% | 2.620-2.870% |

Isolated down improves 8.03-8.24%. Complete FFN remains below the historical
5% standalone gate; the approved portfolio exception does not rewrite that
result. Four-row complete FFN gains are 1.817%, 2.451% and 0.030%; layer35's
interval crosses zero. Do not admit a four-row packet runtime from this run.

| Actual generated-step sample | Vocabulary control / packet ms | Argmax control / partitioned ms |
| --- | ---: | ---: |
| 0 | 1.561965 / 1.429427 | 0.172650 / 0.010358 |
| 1 | 1.562146 / 1.429449 | 0.172754 / 0.010353 |
| 8 | 1.562285 / 1.429433 | 0.172999 / 0.010346 |
| 32 | 1.562584 / 1.429418 | 0.173242 / 0.010396 |
| 128 | 1.562029 / 1.429457 | 0.173409 / 0.010310 |
| 256 | 1.562336 / 1.429479 | 0.173681 / 0.010372 |
| 511 | 1.562780 / 1.429512 | 0.175180 / 0.010460 |

Actual vocabulary improves 9.272-9.323%; argmax throughput is 16.66-16.82x
the control. Every interval is positive and no operator case requests a
variation extension. These repeated isolated calls do not reproduce all
whole-token cache conditions. In particular, actual-input argmax control is
about 0.173 ms here, not the earlier sampled whole-token 0.260 ms.

Use contemporary absolute costs as a conservative alternative to the
earlier profile-transfer estimate. Median complete-FFN saved time across
three layers, multiplied by 36 layers and 512 tokens, gives 169.633 ms.
Seven-sample median vocabulary and argmax savings give 68.020 and 83.377 ms.
Their non-overlapping total is 321.030 ms, a conditional 3.297% HTTP gain.
This assumes representative samples and independent transfer with no
integration overhead. It is neither a measured HTTP gain nor a combined
confidence interval. It does not justify adding the larger historical
argmax estimate or the isolated-down saving to complete FFN a second time.

## Resources And Remaining Gates

One vocabulary packet adds 330,612,736 device bytes. Its measured CPU packing
plus inverse check is 1,032.911 ms, upload/allocation 47.109 ms. Each sampled
down packet adds 23,953,408 bytes, with 64.886-67.215 ms packing/inverse and
3.822-4.001 ms upload. The benchmark retains host packet copies and an
inverse temporary of the same payload size for validation. These are cold
setup costs, not steady-token work.

The intended full portfolio still adds exactly 1,192,935,424 device bytes:
36 down packets plus vocabulary. This full simultaneous allocation and its
setup high-water mark have NOT been measured yet. The operator screens
retain originals in both arms, with only one sampled down packet at a time.

Next implement only the five approved runtime arms: compound control,
down-only, vocabulary-only, argmax-only, all-three. Keep equal retained
allocations across timing arms and restrict packet dispatch to the qualified
single-row path. Require actual IDs/counts/stops, all twelve frozen task
outputs, sampled full logits, valid KV, unchanged compiled controls,
prefill/ownership/cleanup and full residency checks before HTTP timing.
Use paired variation rules and fresh 20-pair confirmation if the combined
result is borderline. Stop on failure without adding portfolio components.

## Artifacts And Reproduction

- [Actual-logit correctness](2026-10-29-portfolio-actual-logits.jsonl)
- [Compound-rebased complete FFN and surrogate-head timing](2026-10-29-compound-packet-operators.jsonl)
- [Actual-input vocabulary/argmax timing](2026-10-29-portfolio-actual-operators.jsonl)
- [Independent audit](2026-10-29-portfolio-operators-audit.jq) and [derived results](2026-10-29-portfolio-operators-audit.json)

Measured operator executables are retained under ignored
`local-artifacts/2026-10-29-portfolio-operators/`. `joint_only` SHA256 is
`6bc43d1713d5a6b80e9ce706abdf8143ae6ae02a7f443cb12650afc63960fe7e`;
`q8_execution_bench` is
`d96f5ec1eff7ef65bb96781a82fac81ceff29c4e2f6a01808454b6089b4d3865`.
The initial correctness-only executable was rebuilt before retention;
the retained operator runner repeats all its checks and matches all seven
logit hashes. The immutable reference binaries were not touched.

Source `scripts/dev-env.sh`, restore `.configuration.environment` from
`2026-10-27-compound-reference.json`, then use the existing model path:

```bash
cargo build --release --features cuda,server --example joint_only
target/release/examples/joint_only --model "$MODEL" \
  --phase portfolio-logits --output /tmp/new-actual-logits.jsonl
target/release/examples/joint_only --model "$MODEL" \
  --phase portfolio-operators --quality-report /tmp/new-actual-logits.jsonl \
  --output /tmp/new-actual-operators.jsonl
```

For the FFN runner, restore the older frozen capture environment, not the
runtime environment. Follow the [packet reproduction](2026-10-26-maxwell-warp-packet-screen.md#reproduction-and-artifacts),
adding `--prepared-compound-control` and using a new output filename.
Do not run these commands concurrently. The exact portfolio is still in
progress; approximate ILP4 and the drafter cost probe have not begun.
