# Fixed Compound FFN Clears The Microbenchmark Gate

## Decision

The single [authorized compound](2026-10-26-maxwell-compound-ffn-proposal.md)
passes the unchanged >=5% complete one-row FFN gate at all three frozen
layers, after twenty fresh confirmation pairs. Total throughput gain over
the qualified split control is **5.43-5.55%**; incremental gain over the
contemporaneous joint-only arm is **0.58-0.67%**. Both comparisons have
positive paired confidence bounds. The individual gains are not additive.

Keep implementation `c629471` **benchmark-only**. Passing this gate allows
separate runtime qualification; it does not prove final-model actual512
equality, HTTP improvement or a promoted configuration. No generation
selector, default change or E2E claim was added. Preserve the qualified
**49.146 HTTP tok/s** configuration. The original **53.06** and **56.037**
HTTP tok/s targets remain unmet.

## Contemporaneous Four-Arm Results

Five initial pairs clear numerical/resource/isolation checks, but one-row
point gains of 5.51-5.96% are within one percentage point of the gate.
Therefore run exactly twenty **fresh** confirmation pairs, without pooling.
Maximum confirmation MAD/median throughput variation is 1.91%, on rows4;
no case exceeds the 5% variation limit. Warm every arm for at least one
second of GPU work before primary trials. Rotate/reverse arm order and
alternate stage order, repeating every timed stage ten times.

Confirmation complete-FFN CUDA-event medians in milliseconds:

| Layer | Rows | Split | Joint only | Shared only | Compound | Total vs split | Increment vs joint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.352203 | 0.335920 | 0.344475 | 0.333693 | +5.55% | +0.67% |
| 17 | 1 | 0.352227 | 0.336030 | 0.344816 | 0.334077 | +5.43% | +0.58% |
| 35 | 1 | 0.352158 | 0.335811 | 0.344562 | 0.333837 | +5.49% | +0.59% |
| 0 | 4 | 1.589674 | 1.528648 | 1.554624 | 1.516752 | +4.81% | +0.78% |
| 17 | 4 | 1.599043 | 1.526546 | 1.550214 | 1.515562 | +5.51% | +0.72% |
| 35 | 4 | 1.539002 | 1.476861 | 1.505718 | 1.467770 | +4.85% | +0.62% |

One-row paired 95% bootstrap intervals, 10,000 resamples, seed zero:

| Layer | Total gain interval | Incremental gain interval |
| --- | ---: | ---: |
| 0 | +5.46% to +5.64% | +0.59% to +0.72% |
| 17 | +5.37% to +5.51% | +0.53% to +0.63% |
| 35 | +5.42% to +5.53% | +0.53% to +0.64% |

Four-row incremental intervals cross zero at layers0/17; do not claim a
consistent four-row incremental win. Four-row results cannot substitute
for the one-row gate. Isolated gate/up/SwiGLU is about 0.20315-0.20319 ms
compound versus 0.20512-0.20520 joint and 0.22132-0.22144 split. Prepared
down is unchanged. Separate stage medians need not sum to complete FFN.

## Numerical, Resource And Memory Gates

All four arms preserve gate, up, SwiGLU and completed-down bits on genuine
distinct rows1/4 at layers0/17/35. CPU-reference errors remain within the
original tolerance, `1e-3 + 2e-5 * abs(reference)`: maxima are 2.07e-6 gate,
3.41e-6 up, 1.72e-5 SwiGLU and 5.01e-5 down. Optional intermediate capture
and ordinary output match. Primitive tests include signed extremes,
zero/negative-zero scales, K=32/96/2048/4096, odd output tails, producer
stream waits, graph replay, guards, invalid spans and rejected K=4128.

Only one new CUDA consumer is compiled. All 135 original entry points retain
identical encoded words, including the 560-word split control hash
`31cd38f6ca0d4a0bebd03206f484bf8045834d8af592a645cdb168cfaf02e321`.
The compound has 544 words, hash
`d50d6a72d63469a4405a5991e5517770da6a681737abe18e85c81066adbef9ac`.
All four consumers use 32 registers/thread without reported stack/local
storage. At K=2048, split/joint use 1024 dynamic shared bytes and one block
barrier; shared/compound use 1536 and two. This is static resource evidence,
not measured occupancy, warp stalls, cache transactions or DRAM bandwidth.

Both split and joint owners remain resident in **every** arm. Each pair is
47,906,816 bytes; combined preparation is 95,813,632, below 128 MiB. Joint
scales occupy 2,818,048 bytes and payloads 45,088,768. Source GGUF and all
37 qualified down/vocabulary identities remain unchanged. Guarded scratch
is 201,152 / 803,264 bytes for rows1/4, and tracked totals are
4,568,469,952 / 4,569,072,064. Host prepared owners retain another
95,813,632 bytes. Packing/upload costs are recorded outside primary timing.
No timed enqueue allocates, frees, copies or synchronizes, nor adds cuBLAS
calls. Final tracked bytes return zero to zero, and GPU is empty after exit.

## Validation And Reproduction

- Primitive tests: 2/2 with cuBLAS on and off. Compound example checks: 3/3
  with each setting, including actual complete-FFN dispatch and ownership.
- Example CPU/CLI suite: 40 pass, 18 hardware tests ignored in that invocation;
  the new hardware test was explicitly included separately.
- Attention 11, forward 20, MLP 1 and projection 2 pass. Four optional graph
  tests remain ignored. Non-CUDA warning-as-error tests: 217 pass. CUDA/server
  Clippy, formatting and diff checks pass.
- Initial generic regression run rejected the frozen head128-only override
  on head64 fixtures. An attempted clean environment also removed dev-env's
  compiler sysroot and failed during compilation. Both logs are preserved.
  Re-source dev-env after clearing runtime overrides; all regressions then
  pass. No kernel or threshold was changed to resolve either setup issue.
- Independent raw-record audits validate source/fixture/cache identity,
  all-arm coverage, raw summaries, dispatch, memory and serial timestamps.
  Twelve deliberate corruption controls are rejected for each run.

Clean source: `c6294717e423057cec566cd7951fb0031e12d18e`; release executable
SHA-256 `a660dd9184307ad3440b5913007ee2f241c003b563f9b20b377423ed5b44ca4a`.
NVCC 12.8.93, `-O3`, sm_52 SASS/compute_52 PTX, no fast math. GPU boundaries
show only the owned benchmark PID, P0, 1113 MHz SM / 3004 MHz memory. Each
run is serial: screen 360 primary intervals/60 boundaries, confirmation
1440/240. No build, test, MemPalace call or other GPU work overlaps timing.

Source dev-env and restore `.execution_configuration.environment` from the
frozen [capture](2026-10-25-maxwell-ffn-capture.jsonl), then build release
`--features cuda --example q8_execution_bench`. Use the same model/fixture
arguments as the [shared-scale screen](2026-10-26-maxwell-shared-scales-ffn-screen.md#reproduction-and-artifacts),
replacing `--ffn-f32-shared-scales` with `--ffn-f32-compound`. Keep
`--rows 1,4 --iterations 10`, run `--trials 5` and twenty fresh confirmation
pairs to a different output path. No new search or extension is authorized.

Artifacts: [screen raw](2026-10-27-maxwell-compound-ffn-screen.jsonl),
[screen audit](2026-10-27-maxwell-compound-ffn-screen-audit.json),
[confirmation raw](2026-10-27-maxwell-compound-ffn-confirm.jsonl),
[confirmation audit](2026-10-27-maxwell-compound-ffn-confirm-audit.json),
[audit program](2026-10-27-maxwell-compound-ffn-audit.jq),
[corruption controls](2026-10-27-maxwell-compound-ffn-negative-controls.sh),
[instruction audit](2026-10-27-maxwell-compound-ffn-sass-audit.json),
[candidate SASS](2026-10-27-maxwell-compound-ffn-candidate.sass.gz),
[resources](2026-10-27-maxwell-compound-ffn-resources.txt.gz),
[orchestration](2026-10-27-maxwell-compound-ffn-orchestration.tar.gz), and
[checksums](2026-10-27-maxwell-compound-ffn.sha256).

## Next

The user requested the independent
[exact multi-token verification study](2026-10-27-maxwell-exact-verification-plan.md)
after this checkpoint. Start with a perfect-drafter upper bound and causal,
KV ownership/rollback and exact greedy-token parity. Runtime integration of
the compound remains a separate eligible step, not an assumed E2E gain.
