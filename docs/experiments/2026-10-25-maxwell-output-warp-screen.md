# Native Output Warp Reduction Screen

## Decision

The fixed four-column/256-thread output kernel passes its independent
stage gate: **30.42-31.57%** one-request throughput gain with positive
paired intervals and bitwise-equal control outputs. Combined with the
prespecified positive QKV result, its optimistic profile-weighted
whole-token gain is **4.68-4.72%**, above the unchanged 3% feasibility gate.
This permits a combined opt-in runtime experiment, not an HTTP speed claim.

No generation dispatch or defaults changed. The qualified split-FFN/SGEMM
runtime remains **45.075 HTTP tok/s** versus fresh Ollama **43.119**. Both
original objectives remain unmet: 53.06 and 1.30x fresh Ollama, currently
56.055. Before timing any combined runtime, require actual 512-token
equality, frozen dense-valid quality, identical prepared gate/up/down/head,
and the current split-FFN/SGEMM control. Original measured HTTP gates and a
fresh official Ollama comparison still apply.

## Reproduction And Paired Results

Clean release source `12191f55c1aedd917e8a83048b989bbc89802384`, executable
SHA-256 `8d0995e317beaa17e37dcf4d1632abf61c9b867f2dc46746951a3cfeef82241d`.
The [pre-implementation hypothesis](2026-10-25-maxwell-output-warp-hypothesis.md)
fixes the original four-column, split4, 256-thread assignment, FP32
activations, raw 34-byte weights, scale/FMA operations and reduction tree.
Only the finish changes to a single publication barrier and warp reduction.

The existing QKV probe is generalized internally, not copied. Both modes
share ownership, CPU validation, immutable-span/guard checks, event timing
and operation-count verification. The output mode binds the original QKV
raw report SHA-256
`9c1468a93abc8b4828e04d5880f1d1fbb70c36fe4790c73c27087a0098217b73`.
That peer is fixed evidence, not a newly selected arm or trial subset.

```bash
source scripts/dev-env.sh
while IFS=$'\t' read -r key value; do
  export "$key=$value"
done < <(head -n 1 docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl |
  jq -r '.execution_configuration.environment | to_entries[] | [.key,.value] | @tsv')
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --output-f32-warp \
  --qkv-warp-report docs/experiments/2026-10-25-maxwell-qkv-warp-screen.jsonl \
  --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-output-warp-screen.jsonl
```

Real layer 0/17/35 output weights have K=N=2048. Frozen normalized **FFN**
inputs are disclosed projection surrogates, not captured attention outputs.
Four rows means four distinct independent one-token requests and four
launches, not one batched GEMM. Source and prepared down/vocabulary remain
resident in both arms; no gate/up or output weights are newly prepared.
This is not the full selected runtime's memory footprint.

Five alternating pairs use ten repetitions per arm. Times are median group
GPU milliseconds, including launch/dependency scheduling. JSONL also
reports per-request time. Gain is control/candidate minus one; paired
bootstrap uses 10,000 resamples and seed zero.

| Layer | Requests | Original ms | Warp ms | Throughput gain | Paired 95% interval | Combined optimistic gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.048451 | 0.037149 | +30.42% | +28.19% to +31.41% | +4.680% |
| 17 | 1 | 0.048429 | 0.036810 | +31.57% | +25.47% to +35.49% | +4.718% |
| 35 | 1 | 0.048899 | 0.037286 | +31.14% | +25.51% to +32.27% | +4.709% |
| 0 | 4 | 0.192675 | 0.151878 | +26.86% | +26.34% to +27.69% | +4.405% |
| 17 | 4 | 0.192499 | 0.150950 | +27.52% | +26.50% to +28.12% | +4.419% |
| 35 | 4 | 0.193446 | 0.151834 | +27.41% | +24.97% to +28.15% | +4.434% |

One-request latency falls about 23.3-24.0%, not 30-32%. The point gains
and lower intervals are well above the 5% stage gate; no borderline trial
extension or geometry search is warranted. Combined estimates use the
matching layer/request QKV gain and fixed profile fractions:

`1 / (1 - 0.1172 - 0.0825 + 0.1172/(1+qkv_gain) + 0.0825/(1+output_gain)) - 1`.

This is an optimistic model based on an instrumented middle token and
isolated surrogate-input kernels. It is neither a forecast nor measured
whole-token gain. Only the three one-request cases authorize proceeding;
four-row work is independent evidence, not runtime batching qualification.

## Exactness, Instructions And Ownership

All six real-weight cases preserve original output bits. Maximum CPU
absolute/mean errors are 1.18519e-6 / 1.42744e-7, within existing tolerance.
Input, source weights and guard regions remain unchanged. Primitive tests
cover every four-column tail, zero scales, signed extrema, two-byte-aligned
raw weights, ordered asynchronous producers on both streams, rejected
invalid spans without effects, and complete RAII cleanup.

Both output kernels use 40 registers, zero reported stack/local storage,
and 4096 dynamic shared bytes. The candidate has one `BAR.SYNC` and twenty
`SHFL.DOWN` sites. The control has two barrier sites; its reduction loop
executes eight times, giving nine dynamic barriers including publication.
Original output SASS stays byte-identical, SHA-256
`a122594fde39a84160359f5035fbd40882a1df7098a5a4c45988858ed6e9dc1e`.
Original QKV SASS also stays unchanged,
`b40f55cf110eeff0d125b246b1e8fc54fd517e12cd98d1ca51677f7d9dca509f`.
These are instruction/resource observations, not measured occupancy or DRAM
traffic. No upstream code was copied for this candidate.

Original weights use 3,279,519,744 bytes and prepared down/vocabulary
1,192,935,424 bytes. Guarded scratch is **16,512 / 65,664 bytes** for one/
four requests; tracked totals are 4,472,471,680 / 4,472,520,832 bytes.
There is no KV, activation quantization, extra execution layout, or timed
allocation/free/copy/cuBLAS/stream-wait/stream-sync. Each arm checks the
actual operation counter, ten or forty launches, not just its label.
Event completion is included in GPU elapsed time but outside enqueue
counters. Tracked payload returns zero to zero. Driver context memory
remains visible in-process; the compute-process list is empty after exit.

All sixty primary boundaries show the sole benchmark PID 1444020, P0,
1113 MHz SM / 3004 MHz memory. Sixty primary intervals do not overlap.
No tests, builds, Ollama or MemPalace work overlapped the timed run.

## Validation And Artifacts

CuBLAS on/off pass two output tests, two native QKV tests, one prepared-QKV
test and fifteen shared-probe tests. The broader CUDA run passes eleven
attention-grid, twenty-four forward, ten FFN-runtime, two FP32-FFN, two
projection-wrapper and one MLP-wrapper tests. Non-CUDA warning-as-error
tests pass 210 cases; CUDA/server all-target Clippy and formatting pass.
The initial peer-binding test had a mistyped capture hash; it was corrected
before the implementation commit. No timed data was collected from that
failed test run. These tests do not substitute for actual-model generation.

- [Raw pairs and provenance](2026-10-25-maxwell-output-warp-screen.jsonl)
- [Audit program](2026-10-25-maxwell-output-warp-audit.jq) and [result](2026-10-25-maxwell-output-warp-audit.json)
- [SASS](2026-10-25-maxwell-output-warp-sass.txt.gz), [resources](2026-10-25-maxwell-output-warp-resources.txt), [log](2026-10-25-maxwell-output-warp-screen.log.gz) and [checksums](2026-10-25-maxwell-output-warp.sha256)
- [Fixed QKV peer](2026-10-25-maxwell-qkv-warp-screen.md)

The audit verifies bindings, actual pairs/dispatch, original stage and
combined gates, residency, numerical evidence, clocks and isolation.
Wrong operation, timed allocation, and inflated combined-gain negative
controls are rejected. Next implement one immutable combined QKV/output
choice while retaining current split-FFN/SGEMM as explicit control, then
run actual-token/frozen-quality checks before any new HTTP qualification.
