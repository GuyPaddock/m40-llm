# Maxwell Runtime Quality Qualification

Historical checkpoint before the sequential-prefix ownership repair.
Use the [corrected Q8_1 follow-up](2026-10-26-maxwell-q81-corrected-quality.md)
for current numerical results and completed-prefix reports for exact-path
qualification. Raw records below remain unchanged; step 42 is historical.

M08 extends the existing `model_quality` runner rather than adding another
evaluation framework. This checkpoint qualifies candidates for controlled
HTTP timing. It does not establish an E2E improvement or change defaults.

## Configuration and Scope

Release source `3d6833a`, executable SHA-256
`efb679b703df400adaad36f84892f96d65a80b930113e7509606ced8d289ca6c`.
The original Qwen2.5-3B Q8_0 blob has SHA-256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
Both models retain split prepared down/vocabulary, the 4-column/256-thread
native-order down reduction, full dense FP16 KV/context 1024, and the frozen
M02 environment. Uploaded weights are 3,279,519,744 bytes/model, prepared
weights 1,192,935,424, total weight payload 4,472,455,168. This is not a
full FP32 materialization experiment.

Every case independently loads the native/sequential control and one explicit
candidate, warms both, resets KV per input, and generates the canonical
65-token browser prompt with greedy top-k=1 and at most 512 actual tokens.
Stopping remains natural. Full actual IDs, text, stop reason and dispatch
counters are retained. The current control matches all 512 IDs in the earlier
[qualified prepared confirmation](2026-10-25-q8-native-order-http.jsonl).

Read-only full-vocabulary logit diagnostics then replay identical reference
prefixes at generated steps 0, 1, 8, 32, 128, 256 and 511, plus the actual
divergence step if different. These generic-head logits are not substituted
for the production GPU-greedy head. Samples do not establish logit parity at
every browser step.

Both fixed corpus splits are evaluated without rubric or prompt changes.
Corpus SHA-256 is
`2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38`.
Sequential candidates additionally compare full-logit teacher-forced loss
over 102 calibration and 101 held-out tokens. Packed-prefill rows do not
misrepresent sequential passage loss as evidence for their packed path.

Runs are serial and check for unrelated GPU processes before each case.
The active Q8_1/256 check observed only its own PID, SM/memory clocks
1113/3004 MHz and 73 C. No builds, tests, Ollama inference or MemPalace
embedding overlapped the GPU quality runs. HTTP-only source/doc edits during
the later runs did not rebuild or replace the committed quality executable;
its hash above is authoritative. Worktree snapshots record those edits.

## Decode Candidates

| FFN candidate | Browser IDs | First divergence | Dense-valid regressions | Held-out PPL | Sampled maximum logit difference |
| --- | --- | --- | --- | --- | --- |
| Packed FP32, 256 threads | All 512 equal | None | 0 / 7 | 39.612157, unchanged | 0 |
| Packed FP32, 128 threads | All 512 equal | None | 0 / 7 | 39.611313 | 0.005404 |
| Q8_1, 128 threads | Both reach 512, different continuation | 42 | 0 / 7 | 40.670991 | 1.375205 |
| Q8_1, 256 threads | Both reach 512, different continuation | 42 | 0 / 7 | 40.776474 | 1.264428 |

Both FP32 candidates preserve all twelve actual task outputs, not only their
rubric outcomes. Packed256 also preserves the sampled browser and passage
logits exactly. Packed128 has small nonzero drift, including a maximum
0.007503 across the teacher-forced passages; it is not bitwise logit parity.
Its actual 512-token equality and dense-valid task gate pass.

The frozen control passes seven tasks and fails five. A failed reference task
is inconclusive for candidate quality, not a compression/kernel regression.
Q8_1 does not introduce dense-valid task failures here, but calibration PPL
worsens from 31.556846 to 36.478550 / 36.609940 at 128 / 256 threads.
The changed continuation and higher loss keep Q8_1 on its separate numerical
research track regardless of any timing result.

## Independent Prefill Candidates

Decode remains native for both prefill candidates. Both complete 512 tokens
and preserve the seven dense-valid task passes, but both change the first
browser token. Their complete candidate continuations match each other,
not the sequential control. Prompt full-logit maximum/mean differences versus
sequential are 9.957545 / 1.072239 for native packed and 9.957608 / 1.072258
for SGEMM. Neither satisfies the exact-output gate.

| Path | Control prompt interval | Candidate prompt interval | Candidate decode | Actual temporary weight |
| --- | --- | --- | --- | --- |
| Native packed prefix | 1569.36 ms | 3216.44 ms | 12725.53 ms | None |
| Q8-to-FP32 SGEMM prefix | 1565.52 ms | 562.19 ms | 12717.18 ms | 90,177,536 bytes |

These are single resident diagnostic observations, not qualified HTTP
percentages. The SGEMM row includes all 252 weight conversions and 252 cuBLAS
calls; no full-model FP32 cache is retained. It records 42 allocations/frees
over the request, all released. Normal decode keeps the prepared control.
The faster prompt interval does not override the failed exact-output gate.

The similar drift and identical continuations across both packed paths point
to their common prefill execution rather than evidence uniquely blaming the
new converter. Causation is unresolved. Source inspection identifies a
precision difference worth isolating: packed attention consumes FP32
`ws.dk/ws.dv`, while one-token attention reads FP16 KV storage. Do not label
that the measured cause without a controlled ablation. No combined
decode/prefill candidate is advanced while this gate fails, and no tolerance,
prompt, stop behavior or default has been changed to make it pass.

## Evidence and Next Gate

Raw reports:

- [Packed FP32/256](2026-10-25-maxwell-runtime-packed256-quality.jsonl)
- [Packed FP32/128](2026-10-25-maxwell-runtime-packed128-quality.jsonl)
- [Q8_1/128](2026-10-25-maxwell-runtime-q81128-quality.jsonl)
- [Q8_1/256](2026-10-25-maxwell-runtime-q81256-quality.jsonl)
- [Native packed-prefix](2026-10-25-maxwell-runtime-native-packed-quality.jsonl)
- [Temporary-dequantization SGEMM](2026-10-25-maxwell-runtime-dequant-sgemm-quality.jsonl)

The [derived audit](2026-10-25-maxwell-runtime-quality-audit.jsonl) verifies
completed rows, zero tracked cleanup and each reference's equality with the
qualified 512-ID trace. [SHA-256 checksums](2026-10-25-maxwell-runtime-quality.sha256)
cover all six reports, compressed process logs and the audit.

All six completed cases return tracked device allocations to zero and retain
the original prepared down/vocabulary payload. Q8_1 dispatch includes one
activation quantization per layer per one-row step; Qwen scratch is 2304 bytes.
These are not kernel-only timings with quantization excluded.

Reproduction uses the frozen environment from the first row of
`2026-10-25-maxwell-ffn-capture.jsonl`, original Ollama model blob and:

```bash
target/release/examples/model_quality \
  --reference-model "$MODEL" --model "$MODEL" --variant-id m08-packed128 \
  --q8-runtime-candidate packed-f32-128 --q8-runtime-prefill sequential \
  --split all --output /path/to/new-quality.jsonl
```

The other configurations are the two 128/256-thread FFN families above and
`--q8-runtime-candidate native --q8-runtime-prefill native-packed|dequant-sgemm`.
Use a separate explicit command for each prefill value. Source
`scripts/dev-env.sh`, restore all frozen flags and request GPU escalation;
never overlap these commands. Each report records the complete configuration,
source and executable hashes, actual output and current worktree.

The resident generation times in raw rows are single diagnostic observations,
not paired HTTP claims. Run the existing comparison tool on qualified exact
candidates with five alternating pairs, variation extension and fresh
borderline confirmation. Retain the 3%/twice-variation threshold and positive
95% paired confidence interval. Only then test a qualifying decode/prefill
combination and obtain a fresh controlled Ollama comparison. Both 53.06 HTTP
tok/s and 1.30x the fresh Ollama median remain unachieved.
