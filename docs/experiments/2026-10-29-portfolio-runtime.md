# Exact Portfolio Runtime Qualification

## Decision

The fixed three-component portfolio passes its approved exact-output,
ownership and combined buffered HTTP performance gates. Retain it as an
explicit research configuration. Ordinary runtime defaults and joint-only
preferred ownership are unchanged. No new CUDA kernel was needed.

Fresh 20-pair confirmation measures **52.548523 versus 50.523256 HTTP tok/s**,
**+4.008582%**, with paired 95% interval **[3.933599%, 4.057069%]**.
This is a measured combination, not a sum of historical or standalone gains.
The contemporary control is lower than the immutable 50.910426 checkpoint.
The candidate is 3.217606% above that historical median, but that comparison
is not paired and must not be labeled a 4.01% gain over the immutable result.

Neither 53.06 nor 56.153149 HTTP tok/s is reached. Relative to the unchanged
historical official Ollama 43.194730 result, this candidate is 21.654940%
faster; no new Ollama run was performed. Preserve the immutable reference,
original evidence and larger exact-dense goal.

This completes the exact-portfolio experiment. The next approved experiment
is the single fixed approximate ILP4 operator, with its separate numerical
contract and 5% complete-FFN gate. Do not expand this portfolio or treat an
approximate result as progress on the exact-output milestone.

## Implementation And Correctness

Runtime adapter `0ad6119` installs one immutable, model-local research arm
before sharing the model. Runner `1b1890a` adds the five-arm quality and HTTP
phases. Both are explicit APIs, not environment or generation defaults.
All arms retain original down/head owners, joint gate/up pairs, and all 37
packet owners. Every research session allocates the same 384-byte argmax
scratch once, including control and arms that do not use it.

Packet dispatch changes only single-row down or vocabulary projections.
Multi-row packed-prefix projections keep the original consumers. Partitioned
argmax replaces the final reduction only; output norm and selected full-head
logits still execute. Model-local ownership avoids process-global selection
and per-token scratch allocation. Unsupported graph, compressed-KV and
batched-server configurations fail rather than silently changing behavior.

All five actual Qwen arms pass:

- The frozen 65 prompt IDs, all 512 generated IDs, output, count and length stop.
- Seven sampled 151,936-entry full-vocabulary vectors at steps 0, 1, 8, 32,
  128, 256 and 511. All 35 vectors are bit-identical to the frozen original
  hashes and contemporary control vectors; diagnostics use the selected head.
- All 10,616,832 valid FP16 KV words at sequence length 576, including replay.
- All twelve frozen task outputs and contemporary token IDs. Seven tasks are
  dense-valid; five remain inconclusive, not newly counted as quality passes.
- Original joint hashes, packet source identities/inverse checks, packet
  readback hashes, warm allocation checks and zero tracked teardown per arm.
- Exactly the admitted dispatch changes. The packed SGEMM prefix, 90,177,536
  temporary weight bytes, copies, synchronizations and stream waits remain
  unchanged. Research requests add one scratch allocation/free; warm tokens
  add none. Partitioned argmax adds one launch per generated token.

All 138 compiled CUDA instruction streams match the preserved controls.
This establishes instruction identity, not achieved occupancy or DRAM traffic.
All 34 immutable-reference artifact, executable, source archive and model
checksums still pass. No immutable file or measured binary was replaced.

## Buffered HTTP Results

Release Rust, CUDA `-O3`, `sm_52`/`compute_52`, no fast math, and the frozen
compound environment are unchanged. Dense KV and joint-only are explicit.
Both resident models hold equal packet storage. Requests execute serially on
the isolated M40; no builds, inference elsewhere, or MemPalace writes overlap
timing. Every recorded before/after state is P0, 1113 MHz SM, 3004 MHz memory.

Warmup, model loading, packet preparation, readback audits and final replay
are outside HTTP timing. Each timed interval includes the complete buffered
response for 65 prompt tokens and 512 generated tokens. All 120 timed requests
preserve output, counts, stop reason and the corresponding quality dispatch.
Actual IDs and packet hashes are rechecked after timing; all four comparisons
release both servers/models and return tracked allocations to zero.

| Candidate | Final cohort | Control HTTP tok/s | Candidate HTTP tok/s | Gain | Paired 95% interval |
| --- | --- | ---: | ---: | ---: | ---: |
| Down only | 20 fresh pairs | 50.824655 | 51.941912 | 2.198257% | 2.133825-2.245313% |
| Vocabulary only | 5 pairs | 50.861317 | 51.236681 | 0.738016% | 0.510423-1.011531% |
| Argmax only | 5 pairs | 50.854134 | 51.543311 | 1.355204% | 1.221538-1.463745% |
| All three | 20 fresh pairs | 50.523256 | 52.548523 | 4.008582% | 3.933599-4.057069% |

The down screen was +2.274911%, interval [2.168066%, 2.305403%]. The combined
screen was +3.982844%, interval [3.736892%, 4.258221%]. Both fell inside the
predefined one-percentage-point confirmation band around the 3% threshold,
so each received twenty fresh pairs. Screens and confirmations are retained
separately and never pooled. No case needed a variation extension. Combined
confirmation MAD/median is 0.058990% control and 0.104849% candidate, leaving
the promotion threshold at 3%. Only the combined arm must exceed that gate.

For combined confirmation, median reported prefill is 504.698/502.573 ms,
decode 9578.993/9193.193 ms, and generation total 10082.954/9695.818 ms for
control/candidate. Median buffered HTTP time is 10133.947/9743.376 ms.
These phase medians are descriptive and are not added to reconstruct a
request median. This is not a separate resident-only benchmark.

## Resource Cost

Each arm adds exactly **1,192,935,424 device bytes**, about 1.111 GiB:
862,322,688 for 36 down packets and 330,612,736 for vocabulary. The original
37 split owners and 36 joint pairs remain, so prepared payload is
4,110,516,224 bytes within the unchanged 6,442,450,944-byte budget.
Original uploaded GGUF weight payload is 3,279,519,744 bytes and is preserved
for prefill. Cold tracked weight residency is therefore 7,390,035,968 bytes.

The native dense KV payload is separately reported as 37,748,736 bytes.
Do not call the Rust tracked-weight counter total driver residency: native
KV/control storage, CUDA/cuBLAS state and later workspaces also consume VRAM.
Raw HTTP GPU process observations retain driver-reported resident memory.

Across the five quality loads, original loading/preparation costs 16.4-17.8 s.
Additional packet preparation costs 6.389-6.621 s, including approximately
6.194-6.420 s packing/inverse validation/hashing and 0.174-0.190 s upload.
Cold hash audits add work outside these preparation timers. This is not free
setup and is not included in warm HTTP throughput.

Packet construction retains only one host packet plus its inverse temporary,
with a maximum payload of 661,225,472 bytes. It never holds all host packets.
The original preparation payload bound remains 6,895,608,736 bytes. The
process-lifetime host high-water observations are 7,007,420-7,135,040 KiB;
they include the model, allocator and prior work, not just packet scratch.
Packet setup device allocation is monotonic, 74 allocations and no frees,
so its final tracked level is its tracked setup high-water level. This is not
an instrumentation claim about all native CUDA allocation peaks.

## Validation And Reproduction

Synthetic runtime coverage includes sequential and packed-prefix prefill,
exact logits/greedy/KV bits, stream routing, allocation-free warm calls,
immutable owners, budget/upload failure cleanup and unmodified multi-row
dispatch. The three runtime tests pass with cuBLAS enabled and disabled.
The attention grid passes 11 cases; affected compound 3, logits-stream 1,
prepared-greedy 1 and packet 2 cases pass. Two portfolio library cases pass.
The research runner passes 10 non-hardware cases; its separate actual-logit
hardware probe passed at the preceding operator checkpoint. Non-CUDA
warning-as-error tests, CUDA/server Clippy and formatting pass. The updated
non-CUDA example has zero tests, so its result is a build check only.

Independent quality and HTTP audits pass and reject nine corrupted evidence
variants: incomplete reports, changed logits/IDs/resources, negative paired
bounds, incorrect arm binding, output and timing-rate changes. The HTTP audit
recomputes medians and gains from raw pairs; paired bootstrap intervals come
from the existing seeded, tested 10,000-resample Rust protocol.

- [Quality JSONL](2026-10-29-portfolio-runtime-quality.jsonl),
  [audit](2026-10-29-portfolio-quality-audit.jq),
  [derived results](2026-10-29-portfolio-quality-audit.json).
- [HTTP JSONL](2026-10-29-portfolio-runtime-http.jsonl),
  [audit](2026-10-29-portfolio-http-audit.jq),
  [derived results](2026-10-29-portfolio-http-audit.json).
- [Compiled-control audit](2026-10-29-portfolio-compiled-audit.json) and
  [artifact checksums](2026-10-29-portfolio-runtime.sha256).

The measured source is `1b1890ad7f2dac99b10536dccdf2a01ae5d6d623`. The executable
is retained at `local-artifacts/2026-10-29-portfolio-runtime/joint_only`, SHA256
`21cb3e53fe9c9a9f7d9bbcedc93da4e84fd4fb2fd50807b676815ab534f5ab46`.
That ignored directory also retains stderr and disassembly. The initial
quality preflight rejected an untracked binary copy before loading a model;
moving the copy outside the worktree resolved bookkeeping without a code or
gate change. It was not a failed numerical/performance experiment.

Use a clean checkout of the measured source, source `scripts/dev-env.sh`, and
restore the environment entries from `2026-10-27-compound-reference.json`.
Use the existing content-addressed Qwen model from that manifest:

```bash
cargo build --release --features cuda,server --example joint_only
target/release/examples/joint_only --model "$MODEL" \
  --phase portfolio-quality --output /tmp/new-portfolio-quality.jsonl
target/release/examples/joint_only --model "$MODEL" \
  --phase portfolio-http --quality-report /tmp/new-portfolio-quality.jsonl \
  --output /tmp/new-portfolio-http.jsonl
```

The HTTP phase requires quality from the same source and executable. Do not
overwrite old reports. No training, model download, A4000 switch, new Ollama
comparison, approximate-numerics qualification or runtime-default promotion
occurred in this experiment.
