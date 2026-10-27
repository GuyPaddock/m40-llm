# Qualified FFN Has Small Tested Reuse-Order Effects

## Finding

Grouping ten calls to each layer versus alternating those same calls has
no material effect on the tested frozen-input FFN work. Complete-FFN median
latency changes **-0.006% to -0.060%**. All nine stage confidence intervals
cross zero and lie inside **+/-0.45%**, well within the prespecified +/-3%
diagnostic threshold. No fresh borderline confirmation is needed.

This closes a material temporal-order bias for these three pairs, not every
cache condition or a true 36-layer forward. There is no hidden-state
dependency between the independent pipelines, attention, logits, KV or
generated tokens. No measured cache traffic, DRAM saturation or occupancy
claim follows. The result cannot reopen rejected candidates or bypass
original complete-FFN, actual512 quality and HTTP gates.

## Five-Pair Release Results

CUDA-event milliseconds normalized by **20 actual layer operations** per
interval: ten calls to each layer. Grouped order is A^10 then B^10;
alternating is (AB)^10. Both retain the same weights, pointers and scratch.
Five contemporaneous trials reverse order within each stage on odd trials.
No data from the three pairs is pooled.

| Layers | Stage | Grouped ms/layer | Alternating ms/layer | Latency change | Paired 95% interval |
| --- | --- | ---: | ---: | ---: | ---: |
| 0,17 | Gate/up/SiLU | 0.221426 | 0.221230 | -0.088% | -0.206% to +0.447% |
| 0,17 | Down | 0.131611 | 0.131702 | +0.069% | -0.054% to +0.306% |
| 0,17 | Complete FFN | 0.351819 | 0.351746 | -0.021% | -0.055% to +0.139% |
| 17,35 | Gate/up/SiLU | 0.221291 | 0.221202 | -0.040% | -0.156% to +0.213% |
| 17,35 | Down | 0.131290 | 0.131470 | +0.138% | -0.038% to +0.340% |
| 17,35 | Complete FFN | 0.351870 | 0.351850 | -0.006% | -0.038% to +0.038% |
| 35,0 | Gate/up/SiLU | 0.221307 | 0.221229 | -0.035% | -0.309% to +0.125% |
| 35,0 | Down | 0.131298 | 0.131278 | -0.015% | -0.152% to +0.226% |
| 35,0 | Complete FFN | 0.351886 | 0.351674 | -0.060% | -0.099% to +0.065% |

Intervals come from 10,000 paired bootstrap resamples with seed zero,
transformed to alternating/grouped latency change. The audit checks raw
pairs, medians, normalization, feasible bounds and decision rules; it does
not independently recreate the RNG's full resample sequence. This is
diagnostic uncertainty, not a throughput promotion.

## Work, Numerics And Ownership

Both orders execute qualified split-f32x4-128 gate/up and original prepared
down on the decode stream. Every stage interval has 20 gate/up or down
calls, or 20 of each for complete FFN. Enqueue counters have zero device
allocation/free, copy, cuBLAS, stream wait or stream synchronization.
Completion uses events outside enqueue counters. All 90 primary intervals
are serial, with no overlapping inference, builds/tests or MemPalace work.

Both layers independently pass original full-output CPU bounds and
intermediate-capture parity. Every timed arm's outputs match its reference
bits. Inputs, guards and all 37 qualified down/vocabulary cache identities
remain unchanged. Each order warms complete FFN for at least 1000 GPU ms
under the existing finite budget, separate from primary data.

Two original gate/up preparations occupy **95,813,632 bytes**, below
128 MiB. Separate one-row scratch totals **402,304 bytes**. Source plus the
unchanged prepared cache is **4,472,455,168 bytes**; tracked resident total
is **4,568,671,104 bytes** in both arms. Each pair releases its extra owners
before the next pair; final tracked bytes return zero to zero. These are
requested payloads, not peak VRAM or traffic.

## Provenance And Validation

Clean source `75ffdf78d37a3ff1fd691b1ab611ac4633818509`; release executable
SHA-256 `db70d00408f9444b753b8b1a81a1692d22d5178c09823e7aae812ce67066c554`.
The frozen original Qwen2.5-3B Q8_0 model, M02 capture, fixture and execution
configuration are unchanged. No CUDA or inference code is added. All
**100,400 original instruction words / 135 entries** retain aggregate hash
`7a954317700bf84cdc40440471d00b2dfb2b5b126cfb562e88c5566b8449970a`.
NVCC 12.8.93 keeps `-O3`, sm_52/compute_52 and no fast math.

CPU tests pass **38**, with 17 hardware checks ignored in that invocation,
not counted as passes. The explicit new two-layer hardware test passes
with cuBLAS on and off: distinct outputs, order parity, work counts, guards,
reuse and cleanup. Warning-as-error non-CUDA **217**, Clippy and formatting
pass. The independent artifact audit rejects twelve corrupted reports.

The first command stopped before loading the model because the shared
entry guard still required rows1/4. `75ffdf7` fixes admission and adds a
regression without weakening baseline or independent-mode validation.
No GPU timing/output file existed for that failed preflight; none is pooled.

Successful session **11163**, PID **3614633**, terminates zero. All thirty
measured GPU boundaries show only that process, P0, 1113 MHz SM / 3004 MHz
memory. The final in-process record distinguishes zero Rust payloads from
remaining driver context memory; a separate post-exit query finds no GPU
compute process. No clocks or driver settings are changed.

## Reproduction And Evidence

Source `scripts/dev-env.sh` and restore the frozen M02 environment as in the
[original FFN read reproduction](2026-10-26-maxwell-ffn-read-screen.md#reproduction-and-evidence).
Build release `q8_execution_bench`, then use the original explicit model,
fixture and capture paths with:

```bash
target/release/examples/q8_execution_bench \
  --model "$MODEL" --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-layer-reuse --rows 1 --trials 5 --iterations 10 \
  --output /tmp/new-layer-reuse.jsonl
```

The output must not exist. Request escalation and verify GPU exclusivity.
Four-row work is rejected, not silently measured on a different stream.

- [Contract](2026-10-26-maxwell-layer-reuse-hypothesis.md), [raw report](2026-10-26-maxwell-layer-reuse-screen.jsonl), [audit](2026-10-26-maxwell-layer-reuse-audit.sh) and [derived results](2026-10-26-maxwell-layer-reuse-audit.json)
- [Timing log](2026-10-26-maxwell-layer-reuse-screen.log.gz), [failed preflight](2026-10-26-maxwell-layer-reuse-preflight-failed.log.gz), [build](2026-10-26-maxwell-layer-reuse-build.log.gz) and [checksums](2026-10-26-maxwell-layer-reuse.sha256)
- [CPU tests](2026-10-26-maxwell-layer-reuse-cpu-tests.log.gz), [CUDA on](2026-10-26-maxwell-layer-reuse-cuda-on.log.gz), [CUDA off](2026-10-26-maxwell-layer-reuse-cuda-off.log.gz), [non-CUDA](2026-10-26-maxwell-layer-reuse-noncuda.log.gz) and [Clippy](2026-10-26-maxwell-layer-reuse-clippy.log.gz)

Keep the reusable diagnostic; do not change kernels or resurrect closed
candidates from a negligible ordering effect. Qualified exact status stays
**49.146 HTTP tok/s** versus **43.106 official Ollama**. Both **53.06** and
**56.037 HTTP tok/s** targets remain unmet; runtime defaults are unchanged.
