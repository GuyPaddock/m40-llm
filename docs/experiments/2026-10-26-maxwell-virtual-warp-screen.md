# Register-Owned Virtual FFN Warps Regress

## Decision

The [prespecified ownership experiment](2026-10-26-maxwell-virtual-warp-hypothesis.md)
preserves the qualified control's gate/up/SwiGLU and complete-output bits,
but **reduces one-row complete-FFN throughput by 5.36-5.40%**. All paired
intervals are negative. Four-row measurements also regress. The fixed
mapping fails every 5% complete-FFN gate and stays benchmark-only.

| Layer | Rows | Control complete ms | Virtual-warp complete ms | Throughput change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 1 | 0.351837 | 0.371747 | -5.356% | [-5.601%, -5.243%] |
| 17 | 1 | 0.352077 | 0.372192 | -5.405% | [-5.548%, -5.283%] |
| 35 | 1 | 0.351907 | 0.371917 | -5.380% | [-5.584%, -5.320%] |
| 0 | 4 | 1.511290 | 1.598029 | -5.428% | [-6.473%, -4.385%] |
| 17 | 4 | 1.521376 | 1.589174 | -4.266% | [-5.788%, -3.990%] |
| 35 | 4 | 1.603117 | 1.676877 | -4.399% | [-5.794%, -0.822%] |

Five alternating pairs per shape, ten iterations per stage, follow bounded
warmup. These are CUDA-event microbenchmarks, not token or HTTP throughput.
No borderline extension, geometry search, register cap, runtime integration
or combination with another failed candidate is warranted.

## What Changed

One warp now owns one output. Each lane keeps the original partials for
virtual IDs `lane+{0,32,64,96}`, visiting four-value tasks at stride 128.
It combines them in the original `(p0+p2)+(p1+p3)` tree before the existing
five warp shuffles. Four independent output warps share a 128-thread block.
There is no shared publication or block barrier. Weight planes, FP32
activation values, scale conversion, FMA order, SwiGLU, prepared down and
vocabulary are unchanged. No activation quantization is performed.

This is original project code, not a borrowed Ollama kernel. The pinned
generic Maxwell Q8 MMVQ path actually uses four physical warps per output
at these K widths; its independent-warp MoE path does not explain dense
Qwen execution. See the hypothesis's source audit and the earlier
[observed dispatch](2026-10-25-maxwell-q8-dispatch.md).

Gate/up/SwiGLU alone grows from 0.2216-0.2222 ms to 0.2410-0.2413 ms
at one row. Isolated down stays around 0.1312-0.1317 ms. This locates the
regression in the changed stage without claiming a hardware-counter cause.
Four-row down timings vary more; all raw pairs are retained.

## Compiled And Numerical Evidence

The selected release sm_52 entry uses **56 registers**, versus **32** for
the qualified control. Both report zero stack/local storage. The candidate
has ten shuffle instructions, no `BAR.SYNC`, and no shared publication;
the control retains 1024 dynamic shared bytes and its publication barrier.
`DEPBAR` dependency instructions in the candidate are not block barriers.
The register increase and changed work ownership are observations, not
measured occupancy, DRAM traffic or a quantitative attribution of the loss.

Full extracted control SASS is unchanged, SHA-256
`7c2d3c8e46c37367a7ce14d14d865ea5a6476b4ab5baf4e2578b45adfa182b22`.
Candidate SASS SHA-256 is
`fc291a43af624b3500ba4d2747050186ffbb3e9b7c4839b4d21c810dfb428a84`.

All six real-input shapes preserve hidden and completed-down control bits.
Intermediate capture matches ordinary enqueue. Behavioral CUDA tests also
compare gate/up bits, signed bytes, zero scales, partial output-warps,
distinct rows, guards and unsupported arguments. Maximum absolute error
against the scalar FP64 references across real-input shapes is 2.064e-6
gate, 3.409e-6 up, 1.713e-5 SwiGLU and 5.008e-5 complete down, within
the existing tolerance. These are reference errors, not differences from
the bit-identical GPU control. No actual-512 runtime qualification is
claimed for a candidate that failed its prerequisite screen.

## Identity, Isolation And Memory

Clean source `21ca3491412f1782736e6f0a3aba2d32be3e4700`, release benchmark
SHA-256 `59ab040d07ca1af396bea37acc0b13f9da0a8e1053cd6e0adc0e097fcb26975f`.
The frozen M02 model, fixture and capture hashes remain unchanged. Real
gate/up and down tensors come from Qwen layers 0, 17 and 35, with genuine
distinct rows 1 and 4. CUDA uses `-O3`, sm_52/compute_52, no fast math,
CUDA 12.8.93, and the captured environment.

Both arms retain 37 qualified down/vocabulary owners, 1,192,935,424 prepared
bytes plus 3,279,519,744 source bytes. One 47,906,816-byte split gate/up
pair is prepared per shape and shared by both arms. Scratch is 201,152
bytes at one row and 803,264 at four. Tracked per-shape totals are
4,520,563,136 and 4,521,165,248 bytes. The candidate adds no allocation,
weight payload or staging capacity. No KV is allocated in this FFN screen.

Thirty primary pairs contain 180 strictly serial stage intervals. All sixty
GPU boundary samples show P0, 1113 MHz SM and 3004 MHz memory, with only
benchmark PID 2456852. No other inference, tests, builds or MemPalace calls
overlap measurement. Session 39228 exits zero, tracked allocations return
zero to zero, and the subsequent GPU process query is empty. Driver context
memory present in the in-process completion record is not a payload leak.

## Validation And Reproduction

The two FFN parity/rejection tests pass with cuBLAS enabled and disabled.
Attention grid, projection, MLP and forward regressions pass in both builds;
the enabled forward suite retains four explicitly ignored tests, not passes.
The new adapter/argument tests pass, including unchanged cache/scratch and
actual candidate launches. Existing benchmark CPU tests pass. Warning-as-error
non-CUDA tests pass 211 checks; CUDA/server all-target Clippy and formatting
pass. The independent audit checks actual paired timings, stage dispatch,
frozen identities, exact bits, memory, clocks, process isolation and cleanup.

Source `scripts/dev-env.sh`, export the frozen environment from the first
record of `2026-10-25-maxwell-ffn-capture.jsonl`, and build release with CUDA.
Use a new output path:

```bash
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-f32-virtual-warp --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-virtual-warp-screen.jsonl
```

- [Raw measurements](2026-10-26-maxwell-virtual-warp-screen.jsonl)
- [Independent audit](2026-10-26-maxwell-virtual-warp-audit.jq) and [derived results](2026-10-26-maxwell-virtual-warp-audit.json)
- [Control SASS](2026-10-26-maxwell-virtual-warp-control.sass.gz), [candidate SASS](2026-10-26-maxwell-virtual-warp-candidate.sass.gz), [resources](2026-10-26-maxwell-virtual-warp-resources.txt.gz)
- [Run log](2026-10-26-maxwell-virtual-warp-screen.log.gz), [validation log](2026-10-26-maxwell-virtual-warp-validation.log.gz), [checksums](2026-10-26-maxwell-virtual-warp-screen.sha256)

Keep `21ca349` before any later removal. Qualified runtime remains
**49.146 HTTP tok/s versus 43.106 official Ollama**, below both **53.06**
and **56.037** requirements. Defaults remain unchanged. The next task
must establish an independent measured/source-backed hypothesis; do not
reopen this virtual-thread mapping or the prior closed FFN variants.
