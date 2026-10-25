# Joint Gate/Up Packing Hypothesis

## Control Instruction Audit

This audit precedes implementation and timing. The retained one-column
`gate_up_split_f32x4` is the control, not the rejected two-column kernel.
Its sm_52 instructions are preserved in the
[two-column experiment's SASS](2026-10-25-maxwell-split-cols2-sass.txt.gz).
Extract the function whose mangled name contains `19gate_up_split_f32x4`.
Its extracted SHA-256 is
`84a8cfe7d64c3ac38fc52666e7f509a79893422e50f683db591f7e4c4a1c69fd`.
That function was byte-identical before and after the two-column experiment.

In the first four-value task body, the issued global loads are:

| Offset | Instruction | Data |
| --- | --- | --- |
| `02c8` | `LDG.E.CI` | Four signed gate bytes |
| `0318` | `LDG.E.CI` | Four signed up bytes |
| `0338` | `LDG.E.CI.U16` | Original gate scale |
| `0350` | `LDG.E.CI.128` | Four FP32 activations |
| `0358` | `LDG.E.CI.U16` | Original up scale |

The same five-load pattern occurs in each subsequent task body. The compiler
peels the first task and emits two task bodies around the backward branch at
`0a90`; static site totals must not be mistaken for dynamic execution counts.
For Qwen's K=2048, each thread handles four tasks, so the source work is
20 global loads per participating thread before reduction.

The first body also contains eight signed-byte-to-FP32 conversions, two
FP16-to-FP32 conversions, eight scale multiplies and eight FP32 FMAs. Gate
and up have separate address-base additions with carry: payload pairs at
`0298/02b0` and `02e8/0308`, scale pairs at `0310/0328` and `0330/0348`.
Those are concrete removable load/address sites. Pairing does not eliminate
the numerical conversions or the per-weight arithmetic.

## One Bounded Candidate

Store one `uint2` record per four-value task, containing the original gate
bytes followed by the original up bytes. Store one 32-bit scale record per
quantization block, containing both original FP16 bit patterns. Seek one
64-bit payload load and one 32-bit scale load in place of the four separate
weight loads. Keep the 128-bit activation load. The intended total is three
global loads per task rather than five; emitted code must confirm this.

Keep one output column, 128 threads, four FP32 inputs per task, original
signed conversions, per-weight FP32 scaling/FMA, accumulation order,
publication and reduction tree. No activation quantization, scale factoring,
extra column count or geometry search. Preserve both semantic source
identities and reject shape/model mismatches before allocation.

A Qwen layer pair still holds 45,088,768 payload bytes and 2,818,048 scale
bytes, totaling 47,906,816. Microbenchmarks retain both the control and joint
representations for both arms, so their extra residency is 95,813,632 bytes,
within the existing 128 MiB per-case preparation bound. Report that doubled
experimental residency rather than implying runtime memory savings.

## Gate And Limitations

Use the existing frozen inputs, layers 0/17/35, genuine rows 1/4, five
alternating pairs and ten iterations. Compare against one-column split,
require hidden/completed-output bit equality plus existing CPU tolerance,
guards, ownership and no timed allocation/copy/synchronization. Inspect
registers and spills without retuning geometry.

Only a complete one-row FFN improvement of at least 5% with positive paired
confidence intervals justifies runtime integration and the existing actual
512-token quality and HTTP gates. A failed microbenchmark remains recorded
and benchmark-only. Fewer static load sites do not establish lower DRAM
traffic, occupancy or faster execution; CUPTI counters remain unavailable.

The qualified runtime is still split-f32x4-128 plus SGEMM at 45.075 HTTP
tok/s. Both original targets remain unmet: 53.06 and 1.30x fresh Ollama,
currently 56.055 tok/s. Runtime defaults and prepared down/vocabulary do not
change in this experiment.
