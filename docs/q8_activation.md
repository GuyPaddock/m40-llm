# Q8_1 Activation Experiment

This is M04 of the [Maxwell Q8 execution plan](maxwell_q8_execution.md),
not a new GGUF weight format or a generation default. It quantizes FP32
activation rows for the separate numerical Q8_1 FFN experiment. The qualified
prepared FP32-activation down/vocabulary path remains unchanged.

## Pinned Contract

The encoding is adapted from llama.cpp commit
`9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e`, pinned by Ollama 0.40.11:
`ggml/src/ggml-common.h::block_q8_1`,
`ggml/src/ggml-cuda/quantize.cu::quantize_q8_1`, and the sm_52
`warp_reduce_max` / `warp_reduce_sum` branches in `common.cuh`.
Copyright (c) 2023-2026 The ggml authors. The complete MIT notice is in
[licenses/llama.cpp-MIT.txt](../licenses/llama.cpp-MIT.txt); distribute that
notice with copies or binaries containing these adapted portions. Only the
Maxwell-relevant encoding/reduction logic is adapted.

Each row has explicit column and padded-column counts. This bounded
experiment supports rows 1/4; padded columns cover the input and are a
multiple of 32. Padding is zero, never read beyond the input row. Every
36-byte block contains two FP16 values followed by 32 signed bytes:

1. Find the maximum absolute original value in the 32-value block.
2. Compute the FP32 scale as maximum / 127, before converting it to FP16.
3. Encode each value with `roundf(value / scale)`, ties away from zero.
   An all-zero block encodes zero integers and scale zero.
4. Store the scale as FP16 and the original-input FP32 XOR-tree sum as FP16.
   The sum field is not recomputed from the quantized integers. Preserve
   the 16/8/4/2/1 XOR addition tree, including negative-zero behavior.

The CPU reference rejects nonfinite inputs and FP32 scale underflow. The
unsafe asynchronous device API requires inputs satisfying those conditions;
it does not add host readback or validation synchronization to the warm path.
FP16 scale/sum rounding and overflow remain part of the pinned storage
contract, not an assertion that every FP32 input is losslessly representable.

## Ownership and Verification

`infer::q8_activation` defines checked shapes, explicit byte encoding/decoding
and an independent CPU encoder. `CudaContext::quantize_q8_1_async` writes into
caller-owned reusable storage on decode: one launch, no allocation, host
transfer or synchronization. Input/output spans must be disjoint and remain
alive through completion. Other-stream producers require an explicit wait.
Consumers will reuse one quantized activation row for both gate and up.

Three CPU tests cover zero/ties, FP32-before-FP16 scaling, original-input
sum semantics, row padding, serialization and invalid inputs. Two explicit
M40 tests compare every output byte to the CPU contract, protect input/output
guards, reuse allocations and reject invalid buffers before device activity.
Release M40 tests pass for rows 1/4, short/tail inputs, K=2048 and K=11008.
This establishes the encoding contract, not model quality or a speedup.

## Packed FFN Consumer

`CudaContext::ffn_q8_1_async` consumes an already encoded row with native
34-byte Q8_0 gate/up weights. `Q8FfnQ81Launch` has exactly two geometries,
128 and 256 threads. Each thread handles an eight-value contribution using
two packed words, a signed scalar-integer dot and FP32 block-scale products.
Gate/up share the same quantized activation and then use the tested shared
publication/final-warp reduction and FP32 SwiGLU. No DP4A, Tensor Core,
half-arithmetic projection, per-call allocation or synchronization is used.

The packed weight-word access and signed dot are adapted from the same pinned
`vecdotq.cuh::get_int_b2` / `vec_dot_q8_0_q8_1_impl` and
`common.cuh::ggml_cuda_dp4a` pre-sm_61 fallback, under the MIT notice above.
Weight loads require only two-byte alignment; activation words require four.
FP16 stored activation scales are widened to FP32 before multiplication.
The stored original-input sum is not needed for Q8_0 dot products.

Optional diagnostic gate/up output spans use separately instantiated kernels;
timed calls omit them. Release M40 tests compare every gate/up/SwiGLU output
to independent scalar CPU projections of reconstructed activation bytes.
Diagnostic output and timed-kernel SwiGLU bits agree. Tests cover distinct
rows 1/4, padding, zero scales, signed weight values, K=32/96/2048/11008,
hidden tails, two-byte-offset weights, guards, immutable inputs, repeated
scratch use, enqueue counters, invalid arguments and tracked allocation cleanup.

NVCC 12.8.93 release sm_52 reports 17 quantizer registers and 32 registers for
both FFN geometries, including diagnostic variants, with zero stack/local
bytes. Dynamic FFN shared storage is 1024/2048 bytes at 128/256 threads.
The inspected 128-thread consumer has 64 XMAD, ten shuffle and one barrier
instruction sites, with no DP4A or Tensor Core instructions. Static counts
are not latency, measured occupancy or DRAM traffic.

## Quantization-Inclusive Benchmark

The existing `q8_execution_bench` accepts `--ffn-q81` with the same frozen
`--ffn-inputs` and `--ffn-capture-report` used by the
[FP32 screen](ffn_benchmark_inputs.md#bounded-fp32-ffn-screen). It is mutually
exclusive with `--ffn-f32-ablation`. Substitute `--ffn-q81` in that command;
preserve the exact capture environment, release build, rows 1/4, five initial
alternating native/candidate pairs and ten iterations per repeated stage.

Only the two fixed Q8_1 geometries execute. Reserve one guarded activation
buffer before validation/timing and reuse it across both arms. Native owns
the same scratch for matched residency but does not quantize its input.
Record payload and allocated capacity separately, including guard bytes.
Keep qualified prepared down/vocabulary fixed. Gate/up and complete-FFN
candidate totals include quantization on every iteration, not one cached
quantization reused across an entire trial. A separate `quantize` stage
measures its cost; do not add that cost to totals that already include it.

Numerical validation compares encoded bytes and gate/up/SwiGLU against the
independent quantized-input CPU reference using the existing strict tolerance.
Reconstruction and original-FP32 projection/hidden drift are separate measured
fields, not grounds to weaken the correctness tolerance. Native and initial
256-thread FP32 variants retain frozen one-row bit requirements. All expensive
CPU references and readbacks precede active warmup and primary pairs.

Release M40 runner tests pass, including guarded reuse, exact quantizer launch
accounting, cross-stream four-row execution, unchanged down, rejection before
scratch reservation and native/FP32 regression checks. Real-input measurement
is still pending. Do not enable a runtime path from these tests or treat Q8_1
as satisfying exact-output acceptance.

## Atomic Delivery

| Unit | Purpose and Dependencies | Draft Commit |
| --- | --- | --- |
| Encoding | CPU/CUDA block contract, reusable-storage API, license and byte parity; depends on M02 input contract | `feat(cuda): define pinned Q8_1 activations` |
| Dot consumer | Packed sm_52 integer products, fixed 128/256 geometry and projection/SwiGLU CPU tests; depends on encoding | `feat(cuda): probe packed Q8_1 FFN` |
| Shared benchmark | Quantizer cost, complete FFN pairs, reconstruction/projection drift and memory accounting; depends on dot consumer | `test(bench): compare quantized FFN work` |
| Evidence | Preserve every measured candidate and update M04/M05 decision before runtime integration | `docs(perf): record Q8_1 FFN screen` |
