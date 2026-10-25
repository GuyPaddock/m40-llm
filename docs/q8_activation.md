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
This does not establish projection parity, model quality or a speedup.

Next add the two fixed 128/256-thread packed integer-dot FFN consumers and
numerical tests, then measure quantization-inclusive complete FFN costs using
the existing real-input benchmark. Do not enable a runtime path from encoder
correctness alone or treat Q8_1 as satisfying exact-output acceptance.

## Atomic Delivery

| Unit | Purpose and Dependencies | Draft Commit |
| --- | --- | --- |
| Encoding | CPU/CUDA block contract, reusable-storage API, license and byte parity; depends on M02 input contract | `feat(cuda): define pinned Q8_1 activations` |
| Dot consumer | Packed sm_52 integer products, fixed 128/256 geometry and projection/SwiGLU CPU tests; depends on encoding | `feat(cuda): probe packed Q8_1 FFN` |
| Shared benchmark | Quantizer cost, complete FFN pairs, reconstruction/projection drift and memory accounting; depends on dot consumer | `test(bench): compare quantized FFN work` |
| Evidence | Preserve every measured candidate and update M04/M05 decision before runtime integration | `docs(perf): record Q8_1 FFN screen` |
