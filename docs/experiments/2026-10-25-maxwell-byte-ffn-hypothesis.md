# Exact Signed-Byte Conversion In The Qualified FFN

## Evidence Before Implementation

The qualified split-plane fused gate/up/SwiGLU accounts for **39.43%** of
sampled middle-token GPU work. Its control SASS is preserved with SHA-256
`7c2d3c8e46c37367a7ce14d14d865ea5a6476b4ab5baf4e2578b45adfa182b22`.
In the first task, addresses `0370` through `0478` contain eight signed
integer-to-float conversions across gate/up. Byte 1 also uses BFE, shift
and S16 conversion; bytes 2/3 use shifts plus S8 conversion. Two FP16 scale
conversions, eight FP32 scale multiplies and eight FP32 FMAs follow.
This is compiled instruction evidence, not a measured conversion stall.

The [CUDA 12.8 throughput table](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#arithmetic-instructions)
lists lower throughput for conversions than FP32 arithmetic on sm_52.
That does not establish the bottleneck or predict end-to-end gains. The
[byte-permutation intrinsic](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html)
can construct a float bit pattern from one packed byte and constant bytes.
Test whether that exact representation change reduces the conversion
dependency cost without changing weights or FP32 activation precision.

## One Fixed Candidate

For a four-byte packed word, XOR `0x80808080` once. Each resulting byte
is `u = signed_byte + 128`, in `[0,255]`. For each of its four positions,
use `__byte_perm(biased_word, 0x4b000000, 0x7650 + byte_index)` to construct
`0x4b000000 | u`. Its FP32 value is exactly `8388608 + u`. Subtract the
exact FP32 integer `8388736` using `__fsub_rn` to obtain the original signed
byte in `[-128,127]`, including positive zero. All values and differences
are exactly representable. Explicit rounded subtraction prevents compiler
contraction with the subsequent original scale multiply.

This changes neither quantization nor the scale/FMA association. Keep the
qualified one-column/four-value/128-thread fused kernel, runtime K, read-only
loads, split planes, original reduction, SwiGLU, optional gate/up capture,
streams, scratch and qualified prepared down/vocabulary. Do not combine
failed cache hints, layouts, fixed-K or unfused variants. Do not add another
conversion scheme, geometry, register cap or unroll configuration.

Compile first for sm_52 and inspect conversion instructions, registers,
spills and control SASS. If the compiler returns the original conversion
path, stop without timing. Otherwise extend the existing explicit FFN
benchmark and behavioral tests. Cover every signed byte in every packed
position, signed extremes, zero scales, distinct rows, width/hidden tails,
optional capture, invalid arguments, guards, scratch reuse and cleanup.
Require gate/up, hidden and completed down bits equal to current control.

Use frozen M02 model/inputs/environment, layers 0/17/35, genuine rows 1/4,
five alternating pairs and ten iterations. Include full gate/up/SwiGLU and
complete FFN, actual dispatch and dependencies, not conversion-only timing.
No timed allocations, transfers or internal synchronization. Keep all
original gates: >=5% complete one-row gain and positive paired 95% interval;
only borderline results justify twenty fresh confirmation pairs. Do not
rescue a loser with another conversion/geometry search.

## Atomic Delivery

1. `docs(perf): scope exact byte-conversion test`: this instruction finding
   and fixed candidate, before implementation.
2. `feat(cuda): probe exact byte-permute FFN`: distinct compiled code,
   explicit benchmark-only API and behavioral tests.
3. `docs(perf): record byte-conversion screen`: clean release provenance,
   raw pairs, numerical/resource/memory audit and decision.

Only a qualifying screen advances to actual-512/frozen-quality, incremental
HTTP and fresh official Ollama gates. Defaults remain unchanged. The current
qualified **47.379 HTTP tok/s** still misses both **53.06** and **56.026**
for 1.30x the fresh Ollama median. No numerical experiment or kernel-only
gain substitutes for those goals.
