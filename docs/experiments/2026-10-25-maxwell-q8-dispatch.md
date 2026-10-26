# Pinned Ollama Maxwell Q8 Dispatch

2026-10-26 follow-up: [actual official-runner JIT capture](2026-10-26-maxwell-ollama-jit-inspection.md)
now confirms the selected one-token MMVQ words/resources match the offline
assembly below. Historical data and limitations remain labeled as originally
observed; this follow-up is not a new throughput result.

## M01 Finding and Limits

The **official shipped** Ollama 0.40.11 CUDA 12 backend executes Q8_1
activation quantization followed by unfused Q8 MMVQ on the Tesla M40. The
single-token MMVQ specialization uses four 32-thread warps, 128 threads total.
The same diagnostic also observes Q8-to-FP32 dequantization and Maxwell SGEMM
kernels. These are actual launch observations, not just available source.

M01 establishes source provenance and execution differences. It does **not**
attribute the throughput gap quantitatively, qualify a new kernel, relax
token equality, or change runtime defaults. The response's API durations are
perturbed diagnostic data, not a new performance benchmark. The qualified
prepared down/vocabulary control remains 35.724705 HTTP tok/s; the latest
separate timed Ollama baseline remains 43.162711.

## Source and Binary Identity

Ollama tag `v0.40.11` reads `LLAMA_CPP_VERSION=b11845`, resolved to
`9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e`. The source archives are downloaded
into ignored `target/research-sources/`; the user's older Ollama checkout is
unchanged. Archive SHA-256 values are:

- Ollama: `9733d6230d8c8a75ccdf368c53132d90ee49009ce677776d897e96b50cc1c654`.
- llama.cpp: `d8552e1b309c1571e49baf7a53d429b9709fa0c1d6907c446f6cecb87087d07b`.

Ollama's compatibility patch touches model-loader and MTMD hooks, not the
audited CUDA projection code. Its separate compatibility source files are
linked into the llama target. The [per-file source hashes](2026-10-25-maxwell-q8-source-hashes.json)
bind those inputs, CMake setup and the relevant CUDA files. The runner reports
build `9ff0d7e90`, agreeing with the resolved pin.

| Shipped Artifact | SHA-256 |
| --- | --- |
| Ollama executable | `b0a4becfaf872e3ad5d2e9051244e0dfe55409f03d6e2cd9bd9c34400e201b4c` |
| llama-server executable | `234b05b2138264f8fb263c3205e85f4c290e8afe5067e280a4f6f90cdac5696b` |
| CUDA 12 libggml-cuda.so | `7a7b6696ffb43ca00f78504a0b2e76db0c6554320ed64e405559e3433580d9b3` |

Original model bytes are rehashed as
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
No model conversion, pruning, activation threshold or attention approximation
occurs. The [llama.cpp MIT license](2026-10-25-maxwell-q8-llama-license.txt)
is retained with the derived PTX/disassembly artifacts. The observer is
original code using SDK declarations; no NVIDIA sample code is copied.

## Observed Request

Use the isolated pinned binary, one runner/slot, CUDA UUID
`GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`, raw canonical ChatML prompt,
context 1024, greedy sampling and **four generated tokens**. This bounded
request identifies prefill/decode dispatch; it is not the 512-token speed gate.
The response is `Sure, I'll`, with 65 prompt tokens, four generated tokens
and a length stop. Runner diagnostics show an empty slot and zero cached
tokens before the request. Flash attention is disabled in this diagnostic.

The observer is compiled with g++ 14.3.1, C++17, -O2, warnings as errors and
CUPTI 12.8.90. Source base is `3e9a063` plus the then-uncommitted observer,
now preserved unchanged in `723a456`; no clean-source claim is made for the
observation. Observer source SHA-256 is
`831ac896894a854d1b7c0999de147817b5aa6694f53ae88fb9930f4297609f9f`,
library SHA-256
`1a573ea4c119b4ae0b0d9523d008e8569f35ea78ff8eb38b40d65eb3012e0841`.

The sole model runner trace contains 6330 driver and 6330 runtime launch
records. Both domains describe the same launches and have matching
context/correlation pairs; **do not add them**. Graceful runner SIGINT shutdown
produces the final completeness record with zero drops, unknown symbols or
launch errors and successful unsubscribe. Parent/helper processes have no
model launch evidence and are not used for kernel attribution.

Whole-process counts include startup and warmup. They must not be divided
by four output tokens or presented as request stage costs:

| Observed Driver Specialization | Whole-Process Count | Block Dimensions |
| --- | ---: | --- |
| Q8_1 activation quantizer | 1269 | `[256,1,1]` |
| Q8_0 MMVQ, one output token, fusion false | 771 | `[32,4,1]` |
| Q8_0 MMVQ, two output tokens, fusion false | 498 | `[32,4,1]` |
| Q8_0 dequantization to FP32 | 249 | `[256,1,1]` |
| `maxwell_sgemm_128x64_tn` | 179 | `[128,1,1]` |
| `sgemm_32x32x32_NT_vec` | 142 | `[128,1,1]` |

The [full symbol/geometry summary](2026-10-25-maxwell-q8-launch-summary.json)
retains other operations too. No other inference/tests run during this
request. The verified runner and isolated parent exit afterward; final M40
usage is 0 MiB / 0%. This is not a peak-VRAM or GPU-duration measurement.

## Source-to-Runtime Comparison

| Aspect | Qualified M40-LLM Control | Pinned Ollama / llama.cpp |
| --- | --- | --- |
| Gate/up activation arithmetic | FP32 activations times dequantized Q8 values, FP32 accumulation | FP32 input quantized into 32-value Q8_1 blocks, signed integer products, FP32 scaled accumulation |
| Weight access | Native 34-byte Q8_0 blocks, scalar byte loads | Same Q8_0 storage, alignment-safe two-ushort packed loads; aligned packed activation loads |
| Gate/up scheduling | One native fused gate/up/SwiGLU launch | Observed unfused MMVQ specialization; architecture guard rejects MMVQ fusion on Pascal and older |
| Reduction | 256 threads, shared publication and eight block-reduction iterations | 128 threads; non-final warps publish partials once, final warp combines them then uses shuffles |
| Activation quantization reuse | None needed | Source quantizes inside each MMVQ call; observed quantizer and MMVQ counts agree, not a demonstrated cross-projection shared quantizer |
| Single-token down/head | Qualified prepared FP32-activation split planes | Q8_1/MMVQ path; not evidence that changing the qualified control is safe |
| Prompt processing | Sequential in the qualified benchmark | Batched projection dispatch; actual Q8 dequantization and SGEMM launches observed |
| Graphs | Experimental/off | Source disables CUDA graphs below Volta; the log's host graph-reuse count is not proof of CUDA Graph replay |

At the pinned source, `common.cuh::ggml_cuda_dp4a` falls back below cc 610 to
four signed scalar products. `vecdotq.cuh::get_int_b2` handles two-byte-aligned
weights, while `get_int_b4` loads aligned activation words. Q8 dot helpers
consume two packed words per thread contribution. Q8_1 uses warp max/sum,
`amax/127`, `roundf`, explicit zero handling and FP16 scale/sum metadata.
This changes arithmetic and rounding; unchanged weights do not imply equal
logits or continuations.

`ggml_cuda_should_use_mmvq` selects small quantized matrix-vector work.
For Maxwell, the MMQ prefill alternative is rejected without native DP4A,
leaving cuBLAS. `ggml_cuda_mul_mat_cublas_impl` temporarily converts Q8 weights
to FP32 and chooses `cublasSgemm` for ordinary FP32 matrices. Pool ownership
is not a persistent all-model FP32 materialization cache. The earlier failed
M40 packed-prefix run used another projection backend; it does not test this
bounded dequant-plus-SGEMM strategy.

## Compiled-Code Check

The shipped library contains sm_52 PTX 8.7, but **no sm_52 cubin**. The M40
therefore requires JIT of compatible PTX. Preserve this distinction when
interpreting compiler resources. See [cubin](2026-10-25-maxwell-q8-cubin-inventory.txt)
and [PTX](2026-10-25-maxwell-q8-ptx-inventory.txt) inventories.

The observed one-token, unfused Q8 MMVQ entry is extracted from shipped sm_52
PTX and assembled offline with toolkit 12.8.93 `ptxas -arch=sm_52 -v`.
It uses 48 registers, 384 static shared bytes and no stack/spills. SASS has
one static barrier, five shuffle instructions, packed ushort loads and scalar
integer XMAD operations, not DP4A. This is a reproducible offline assembly,
**not the actual driver's JIT cubin or a measured occupancy result**.

The unchanged qualified M40 benchmark executable, SHA-256
`36df94961ca5fa042ebb2da4dc45ed91877086e675041be541867e5bcb807af9`,
has native sm_52 SASS. Its fused gate/up split4 entry uses 32 registers,
zero stack/local storage, and 2048 dynamic shared bytes at the selected
256-thread launch. It has byte loads and no shuffle. Two static barrier
instructions represent the initial publication plus the loop's repeated
barrier, nine executions for the selected reduction.

These are different fused/unfused kernels; lower register count alone does
not explain whole-FFN performance. Their generated-code differences justify
the bounded packed-load/reduction ablation, not a speed prediction.

## Validation, Reproduction and Next Step

The standalone fixture verifies actual GPU result 96 from ordinary/extended
launches and deliberate invalid-grid status 9. The host-only subscription
works with zero launches. Explicit hidden-device invocation fails with
`no CUDA-capable device is detected`, not a successful skip. g++ warnings as
errors, shell syntax, fmt and whitespace checks pass. No production Rust or
CUDA kernel is changed, so no new inference parity claim follows.

Follow [the observer instructions](../cuda_launch_trace.md), use the isolated
official server environment from [the controlled protocol](../controlled_benchmark.md),
and submit its canonical raw prompt with `num_predict=4`, `num_ctx=1024`,
temperature zero and neutral penalties. All children inherit the observer;
select the verified GPU runner's file, not an empty discovery/helper trace.
Ordinary timing must omit the observer completely.

Next M02 captures read-only normalized FFN inputs and completed SwiGLU vectors
from frozen calibration prompts and uses real layer 0/17/35 weights. Measure
genuine rows 1/4, gate/up plus activation, isolated down and complete FFN using
the existing benchmark. Hold prepared down/vocabulary fixed. No per-component
speed attribution or runtime integration is justified until those measurements.

## Raw Evidence

- [Trace gzip](2026-10-25-maxwell-q8-launches.jsonl.gz), SHA-256
  `3b7496f147fb6e4267706ea82ec0304155485245169a674940fcb2916d8b96a0`;
  uncompressed SHA-256
  `0bcdadb34ce55804440c51aadf14529e4614bb482c0b9b31c44ba137c8a80550`.
- [Audit/identities](2026-10-25-maxwell-q8-dispatch-audit.json),
  [response](2026-10-25-maxwell-q8-dispatch-response.json) and
  [runner log](2026-10-25-maxwell-q8-dispatch-ollama.txt).
- [Observer smoke](2026-10-25-cuda-launch-trace-smoke.jsonl),
  [actual output](2026-10-25-cuda-launch-trace-smoke-output.json) and
  [no-device failure](2026-10-25-cuda-launch-trace-no-device.txt).
- [Selected shipped PTX](2026-10-25-maxwell-q8-ollama-selected.ptx), SHA-256
  `5112dbca18c28fa506202f3156261e3cf2bfd0a4818f3372f7d25acc5c3ceb86`,
  [offline ptxas](2026-10-25-maxwell-q8-ollama-ptxas.txt) and
  [offline SASS](2026-10-25-maxwell-q8-ollama-q8-offline-sass.txt).
- [Native gate/up SASS](2026-10-25-maxwell-q8-native-gate-up-sass.txt) and
  [native resources](2026-10-25-maxwell-q8-native-gate-up-resources.txt).

JSON/trace/PTX bytes are unchanged; text logs trim trailing whitespace and
redundant final blank lines only. Full downloaded sources and offline cubin
stay in ignored research storage. Source URLs use the pinned commit/tag:
[llama.cpp](https://github.com/ggml-org/llama.cpp/tree/9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e/ggml/src/ggml-cuda),
[Ollama setup](https://github.com/ollama/ollama/blob/v0.40.11/llama/server/CMakeLists.txt)
and [compatibility patch](https://github.com/ollama/ollama/blob/v0.40.11/llama/compat/001-llama-cpp-hooks.patch).
