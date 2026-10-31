# Driver JIT Does Not Change Current Projection Code

## Decision

On this M40/driver/toolkit combination, forced driver PTX JIT produces
**identical encoded gate/up and prepared down/head code** to the embedded
sm_52 SASS. The whole captured project module has the same **100,400 encoded
words across 135 disassembled entries**, and resource reports are identical.
Close this standalone compiler hypothesis without another timing matrix or
runtime loader/selector. This is compiled-code evidence, not a speed claim.

Global forced JIT also breaks the cuBLAS-enabled context path. The explicit
no-cuBLAS inspection below is a diagnostic isolation, not a runtime fallback
or replacement for qualified SGEMM prefill. Do not recommend global
`CUDA_FORCE_PTX_JIT=1` for generation.

Qualified performance remains **49.146 HTTP tok/s** versus **43.106 official
Ollama**, not remeasured. Both **53.06** and **56.037 HTTP tok/s** requirements
remain unmet. All runtime/KV defaults, arithmetic, prepared weights, prefix
and attention paths stay unchanged.

## Actual Loaded Code

The [contract](2026-10-26-maxwell-driver-jit-hypothesis.md) asked whether the
installed driver generates a useful alternative before any new inference
code. The existing optional observer now copies actual loaded ELF cubins
inside CUPTI resource callbacks, with bounded size/count, exclusive files
and no CUDA call from the callback. It remains outside Cargo/inference.

Observation checkout was clean `b266a7329468470957f596a955f3b8de8a21a6bd`.
The cuBLAS-enabled fixture was built at `5f44f8911cb82ae7db1ae33aafbde6697fcd1466`,
before the provenance-only observer commit; Rust/CUDA fixture sources did not
change. The disabled rebuild was done at `b266a73`. Both compile sm_52 SASS
and compute_52 PTX with NVCC 12.8.93, `-O3`, no fast math. GPU is Tesla M40
24GB, UUID `GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`, driver **580.159.04**.

| Exact executable | SHA-256 |
| --- | --- |
| Release FFN tests, cuBLAS enabled | `88e0cd0221bfb739b927f1fabf653dd889d250564bb0b7efa1f3d46c1627ac47` |
| Release FFN tests, cuBLAS disabled | `7696b754bf3cb9ec70833e6ca265f3dea0f220899a3464402eb138edfbb38189` |
| Observer source | `f6d738c4dd65ff39e407ca3b4b7150181d64d9d3d47e8460b49dc4f64efafa43` |
| Observer library | `cc155e8fdda002984c2354eac831a0d97be2d699b48244289072efe32a3bda26` |

Native and JIT project cubins differ in container bytes, **1,100,416** versus
**1,100,960**. That is not evidence of changed executable instructions.
Their SHA values, retained in [identities](2026-10-26-maxwell-driver-jit-identities.json),
bind the actual images. Both are sm_52 ELF modules. Original images remain
under ignored `target/research-tools/driver-jit-inspection/`; no binary cubin
is committed.

| Inspected entry | Encoded words | Registers/thread | Stack/local | Dynamic shared |
| --- | ---: | ---: | --- | ---: |
| Qualified split gate/up | 560 | 32 | 0 / 0 | 1024 bytes |
| Original prepared projection | 552 | 40 | 0 / 0 | selected by existing launch |

The gate/up word SHA remains
`31cd38f6ca0d4a0bebd03206f484bf8045834d8af592a645cdb168cfaf02e321`;
prepared down/head remains
`fafef036cf36582ad7c230dd9b2502575a53356ada1396aaeac8e9baf0872397`.
The whole project word stream SHA is
`7a954317700bf84cdc40440471d00b2dfb2b5b126cfb562e88c5566b8449970a`
for native cuBLAS-on, native cuBLAS-off and forced-JIT cuBLAS-off images.
Scheduling words are included. This does not measure achieved occupancy,
cache hits, stalls or bandwidth.

## Correctness And Compatibility

All GPU work was serial and explicitly elevated. Every run uses
`CUDA_CACHE_DISABLE=1`. Force-JIT settings are recorded inside the process
in `cuda_trace_environment`, not inferred from image size or a shell label.

| cuBLAS build | Force JIT | Primitive tests | Launch records | Module images |
| --- | --- | --- | ---: | ---: |
| Enabled | 0 | 2 pass | 1430 | 2 |
| Enabled | 1 | 2 fail at context creation | 0 | 0 |
| Disabled | 0 | 2 pass | 1430 | 1 |
| Disabled | 1 | 2 pass | 1430 | 1 |

Driver and runtime domains describe the same launches; do not sum them.
Each successful process records **715 driver calls**, including **36**
qualified gate/up calls. Normalized symbols, geometry, shared memory and
statuses match exactly across all three successful runs. The tests verify
CPU references, native/control/capture parity, guards, distinct rows,
dynamic K, invalid spans, producer ordering and allocation cleanup.
Prepared down/head code is inspected, not exercised by this gate/up fixture.
There is no actual512 continuation, model quality or HTTP qualification.

The standalone [cuBLAS API probe](2026-10-26-maxwell-driver-jit-cublas-probe.cpp)
uses the same `/opt/cuda` runtime/cuBLAS libraries without a model. Native
selection, initialization, handle creation and destruction all return zero.
Forced JIT still initializes CUDA, but `cublasCreate` returns **3**,
`CUBLAS_STATUS_ALLOC_FAILED`; destruction is not attempted. This confirms
the cuBLAS compatibility failure independently of FFN kernels, but does not
diagnose its internal cause or establish VRAM exhaustion. The failed
cuBLAS-enabled run is not counted as numerical coverage.

Sessions **46508 / 17017 / 65593** terminate successfully for enabled-native,
disabled-native and disabled-JIT respectively. PIDs are **3437286 / 3439890 /
3440182**. Enabled forced-JIT PID **3437808** exits 101 and is retained as a
failure. No inference, build, other GPU test or MemPalace call overlaps these
observations. Final GPU compute is empty. Test wall times include tracing
and possible compilation and must not be used as throughput measurements.

## Observer Validation

The existing GPU smoke completes with actual/expected **96** for ordinary,
module-enabled and forced-JIT capture. Its deliberate invalid launch is
the only expected API failure. Plain tracing records zero modules; capture
records one valid ELF image with mode 0600. Wrong settings exit 2 with an
explicit module error. A prefix that permits a trace filename but exceeds
the filesystem name limit for image filenames also exits 2, without a false
completion record. Hidden GPU invocation exits 1, not a successful skip.

An initial unelevated subscription failed with CUPTI not initialized; all
accepted hardware results use escalation. The observer compiles with C++17,
`-O2 -Wall -Wextra -Werror`; the API probe does too. Shell syntax, Rust
formatting and whitespace checks pass. No CUDA/inference source changes, so
unrelated Rust/CUDA suites were not rerun for this external-tool change.

Independent audit verifies settings, complete dispatch, raw module sizes,
the retained compiled witnesses, actual GPU smoke output and explicit errors.
Ten corrupted inputs must fail. These are observation-tool tests, not
kernel-speed acceptance gates.

## Reproduction And Artifacts

Use [observer instructions](../cuda_launch_trace.md#loaded-module-images),
an idle M40 and escalation. Build the release `q8_ffn_fp32` test once with
`M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=0`, then use its emitted path:

```bash
source scripts/dev-env.sh
export CUPTI_ROOT=/tmp/m40-llm-tools/cupti-12.8.90/cuda_cupti-linux-x86_64-12.8.90-archive
CUDA_FORCE_PTX_JIT=0 CUDA_CACHE_DISABLE=1 M40LLM_CUDA_TRACE_MODULES=1 \
  bash scripts/trace_cuda_launches.sh /tmp/new-ffn-native -- \
  target/release/deps/q8_ffn_fp32-5183cc55dc7ad54e \
  --include-ignored --nocapture --test-threads=1
CUDA_FORCE_PTX_JIT=1 CUDA_CACHE_DISABLE=1 M40LLM_CUDA_TRACE_MODULES=1 \
  bash scripts/trace_cuda_launches.sh /tmp/new-ffn-jit -- \
  target/release/deps/q8_ffn_fp32-5183cc55dc7ad54e \
  --include-ignored --nocapture --test-threads=1
```

Use new prefixes. For each captured project image run `cuobjdump --dump-sass`
and `--dump-resource-usage`; compare per-function words and resources against
the exact executable's embedded image. Do not force JIT globally for ordinary
benchmarks. The diagnostic API probe compiles with `g++ -std=c++17 -O2
-Wall -Wextra -Werror -I/opt/cuda/include`, links `/opt/cuda/lib64` with
`-lcudart -lcublas` and records its force/cache settings in JSON.

- [Audit command](2026-10-26-maxwell-driver-jit-audit.sh), [rules](2026-10-26-maxwell-driver-jit-audit.jq), [result](2026-10-26-maxwell-driver-jit-audit.json) and [checksums](2026-10-26-maxwell-driver-jit.sha256). Run the command with `--negative-controls` for corruptions.
- [Native trace](2026-10-26-maxwell-driver-jit-no-cublas-native-trace.jsonl.gz), [forced-JIT trace](2026-10-26-maxwell-driver-jit-no-cublas-forced-trace.jsonl.gz), [enabled native trace](2026-10-26-maxwell-driver-jit-native-trace.jsonl.gz) and [enabled failure trace](2026-10-26-maxwell-driver-jit-forced-trace.jsonl.gz).
- [Gate/up SASS](2026-10-26-maxwell-driver-jit-gate.sass.gz), [prepared SASS](2026-10-26-maxwell-driver-jit-down.sass.gz), [common project word stream](2026-10-26-maxwell-driver-jit-project.words.gz) and [common resource report](2026-10-26-maxwell-driver-jit-resources.txt.gz).
- [Native API status](2026-10-26-maxwell-driver-jit-cublas-native.json), [forced API status](2026-10-26-maxwell-driver-jit-cublas-forced.json), [native test log](2026-10-26-maxwell-driver-jit-no-cublas-native.log.gz), [JIT test log](2026-10-26-maxwell-driver-jit-no-cublas-forced.log.gz) and [enabled failure log](2026-10-26-maxwell-driver-jit-forced.log.gz).
- [Forced smoke trace](2026-10-26-maxwell-driver-jit-smoke-provenance-trace.jsonl.gz), [actual output](2026-10-26-maxwell-driver-jit-smoke-provenance-output.json), [invalid-setting trace](2026-10-26-maxwell-driver-jit-invalid-setting-trace.jsonl.gz), [image-write failure](2026-10-26-maxwell-driver-jit-image-write-error-trace.jsonl.gz) and [hidden-device log](2026-10-26-maxwell-driver-jit-no-device.txt).

This does not capture the official Ollama runner's actual JIT cubin or replace
M01's explicit offline-assembly caveat. It rules out a compiler shortcut for
the current M40-LLM project module on this stack. The next exact optimization
still needs independent dominant execution-cost evidence, not another JIT
flag or closed geometry/cache/register variant.
