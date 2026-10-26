# Actual Official Ollama JIT Matches M01's Selected Assembly

## Finding

The actual driver-loaded one-token Q8 MMVQ entry from pinned official Ollama
0.40.11 matches M01's offline assembly **bit for bit**, including scheduling
words. It uses **48 registers, 384 static shared bytes, zero stack/local
storage, one barrier site and five shuffle sites** on Tesla M40. This
validates the selected compiled-code assumption; it is not a performance
measurement, a new kernel or permission to revive rejected candidates.

The loaded activation quantizer is also captured. It uses 23 registers,
zero static shared/stack/local storage, and launches 256 threads. MMVQ
launches `[32,4,1]`, 128 threads total, with no dynamic shared allocation.
The MMVQ SASS contains packed loads and scalar integer `XMAD`, not DP4A or
Tensor Core operations. Its 102 static XMAD sites also include addressing;
that count is not dynamic dot-product work or a latency attribution.

Qualified exact performance remains **49.146 HTTP tok/s** versus **43.106
official Ollama**, not rerun. Both original **53.06** and **56.037 HTTP
tok/s** requirements remain unmet. No runtime/default, prefill, attention,
model, precision or scheduling changes follow.

## Fixed Request And Identity

Observation checkout is clean `3b9eb51806f277b9deeef2aab0c1f6b3f3dc7df0`.
The [bounded contract](2026-10-26-maxwell-ollama-jit-hypothesis.md) uses the
unchanged official binary and backend from M01, not a rebuilt substitute.
Rechecked hashes are:

| Artifact | SHA-256 |
| --- | --- |
| Ollama executable | `b0a4becfaf872e3ad5d2e9051244e0dfe55409f03d6e2cd9bd9c34400e201b4c` |
| Actual runner executable | `234b05b2138264f8fb263c3205e85f4c290e8afe5067e280a4f6f90cdac5696b` |
| Loaded CUDA 12 backend | `7a7b6696ffb43ca00f78504a0b2e76db0c6554320ed64e405559e3433580d9b3` |
| Original model blob | `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d` |
| Observer library | `cc155e8fdda002984c2354eac831a0d97be2d699b48244289072efe32a3bda26` |

Ollama pins llama.cpp `b11845`, resolved to
`9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e`. Existing source/patch provenance
and the new [MIT notice](2026-10-26-maxwell-ollama-jit-license.txt) accompany
the derived disassembly. No unrelated architecture or proprietary cuBLAS
disassembly is redistributed. Full loaded binary images remain ignored.

The owned localhost server uses port 18435, one parallel request, one loaded
model, cloud disabled and `LLAMA_ARG_CACHE_RAM=0`. The actual CUDA 12 runner
uses context 1024, full GPU offload, flash attention off and raw ChatML.
GPU is M40 UUID `GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`, driver 580.159.04.

The request is copied from the qualified canonical browser prompt and
changes only the generation bound to four. Greedy temperature/top-k are
0/1 with neutral penalties and seed zero. It returns **`Sure, I'll`**, four
natural tokens, `done_reason=length`. The runner tokenizer's **65 actual IDs
exactly equal** the qualified prompt IDs. Both API and runner logs report
zero reused prompt tokens, and the log confirms all 65 prompt tokens were
processed. No second generation, forced continuation or cache replacement
request is used. This limited deterministic output is not a quality suite.

## Actual Image And Launch Evidence

The existing observer runs only in the isolated diagnostic processes.
`M40LLM_CUDA_TRACE_MODULES=1` captures actual loaded images; driver disk cache
is disabled. Force-JIT and disable-JIT are **unset**, recorded inside the
runner, so cuBLAS retains its normal native path. Official sm_52 MMVQ comes
from the backend's shipped PTX, not from a global forced-JIT override.

The complete model-runner trace has **12,660 records**, forming **6330 matched
driver/runtime pairs**, zero drops, unknown symbols or launch errors, and
successful unsubscribe. Sixteen module images total **13,956,848 bytes**,
within unchanged observer limits. Parent/discovery traces are not accepted
as model coverage. Counts include startup/warmup and must not be divided by
four generated tokens or presented as per-token stage costs.

| Observed specialization | Driver count | Block |
| --- | ---: | --- |
| Q8_1 quantizer | 1269 | `[256,1,1]` |
| One-token unfused Q8 MMVQ | 771 | `[32,4,1]` |
| Two-token unfused Q8 MMVQ | 498 | `[32,4,1]` |
| Q8-to-FP32 dequantization | 249 | `[256,1,1]` |
| `maxwell_sgemm_128x64_tn` | 179 | `[128,1,1]` |
| `sgemm_32x32x32_NT_vec` | 142 | `[128,1,1]` |

These reproduce M01's counts exactly. Module index 2 contains the selected
MMVQ entry, index 1 the quantizer. Both selected images are sm_52 cubins.
Image hashes and retained paths are in the [module manifest](2026-10-26-maxwell-ollama-jit-modules.json).
Images are copied to ignored `target/research-tools/ollama-jit-inspection/`;
only relevant disassembly/resources and hashes are committed.

The 480-word MMVQ stream, actual and historical offline, has SHA-256
`9c27a7b45565707b1f4094d1d3c022bce84c238c54a9e3faf2b5f9830742f5cb`.
The actual quantizer word SHA is
`972c76d2c6c57d489d1d3bf9f71131795b12b7af22632efb3794bb8a8b3430f1`.
One Ollama MMVQ entry computes one projection; M40's qualified 32-register
entry fuses gate, up and SwiGLU. Static instruction/register counts are not
comparable complete-FFN timings. No achieved occupancy, DRAM traffic, cache
hit, warp-stall or GPU event measurement is added here.

## Process Ownership And Validation

Server session **39091**, parent PID **3469761**, owns model runner PID
**3470246**, native port 42173. The API request session **12903** completes
successfully. `/proc` executable hashes, parent/child command lines, loaded
CUDA 12 backend mappings and the GPU process listing verify the actual
runner rather than a discovery process. No other inference, build, test or
MemPalace work overlaps the observation.

Only the verified owned runner is sent SIGINT, producing normal cleanup
and the complete observer record. The verified parent is then stopped;
its session exits zero. The final owned-process and GPU compute lists are
empty. An immediate read before the final flush lacked the completion row;
the same file was rechecked and completed. No workload was restarted.

The artifact audit verifies pins, exact prompt IDs, cache evidence,
module conservation, selected-code fingerprints, successful paired launch
coverage and bounded completed output. Ten corrupted inputs must be rejected.
The observer was already validated in the preceding driver-JIT checkpoint;
no implementation changed for this run. Shell syntax, formatting and
whitespace checks pass. Unrelated numerical/Rust suites are not rerun for
this documentation/artifact-only update.

## Reproduction And Decision

Use the [controlled isolated server setup](../controlled_benchmark.md#ollama-setup)
and [module observer](../cuda_launch_trace.md#loaded-module-images). Launch
the pinned server under the observer, `CUDA_CACHE_DISABLE=1`, no force-JIT
override, and a new absolute prefix. Submit the [saved request](2026-10-26-maxwell-ollama-jit-request.json)
once. Verify `/tokenize` with `add_special=false,parse_special=true`, then
stop only verified owned processes gracefully and require a complete trace.

For the selected image, run `cuobjdump --function SYMBOL --dump-sass` and
`--dump-resource-usage`; the exact symbol is recorded in the raw trace and
M01's saved PTX. Compare the encoded word stream against the historical
offline SASS. Never use observer-perturbed API durations for speed claims.
The server log's reused logical graphs do not override the pinned CUDA
backend's pre-Volta graph restriction.

- [Actual runner trace](2026-10-26-maxwell-ollama-jit-runner-trace.jsonl.gz), [server log](2026-10-26-maxwell-ollama-jit-server.log.gz), [launch summary](2026-10-26-maxwell-ollama-jit-launch-summary.json), [identities](2026-10-26-maxwell-ollama-jit-identities.json) and [checksums](2026-10-26-maxwell-ollama-jit.sha256).
- [MMVQ SASS](2026-10-26-maxwell-ollama-jit-mmvq1.sass.gz), [quantizer SASS](2026-10-26-maxwell-ollama-jit-quantizer.sass.gz), [resources](2026-10-26-maxwell-ollama-jit-resources.txt) and [MIT license](2026-10-26-maxwell-ollama-jit-license.txt).
- [Request](2026-10-26-maxwell-ollama-jit-request.json), [response](2026-10-26-maxwell-ollama-jit-response.json), [tokenizer IDs](2026-10-26-maxwell-ollama-jit-tokenize-response.json), [equality check](2026-10-26-maxwell-ollama-jit-tokenizer-check.json), [backend mappings](2026-10-26-maxwell-ollama-jit-backend-maps.txt), [owned processes](2026-10-26-maxwell-ollama-jit-owned-processes.txt) and [final GPU list](2026-10-26-maxwell-ollama-jit-final-gpu.csv).
- [Audit command](2026-10-26-maxwell-ollama-jit-audit.sh), [rules](2026-10-26-maxwell-ollama-jit-audit.jq) and [result](2026-10-26-maxwell-ollama-jit-audit.json). Pass `--negative-controls` to test corruptions.

The selected M01 offline-code limitation is now resolved by actual loaded
code. It does not establish which instruction family explains a performance
gap, generalize to every backend entry or qualify a new optimization. Keep
Q8_1 numerical-only and retain all exact-output/E2E gates. Further exact work
still needs independent dominant-cost evidence, not another compiler flag,
closed variant or a smaller definition of the original target.
