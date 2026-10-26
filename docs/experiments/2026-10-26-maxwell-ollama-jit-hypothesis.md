# Pinned Ollama Actual JIT Inspection

## Purpose

M01 observes official Ollama Q8_1 quantization and Q8 MMVQ launches, but its
compiled-code evidence is offline ptxas, not the loaded driver image.
The new module observer permits a bounded check of that limitation.
Capture actual official-runner code before proposing another exact kernel.

Pin the existing official 0.40.11 binary, CUDA 12 backend and llama.cpp
`9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e`. Verify current hashes against
M01. Use the same Qwen2.5-3B Q8_0 model bytes, canonical 65-ID raw ChatML
prompt, context 1024 and four natural greedy output tokens. No network or
model download is needed. This is untimed launch/code observation, not
a new Ollama baseline or a 512-token throughput result.

## Fixed Observation

- Use an isolated owned localhost server, one model/slot, cloud disabled and
  RAM prompt caching disabled. Confirm an idle GPU and unused port first.
- Enable existing `M40LLM_CUDA_TRACE_MODULES=1`, disable driver disk cache,
  leave force/disable-PTX-JIT unset. Do not force JIT globally: that breaks
  cuBLAS initialization on this stack. Observe the backend's normal choice.
- Keep observer image/count/byte bounds fixed. Explicit capture failures or
  incomplete traces are limitations, not successful inspection. No limit
  expansion, alternate compiler or architecture rescue is authorized.
- Verify actual selected CUDA 12 runner, executable/library hashes, GPU,
  prompt IDs through the native tokenizer, raw request/response, zero reused
  prompt tokens and actual 65-token processing where available.
- Identify the actual model runner by process identity and its launch trace,
  not an empty discovery/parent trace. Save loaded module identities and
  inspect only observed M40-relevant Q8 MMVQ/activation quantizer entries.
- Compare observed one-token MMVQ words/resources against M01 offline
  assembly: registers, shared storage, stack/local storage, packed loads,
  scalar integer multiply path, shuffle and barrier sites. Record actual
  differences without inferring DRAM traffic, occupancy or stage latency.
- Retain the upstream MIT notice with extracted source-derived code/SASS.
  Keep full binary images in ignored research storage. Do not copy unrelated
  CUDA architectures or introduce new inference code.
- Stop only the verified owned model runner/server gracefully; require a
  complete runner trace and final empty owned GPU/process state. Do not
  overlap other inference, builds, tests or MemPalace with observation.

## Decision And Atomic Delivery

1. `docs(research): bound official JIT inspection`: this contract.
2. `docs(research): record official JIT evidence`: actual code/dispatch
   comparison, raw artifacts, independent audit, current status and diary.

If the offline assumptions hold, close that evidence gap without another
kernel/timing search. If different, document a specific independent cost
hypothesis before any bounded experiment. No runtime/default promotion or
revival of closed candidates follows merely from static code differences.

Both original exact requirements remain: 53.06 HTTP tok/s and 1.30x fresh
official Ollama. Qualified M40 stays at 49.146 HTTP tok/s; observation of
four official output tokens cannot satisfy either unmet target.
