# Packed QKV/Output Runtime Quality Gate

## Decision

The fixed packed-load QKV/output combination passes full-model quality
against the qualified scalar-load warp control. All **512 actual generated
IDs** and all **twelve frozen task outputs** match. There are zero
regressions across **seven dense-valid tasks**; the five dense-failed tasks
remain inconclusive. Sampled full-vocabulary logits have **zero maximum
and mean difference**, with top-10 overlap ten at every sampled step.

This permits controlled HTTP timing, not a new speed claim or default.
The last qualified result remains **47.379 versus 43.097 HTTP tok/s**;
both original goals remain unmet: **53.06** and at least **1.30x fresh
official Ollama**, most recently 56.026 tok/s. The independent
[stage screens](2026-10-25-maxwell-packed-projection-screen.md) established
feasibility, not E2E performance.

## Configuration And Reproduction

Runtime integration is `c264c89`; qualification tooling and clean measured
source are `43ea00a84c4e0d526347fbe40d843bc1a06dfcd8`. Release quality
executable SHA-256 is
`e178fa4c5010f20c06bf78d91714cceab3a9db41d0b1d9529b0849a07cd3062d`.
The same build's HTTP executable is
`4b7f15aa9361a29e42472e7c0194c92863b050ed1c3ab436f2f68514f4ff9cf7`.
Build flags remain CUDA/server, NVCC 12.8.93, sm_52 SASS/compute_52 PTX,
the existing CUDA `-O3`, and no fast math.

Both arms use unchanged Qwen2.5-3B Q8_0 bytes, real GGUF Qwen tokenizer
and ChatML, all 36 layers, Q/K/V biases, split-half RoPE, exact head128
attention, dense FP16 KV/context 1024, greedy sampling and natural stops.
Both retain split-f32x4-128 FFN, prepared down/vocabulary and dequant-SGEMM
prefill. Only the single-row QKV/output loads differ: `warp-256` control,
`packed-warp-256` candidate. Geometry, arithmetic and reduction are fixed.

```bash
source scripts/dev-env.sh
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo build --release --features cuda,server \
    --example model_quality --example compare_ollama
while IFS=$'\t' read -r key value; do
  export "$key=$value"
done < <(head -n 1 docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl |
  jq -r '.execution_configuration.environment | to_entries[] | [.key,.value] | @tsv')
MODEL="$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
target/release/examples/model_quality \
  --reference-model "$MODEL" --model "$MODEL" \
  --variant-id native-qkv-output-packed-warp-256-split-sgemm \
  --split all --context-tokens 1024 \
  --q8-runtime-candidate split-f32x4-128 \
  --q8-runtime-prefill dequant-sgemm \
  --q8-runtime-control qualified-warp-sgemm \
  --q8-runtime-attention packed-warp-256 \
  --output /tmp/new-packed-projection-runtime-quality.jsonl
```

Typed options preserve the frozen environment. The shared model loader
checks the actual selected backend before inference; records require
actual dispatch rather than labels alone. No activation quantization,
additional cache or approximate attention is introduced.

## Behavior And Memory

The canonical 65 prompt IDs and actual 512-token continuation match both
the current control and the previously qualified warp runtime. Both stop
with `length`. Full 151,936-element logits at generated steps
0/1/8/32/128/256/511 are equal. All frozen outputs and scores also match
the prior checkpoint, not just the current pair.

Each selected QKV/output kernel launches **18,432 times**, exactly 36
layers times 512 one-row forwards. The control uses the scalar-load warp
operations; the candidate uses their packed-load equivalents. Every other
operation count and aggregate enqueue counter matches: 237,013 launches,
252 SGEMMs, 1097 synchronizations, 324 waits, and 42 allocations/frees per
whole generation. The allocation count includes admission/session work;
it is not a per-token allocation count. The 64-token prefix is unchanged.

All **109 prepared tensor identities and accounting fields** match both
the control and the prior checkpoint, apart from preparation/hash times.
Each model retains 3,279,519,744 source bytes and 2,917,580,800 prepared
bytes: **6,197,100,544 source-plus-prepared bytes** before KV/workspace.
Dense KV is **37,748,736 bytes** per model. Existing temporary prefill
capacity is **90,177,536 bytes**, capped at 128 MiB and released after
the prefix. Packed loads add no device weights or scratch. These are
requested payload capacities, not peak VRAM or measured DRAM traffic.

Quality PID 1717188 ran serially with no Ollama, build, test or MemPalace
work overlapping inference. Session 91271 exited zero; the GPU compute
list was empty afterward. Both model owners are released and tracked
device bytes return from zero to zero. Diagnostic resident-generation
times of 10,896.794 / 10,465.656 ms are unpaired and **not** HTTP evidence.

Tool validation: compare example 20 CPU tests, quality example 24 CPU
tests with nine hardware tests ignored, followed by the explicitly invoked
runtime-prefix CUDA fixture test passing. CUDA/server all-target Clippy
and formatting pass. Runtime-selector validation in `c264c89` separately
covers both cuBLAS modes, model ownership, exact fixture logits and invalid
requests; its warning-as-error non-CUDA suite passes 211 tests.

## Evidence And Next Gate

- [Raw quality JSONL](2026-10-25-maxwell-packed-projection-runtime-quality.jsonl)
- [Independent audit](2026-10-25-maxwell-packed-projection-runtime-quality-audit.jq) and [derived result](2026-10-25-maxwell-packed-projection-runtime-quality-audit.json)
- [Run log](2026-10-25-maxwell-packed-projection-runtime-quality.log.gz), [validation](2026-10-25-maxwell-packed-projection-runtime-quality-validation.log.gz) and [checksums](2026-10-25-maxwell-packed-projection-runtime-quality.sha256)
- [Previous qualified quality checkpoint](2026-10-25-maxwell-projection-runtime-quality.md)

The independent audit binds source/executable, frozen environment, actual
IDs, all prepared identities, selected dispatch, other counters and cleanup.
Mutated dispatch, preparation, token IDs and logits must each be rejected.
Next run five alternating HTTP pairs against the current qualified warp
control. Retain only a gain above the original variation-adjusted 3%
threshold with positive paired interval, extending trials when required.
Only after that gate passes, measure fresh official Ollama. Defaults stay
unchanged and neither original throughput objective is narrowed.
