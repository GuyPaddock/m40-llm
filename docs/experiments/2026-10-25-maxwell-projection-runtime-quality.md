# Combined QKV/Output Runtime Quality Gate

## Decision

The model-local QKV/output warp combination passes actual-model quality
against the **current split-FFN/SGEMM control**. All **512 actual generated
IDs** and all **twelve frozen task outputs** match. There are zero
regressions across **seven dense-valid tasks**; five dense-failed tasks
remain inconclusive. All seven sampled full-vocabulary logit comparisons
have **zero maximum and mean difference**, with top-10 overlap ten.

This qualifies one opt-in combination for controlled HTTP timing. It does
not establish an HTTP gain, change defaults or complete either throughput
objective. The last qualified runtime remains 45.075 HTTP tok/s versus
fresh official Ollama 43.119. Original targets remain 53.06 and at least
1.30x fresh Ollama, currently 56.055.

The preceding [QKV](2026-10-25-maxwell-qkv-warp-screen.md) and
[output](2026-10-25-maxwell-output-warp-screen.md) screens independently
passed their stage gates, and their prespecified combined optimistic
ceiling cleared 3%. This quality check does not substitute for the original
five-pair measured HTTP acceptance gate.

## Configuration And Reproduction

Runtime integration is `b7edf08`; comparison/evidence tooling is
`1349613b4844a6e097eeb4fdddafa8ead587531d`. The actual run uses that clean
release source, CUDA/server features, NVCC 12.8.93, sm_52 SASS/compute_52
PTX, existing CUDA `-O3` flags and no fast math. Quality executable SHA-256:
`6b01f34771bb491ba95f6e0b28ffe5acb49dcb63d0e93936b9948881f6743c71`.
The same build's HTTP executable is
`7ffa5b7898affb3c67a93fd7040a1b590801c89fa66c6daf6ba3ffdef341a360`.

Both arms use the identical Qwen2.5-3B Q8_0 blob, real Qwen2 tokenizer,
ChatML, Q/K/V biases, split-half RoPE, head128 attention, explicit dense
FP16 KV/context 1024, greedy sampling and natural stopping. The control
is `split-f32x4-128` with `dequant-sgemm`; the candidate changes only
attention projections to `warp-256`.

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
  --variant-id native-qkv-output-warp-256-split-sgemm \
  --split all --context-tokens 1024 \
  --q8-runtime-candidate split-f32x4-128 \
  --q8-runtime-prefill dequant-sgemm \
  --q8-runtime-control qualified-split-sgemm \
  --q8-runtime-attention warp-256 \
  --output /tmp/new-projection-runtime-quality.jsonl
```

The typed choices do not alter the frozen environment or calibration inputs.
Generic runtime loading also supports the explicit opt-in
`M40LLM_Q8_ATTN_PROJECTIONS=warp-256`, captured once per model. Older explicit
reference constructors stay native. There is no per-token selector lookup,
activation quantization, new weight layout or change to sampling/attention.
Unsupported raw weights, CUDA builds, device capability, prepared QKV/output
conflicts and wrong model spans fail clearly rather than silently falling back.

## Actual Behavior And Memory

The canonical 65 prompt IDs and 512-token continuation match both the new
control and the previously qualified split runtime. Both stop with `length`.
Sampled generated steps 0, 1, 8, 32, 128, 256 and 511 have identical full
151,936-element logits, greedy IDs and top ten. All twelve task outputs and
scores match. Dense-valid calibration cases are chat reminder, depot fact
and config retry; dense-valid held-out cases are chat tour, observatory
fact, multi-workshop and config cache.

The candidate executes **18,432 QKV warp launches and 18,432 output warp
launches**, exactly 36 layers times 512 one-row forwards. The control uses
the corresponding original kernels. All other per-operation counts and
aggregate enqueue counters are identical. Both retain 18,432 split-FFN
launches, 18,944 prepared down/vocabulary launches, 252 prefix conversions
and 252 SGEMMs. The 64-token prefix remains on the existing SGEMM path;
the final prompt token and subsequent decode use the selected reductions.

All **109 prepared tensor identities, layouts, source/payload byte counts
and accounting fields** match the current control and the previous selected
split runtime, excluding preparation/hash duration fields. Each arm has
3,279,519,744 source bytes and 2,917,580,800 prepared bytes, for
**6,197,100,544 source-plus-prepared bytes** before KV/workspace. There is
no extra gate/up, QKV/output preparation or activation storage.

Dense FP16 KV is **37,748,736 bytes per model**. The existing reusable
temporary SGEMM weight buffer stays **90,177,536 bytes**, capped at 128 MiB
and released after prefix work. These are requested payload capacities,
not peak VRAM. Each whole generation records 42 allocations/frees for
session/admission work; this is not an allocation-free whole-request claim.
The warmed-token fixture separately confirms no per-token device allocation.
Tracked payload returns zero to zero after both models are dropped, and
the compute-process list is empty after exit.

Single-run resident measurements below include no HTTP and are diagnostic
only. They cannot qualify the performance estimate or replace paired trials.

| Arm | Prefill ms | Decode ms | Generation metrics total ms | Resident generation wall ms |
| --- | ---: | ---: | ---: | ---: |
| Current split-FFN/SGEMM | 498.443 | 10834.884 | 11333.367 | 11381.893 |
| Same, QKV/output warp | 501.005 | 10300.083 | 10801.129 | 10851.158 |

## Tests, Audit And Next Gate

Four new runtime tests pass with cuBLAS on/off, covering model-local
selection after environment changes, full Llama/Qwen fixture logits,
prepared identities, warmed session reuse, multi-row native prefill,
wrong spans and unsupported configurations. Existing QKV/output primitive,
attention-grid, forward, FFN-runtime and wrapper checks pass. Non-CUDA
warning-as-error tests pass 211 cases; CUDA/server all-target Clippy and
formatting pass. Comparison-tool checks pass 19 HTTP-tool and 32
model-quality tests, plus eleven non-CUDA example tests. The unrelated
ignored real compact-model fixture requires an external model; it was
explicitly excluded, not counted as a pass. No model download or pruning
experiment was added. The optional real-Qwen mixed-prefill diagnostic was
also unconfigured; this checkpoint makes no new mixed-prefill claim.

- [Raw quality report](2026-10-25-maxwell-projection-runtime-quality.jsonl)
- [Independent audit](2026-10-25-maxwell-projection-runtime-quality-audit.jq) and [result](2026-10-25-maxwell-projection-runtime-quality-audit.json)
- [Run log](2026-10-25-maxwell-projection-runtime-quality.log.gz) and [checksums](2026-10-25-maxwell-projection-runtime-quality.sha256)
- [Previously qualified control](2026-10-25-maxwell-split-runtime-quality.md)

The audit checks source/executable/input bindings, actual prior/current IDs,
all task outcomes, full-vocabulary equality, all existing cache identities,
actual projection/other dispatch and cleanup. Changed dispatch, prepared
gate/up payload and actual-token negative controls are rejected.

Next run incremental HTTP pairs against this same current split-FFN/SGEMM
control. Require at least `max(3%, twice observed variation)` and a positive
95% paired-bootstrap interval, with five initial alternating pairs and the
original extension rules. Only a passing result proceeds to fresh official
Ollama qualification. Do not add another kernel variant or change defaults
to bypass either gate.
