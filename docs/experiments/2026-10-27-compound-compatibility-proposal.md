# Compound Runtime Compatibility And Research Order

## Status And Sources

This is a metadata-first compatibility assessment, not cross-model runtime
qualification. The [immutable reference](2026-10-27-compound-reference.md)
remains Qwen2.5-3B Q8_0 at 50.910426 HTTP tok/s. No model weights were
downloaded, no cache directories were searched, and no GPU work was run
for this assessment. No model-specific speed claim transfers to another row.

Use explicit Q8_0 GGUF paths/hashes for Qwen2.5 Instruct 0.5B, 1.5B, 3B,
7B and Llama-3.2-1B-Instruct. The current 3B artifact is already bound by
the reference. Other rows need explicit local artifacts before loader,
quality or performance conclusions. A model name or F16 harness result
does not verify a Q8_0 artifact.

Planning shapes below were checked against the official configuration files
on 2026-10-27:
[Qwen0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json),
[Qwen1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json),
[Qwen3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/config.json),
and [Qwen7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json).
Those upstream files describe architecture, not a particular GGUF's dtypes
or storage. Recheck every tensor against the actual supplied artifact.

| Model | Layers | Hidden / FFN | Q / KV heads | Head dim | Vocabulary | Tied output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-0.5B | 24 | 896 / 4864 | 14 / 2 | 64 | 151936 | yes |
| Qwen2.5-1.5B | 28 | 1536 / 8960 | 12 / 2 | 128 | 151936 | yes |
| Qwen2.5-3B | 36 | 2048 / 11008 | 16 / 2 | 128 | 151936 | yes |
| Qwen2.5-7B | 28 | 3584 / 18944 | 28 / 4 | 128 | 152064 | no |
| Llama-3.2-1B | pending explicit Q8 GGUF metadata | | | | | |

Qwen's checked configurations have model context 32768 and disabled native
sliding-window attention. Do not confuse that model context with allocated
KV context or the compressed recent window. The official gated Llama config
could not be fetched in this session, so leave its numeric estimates pending
an explicit GGUF rather than inventing verified metadata. Llama remains a
required independent-family validation row.

## Kernel And Budget Matrix

At measured source `63240c5`, check each subsystem independently:

- Compound FFN requires sm_52, uploaded standard-layer Q8_0 gate/up,
  matching pair shapes, positive K divisible by 32 and K<=4096. The fixed
  consumer is not intrinsically limited to Qwen3B's K=2048, but another
  shape still needs resource, numerical and runtime validation.
- Prepared down/vocabulary require their actual Q8_0 sources, valid source
  identity and the existing launch/layout guards. Count a tied vocabulary
  source once. Preserve its original GGUF ownership as well as preparation.
- Packed QKV/output requires raw Q8_0 transformer projections, bounded
  complete input blocks and valid optional FP32 Q/K/V biases. Do not include
  QKV/output in prepared families when selecting that backend.
- Qualified parallel dense attention is head128-only, with valid GQA and
  sequence length <=8192. Qwen0.5B needs existing head64 attention instead;
  it cannot inherit the full head128-qualified configuration. Classify the
  supplied Llama artifact's head dimension before choosing its attention.
- Bounded dequant-SGEMM prefix requires head64/128, all seven transformer
  projections per layer Q8_0, aligned valid source pointers and no tensor
  requiring more than 128 MiB temporary FP32 weight storage. The vocabulary
  projection is not part of that prefix-temporary limit.

See the loader contracts in [FFN](../../src/infer/q8_ffn.rs),
[packed projections](../../src/infer/q8_attention_projection.rs),
[prefix](../../src/infer/q8_prefill.rs) and
[prepared weights](../../src/infer/execution_weights/runtime.rs).
The proposed no-old compressed-cache adapter is not implemented or qualified.

For the checked Qwen shapes, let D=hidden, H=FFN, L=layers, V=vocabulary and
`Q(K,N)=34*(K/32)*N` bytes. All relevant dimensions are multiples of 32.
Original split gate/up/down plus vocabulary is `3*L*Q(D,H)+Q(D,V)`.
Additional joint pairs are `2*L*Q(D,H)`. Proposed joint-only combined
preparation equals the original split byte total, but has different owners.
The largest transformer conversion here is `4*D*H` bytes.

| Model | Max prefix temporary B / MiB | Retained-split compound prepared B / MiB | Proposed joint-only prepared B / MiB |
| --- | ---: | ---: | ---: |
| Qwen0.5B | 17,432,576 / 16.625 | 700,306,432 / 667.864 | 478,041,088 / 455.896 |
| Qwen1.5B | 55,050,240 / 52.500 | 2,295,140,352 / 2188.816 | 1,476,268,032 / 1407.879 |
| Qwen3B | 90,177,536 / 86.000 | 4,642,226,176 / 4427.172 | 2,917,580,800 / 2782.422 |
| Qwen7B | 271,581,184 / 259.000 | 10,678,484,992 / 10183.797 | 6,638,714,880 / 6331.172 |
| Llama1B | pending GGUF metadata | pending GGUF metadata | pending GGUF metadata |

These are checked shape estimates, not allocation measurements for absent
GGUFs. Except for the qualified 3B retained-split row, neither budget fit nor
kernel qualification has been exercised with an actual model in this task.
Joint-only is a proposal for every row, including 3B.

Qwen7B fails both unchanged limits. Its 259 MiB projection cannot enter the
128 MiB SGEMM path; reducing prompt chunk size does not shrink this weight
matrix. Even proposed joint-only preparation exceeds the 6144 MiB budget
by 196,263,936 bytes. Do not raise either cap, silently use native prefix,
or describe it as full-configuration compatible because total GPU memory
might otherwise suffice. Any alternative backend/budget is a separate
explicit proposal, not this checkpoint.

## Model Admission And Evidence

For each supplied artifact, record path/hash, architecture, context, every
projection dtype/shape/offset, tokenizer/special-token metadata, RoPE layout,
Q/K/V biases, output tying and the effective backend selections. Mark each
subsystem as metadata-eligible, loader-accepted, correctness-validated or
performance-qualified; do not collapse those states into a single pass.

Measure original GGUF upload bytes, every prepared duplicate, combined
budget, final KV, reusable scratch, temporary prefix/host staging and
setup high-water marks. Include driver/library headroom before admitting
a model. The preparation budget is not a total-VRAM budget. Estimate KV
from the actual layer/KV-head/head-dimension/context metadata, not 3B's
allocation or aggregate parameter count.

After separate approval and artifact supply, test Qwen1.5B first, then
Llama1B for architecture diversity, then Qwen0.5B for head64/smaller-K
coverage. Qwen7B stays metadata-rejected under current limits. Do not
download an unavailable model or substitute an F16 artifact in a Q8 row.

Use explicit dense KV and retained-split ownership first. Validate each
candidate against that model's existing path with the same artifact and
prompt IDs, full-layer/prefix/projection parity and greedy output checks.
Use a same-model oracle where available. Dense-failed task rows remain
inconclusive for optimization quality, not passes or backend failures.
Run the affected CUDA/cuBLAS on/off and ownership checks before timing.

Then use release, serial paired comparisons within each model. Record
actual generated counts, stop reasons, prefill/decode/HTTP, source and
executable hashes, memory and variation. Do not force a model to generate
512 tokens after EOS; a shorter run cannot qualify the fixed 512-token goal.
Keep per-model results separate from the immutable Qwen3B/Ollama claim.
Test joint-only only after its own new ownership contract passes, and test
compressed routing independently rather than stacking unqualified changes.

## Recommended Order And Review Stop

1. Preserve the immutable compound reference and its measured binaries.
2. Review the [no-old compressed-cache proposal](2026-10-27-compressed-no-old-proposal.md).
   Its candidate-reduction work is concrete, but cache/output equivalence is
   still an unresolved gate. Authorize only its bounded existing-kernel
   adapter and correctness checks if that risk is acceptable.
3. Review the [joint-only ownership proposal](2026-10-27-joint-only-ownership-proposal.md).
   It has a precise memory hypothesis and no speed hypothesis. Validate it
   independently with new residency audits, not frozen 109-owner audits.
4. Admit explicit smaller-model artifacts in the order above. This metadata
   assessment can inform reviews immediately; GPU qualification waits for
   scope approval, input artifacts and successful prerequisite validation.
5. Require a new structural proposal before another exact-dense speed kernel.
   State supported expensive work, how it is avoided, total overhead/memory,
   plausible full-request scope, fixed correctness and predefined stop rules.
   The stronger goal remains 56.153149 HTTP tok/s and 938.958 ms/request
   savings. No candidate in these documents is presumed to supply it.

The four documentation commits are the current deliverable. Runtime/CUDA
implementation and benchmarks require a subsequent feasibility review.
After approval, keep implementation plus behavioral tests atomic, then commit
measured evidence separately. Preserve a failed experiment before any separate
removal. Do not combine changes before independently qualifying them.

All research remains opt-in. Keep runtime defaults, gates, model semantics
and the qualified prepared down/vocabulary control unchanged. No scheduler,
selection-policy, sparsity/pruning, approximate attention, EAGLE or rejected
verifier work is authorized. No hardware-ceiling claim is supported.
