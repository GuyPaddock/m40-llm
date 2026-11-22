# Q4 down projection from an available activation

## Frozen hypothesis and controls

On one Tesla M40, known zeros in a completed FFN activation may justify a
different layout for a single Q4 down projection. Examine Qwen2.5-3B Q4_0
layers 0, 17 and 35 (output 2048, input 11008). Validate GGUF type, shape
and model SHA-256 before timing. If a model tensor has a different type,
report the unsupported case. Do not substitute synthesized weights for it.

Use one fixed 32-output group, 32-way reduction and the Phase 4 selected-Q4
geometry. The primary inputs have 25% or 50% scattered nonzero entries.
Include 100% as a negative control and 12.5% as a diagnostic. Construct
deterministic sorted masks with the same live values in every arm.
No search over geometry or predictors. Masked activations remain synthetic
until the complete charged screen passes.

| Arm | Work |
|---|---|
| Dense native | Current CUDA Q4 decode GEMV on the full zero-filled vector. |
| Sparse native | October selected down partial and reduction with GGUF reads. |
| Lossless channel-major | Same selected operations on the existing exact repack, including native scale sidecar. |
| Zigzag | Quantize 32 adjacent outputs at fixed input, then store those blocks by output tile and input. One FP16 scale and 16 bytes of signed Q4 per group. Charge conversion and record error against the native source and a matching dense evaluation of the same zigzag weights. This is a bounded Q4_0 adaptation of the paper's axis/order, independent of its Metal K-quant kernels. |

Price the selection-free oracle and the charged path separately. Charge
thresholding, stable compaction, device count consumption, partials and
reduction every invocation. No hidden host count read or launch based on a
secret oracle count in the charged arm. Record CUDA event spans and host
submission plus synchronized elapsed time, actual allocation sizes, process
residency, workspace, conversion duration and input identities. Warm arms;
rotate order across five paired trials. Extend unstable evidence according
to the common protocol. Preserve raw timings and maximum discrepancies.

Require every full output to agree with the corresponding masked-weight
FP64 reference within 0.0005 + 0.0005 times the sum of absolute products per
output. Check stable indices and live count, including empty and full masks.
Lossless packed/native selected arms with identical accumulation must agree
bit-for-bit. Measure regrouping error separately. Admit live thresholding
only if the charged path is at least 10% faster than dense in at least two
layers at a primary density, exceeds twice the larger MAD/median, and has
a positive paired lower bound. Require a real-model accuracy suite and
end-to-end evidence before runtime promotion. Close geometry search if the
charged screen fails. A win confined to 12.5% is diagnostic.

Baseline: 48c63332462e009964e891be1cabaf139e3b50f8. The native Q4 GGUF has
SHA-256 670d82d0fbee6289b661eb323612c65bd2e97ec8c65cdbd9131aa44d30e9816a.
The selected CUDA controls use `cuda/q4_channels.cuh`; the existing stable
device selector is in `cuda/activation_sparsity.cuh`. The alternate grouping
uses a reconstruction of the known native Q4 weights as its input. Its
incremental quantization error is specific to this conversion, and it cannot
establish the quality of quantizing from original high-precision weights.
[SpQt §4.1–4.2](https://arxiv.org/html/2511.04477v1)

### CPU fixture preparation

All three GGUF tensors have the expected type and dimensions. Each native,
losslessly repacked or zigzag weight tensor occupies 12,681,216 payload
bytes. Keeping either alternate in addition to native duplicates that payload.
The lossless repack round-tripped all source bytes. Regrouping changes the
weights, with per-layer RMS differences of 0.0010901, 0.0010117 and
0.0007504 and maximum absolute differences of 0.04031, 0.09436 and 0.12903.
These are weight differences, not model-quality or speed results. Timings in
[prepare.jsonl](prepare.jsonl) include packing and conversion separately
and exclude file I/O. Host contention may affect the single observations.

From the worktree, source `scripts/dev-env.sh`, then build with
`cargo build --locked --release --no-default-features --example sparse_q4_inputs`.
Run `target/release/examples/sparse_q4_inputs MODEL NEW_DIRECTORY` and
compare its four named files per layer with [planes.sha256](planes.sha256).
The original run directory is `../raw/sparse-q4-inputs-01`.

### Operator replay

Build the standalone native probe with `bash docs/experiments/2026-11-21-sparse-q4/build.sh`
and the analyzer with `cargo build --locked --release --no-default-features --example sparse_q4_screen`.
The probe compiles the repository's exact dense GEMV, stable selector and
selected partials in one translation unit. Its charged variant reads the
valid count on the device. Every arm runs with the same buffers resident.
The fixture records allocation sizes, driver free bytes and release
observations. The telemetry sampler records process residency. The
standalone probe measures operator cost without loading the rest of the model.

With exclusive M40 access, run
`bash docs/experiments/2026-11-21-sparse-q4/run.sh NEW_OUTPUT_DIRECTORY`.
The synthetic input uses a deterministic shuffle per layer and a signed
nonzero value computed from its input index. The harness checks all 2048
outputs in each arm and measures 20 enqueues per event pair. Every case
has a separate warm pass. Raw wall time includes the ending synchronization;
CPU submission and CUDA event spans have distinct fields. The analyzer uses
the common paired-bootstrap implementation with operator calls/second. A
10% latency reduction requires at least 11.111% higher service rate.
