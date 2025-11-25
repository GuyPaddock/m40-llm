# Minimal forward path (one layer, seq_len=1..2)

Status: minimal, correct, and test‑covered. Intended for validation and wiring, not final performance.

What it runs
- Embedding load: FP16 row from tok_embeddings → host convert → device FP32 buffer
- Pre‑attn RMSNorm: host fallback (operates on device buffers via copy)
- Q/K/V projections: f32×f16→f32 GEMM
- KV append: FP32→KV cache
- Attention: last‑token attention against KV (CUDA path)
- Out projection: f32×f16→f32 GEMM
- Residual add: host fallback (x + y_attn)
- Post‑attn RMSNorm: host fallback
- MLP gates/up: f32×f16→f32 GEMM, then host SiLU(gate) * up
- Down projection: f32×f16→f32 GEMM
- Final residual add: host fallback (x1 + y_mlp)

Assumptions/limits
- Batch = 1, seq_len ∈ {1, 2} validated by tests
- FP16 storage, FP32 compute; cuBLAS used if enabled via M40LLM_ENABLE_CUBLAS=1
- KV cache must be allocated with allocate_kv_cache_with_layout so that dim = num_heads * head_dim
- RoPE is not applied in this minimal path
- Norm/residual/activation are currently host fallbacks to keep scope minimal

Testing
- tests/forward_parity_toy.rs constructs a tiny GGUF model with deterministic FP16 weights
- Validates device output vs a CPU reference
  - One‑token parity at tol ~1e-3
  - Two‑token prefill+decode parity at tol 5e-3 (allows accumulated rounding across QKV+attn+MLP)
- CUDA‑gated; skips gracefully when not on sm_52

Next steps (tracked)
- Optional CUDA RMSNorm/residual (t26-3-impl)
- Device selection and guardrails documented (see docs/device_selection.md)
- Toward full minimal forward across a layer on toy GGUF (t26-min-forward)
