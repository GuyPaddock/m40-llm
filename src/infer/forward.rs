use super::meta::norm_weight_dtype_code;
#[cfg(feature = "cuda")]
use super::workspace::ForwardWorkspacePtrs;
use super::LoadedModel;
#[cfg(feature = "cuda")]
use crate::cuda::CudaStream;
use crate::gguf::GgmlDType;
#[cfg(feature = "cuda")]
use crate::profile;
#[cfg(feature = "cuda")]
use crate::timing;
#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::{anyhow, Result};
use std::ffi::c_void;

#[cfg(feature = "cuda")]
fn log_profiled_op(
    label: &str,
    op: &str,
    before: Option<&profile::ProfileSnapshot>,
    elapsed: std::time::Duration,
) {
    timing::log(&format!("{label}.{op}"), elapsed);
    profile::log_delta(&format!("{label}.{op}"), before, elapsed);
}

impl LoadedModel {
    #[cfg(feature = "cuda")]
    fn with_forward_workspace<R>(
        &self,
        d_model: usize,
        kv_dim: usize,
        hidden_dim: usize,
        f: impl FnOnce(ForwardWorkspacePtrs) -> Result<R>,
    ) -> Result<R> {
        let mut guard = self.forward_workspace.lock().unwrap();
        if !guard
            .as_ref()
            .map(|ws| ws.matches(d_model, kv_dim, hidden_dim))
            .unwrap_or(false)
        {
            if let Some(old) = guard.take() {
                old.free(&self.cuda);
            }
            *guard = Some(super::workspace::ForwardWorkspace::new(
                &self.cuda, d_model, kv_dim, hidden_dim,
            )?);
        }
        let ptrs = guard.as_ref().expect("workspace initialized").ptrs();
        drop(guard);
        f(ptrs)
    }

    pub unsafe fn forward_one_token_minimal_with_norms(
        &self,
        d_x_f32: *const c_void,
        d_model: i32,
        d_wq_f16: *const c_void,
        d_wk_f16: *const c_void,
        d_wv_f16: *const c_void,
        d_wo_f16: *const c_void,
        d_w_gate_f16: *const c_void,
        d_w_up_f16: *const c_void,
        d_w_down_f16: *const c_void,
        hidden_dim: i32,
        seq_id: u32,
        seq_len: u32,
        attn_norm_weight: Option<(*const c_void, GgmlDType)>,
        ffn_norm_weight: Option<(*const c_void, GgmlDType)>,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        if d_model <= 0 || hidden_dim <= 0 {
            anyhow::bail!("forward_one_token_minimal: invalid dims");
        }

        #[cfg(feature = "cuda")]
        {
            let kv = self
                .kv_cache
                .as_ref()
                .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
            let q_heads = self.model_config.attention_head_count;
            let kv_heads = kv.num_heads();
            let head_dim = kv.head_dim();
            let kv_dim = kv_heads
                .checked_mul(head_dim)
                .context("KV projection dim overflow")?;
            if q_heads == 0 || q_heads % kv_heads != 0 {
                anyhow::bail!(
                    "forward_one_token_minimal: query heads {} must be a multiple of kv heads {}",
                    q_heads,
                    kv_heads
                );
            }
            if d_model as u32 != q_heads.saturating_mul(head_dim) {
                anyhow::bail!(
                    "forward_one_token_minimal: d_model {} != query heads {} * head_dim {}",
                    d_model,
                    q_heads,
                    head_dim
                );
            }

            self.with_forward_workspace(
                d_model as usize,
                kv_dim as usize,
                hidden_dim as usize,
                |ws| -> Result<()> {
                    let label = format!("forward.layer.{seq_id}.seq_len.{seq_len}");
                    // Pre-norm (RMSNorm) on x -> xn
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    if let Some((d_weight, dtype)) = attn_norm_weight {
                        self.run_rms_norm_weighted(
                            d_x_f32,
                            d_weight,
                            dtype,
                            ws.d_xn,
                            1,
                            d_model as u32,
                            self.model_config.layer_norm_epsilon,
                        )?;
                    } else {
                        self.run_rms_norm(
                            d_x_f32,
                            ws.d_xn,
                            1,
                            d_model as u32,
                            self.model_config.layer_norm_epsilon,
                        )?;
                    }
                    log_profiled_op(
                        &label,
                        "attn_norm",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );

                    // Q uses query heads; K/V use KV heads for GQA models.
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    self.qkv_project_f32xf16_gguf_f32(
                        ws.d_xn,
                        1,
                        d_model,
                        d_wq_f16,
                        d_model,
                        d_wk_f16,
                        kv_dim as i32,
                        d_wv_f16,
                        kv_dim as i32,
                        ws.dq,
                        ws.dk,
                        ws.dv,
                    )?;
                    log_profiled_op(
                        &label,
                        "qkv_project",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );

                    let pos = seq_len.saturating_sub(1);
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    self.cuda.rope_f32_inplace(
                        ws.dq,
                        1,
                        q_heads,
                        head_dim,
                        pos,
                        self.model_config.rope_freq_base,
                        self.model_config.rope_freq_scale,
                    )?;
                    log_profiled_op(
                        &label,
                        "rope_q",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );

                    // Append K/V for this token, rotating K into the cache.
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    self.append_kv_token_f32_rope_k(
                        seq_id,
                        ws.dk as *const c_void,
                        ws.dv as *const c_void,
                        pos,
                        self.model_config.rope_freq_base,
                        self.model_config.rope_freq_scale,
                    )?;
                    log_profiled_op(
                        &label,
                        "kv_append_rope_k",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );

                    // Use KV cache layout to validate/run attention
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    self.run_attention(
                        ws.dq as *const c_void,
                        ws.datt,
                        seq_id,
                        seq_len,
                        d_model as u32,
                        q_heads,
                        head_dim,
                    )?;
                    log_profiled_op(
                        &label,
                        "attention",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );

                    // Output projection of attention
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    self.out_proj_f32xf16_gguf_f32(
                        ws.datt as *const c_void,
                        d_wo_f16,
                        ws.dy_attn,
                        1,
                        d_model,
                        d_model,
                    )?;
                    log_profiled_op(
                        &label,
                        "out_project",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );

                    // Residual add y_attn: x1 = x + y_attn.
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    self.cuda.residual_add_f32(
                        d_x_f32,
                        ws.dy_attn as *const c_void,
                        ws.d_x1,
                        d_model as usize,
                    )?;
                    log_profiled_op(
                        &label,
                        "attn_residual",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );

                    // Post-attention norm: x1n = norm(x1)
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    if let Some((d_weight, dtype)) = ffn_norm_weight {
                        self.run_rms_norm_weighted(
                            ws.d_x1,
                            d_weight,
                            dtype,
                            ws.d_x1n,
                            1,
                            d_model as u32,
                            self.model_config.layer_norm_epsilon,
                        )?;
                    } else {
                        self.run_rms_norm(
                            ws.d_x1,
                            ws.d_x1n,
                            1,
                            d_model as u32,
                            self.model_config.layer_norm_epsilon,
                        )?;
                    }
                    log_profiled_op(
                        &label,
                        "ffn_norm",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );

                    // MLP gates and up (now feed post-attn normalized x1n)
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    self.mlp_gates_f32xf16_gguf_f32(
                        ws.d_x1n,
                        1,
                        d_model,
                        d_w_gate_f16,
                        d_w_up_f16,
                        hidden_dim,
                        ws.dgate,
                        ws.dup,
                    )?;
                    log_profiled_op(
                        &label,
                        "mlp_gate_up",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );

                    // hidden = SiLU(gate) * up, where SiLU(x) = x * sigmoid(x).
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    self.cuda.swiglu_f32_async(
                        ws.dgate as *const c_void,
                        ws.dup as *const c_void,
                        ws.dhid,
                        hidden_dim as usize,
                    )?;
                    log_profiled_op(
                        &label,
                        "swiglu",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );

                    // Down projection
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    self.cuda.stream_wait_for_stream(
                        CudaStream::Prefill,
                        CudaStream::Decode,
                        "swiglu_to_mlp_down",
                    )?;
                    self.mlp_down_proj_f32xf16_gguf_f32(
                        ws.dhid as *const c_void,
                        1,
                        hidden_dim,
                        d_w_down_f16,
                        d_model,
                        ws.dy_mlp,
                    )?;
                    log_profiled_op(
                        &label,
                        "mlp_down",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );

                    // Final residual add per pre-norm layout: out = x1 + y_mlp.
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    self.cuda.residual_add_f32(
                        ws.d_x1 as *const c_void,
                        ws.dy_mlp as *const c_void,
                        d_out_f32,
                        d_model as usize,
                    )?;
                    log_profiled_op(
                        &label,
                        "mlp_residual",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );

                    Ok(())
                },
            )
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                d_x_f32,
                d_model,
                d_wq_f16,
                d_wk_f16,
                d_wv_f16,
                d_wo_f16,
                d_w_gate_f16,
                d_w_up_f16,
                d_w_down_f16,
                hidden_dim,
                seq_id,
                seq_len,
                attn_norm_weight,
                ffn_norm_weight,
                d_out_f32,
            );
            Ok(())
        }
    }

    /// Minimal forward pass for one token without learned RMSNorm weights.
    ///
    /// # Safety
    /// All device pointers must be valid on this context's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_one_token_minimal(
        &self,
        d_x_f32: *const c_void,
        d_model: i32,
        d_wq_f16: *const c_void,
        d_wk_f16: *const c_void,
        d_wv_f16: *const c_void,
        d_wo_f16: *const c_void,
        d_w_gate_f16: *const c_void,
        d_w_up_f16: *const c_void,
        d_w_down_f16: *const c_void,
        hidden_dim: i32,
        seq_id: u32,
        seq_len: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        self.forward_one_token_minimal_with_norms(
            d_x_f32,
            d_model,
            d_wq_f16,
            d_wk_f16,
            d_wv_f16,
            d_wo_f16,
            d_w_gate_f16,
            d_w_up_f16,
            d_w_down_f16,
            hidden_dim,
            seq_id,
            seq_len,
            None,
            None,
            d_out_f32,
        )
    }

    /// # Safety
    pub unsafe fn run_rms_norm(
        &self,
        d_in: *const c_void,
        d_out: *mut c_void,
        seq_len: u32,
        dim: u32,
        eps: f32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.cuda.rms_norm_f32(d_in, d_out, seq_len, dim, eps)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_in, d_out, seq_len, dim, eps);
            Ok(())
        }
    }

    /// # Safety
    /// Device pointers must reference buffers sized for `seq_len * dim` f32 values and
    /// a norm weight vector of `dim` F16/F32 values on the same CUDA context.
    pub unsafe fn run_rms_norm_weighted(
        &self,
        d_in: *const c_void,
        d_weight: *const c_void,
        weight_dtype: GgmlDType,
        d_out: *mut c_void,
        seq_len: u32,
        dim: u32,
        eps: f32,
    ) -> Result<()> {
        let dtype_code = norm_weight_dtype_code(weight_dtype)?;
        #[cfg(feature = "cuda")]
        {
            self.cuda
                .rms_norm_f32_weighted(d_in, d_weight, d_out, seq_len, dim, eps, dtype_code)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_in, d_weight, d_out, seq_len, dim, eps, dtype_code);
            Ok(())
        }
    }

    /// # Safety
    /// Applies RoPE to Q/K in-place. Buffers must be sized for `rows * num_heads * head_dim` f32 values.
    pub unsafe fn apply_rope_f32(
        &self,
        d_q: *mut c_void,
        d_k: *mut c_void,
        rows: u32,
        num_heads: u32,
        head_dim: u32,
        past_len: u32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.cuda.rope_f32(
                d_q,
                d_k,
                rows,
                num_heads,
                head_dim,
                past_len,
                self.model_config.rope_freq_base,
                self.model_config.rope_freq_scale,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_q, d_k, rows, num_heads, head_dim, past_len);
            Ok(())
        }
    }
}

impl LoadedModel {
    /// Helper to run forward_one_token_minimal using mapped layer weights.
    /// Assumes embeddings have already produced x (f32) for the token index.
    /// The KV cache is addressed explicitly as KV[layer][sequence][position].
    /// # Safety
    /// - d_x_f32 and d_out_f32 must be valid device pointers on this model's CUDA context
    /// - Pointers must reference buffers sized for the provided d_model and hidden dims of the mapped layer
    pub unsafe fn forward_one_token_with_layer(
        &self,
        d_x_f32: *const c_void,
        layer: usize,
        sequence_id: u32,
        seq_len: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let w = self.map_standard_layer(layer)?;
        let layer_id: u32 = layer
            .try_into()
            .map_err(|_| anyhow::anyhow!("layer index {} does not fit in u32", layer))?;
        let position = seq_len.saturating_sub(1);
        if let Some(kv) = &self.kv_cache {
            if position >= kv.max_seq_len() {
                anyhow::bail!(
                    "KV position {} out of range for max_seq_len {}",
                    position,
                    kv.max_seq_len()
                );
            }
        }
        #[cfg(feature = "cuda")]
        {
            let kv_slot = self.kv_physical_slot_for_layer_sequence(layer_id, sequence_id)?;
            let wq_ptr = self.tensor_device_ptr("wq", &w.wq)?;
            let wk_ptr = self.tensor_device_ptr("wk", &w.wk)?;
            let wv_ptr = self.tensor_device_ptr("wv", &w.wv)?;
            let wo_ptr = self.tensor_device_ptr("wo", &w.wo)?;
            let w_gate_ptr = self.tensor_device_ptr("w_gate", &w.w_gate)?;
            let w_up_ptr = self.tensor_device_ptr("w_up", &w.w_up)?;
            let w_down_ptr = self.tensor_device_ptr("w_down", &w.w_down)?;
            let attn_norm_ptr = w
                .attn_norm
                .as_ref()
                .map(|view| {
                    self.tensor_device_ptr("attn_norm", view)
                        .map(|ptr| (ptr, view.dtype))
                })
                .transpose()?;
            let ffn_norm_ptr = w
                .ffn_norm
                .as_ref()
                .map(|view| {
                    self.tensor_device_ptr("ffn_norm", view)
                        .map(|ptr| (ptr, view.dtype))
                })
                .transpose()?;

            self.forward_one_token_minimal_with_norms(
                d_x_f32,
                w.d_model as i32,
                wq_ptr,
                wk_ptr,
                wv_ptr,
                wo_ptr,
                w_gate_ptr,
                w_up_ptr,
                w_down_ptr,
                w.hidden_dim as i32,
                kv_slot,
                seq_len,
                attn_norm_ptr,
                ffn_norm_ptr,
                d_out_f32,
            )?;

            return Ok(());
        }
        #[cfg(not(feature = "cuda"))]
        {
            use half::f16;
            use std::slice;
            let d_model = w.d_model;
            let hidden_dim = w.hidden_dim;

            // Read input x (f32) from host pointer
            let x_slice = slice::from_raw_parts(d_x_f32 as *const f32, d_model);

            // RMSNorm
            let eps = 1e-6f32;
            let mut mean_sq = 0f32;
            for &v in x_slice {
                mean_sq += v * v;
            }
            mean_sq /= d_model as f32;
            let scale = 1.0f32 / (mean_sq + eps).sqrt();
            let mut x_n = vec![0f32; d_model];
            for i in 0..d_model {
                x_n[i] = x_slice[i] * scale;
            }

            // Helper to multiply f32 row (1xK) by f16 matrix (KxN) into f32 (1xN)
            let dot_row = |row: &[f32], t: &crate::infer::DeviceTensorView, n: usize| -> Vec<f32> {
                let k = t.shape[0] as usize;
                debug_assert_eq!(k, row.len());
                let off = t.byte_offset as usize;
                let bytes = &self.host_weights[off..off + k * n * 2];
                let mut out = vec![0f32; n];
                for (j, out_j) in out.iter_mut().enumerate().take(n) {
                    let mut acc = 0f32;
                    let mut idx = j * 2; // column-major stride over rows in row-major [k x n]
                    for &row_val in row.iter().take(k) {
                        let lo = bytes[idx] as u16;
                        let hi = bytes[idx + 1] as u16;
                        let w = f16::from_bits(lo | (hi << 8)).to_f32();
                        acc += row_val * w;
                        idx += n * 2;
                    }
                    *out_j = acc;
                }
                out
            };

            // Helper to multiply f32 row (1xK) by Q5_1 matrix (KxN) into f32 (1xN)
            let dot_row_q5_1 =
                |row: &[f32], t: &crate::infer::DeviceTensorView, n: usize| -> Vec<f32> {
                    let k = t.shape[0] as usize;
                    debug_assert_eq!(k, row.len());
                    let off = t.byte_offset as usize;
                    let mut out = vec![0f32; n];

                    let blocks = k.div_ceil(32);
                    for (j, out_j) in out.iter_mut().enumerate().take(n) {
                        let mut acc = 0f32;
                        for (r, &row_val) in row.iter().enumerate().take(k) {
                            let b = r / 32;
                            let idx = r % 32;
                            let block_base = off + j * blocks * 6 + b * 6;
                            let scale_bytes = &self.host_weights[block_base..block_base + 4];
                            let scale = f32::from_le_bytes([
                                scale_bytes[0],
                                scale_bytes[1],
                                scale_bytes[2],
                                scale_bytes[3],
                            ]);
                            let q = self.host_weights[block_base + 4 + idx] as i8 as f32;
                            acc += row_val * scale * q;
                        }
                        *out_j = acc;
                    }

                    out
                };

            // Q, K, V
            let nq = w.wq.shape[1] as usize;
            let nk = w.wk.shape[1] as usize;
            let nv = w.wv.shape[1] as usize;
            let q = if w.wq.dtype == GgmlDType::Q5_1 {
                dot_row_q5_1(&x_n, &w.wq, nq)
            } else {
                dot_row(&x_n, &w.wq, nq)
            };
            let k_vec = if w.wk.dtype == GgmlDType::Q5_1 {
                dot_row_q5_1(&x_n, &w.wk, nk)
            } else {
                dot_row(&x_n, &w.wk, nk)
            };
            let v_vec = if w.wv.dtype == GgmlDType::Q5_1 {
                dot_row_q5_1(&x_n, &w.wv, nv)
            } else {
                dot_row(&x_n, &w.wv, nv)
            };

            // Append KV (host path)
            self.append_kv_token_f32_from_host_for_layer(
                layer_id,
                sequence_id,
                position,
                &k_vec,
                &v_vec,
            )?;

            // Attention over last token using KV cache helper
            let kv = self
                .kv_cache
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("kv_cache not allocated"))?;
            let num_heads = kv.num_heads();
            let head_dim = kv.head_dim();
            let mut attn_out = vec![0u8; d_model * 4];
            self.run_attention_for_layer(
                q.as_ptr() as *const c_void,
                attn_out.as_mut_ptr() as *mut c_void,
                layer_id,
                sequence_id,
                seq_len,
                d_model as u32,
                num_heads,
                head_dim,
            )?;
            let attn: Vec<f32> = attn_out
                .chunks_exact(4)
                .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
                .collect();
            let _ = (num_heads, head_dim); // already validated during allocation

            // Out projection and residual add: x1 = x + attn·Wo
            let no = w.wo.shape[1] as usize;
            debug_assert_eq!(no, d_model);
            let y_attn = if w.wo.dtype == GgmlDType::Q5_1 {
                dot_row_q5_1(&attn, &w.wo, no)
            } else {
                dot_row(&attn, &w.wo, no)
            };
            // Residual add on host: x1 = x + y_attn (fallback)
            let bytes_d = d_model * 4;
            let mut h_x = vec![0u8; bytes_d];
            let mut h_y_attn = vec![0u8; bytes_d];
            self.cuda
                .memcpy_d2h(h_x.as_mut_ptr() as *mut c_void, d_x_f32, bytes_d)?;
            // In non-CUDA path, we don't have d_y_attn, so we use y_attn directly
            for i in 0..y_attn.len() {
                let bytes = y_attn[i].to_le_bytes();
                h_y_attn[i * 4..i * 4 + 4].copy_from_slice(&bytes);
            }
            let mut x_f = Vec::with_capacity(d_model);
            let mut y_attn_f = Vec::with_capacity(d_model);
            for ch in h_x.chunks_exact(4) {
                x_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
            }
            for ch in h_y_attn.chunks_exact(4) {
                y_attn_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
            }
            let mut x1_f = Vec::with_capacity(d_model);
            for i in 0..d_model {
                x1_f.push(x_f[i] + y_attn_f[i]);
            }
            let mut x1_bytes = Vec::with_capacity(bytes_d);
            for v in &x1_f {
                x1_bytes.extend_from_slice(&v.to_le_bytes());
            }
            self.cuda.memcpy_h2d(
                d_x_f32 as *mut c_void,
                x1_bytes.as_ptr() as *const c_void,
                bytes_d,
            )?;

            // Post-attention norm
            let mut h_x1 = vec![0u8; bytes_d];
            self.cuda.memcpy_d2h(
                h_x1.as_mut_ptr() as *mut c_void,
                h_x1.as_ptr() as *const c_void,
                bytes_d,
            )?;
            let mut x1_f = Vec::with_capacity(d_model);
            for ch in h_x1.chunks_exact(4) {
                x1_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
            }
            let mut mean_sq2 = 0f32;
            for v in &x1_f {
                mean_sq2 += v * v;
            }
            mean_sq2 /= d_model as f32;
            let scale2 = 1.0f32 / (mean_sq2 + eps).sqrt();
            let mut x1n = vec![0f32; d_model];
            for i in 0..d_model {
                x1n[i] = x1_f[i] * scale2;
            }

            // MLP: gate/up -> SiLU(gate)*up -> down -> residual add with x1n
            let h = w.w_up.shape[1] as usize;
            debug_assert_eq!(h, hidden_dim);
            let gate = if w.w_gate.dtype == GgmlDType::Q5_1 {
                dot_row_q5_1(&x1n, &w.w_gate, h)
            } else {
                dot_row(&x1n, &w.w_gate, h)
            };
            let up = if w.w_up.dtype == GgmlDType::Q5_1 {
                dot_row_q5_1(&x1n, &w.w_up, h)
            } else {
                dot_row(&x1n, &w.w_up, h)
            };
            fn sigmoid(v: f32) -> f32 {
                1.0 / (1.0 + (-v).exp())
            }
            let mut hidden = vec![0f32; h];
            for i in 0..h {
                let g = gate[i];
                hidden[i] = (g * sigmoid(g)) * up[i];
            }
            let y_mlp = if w.w_down.dtype == GgmlDType::Q5_1 {
                dot_row_q5_1(&hidden, &w.w_down, d_model)
            } else {
                dot_row(&hidden, &w.w_down, d_model)
            };
            let mut y = vec![0f32; d_model];
            for i in 0..d_model {
                y[i] = x1n[i] + y_mlp[i];
            }

            // Write to output pointer
            let out_slice = slice::from_raw_parts_mut(d_out_f32 as *mut f32, d_model);
            out_slice.copy_from_slice(&y);

            // Free scratch buffers

            Ok(())
        }
    }
    /// Runs one token through every transformer layer.
    ///
    /// The current single-request server path uses explicit layer/sequence KV
    /// addressing with `sequence_id=0`; internally that still maps each layer
    /// to one physical KV slot until the cache is widened for real batching.
    ///
    /// # Safety
    /// - `d_x_f32` and `d_out_f32` must be valid device pointers on this model's CUDA context.
    /// - Both buffers must contain/be sized for one f32 hidden vector of `embedding_length`.
    pub unsafe fn forward_one_token_all_layers(
        &self,
        d_x_f32: *const c_void,
        seq_len: u32,
        d_out_f32: *mut c_void,
    ) -> Result<usize> {
        let layer_count = self.model_config.block_count as usize;
        if layer_count == 0 {
            anyhow::bail!("forward_one_token_all_layers: model has zero layers");
        }
        let (d_model, _) = self.validate_standard_layers()?;
        let kv = self.kv_cache.as_ref().ok_or_else(|| {
            anyhow!("kv_cache not allocated; call allocate_kv_cache_for_layers first")
        })?;
        if kv.max_batch_size() < self.model_config.block_count {
            anyhow::bail!(
                "kv_cache has {} slots, but full-layer forward needs {} layer slots",
                kv.max_batch_size(),
                self.model_config.block_count
            );
        }

        #[cfg(feature = "cuda")]
        {
            if layer_count == 1 {
                self.forward_one_token_with_layer(d_x_f32, 0, 0, seq_len, d_out_f32)?;
                return Ok(1);
            }
            let hidden_dim = self.model_config.feed_forward_length as usize;
            let kv_dim = (self.model_config.attention_head_count_kv as usize)
                .checked_mul(self.model_config.attention_key_length as usize)
                .context("forward workspace kv dim overflow")?;

            return self.with_forward_workspace(d_model, kv_dim, hidden_dim, |ws| {
                let mut current = d_x_f32;
                let sequence_id = 0u32;
                for layer in 0..layer_count {
                    let next = if layer + 1 == layer_count {
                        d_out_f32
                    } else if layer % 2 == 0 {
                        ws.scratch_a
                    } else {
                        ws.scratch_b
                    };
                    self.forward_one_token_with_layer(current, layer, sequence_id, seq_len, next)?;
                    current = next as *const c_void;
                }
                Ok(layer_count)
            });
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_x_f32, seq_len, d_out_f32, d_model);
            anyhow::bail!("forward_one_token_all_layers requires CUDA device buffers")
        }
    }
}
