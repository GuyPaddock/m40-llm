#[cfg(feature = "cuda")]
use super::gemm::decode_cublas_single_stream_enabled;
use super::meta::norm_weight_dtype_code;
#[cfg(feature = "cuda")]
use super::workspace::ForwardWorkspacePtrs;
use super::LoadedModel;
#[cfg(feature = "cuda")]
use crate::cuda::CudaStream;
#[cfg(feature = "cuda")]
use crate::cuda::KVCache;
use crate::gguf::GgmlDType;
#[cfg(feature = "cuda")]
use crate::infer::{BatchMetadata, BatchSequence, VarlenPrefillPlan};
#[cfg(feature = "cuda")]
use crate::kv_compression;
#[cfg(feature = "cuda")]
use crate::kv_selection;
#[cfg(feature = "cuda")]
use crate::profile;
#[cfg(feature = "cuda")]
use crate::timing;
#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::{anyhow, Result};
use std::ffi::c_void;

#[cfg(feature = "cuda")]
type QkvBiasPtrs = (*const c_void, *const c_void, *const c_void);

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct ForwardBatchItem {
    pub d_x_f32: *mut c_void,
    pub sequence_id: u32,
    pub seq_len: u32,
    pub d_out_f32: *mut c_void,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct ForwardPrefillSequence<'a> {
    pub token_ids: &'a [u32],
    pub sequence_id: u32,
    pub d_out_f32: *mut c_void,
}

#[cfg(feature = "cuda")]
unsafe fn mut_byte_offset(ptr: *mut c_void, bytes: usize) -> *mut c_void {
    unsafe { (ptr as *mut u8).add(bytes).cast::<c_void>() }
}

#[cfg(feature = "cuda")]
unsafe fn const_byte_offset(ptr: *const c_void, bytes: usize) -> *const c_void {
    unsafe { (ptr as *const u8).add(bytes).cast::<c_void>() }
}

#[cfg(feature = "cuda")]
fn log_profiled_op(
    label: &str,
    op: &str,
    before: Option<&profile::ProfileSnapshot>,
    elapsed: std::time::Duration,
) {
    timing::log(format!("{label}.{op}"), elapsed);
    profile::log_delta(&format!("{label}.{op}"), before, elapsed);
}

#[cfg(feature = "cuda")]
fn kv_physical_slot_for_layer_sequence_in(
    kv: &KVCache,
    layer_count: u32,
    layer_id: u32,
    sequence_id: u32,
) -> Result<u32> {
    if layer_count == 0 || layer_id >= layer_count {
        anyhow::bail!("KV layer_id {layer_id} out of range for {layer_count} layers");
    }
    let sequence_capacity = kv.max_batch_size() / layer_count;
    if sequence_id >= sequence_capacity {
        anyhow::bail!(
            "KV sequence_id {} out of range for {} logical sequences",
            sequence_id,
            sequence_capacity
        );
    }
    let physical_slot = sequence_id
        .checked_mul(layer_count)
        .and_then(|base| base.checked_add(layer_id))
        .ok_or_else(|| anyhow!("KV physical slot overflow"))?;
    if physical_slot >= kv.max_batch_size() {
        anyhow::bail!(
            "KV physical slot {} out of range for {} slots",
            physical_slot,
            kv.max_batch_size()
        );
    }
    Ok(physical_slot)
}

#[cfg(feature = "cuda")]
fn forward_finite_log_enabled() -> bool {
    std::env::var("M40LLM_FORWARD_FINITE_LOG").ok().as_deref() == Some("1")
}

impl LoadedModel {
    #[cfg(feature = "cuda")]
    fn qkv_bias_ptrs(&self, w: &super::StandardLayerWeights) -> Result<Option<QkvBiasPtrs>> {
        let bq_ptr =
            w.bq.as_ref()
                .map(|view| self.tensor_device_ptr("bq", view))
                .transpose()?;
        let bk_ptr =
            w.bk.as_ref()
                .map(|view| self.tensor_device_ptr("bk", view))
                .transpose()?;
        let bv_ptr =
            w.bv.as_ref()
                .map(|view| self.tensor_device_ptr("bv", view))
                .transpose()?;
        match (bq_ptr, bk_ptr, bv_ptr) {
            (Some(bq), Some(bk), Some(bv)) => Ok(Some((bq, bk, bv))),
            (None, None, None) => Ok(None),
            _ => anyhow::bail!("Q/K/V attention biases must be all present or all absent"),
        }
    }

    #[cfg(feature = "cuda")]
    fn apply_qkv_bias_async(
        &self,
        qkv_bias: Option<QkvBiasPtrs>,
        rows: usize,
        d_model: usize,
        kv_dim: usize,
        ws: ForwardWorkspacePtrs,
    ) -> Result<()> {
        let Some((d_bq, d_bk, d_bv)) = qkv_bias else {
            return Ok(());
        };
        if rows == 0 {
            return Ok(());
        }
        let bytes_d = d_model
            .checked_mul(std::mem::size_of::<f32>())
            .context("q bias row byte size overflow")?;
        let bytes_kv = kv_dim
            .checked_mul(std::mem::size_of::<f32>())
            .context("kv bias row byte size overflow")?;
        self.cuda.stream_wait_for_stream(
            CudaStream::Decode,
            CudaStream::Prefill,
            "qkv_project_to_bias_add",
        )?;
        for row in 0..rows {
            let q_row = unsafe { mut_byte_offset(ws.dq, row * bytes_d) };
            let k_row = unsafe { mut_byte_offset(ws.dk, row * bytes_kv) };
            let v_row = unsafe { mut_byte_offset(ws.dv, row * bytes_kv) };
            unsafe {
                self.cuda
                    .residual_add_f32_async(q_row as *const c_void, d_bq, q_row, d_model)?;
                self.cuda
                    .residual_add_f32_async(k_row as *const c_void, d_bk, k_row, kv_dim)?;
                self.cuda
                    .residual_add_f32_async(v_row as *const c_void, d_bv, v_row, kv_dim)?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn debug_log_device_f32_finiteness(
        &self,
        label: &str,
        ptr: *const c_void,
        len: usize,
    ) -> Result<()> {
        if !forward_finite_log_enabled() {
            return Ok(());
        }
        self.cuda.synchronize_stream(CudaStream::Decode)?;
        self.cuda.synchronize_stream(CudaStream::Prefill)?;
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .context("debug finiteness byte size overflow")?;
        let mut raw = vec![0u8; bytes];
        unsafe {
            self.cuda
                .memcpy_d2h(raw.as_mut_ptr() as *mut c_void, ptr, bytes)?;
        }
        let mut nonfinite = 0usize;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut first_nonfinite = None;
        for (idx, ch) in raw.chunks_exact(4).enumerate() {
            let value = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
            if value.is_finite() {
                min = min.min(value);
                max = max.max(value);
            } else {
                nonfinite += 1;
                first_nonfinite.get_or_insert((idx, value));
            }
        }
        eprintln!(
            "[cuda] finite_check {label}: len={len} nonfinite={nonfinite} first_nonfinite={first_nonfinite:?} finite_min={min:.6e} finite_max={max:.6e}"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn with_forward_workspace<R>(
        &self,
        d_model: usize,
        kv_dim: usize,
        hidden_dim: usize,
        f: impl FnOnce(ForwardWorkspacePtrs) -> Result<R>,
    ) -> Result<R> {
        self.with_forward_workspace_for_rows(d_model, kv_dim, hidden_dim, 1, f)
    }

    #[cfg(feature = "cuda")]
    fn with_forward_workspace_for_rows<R>(
        &self,
        d_model: usize,
        kv_dim: usize,
        hidden_dim: usize,
        rows: usize,
        f: impl FnOnce(ForwardWorkspacePtrs) -> Result<R>,
    ) -> Result<R> {
        if rows == 0 {
            anyhow::bail!("forward workspace row count must be positive");
        }
        let mut guard = self.forward_workspace.lock().unwrap();
        if !guard
            .as_ref()
            .map(|ws| ws.matches(d_model, kv_dim, hidden_dim, rows))
            .unwrap_or(false)
        {
            if let Some(old) = guard.take() {
                old.free(&self.cuda);
            }
            *guard = Some(super::workspace::ForwardWorkspace::new_with_rows(
                &self.cuda, d_model, kv_dim, hidden_dim, rows,
            )?);
        }
        let ptrs = guard.as_ref().expect("workspace initialized").ptrs();
        drop(guard);
        f(ptrs)
    }

    #[cfg(feature = "cuda")]
    pub unsafe fn forward_one_token_minimal_with_norms(
        &self,
        d_x_f32: *const c_void,
        d_model: i32,
        d_wq_f16: *const c_void,
        d_wk_f16: *const c_void,
        d_wv_f16: *const c_void,
        qkv_bias: Option<QkvBiasPtrs>,
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
            if q_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
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
            let single_stream_decode = decode_cublas_single_stream_enabled();

            self.with_forward_workspace(
                d_model as usize,
                kv_dim as usize,
                hidden_dim as usize,
                |ws| -> Result<()> {
                    let wait_cross_stream =
                        |waiting: CudaStream, signal: CudaStream, op: &'static str| -> Result<()> {
                            if single_stream_decode {
                                Ok(())
                            } else {
                                self.cuda.stream_wait_for_stream(waiting, signal, op)
                            }
                        };
                    let label = format!("forward.layer.{seq_id}.seq_len.{seq_len}");
                    // Pre-norm (RMSNorm) on x -> xn
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    if let Some((d_weight, dtype)) = attn_norm_weight {
                        self.run_rms_norm_weighted_async(
                            d_x_f32,
                            d_weight,
                            dtype,
                            ws.d_xn,
                            1,
                            d_model as u32,
                            self.model_config.layer_norm_epsilon,
                        )?;
                    } else {
                        self.run_rms_norm_async(
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
                    self.debug_log_device_f32_finiteness(
                        &format!("{label}.attn_norm"),
                        ws.d_xn as *const c_void,
                        d_model as usize,
                    )?;

                    // Q uses query heads; K/V use KV heads for GQA models.
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Prefill,
                        CudaStream::Decode,
                        "attn_norm_to_qkv_project",
                    )?;
                    self.qkv_project_f32xf16_gguf_f32_async(
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
                    self.apply_qkv_bias_async(qkv_bias, 1, d_model as usize, kv_dim as usize, ws)?;
                    log_profiled_op(
                        &label,
                        "qkv_project",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );
                    self.debug_log_device_f32_finiteness(
                        &format!("{label}.q"),
                        ws.dq as *const c_void,
                        d_model as usize,
                    )?;
                    self.debug_log_device_f32_finiteness(
                        &format!("{label}.k"),
                        ws.dk as *const c_void,
                        kv_dim as usize,
                    )?;
                    self.debug_log_device_f32_finiteness(
                        &format!("{label}.v"),
                        ws.dv as *const c_void,
                        kv_dim as usize,
                    )?;

                    let pos = seq_len.saturating_sub(1);
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Decode,
                        CudaStream::Prefill,
                        "qkv_project_to_rope_kv",
                    )?;
                    self.cuda.rope_f32_inplace_layout_async(
                        ws.dq,
                        1,
                        q_heads,
                        head_dim,
                        pos,
                        self.model_config.rope_freq_base,
                        self.model_config.rope_freq_scale,
                        self.model_config.rope_layout_code(),
                    )?;
                    log_profiled_op(
                        &label,
                        "rope_q",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );
                    self.debug_log_device_f32_finiteness(
                        &format!("{label}.rope_q"),
                        ws.dq as *const c_void,
                        d_model as usize,
                    )?;

                    // Append K/V for this token, rotating K into the cache.
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Decode,
                        CudaStream::Prefill,
                        "qkv_project_to_kv_append",
                    )?;
                    self.append_kv_token_f32_rope_k_at_async(
                        seq_id,
                        ws.dk as *const c_void,
                        ws.dv as *const c_void,
                        pos,
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
                    self.run_attention_async(
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
                    self.debug_log_device_f32_finiteness(
                        &format!("{label}.attention"),
                        ws.datt as *const c_void,
                        d_model as usize,
                    )?;

                    // Output projection of attention
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Prefill,
                        CudaStream::Decode,
                        "attention_to_out_project",
                    )?;
                    self.out_proj_f32xf16_gguf_f32_async(
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
                    self.debug_log_device_f32_finiteness(
                        &format!("{label}.out_project"),
                        ws.dy_attn as *const c_void,
                        d_model as usize,
                    )?;

                    // Residual add y_attn: x1 = x + y_attn.
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Decode,
                        CudaStream::Prefill,
                        "out_project_to_attn_residual",
                    )?;
                    self.cuda.residual_add_f32_async(
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
                    self.debug_log_device_f32_finiteness(
                        &format!("{label}.attn_residual"),
                        ws.d_x1 as *const c_void,
                        d_model as usize,
                    )?;

                    // Post-attention norm: x1n = norm(x1)
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    if let Some((d_weight, dtype)) = ffn_norm_weight {
                        self.run_rms_norm_weighted_async(
                            ws.d_x1,
                            d_weight,
                            dtype,
                            ws.d_x1n,
                            1,
                            d_model as u32,
                            self.model_config.layer_norm_epsilon,
                        )?;
                    } else {
                        self.run_rms_norm_async(
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
                    self.debug_log_device_f32_finiteness(
                        &format!("{label}.ffn_norm"),
                        ws.d_x1n as *const c_void,
                        d_model as usize,
                    )?;

                    // MLP gates and up (now feed post-attn normalized x1n)
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Prefill,
                        CudaStream::Decode,
                        "ffn_norm_to_mlp_gate_up",
                    )?;
                    self.mlp_gates_f32xf16_gguf_f32_async(
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
                    self.debug_log_device_f32_finiteness(
                        &format!("{label}.gate"),
                        ws.dgate as *const c_void,
                        hidden_dim as usize,
                    )?;
                    self.debug_log_device_f32_finiteness(
                        &format!("{label}.up"),
                        ws.dup as *const c_void,
                        hidden_dim as usize,
                    )?;

                    // hidden = SiLU(gate) * up, where SiLU(x) = x * sigmoid(x).
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Decode,
                        CudaStream::Prefill,
                        "mlp_gate_up_to_swiglu",
                    )?;
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
                    self.debug_log_device_f32_finiteness(
                        &format!("{label}.swiglu"),
                        ws.dhid as *const c_void,
                        hidden_dim as usize,
                    )?;

                    // Down projection
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Prefill,
                        CudaStream::Decode,
                        "swiglu_to_mlp_down",
                    )?;
                    self.mlp_down_proj_f32xf16_gguf_f32_async(
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
                    self.debug_log_device_f32_finiteness(
                        &format!("{label}.mlp_down"),
                        ws.dy_mlp as *const c_void,
                        d_model as usize,
                    )?;

                    // Final residual add per pre-norm layout: out = x1 + y_mlp.
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Decode,
                        CudaStream::Prefill,
                        "mlp_down_to_mlp_residual",
                    )?;
                    self.cuda.residual_add_f32_async(
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
                    self.debug_log_device_f32_finiteness(
                        &format!("{label}.mlp_residual"),
                        d_out_f32 as *const c_void,
                        d_model as usize,
                    )?;

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

    /// Graph-capture-compatible one-token forward. Position-sensitive kernels
    /// read token position and sequence length from device buffers so replay can
    /// update those values without rebuilding the graph.
    ///
    /// # Safety
    /// All device pointers must be valid on this context's device. `d_position`
    /// and `d_seq_len` must each point to one device-resident u32.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_one_token_minimal_with_norms_graph_params(
        &self,
        d_x_f32: *const c_void,
        d_model: i32,
        d_wq_f16: *const c_void,
        d_wk_f16: *const c_void,
        d_wv_f16: *const c_void,
        qkv_bias: Option<QkvBiasPtrs>,
        d_wo_f16: *const c_void,
        d_w_gate_f16: *const c_void,
        d_w_up_f16: *const c_void,
        d_w_down_f16: *const c_void,
        hidden_dim: i32,
        seq_id: u32,
        d_position: *const u32,
        d_seq_len: *const u32,
        attn_norm_weight: Option<(*const c_void, GgmlDType)>,
        ffn_norm_weight: Option<(*const c_void, GgmlDType)>,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        if d_model <= 0 || hidden_dim <= 0 {
            anyhow::bail!("forward_one_token_minimal_graph_params: invalid dims");
        }
        if d_position.is_null() || d_seq_len.is_null() {
            anyhow::bail!("forward_one_token_minimal_graph_params: null device parameter");
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
            if q_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
                anyhow::bail!(
                    "forward_one_token_minimal_graph_params: query heads {} must be a multiple of kv heads {}",
                    q_heads,
                    kv_heads
                );
            }
            if d_model as u32 != q_heads.saturating_mul(head_dim) {
                anyhow::bail!(
                    "forward_one_token_minimal_graph_params: d_model {} != query heads {} * head_dim {}",
                    d_model,
                    q_heads,
                    head_dim
                );
            }
            let single_stream_decode = decode_cublas_single_stream_enabled();

            self.with_forward_workspace(
                d_model as usize,
                kv_dim as usize,
                hidden_dim as usize,
                |ws| -> Result<()> {
                    let wait_cross_stream =
                        |waiting: CudaStream, signal: CudaStream, op: &'static str| -> Result<()> {
                            if single_stream_decode {
                                Ok(())
                            } else {
                                self.cuda.stream_wait_for_stream(waiting, signal, op)
                            }
                        };
                    let label = format!("forward.layer.{seq_id}.graph_params");
                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    if let Some((d_weight, dtype)) = attn_norm_weight {
                        self.run_rms_norm_weighted_async(
                            d_x_f32,
                            d_weight,
                            dtype,
                            ws.d_xn,
                            1,
                            d_model as u32,
                            self.model_config.layer_norm_epsilon,
                        )?;
                    } else {
                        self.run_rms_norm_async(
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

                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Prefill,
                        CudaStream::Decode,
                        "attn_norm_to_qkv_project",
                    )?;
                    self.qkv_project_f32xf16_gguf_f32_async(
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
                    self.apply_qkv_bias_async(qkv_bias, 1, d_model as usize, kv_dim as usize, ws)?;
                    log_profiled_op(
                        &label,
                        "qkv_project",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );

                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Decode,
                        CudaStream::Prefill,
                        "qkv_project_to_rope_kv",
                    )?;
                    self.cuda.rope_f32_inplace_position_dev_layout_async(
                        ws.dq,
                        1,
                        q_heads,
                        head_dim,
                        d_position,
                        self.model_config.rope_freq_base,
                        self.model_config.rope_freq_scale,
                        self.model_config.rope_layout_code(),
                    )?;
                    log_profiled_op(
                        &label,
                        "rope_q",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );

                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Decode,
                        CudaStream::Prefill,
                        "qkv_project_to_kv_append",
                    )?;
                    self.append_kv_token_f32_rope_k_position_dev_async(
                        seq_id,
                        ws.dk as *const c_void,
                        ws.dv as *const c_void,
                        d_position,
                        self.model_config.rope_freq_base,
                        self.model_config.rope_freq_scale,
                    )?;
                    log_profiled_op(
                        &label,
                        "kv_append_rope_k",
                        profile_before.as_ref(),
                        op_start.elapsed(),
                    );

                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    self.run_attention_seq_len_dev_async(
                        ws.dq as *const c_void,
                        ws.datt,
                        seq_id,
                        d_seq_len,
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

                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Prefill,
                        CudaStream::Decode,
                        "attention_to_out_project",
                    )?;
                    self.out_proj_f32xf16_gguf_f32_async(
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

                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Decode,
                        CudaStream::Prefill,
                        "out_project_to_attn_residual",
                    )?;
                    self.cuda.residual_add_f32_async(
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

                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    if let Some((d_weight, dtype)) = ffn_norm_weight {
                        self.run_rms_norm_weighted_async(
                            ws.d_x1,
                            d_weight,
                            dtype,
                            ws.d_x1n,
                            1,
                            d_model as u32,
                            self.model_config.layer_norm_epsilon,
                        )?;
                    } else {
                        self.run_rms_norm_async(
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

                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Prefill,
                        CudaStream::Decode,
                        "ffn_norm_to_mlp_gate_up",
                    )?;
                    self.mlp_gates_f32xf16_gguf_f32_async(
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

                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Decode,
                        CudaStream::Prefill,
                        "mlp_gate_up_to_swiglu",
                    )?;
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

                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Prefill,
                        CudaStream::Decode,
                        "swiglu_to_mlp_down",
                    )?;
                    self.mlp_down_proj_f32xf16_gguf_f32_async(
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

                    let profile_before = profile::snapshot_if_enabled();
                    let op_start = std::time::Instant::now();
                    wait_cross_stream(
                        CudaStream::Decode,
                        CudaStream::Prefill,
                        "mlp_down_to_mlp_residual",
                    )?;
                    self.cuda.residual_add_f32_async(
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
                d_position,
                d_seq_len,
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
    #[cfg(feature = "cuda")]
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
            None,
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
    pub unsafe fn run_rms_norm_async(
        &self,
        d_in: *const c_void,
        d_out: *mut c_void,
        seq_len: u32,
        dim: u32,
        eps: f32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.cuda.rms_norm_f32_async(d_in, d_out, seq_len, dim, eps)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_in, d_out, seq_len, dim, eps);
            Ok(())
        }
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
    pub unsafe fn run_rms_norm_weighted_async(
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
                .rms_norm_f32_weighted_async(d_in, d_weight, d_out, seq_len, dim, eps, dtype_code)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_in, d_weight, d_out, seq_len, dim, eps, dtype_code);
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
        #[cfg(feature = "cuda")]
        kv_selection::set_attention_context(Some(layer_id), Some(position));
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
            let qkv_bias = self.qkv_bias_ptrs(&w)?;
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
                qkv_bias,
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

            Ok(())
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

    /// Runs one layer using device-resident position/sequence-length parameters.
    ///
    /// This is intended for CUDA Graph replay where token position changes
    /// between launches without rebuilding the captured graph.
    ///
    /// # Safety
    /// - `d_x_f32` and `d_out_f32` must be valid device pointers on this model's CUDA context.
    /// - `d_position` and `d_seq_len` must each point to one device-resident u32.
    pub unsafe fn forward_one_token_with_layer_graph_params(
        &self,
        d_x_f32: *const c_void,
        layer: usize,
        sequence_id: u32,
        d_position: *const u32,
        d_seq_len: *const u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        if d_position.is_null() || d_seq_len.is_null() {
            anyhow::bail!("forward_one_token_with_layer_graph_params: null device parameter");
        }
        let w = self.map_standard_layer(layer)?;
        let layer_id: u32 = layer
            .try_into()
            .map_err(|_| anyhow::anyhow!("layer index {} does not fit in u32", layer))?;
        #[cfg(feature = "cuda")]
        kv_selection::set_attention_context(Some(layer_id), None);
        #[cfg(feature = "cuda")]
        {
            let kv_slot = self.kv_physical_slot_for_layer_sequence(layer_id, sequence_id)?;
            let wq_ptr = self.tensor_device_ptr("wq", &w.wq)?;
            let wk_ptr = self.tensor_device_ptr("wk", &w.wk)?;
            let wv_ptr = self.tensor_device_ptr("wv", &w.wv)?;
            let qkv_bias = self.qkv_bias_ptrs(&w)?;
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

            self.forward_one_token_minimal_with_norms_graph_params(
                d_x_f32,
                w.d_model as i32,
                wq_ptr,
                wk_ptr,
                wv_ptr,
                qkv_bias,
                wo_ptr,
                w_gate_ptr,
                w_up_ptr,
                w_down_ptr,
                w.hidden_dim as i32,
                kv_slot,
                d_position,
                d_seq_len,
                attn_norm_ptr,
                ffn_norm_ptr,
                d_out_f32,
            )?;

            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                d_x_f32,
                sequence_id,
                d_position,
                d_seq_len,
                d_out_f32,
                layer_id,
                w,
            );
            anyhow::bail!("forward_one_token_with_layer_graph_params requires CUDA")
        }
    }
    /// Runs one token through every transformer layer.
    ///
    /// The current single-request server path defaults to `sequence_id=0`;
    /// batched callers can use `forward_one_token_all_layers_for_sequence`.
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
        unsafe { self.forward_one_token_all_layers_for_sequence(d_x_f32, 0, seq_len, d_out_f32) }
    }

    /// Runs one token through every transformer layer for one logical sequence.
    ///
    /// # Safety
    /// - `d_x_f32` and `d_out_f32` must be valid device pointers on this model's CUDA context.
    /// - Both buffers must contain/be sized for one f32 hidden vector of `embedding_length`.
    pub unsafe fn forward_one_token_all_layers_for_sequence(
        &self,
        d_x_f32: *const c_void,
        sequence_id: u32,
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
        self.kv_physical_slot_for_layer_sequence((layer_count - 1) as u32, sequence_id)?;

        #[cfg(feature = "cuda")]
        {
            if layer_count == 1 {
                self.forward_one_token_with_layer(d_x_f32, 0, sequence_id, seq_len, d_out_f32)?;
                return Ok(1);
            }
            let hidden_dim = self.model_config.feed_forward_length as usize;
            let kv_dim = (self.model_config.attention_head_count_kv as usize)
                .checked_mul(self.model_config.attention_key_length as usize)
                .context("forward workspace kv dim overflow")?;

            self.with_forward_workspace(d_model, kv_dim, hidden_dim, |ws| {
                let mut current = d_x_f32;
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
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_x_f32, seq_len, d_out_f32, d_model);
            anyhow::bail!("forward_one_token_all_layers requires CUDA device buffers")
        }
    }

    /// Runs one decode token for a batch of logical sequences.
    ///
    /// This path keeps projections and MLP row-batched while using the existing
    /// packed variable-length GQA decode attention primitive for the attention
    /// phase. It is intentionally decode-only: each item represents one query
    /// token with its own already-appended KV length.
    ///
    /// # Safety
    /// - Every pointer in `items` must be a valid device pointer on this model's CUDA context.
    /// - Input and output buffers must each hold one f32 hidden vector.
    #[cfg(feature = "cuda")]
    pub unsafe fn forward_one_token_all_layers_batched_for_sequences(
        &self,
        items: &[ForwardBatchItem],
    ) -> Result<usize> {
        if items.is_empty() {
            anyhow::bail!("batched forward requires at least one item");
        }
        let layer_count = self.model_config.block_count as usize;
        if layer_count == 0 {
            anyhow::bail!("batched forward: model has zero layers");
        }
        let (d_model, hidden_dim) = self.validate_standard_layers()?;
        let kv = self.kv_cache.as_ref().ok_or_else(|| {
            anyhow!("kv_cache not allocated; call allocate_kv_cache_for_layers first")
        })?;
        if kv.head_dim() != 64 && kv.head_dim() != 128 {
            anyhow::bail!(
                "batched decode attention currently requires head_dim=64 or 128, got {}",
                kv.head_dim()
            );
        }
        for item in items {
            if item.seq_len == 0 {
                anyhow::bail!("batched forward item has zero seq_len");
            }
            self.kv_physical_slot_for_layer_sequence((layer_count - 1) as u32, item.sequence_id)?;
        }

        let q_heads = self.model_config.attention_head_count;
        let kv_heads = kv.num_heads();
        let head_dim = kv.head_dim();
        let kv_dim = kv_heads
            .checked_mul(head_dim)
            .context("batched forward KV projection dim overflow")? as usize;
        if q_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
            anyhow::bail!(
                "batched forward: query heads {} must be a multiple of kv heads {}",
                q_heads,
                kv_heads
            );
        }
        if d_model as u32 != q_heads.saturating_mul(head_dim) {
            anyhow::bail!(
                "batched forward: d_model {} != query heads {} * head_dim {}",
                d_model,
                q_heads,
                head_dim
            );
        }

        let rows = items.len();
        let bytes_d = d_model
            .checked_mul(std::mem::size_of::<f32>())
            .context("batched forward d_model byte size overflow")?;
        let bytes_kv = kv_dim
            .checked_mul(std::mem::size_of::<f32>())
            .context("batched forward kv byte size overflow")?;
        self.with_forward_workspace_for_rows(d_model, kv_dim, hidden_dim, rows, |ws| {
            for layer in 0..layer_count {
                let w = self.map_standard_layer(layer)?;
                let layer_id: u32 = layer
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("layer index {} does not fit in u32", layer))?;
                let mut attention_states = Vec::with_capacity(items.len());
                for (idx, item) in items.iter().enumerate() {
                    let physical_slot =
                        self.kv_physical_slot_for_layer_sequence(layer_id, item.sequence_id)?;
                    attention_states.push(crate::decode_batch::DecodeRequestState::active(
                        idx as u64,
                        physical_slot,
                        item.seq_len,
                    ));
                }
                let batch_plan =
                    crate::decode_batch::DecodeBatchPlan::from_requests(&attention_states)?;
                let cuda_plan =
                    crate::decode_batch::CudaDecodeBatchPlan::new(&self.cuda, batch_plan)?;
                let wq_ptr = self.tensor_device_ptr("wq", &w.wq)?;
                let wk_ptr = self.tensor_device_ptr("wk", &w.wk)?;
                let wv_ptr = self.tensor_device_ptr("wv", &w.wv)?;
                let qkv_bias = self.qkv_bias_ptrs(&w)?;
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

                for (row, item) in items.iter().enumerate() {
                    let dst = unsafe { mut_byte_offset(ws.scratch_a, row * bytes_d) };
                    unsafe {
                        self.cuda.memcpy_d2d_async(
                            dst,
                            item.d_x_f32 as *const c_void,
                            bytes_d,
                            CudaStream::Decode,
                        )?;
                    }
                }

                let label = format!("forward.batch.layer.{layer}.rows.{rows}");
                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                if let Some((d_weight, dtype)) = attn_norm_ptr {
                    self.run_rms_norm_weighted_async(
                        ws.scratch_a,
                        d_weight,
                        dtype,
                        ws.d_xn,
                        rows as u32,
                        d_model as u32,
                        self.model_config.layer_norm_epsilon,
                    )?;
                } else {
                    self.run_rms_norm_async(
                        ws.scratch_a,
                        ws.d_xn,
                        rows as u32,
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

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Prefill,
                    CudaStream::Decode,
                    "batch_attn_norm_to_qkv_project",
                )?;
                self.qkv_project_f32xf16_gguf_f32_async(
                    ws.d_xn,
                    rows as i32,
                    d_model as i32,
                    wq_ptr,
                    d_model as i32,
                    wk_ptr,
                    kv_dim as i32,
                    wv_ptr,
                    kv_dim as i32,
                    ws.dq,
                    ws.dk,
                    ws.dv,
                )?;
                self.apply_qkv_bias_async(qkv_bias, rows, d_model, kv_dim, ws)?;
                log_profiled_op(
                    &label,
                    "qkv_project",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Decode,
                    CudaStream::Prefill,
                    "batch_qkv_project_to_rope_kv",
                )?;
                for (row, item) in items.iter().enumerate() {
                    let pos = item.seq_len.saturating_sub(1);
                    let q_row = unsafe { mut_byte_offset(ws.dq, row * bytes_d) };
                    let k_row = unsafe { const_byte_offset(ws.dk, row * bytes_kv) };
                    let v_row = unsafe { const_byte_offset(ws.dv, row * bytes_kv) };
                    let kv_slot =
                        self.kv_physical_slot_for_layer_sequence(layer_id, item.sequence_id)?;
                    self.cuda.rope_f32_inplace_layout_async(
                        q_row,
                        1,
                        q_heads,
                        head_dim,
                        pos,
                        self.model_config.rope_freq_base,
                        self.model_config.rope_freq_scale,
                        self.model_config.rope_layout_code(),
                    )?;
                    self.append_kv_token_f32_rope_k_at_async(
                        kv_slot,
                        k_row,
                        v_row,
                        pos,
                        pos,
                        self.model_config.rope_freq_base,
                        self.model_config.rope_freq_scale,
                    )?;
                }
                log_profiled_op(
                    &label,
                    "rope_q_and_kv_append",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                let kv_runtime_config = kv_compression::runtime_config();
                unsafe {
                    if kv.is_compressed() && kv_runtime_config.is_preferred_batched_runtime() {
                        cuda_plan.dispatch_fp16_k_q4_v_direct_attention_async(
                            &self.cuda,
                            kv,
                            ws.dq as *const c_void,
                            q_heads,
                            kv_runtime_config.recent_window,
                            kv_runtime_config.block_size,
                            kv_runtime_config.top_blocks,
                            ws.datt,
                        )?;
                    } else {
                        cuda_plan.dispatch_attention_async(
                            &self.cuda,
                            kv,
                            ws.dq as *const c_void,
                            q_heads,
                            ws.datt,
                        )?;
                    }
                }
                log_profiled_op(
                    &label,
                    if kv.is_compressed() {
                        "attention_batched_compressed"
                    } else {
                        "attention_batched"
                    },
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Prefill,
                    CudaStream::Decode,
                    "batch_attention_to_out_project",
                )?;
                self.out_proj_f32xf16_gguf_f32_async(
                    ws.datt as *const c_void,
                    wo_ptr,
                    ws.dy_attn,
                    rows as i32,
                    d_model as i32,
                    d_model as i32,
                )?;
                log_profiled_op(
                    &label,
                    "out_project",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Decode,
                    CudaStream::Prefill,
                    "batch_out_project_to_attn_residual",
                )?;
                self.cuda.residual_add_f32_async(
                    ws.scratch_a,
                    ws.dy_attn as *const c_void,
                    ws.d_x1,
                    rows * d_model,
                )?;
                log_profiled_op(
                    &label,
                    "attn_residual",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                if let Some((d_weight, dtype)) = ffn_norm_ptr {
                    self.run_rms_norm_weighted_async(
                        ws.d_x1,
                        d_weight,
                        dtype,
                        ws.d_x1n,
                        rows as u32,
                        d_model as u32,
                        self.model_config.layer_norm_epsilon,
                    )?;
                } else {
                    self.run_rms_norm_async(
                        ws.d_x1,
                        ws.d_x1n,
                        rows as u32,
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

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Prefill,
                    CudaStream::Decode,
                    "batch_ffn_norm_to_mlp_gate_up",
                )?;
                self.mlp_gates_f32xf16_gguf_f32_async(
                    ws.d_x1n,
                    rows as i32,
                    d_model as i32,
                    w_gate_ptr,
                    w_up_ptr,
                    hidden_dim as i32,
                    ws.dgate,
                    ws.dup,
                )?;
                log_profiled_op(
                    &label,
                    "mlp_gate_up",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Decode,
                    CudaStream::Prefill,
                    "batch_mlp_gate_up_to_swiglu",
                )?;
                self.cuda.swiglu_f32_async(
                    ws.dgate as *const c_void,
                    ws.dup as *const c_void,
                    ws.dhid,
                    rows * hidden_dim,
                )?;
                log_profiled_op(
                    &label,
                    "swiglu",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Prefill,
                    CudaStream::Decode,
                    "batch_swiglu_to_mlp_down",
                )?;
                self.mlp_down_proj_f32xf16_gguf_f32_async(
                    ws.dhid as *const c_void,
                    rows as i32,
                    hidden_dim as i32,
                    w_down_ptr,
                    d_model as i32,
                    ws.dy_mlp,
                )?;
                log_profiled_op(
                    &label,
                    "mlp_down",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Decode,
                    CudaStream::Prefill,
                    "batch_mlp_down_to_mlp_residual",
                )?;
                self.cuda.residual_add_f32_async(
                    ws.d_x1 as *const c_void,
                    ws.dy_mlp as *const c_void,
                    ws.scratch_b,
                    rows * d_model,
                )?;
                log_profiled_op(
                    &label,
                    "mlp_residual",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                for (row, item) in items.iter().enumerate() {
                    let src = unsafe { const_byte_offset(ws.scratch_b, row * bytes_d) };
                    unsafe {
                        self.cuda.memcpy_d2d_async(
                            item.d_out_f32,
                            src,
                            bytes_d,
                            CudaStream::Decode,
                        )?;
                    }
                    if layer + 1 != layer_count {
                        unsafe {
                            self.cuda.memcpy_d2d_async(
                                item.d_x_f32,
                                src,
                                bytes_d,
                                CudaStream::Decode,
                            )?;
                        }
                    }
                }
            }
            Ok(layer_count)
        })
    }

    /// Runs full-prompt prefill for a batch of logical sequences.
    ///
    /// This is the packed variable-length prefill counterpart to the batched
    /// decode path. Prompt tokens are packed into contiguous workspace rows,
    /// all projections/MLP work runs over valid rows only, and attention uses
    /// `VarlenPrefillPlan` with true per-sequence query/KV lengths.
    ///
    /// # Safety
    /// - `d_out_f32` for every item must point to one writable f32 hidden vector.
    /// - This mutates the model KV cache for each item sequence and layer.
    #[cfg(feature = "cuda")]
    pub unsafe fn forward_prefill_all_layers_varlen_for_sequences(
        &self,
        items: &[ForwardPrefillSequence<'_>],
    ) -> Result<usize> {
        let kv = self.kv_cache.as_ref().ok_or_else(|| {
            anyhow!("kv_cache not allocated; call allocate_kv_cache_for_layers first")
        })?;
        unsafe { self.forward_prefill_all_layers_varlen_for_sequences_with_kv(items, kv) }
    }

    #[cfg(feature = "cuda")]
    pub unsafe fn forward_prefill_all_layers_varlen_for_sequences_with_kv(
        &self,
        items: &[ForwardPrefillSequence<'_>],
        kv: &KVCache,
    ) -> Result<usize> {
        if items.is_empty() {
            anyhow::bail!("batched prefill requires at least one item");
        }
        let layer_count = self.model_config.block_count as usize;
        if layer_count == 0 {
            anyhow::bail!("batched prefill: model has zero layers");
        }
        let (d_model, hidden_dim) = self.validate_standard_layers()?;
        if kv.head_dim() != 64 && kv.head_dim() != 128 {
            anyhow::bail!(
                "batched prefill attention currently requires head_dim=64 or 128, got {}",
                kv.head_dim()
            );
        }

        let mut sequences = Vec::with_capacity(items.len());
        for item in items {
            if item.token_ids.is_empty() {
                anyhow::bail!("batched prefill item has empty prompt");
            }
            kv_physical_slot_for_layer_sequence_in(
                kv,
                layer_count as u32,
                (layer_count - 1) as u32,
                item.sequence_id,
            )?;
            let len: u32 = item
                .token_ids
                .len()
                .try_into()
                .context("batched prefill prompt length does not fit in u32")?;
            sequences.push(BatchSequence {
                seq_len: len,
                kv_len: len,
                query_len: len,
            });
        }
        let meta = BatchMetadata::new(sequences)?;
        let rows = meta.total_q_tokens() as usize;

        let q_heads = self.model_config.attention_head_count;
        let kv_heads = kv.num_heads();
        let head_dim = kv.head_dim();
        let kv_dim = kv_heads
            .checked_mul(head_dim)
            .context("batched prefill KV projection dim overflow")? as usize;
        if q_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
            anyhow::bail!(
                "batched prefill: query heads {} must be a multiple of kv heads {}",
                q_heads,
                kv_heads
            );
        }
        if d_model as u32 != q_heads.saturating_mul(head_dim) {
            anyhow::bail!(
                "batched prefill: d_model {} != query heads {} * head_dim {}",
                d_model,
                q_heads,
                head_dim
            );
        }

        let bytes_d = d_model
            .checked_mul(std::mem::size_of::<f32>())
            .context("batched prefill d_model byte size overflow")?;
        let bytes_kv = kv_dim
            .checked_mul(std::mem::size_of::<f32>())
            .context("batched prefill kv byte size overflow")?;
        let total_bytes_d = rows
            .checked_mul(bytes_d)
            .context("batched prefill total d_model byte size overflow")?;
        let prefill_plan = VarlenPrefillPlan::new(&self.cuda, meta.clone(), head_dim)?;

        self.with_forward_workspace_for_rows(d_model, kv_dim, hidden_dim, rows, |ws| {
            for (seq_idx, item) in items.iter().enumerate() {
                let q_offset = meta.offsets()[seq_idx].q_offset as usize;
                for (tok_idx, &tok_id) in item.token_ids.iter().enumerate() {
                    let row = q_offset + tok_idx;
                    let dst = unsafe { mut_byte_offset(ws.scratch_a, row * bytes_d) };
                    unsafe {
                        self.load_token_embedding_to_f32(tok_id as u64, dst)?;
                    }
                }
            }

            for layer in 0..layer_count {
                let w = self.map_standard_layer(layer)?;
                let layer_id: u32 = layer
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("layer index {} does not fit in u32", layer))?;
                let wq_ptr = self.tensor_device_ptr("wq", &w.wq)?;
                let wk_ptr = self.tensor_device_ptr("wk", &w.wk)?;
                let wv_ptr = self.tensor_device_ptr("wv", &w.wv)?;
                let qkv_bias = self.qkv_bias_ptrs(&w)?;
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

                let label = format!("forward.prefill.layer.{layer}.rows.{rows}");
                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                if let Some((d_weight, dtype)) = attn_norm_ptr {
                    self.run_rms_norm_weighted_async(
                        ws.scratch_a,
                        d_weight,
                        dtype,
                        ws.d_xn,
                        rows as u32,
                        d_model as u32,
                        self.model_config.layer_norm_epsilon,
                    )?;
                } else {
                    self.run_rms_norm_async(
                        ws.scratch_a,
                        ws.d_xn,
                        rows as u32,
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

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Prefill,
                    CudaStream::Decode,
                    "prefill_attn_norm_to_qkv_project",
                )?;
                self.qkv_project_f32xf16_gguf_f32_async(
                    ws.d_xn,
                    rows as i32,
                    d_model as i32,
                    wq_ptr,
                    d_model as i32,
                    wk_ptr,
                    kv_dim as i32,
                    wv_ptr,
                    kv_dim as i32,
                    ws.dq,
                    ws.dk,
                    ws.dv,
                )?;
                self.apply_qkv_bias_async(qkv_bias, rows, d_model, kv_dim, ws)?;
                log_profiled_op(
                    &label,
                    "qkv_project",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Decode,
                    CudaStream::Prefill,
                    "prefill_qkv_project_to_rope_kv",
                )?;
                for (seq_idx, item) in items.iter().enumerate() {
                    let q_offset = meta.offsets()[seq_idx].q_offset as usize;
                    let kv_slot = kv_physical_slot_for_layer_sequence_in(
                        kv,
                        layer_count as u32,
                        layer_id,
                        item.sequence_id,
                    )?;
                    for tok_idx in 0..item.token_ids.len() {
                        let pos: u32 = tok_idx
                            .try_into()
                            .context("batched prefill token position does not fit in u32")?;
                        let row = q_offset + tok_idx;
                        let q_row = unsafe { mut_byte_offset(ws.dq, row * bytes_d) };
                        let k_row_mut = unsafe { mut_byte_offset(ws.dk, row * bytes_kv) };
                        let k_row = k_row_mut as *const c_void;
                        let v_row = unsafe { const_byte_offset(ws.dv, row * bytes_kv) };
                        kv.append_token_f32_rope_k_at_layout_async(
                            &self.cuda,
                            kv_slot,
                            k_row,
                            v_row,
                            pos,
                            pos,
                            self.model_config.rope_freq_base,
                            self.model_config.rope_freq_scale,
                            self.model_config.rope_layout_code(),
                        )?;
                        self.cuda.rope_f32_inplace_layout_async(
                            q_row,
                            1,
                            q_heads,
                            head_dim,
                            pos,
                            self.model_config.rope_freq_base,
                            self.model_config.rope_freq_scale,
                            self.model_config.rope_layout_code(),
                        )?;
                        self.cuda.rope_f32_inplace_layout_async(
                            k_row_mut,
                            1,
                            kv_heads,
                            head_dim,
                            pos,
                            self.model_config.rope_freq_base,
                            self.model_config.rope_freq_scale,
                            self.model_config.rope_layout_code(),
                        )?;
                    }
                }
                log_profiled_op(
                    &label,
                    "rope_q_k_and_kv_append",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Prefill,
                    CudaStream::Decode,
                    "prefill_rope_kv_to_attention",
                )?;
                unsafe {
                    prefill_plan.dispatch_async(
                        ws.dq as *const c_void,
                        ws.dk as *const c_void,
                        ws.dv as *const c_void,
                        q_heads,
                        kv_heads,
                        ws.datt,
                    )?;
                }
                log_profiled_op(
                    &label,
                    "attention_varlen",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.out_proj_f32xf16_gguf_f32_async(
                    ws.datt as *const c_void,
                    wo_ptr,
                    ws.dy_attn,
                    rows as i32,
                    d_model as i32,
                    d_model as i32,
                )?;
                log_profiled_op(
                    &label,
                    "out_project",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Decode,
                    CudaStream::Prefill,
                    "prefill_out_project_to_attn_residual",
                )?;
                self.cuda.residual_add_f32_async(
                    ws.scratch_a,
                    ws.dy_attn as *const c_void,
                    ws.d_x1,
                    rows * d_model,
                )?;
                log_profiled_op(
                    &label,
                    "attn_residual",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                if let Some((d_weight, dtype)) = ffn_norm_ptr {
                    self.run_rms_norm_weighted_async(
                        ws.d_x1,
                        d_weight,
                        dtype,
                        ws.d_x1n,
                        rows as u32,
                        d_model as u32,
                        self.model_config.layer_norm_epsilon,
                    )?;
                } else {
                    self.run_rms_norm_async(
                        ws.d_x1,
                        ws.d_x1n,
                        rows as u32,
                        d_model as u32,
                        self.model_config.layer_norm_epsilon,
                    )?;
                }
                log_profiled_op(&label, "ffn_norm", profile_before.as_ref(), op_start.elapsed());

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Prefill,
                    CudaStream::Decode,
                    "prefill_ffn_norm_to_mlp_gate_up",
                )?;
                self.mlp_gates_f32xf16_gguf_f32_async(
                    ws.d_x1n,
                    rows as i32,
                    d_model as i32,
                    w_gate_ptr,
                    w_up_ptr,
                    hidden_dim as i32,
                    ws.dgate,
                    ws.dup,
                )?;
                log_profiled_op(
                    &label,
                    "mlp_gate_up",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Decode,
                    CudaStream::Prefill,
                    "prefill_mlp_gate_up_to_swiglu",
                )?;
                self.cuda.swiglu_f32_async(
                    ws.dgate as *const c_void,
                    ws.dup as *const c_void,
                    ws.dhid,
                    rows * hidden_dim,
                )?;
                log_profiled_op(&label, "swiglu", profile_before.as_ref(), op_start.elapsed());

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Prefill,
                    CudaStream::Decode,
                    "prefill_swiglu_to_mlp_down",
                )?;
                self.mlp_down_proj_f32xf16_gguf_f32_async(
                    ws.dhid as *const c_void,
                    rows as i32,
                    hidden_dim as i32,
                    w_down_ptr,
                    d_model as i32,
                    ws.dy_mlp,
                )?;
                log_profiled_op(
                    &label,
                    "mlp_down",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                let profile_before = profile::snapshot_if_enabled();
                let op_start = std::time::Instant::now();
                self.cuda.stream_wait_for_stream(
                    CudaStream::Decode,
                    CudaStream::Prefill,
                    "prefill_mlp_down_to_mlp_residual",
                )?;
                self.cuda.residual_add_f32_async(
                    ws.d_x1 as *const c_void,
                    ws.dy_mlp as *const c_void,
                    ws.scratch_b,
                    rows * d_model,
                )?;
                log_profiled_op(
                    &label,
                    "mlp_residual",
                    profile_before.as_ref(),
                    op_start.elapsed(),
                );

                if layer + 1 != layer_count {
                    unsafe {
                        self.cuda.memcpy_d2d_async(
                            ws.scratch_a,
                            ws.scratch_b as *const c_void,
                            total_bytes_d,
                            CudaStream::Decode,
                        )?;
                    }
                }
                if std::env::var("M40LLM_PREFILL_SYNC_EACH_LAYER")
                    .ok()
                    .as_deref()
                    == Some("1")
                {
                    self.cuda.synchronize_stream(CudaStream::Decode)?;
                    self.cuda.synchronize_stream(CudaStream::Prefill)?;
                }
            }

            for (seq_idx, item) in items.iter().enumerate() {
                let seq = meta.sequences()[seq_idx];
                let final_row =
                    meta.offsets()[seq_idx].q_offset as usize + seq.query_len as usize - 1;
                let src = unsafe { const_byte_offset(ws.scratch_b, final_row * bytes_d) };
                unsafe {
                    self.cuda.memcpy_d2d_async(
                        item.d_out_f32,
                        src,
                        bytes_d,
                        CudaStream::Decode,
                    )?;
                }
            }

            if std::env::var("M40LLM_SERVER_BATCH_PREFILL_LOG")
                .ok()
                .as_deref()
                == Some("1")
            {
                let max_len = meta
                    .sequences()
                    .iter()
                    .map(|seq| seq.query_len)
                    .max()
                    .unwrap_or(0);
                let padded_tokens = max_len as u64 * items.len() as u64;
                let valid_tokens = meta.total_q_tokens() as u64;
                eprintln!(
                    "[server] packed prefill batch_size={} valid_tokens={} padded_tokens={} avoided_tokens={}",
                    items.len(),
                    valid_tokens,
                    padded_tokens,
                    padded_tokens.saturating_sub(valid_tokens)
                );
            }

            // The forward workspace is mutex-protected only while this closure
            // owns it. Since the packed prefill path enqueues asynchronous work
            // on both streams, drain the streams before returning so a following
            // single-row decode cannot resize/free the workspace under queued
            // kernels.
            self.cuda
                .synchronize_stream(CudaStream::Decode)
                .context("batched prefill decode stream synchronization failed")?;
            self.cuda
                .synchronize_stream(CudaStream::Prefill)
                .context("batched prefill prefill stream synchronization failed")?;

            Ok(layer_count)
        })
    }

    /// Runs one token through every transformer layer using device-resident
    /// position and sequence-length parameters for graph replay.
    ///
    /// # Safety
    /// - `d_x_f32` and `d_out_f32` must be valid device pointers on this model's CUDA context.
    /// - `d_position` and `d_seq_len` must each point to one device-resident u32.
    /// - Both hidden buffers must be sized for one f32 hidden vector of `embedding_length`.
    pub unsafe fn forward_one_token_all_layers_for_sequence_graph_params(
        &self,
        d_x_f32: *const c_void,
        sequence_id: u32,
        d_position: *const u32,
        d_seq_len: *const u32,
        d_out_f32: *mut c_void,
    ) -> Result<usize> {
        if d_position.is_null() || d_seq_len.is_null() {
            anyhow::bail!(
                "forward_one_token_all_layers_for_sequence_graph_params: null device parameter"
            );
        }
        let layer_count = self.model_config.block_count as usize;
        if layer_count == 0 {
            anyhow::bail!("forward_one_token_all_layers_graph_params: model has zero layers");
        }
        let (d_model, _) = self.validate_standard_layers()?;
        let kv = self.kv_cache.as_ref().ok_or_else(|| {
            anyhow!("kv_cache not allocated; call allocate_kv_cache_for_layers first")
        })?;
        if kv.max_batch_size() < self.model_config.block_count {
            anyhow::bail!(
                "kv_cache has {} slots, but full-layer graph forward needs {} layer slots",
                kv.max_batch_size(),
                self.model_config.block_count
            );
        }
        self.kv_physical_slot_for_layer_sequence((layer_count - 1) as u32, sequence_id)?;

        #[cfg(feature = "cuda")]
        {
            if layer_count == 1 {
                self.forward_one_token_with_layer_graph_params(
                    d_x_f32,
                    0,
                    sequence_id,
                    d_position,
                    d_seq_len,
                    d_out_f32,
                )?;
                return Ok(1);
            }
            let hidden_dim = self.model_config.feed_forward_length as usize;
            let kv_dim = (self.model_config.attention_head_count_kv as usize)
                .checked_mul(self.model_config.attention_key_length as usize)
                .context("forward graph workspace kv dim overflow")?;

            self.with_forward_workspace(d_model, kv_dim, hidden_dim, |ws| {
                let mut current = d_x_f32;
                for layer in 0..layer_count {
                    let next = if layer + 1 == layer_count {
                        d_out_f32
                    } else if layer % 2 == 0 {
                        ws.scratch_a
                    } else {
                        ws.scratch_b
                    };
                    self.forward_one_token_with_layer_graph_params(
                        current,
                        layer,
                        sequence_id,
                        d_position,
                        d_seq_len,
                        next,
                    )?;
                    current = next as *const c_void;
                }
                Ok(layer_count)
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                d_x_f32,
                sequence_id,
                d_position,
                d_seq_len,
                d_out_f32,
                d_model,
            );
            anyhow::bail!(
                "forward_one_token_all_layers_for_sequence_graph_params requires CUDA device buffers"
            )
        }
    }
}
