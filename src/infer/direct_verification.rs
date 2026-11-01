//! Fixed research schedule. All projections except the gate remain one-row.
use super::{LoadedModel, QkvBiasPtrs};
use crate::{
    gguf::GgmlDType,
    infer::{
        workspace::ForwardWorkspace, Q8AttentionProjectionBackend, Q8FfnKernel, Q8FfnOwnership,
    },
};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

const D: usize = 2048;
const H: usize = 11008;
const KV: usize = 256;

/// Held by the exclusive verification session, never the model's shared scratch.
pub(crate) struct DirectVerificationWorkspace {
    storage: ForwardWorkspace,
    bytes: usize,
}

impl DirectVerificationWorkspace {
    pub(crate) fn new(model: &LoadedModel) -> Result<Self> {
        let cache = model
            .kv_cache
            .as_ref()
            .context("direct verification requires dense KV")?;
        ensure!(
            !cache.is_compressed()
                && cache.has_exclusive_storage()
                && cache.num_heads() == 2
                && cache.head_dim() == 128
                && cache.max_batch_size() == 36,
            "direct verification requires exclusive 36-slot dense KV with two head128 KV heads"
        );
        ensure!(model.model_config.embedding_length==D as u32
            && model.model_config.feed_forward_length==H as u32
            && model.model_config.attention_head_count==16
            && model.model_config.attention_head_count_kv==2
            && model.model_config.attention_key_length==128
            && model.model_config.block_count==36
            && model.q8_ffn.kernel()==Q8FfnKernel::JointSharedScales
            && model.q8_ffn.ownership()==Q8FfnOwnership::JointOnly
            && model.q8_attention_projection.backend()==Q8AttentionProjectionBackend::PackedWarp256
            && super::q8_decode_kernel_decode_stream_enabled(),
            "direct verification requires qualified Qwen D2048/H11008/head128, joint-only compound and packed-warp decode projections");
        ensure!(
            model.ffn_capture.is_none() && model.layer_influence.is_none(),
            "direct verification cannot interleave model observers"
        );
        let stats = model.q8_execution_stats();
        ensure!(
            stats.has_prepared_vocabulary
                && stats.single_row_down_split_launch
                    == Some(crate::infer::execution_weights::Q8SeparateLaunch::new(
                        4, 256
                    )?),
            "direct verification requires original one-row prepared down/vocabulary"
        );
        for layer in 0..36 {
            let w = model.map_standard_layer(layer)?;
            ensure!(
                [&w.wq, &w.wk, &w.wv, &w.wo, &w.w_gate, &w.w_up, &w.w_down]
                    .iter()
                    .all(|w| w.dtype == GgmlDType::Q8_0),
                "direct verification requires all Q8_0 layer projections"
            );
            model.q8_execution.compound_pair(
                [w.w_gate.dptr.cast_const(), w.w_up.dptr.cast_const()],
                H as i32,
                D as i32,
            )?;
            ensure!(
                model
                    .q8_execution
                    .get_separate_weight(w.w_down.dptr)
                    .is_some(),
                "missing direct down owner"
            );
        }
        Ok(Self {
            storage: ForwardWorkspace::new_with_rows(&model.cuda, D, KV, H, 4, model.q8_ffn, 0)?,
            bytes: ForwardWorkspace::estimated_bytes(D, KV, H, 4, model.q8_ffn, 0)?,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

unsafe fn row(ptr: *mut c_void, index: usize, columns: usize) -> *mut c_void {
    ptr.cast::<f32>().add(index * columns).cast()
}

impl LoadedModel {
    /// Fixed layer-major research execution, called only by an exclusive session.
    /// No generic multi-row projection or SGEMM call occurs in this schedule.
    ///
    /// # Safety
    /// The session owns this model/cache and workspace until decode completes.
    /// Output holds four hidden rows on this context and does not alias scratch.
    pub(crate) unsafe fn forward_direct_verification_rows(
        &self,
        tokens: &[u32],
        start: u32,
        output: *mut c_void,
        workspace: &DirectVerificationWorkspace,
    ) -> Result<()> {
        ensure!(
            tokens.len() == 4 && !output.is_null(),
            "direct verification requires exactly four rows"
        );
        let kv = self.kv_cache.as_ref().context("missing direct dense KV")?;
        ensure!(
            !kv.is_compressed()
                && kv.has_exclusive_storage()
                && start.checked_add(4).is_some_and(|n| n <= kv.max_seq_len()),
            "invalid direct dense KV span"
        );
        let ws = workspace.storage.ptrs();
        for (index, &token) in tokens.iter().enumerate() {
            self.load_token_embedding_to_f32(token as u64, row(ws.scratch_a, index, D))?;
        }
        let mut current = ws.scratch_a;
        for layer in 0..36 {
            let w = self.map_standard_layer(layer)?;
            let slot = self.kv_physical_slot_for_layer_sequence(layer as u32, 0)?;
            let bias: Option<QkvBiasPtrs> = self.qkv_bias_ptrs(&w)?;
            let attn_norm = w
                .attn_norm
                .as_ref()
                .map(|v| self.tensor_device_ptr("attn_norm", v).map(|p| (p, v.dtype)))
                .transpose()?;
            let ffn_norm = w
                .ffn_norm
                .as_ref()
                .map(|v| self.tensor_device_ptr("ffn_norm", v).map(|p| (p, v.dtype)))
                .transpose()?;
            for index in 0..4 {
                let x = row(current, index, D);
                let xn = row(ws.d_xn, index, D);
                if let Some((weight, dtype)) = attn_norm {
                    self.run_rms_norm_weighted_async(
                        x,
                        weight,
                        dtype,
                        xn,
                        1,
                        D as u32,
                        self.model_config.layer_norm_epsilon,
                    )?;
                } else {
                    self.run_rms_norm_async(
                        x,
                        xn,
                        1,
                        D as u32,
                        self.model_config.layer_norm_epsilon,
                    )?;
                }
                let q = row(ws.dq, index, D);
                let k = row(ws.dk, index, KV);
                let v = row(ws.dv, index, KV);
                ensure!(
                    self.try_qkv_project_f32xf16_gguf_f32_decode_async(
                        xn,
                        1,
                        D as i32,
                        w.wq.dptr,
                        D as i32,
                        w.wk.dptr,
                        KV as i32,
                        w.wv.dptr,
                        KV as i32,
                        bias.map(|(q, _, _)| q),
                        bias.map(|(_, k, _)| k),
                        bias.map(|(_, _, v)| v),
                        q,
                        k,
                        v
                    )?,
                    "qualified direct QKV unavailable; no SGEMM fallback"
                );
                let position = start + index as u32;
                self.cuda.rope_f32_inplace_layout_async(
                    q,
                    1,
                    16,
                    128,
                    position,
                    self.model_config.rope_freq_base,
                    self.model_config.rope_freq_scale,
                    self.model_config.rope_layout_code(),
                )?;
                self.append_kv_token_f32_rope_k_at_async(
                    slot,
                    k,
                    v,
                    position,
                    position,
                    self.model_config.rope_freq_base,
                    self.model_config.rope_freq_scale,
                )?;
                // Append and consume one row before appending the next. Neither
                // cache metadata nor attention can expose future candidate keys.
                let att = row(ws.datt, index, D);
                self.run_attention_async(q, att, slot, position + 1, D as u32, 16, 128)?;
                let projected = row(ws.dy_attn, index, D);
                self.out_proj_f32xf16_gguf_f32_async(
                    att, w.wo.dptr, projected, 1, D as i32, D as i32,
                )?;
                let residual = row(ws.d_x1, index, D);
                self.cuda
                    .residual_add_f32_async(x, projected, residual, D)?;
                let normalized = row(ws.d_x1n, index, D);
                if let Some((weight, dtype)) = ffn_norm {
                    self.run_rms_norm_weighted_async(
                        residual,
                        weight,
                        dtype,
                        normalized,
                        1,
                        D as u32,
                        self.model_config.layer_norm_epsilon,
                    )?;
                } else {
                    self.run_rms_norm_async(
                        residual,
                        normalized,
                        1,
                        D as u32,
                        self.model_config.layer_norm_epsilon,
                    )?;
                }
            }
            let pair = self.q8_execution.compound_pair(
                [w.w_gate.dptr.cast_const(), w.w_up.dptr.cast_const()],
                H as i32,
                D as i32,
            )?;
            self.cuda.ffn_q8_joint_four_rows_f32_async(
                ws.d_x1n,
                pair.planes()[0].as_ptr(),
                pair.planes()[1].as_ptr(),
                ws.dhid,
                4,
                H as i32,
                D as i32,
                None,
            )?;
            let next = if layer == 35 {
                output
            } else if layer % 2 == 0 {
                ws.scratch_b
            } else {
                ws.scratch_a
            };
            for index in 0..4 {
                let down = row(ws.dy_mlp, index, D);
                self.mlp_down_proj_f32xf16_gguf_f32_async(
                    row(ws.dhid, index, H),
                    1,
                    H as i32,
                    w.w_down.dptr,
                    D as i32,
                    down,
                )?;
                self.cuda.residual_add_f32_async(
                    row(ws.d_x1, index, D),
                    down,
                    row(next, index, D),
                    D,
                )?;
            }
            current = next;
        }
        // Every producer/consumer above is an existing decode-stream enqueue.
        // Ordinary per-row logits/greedy completion drains it in the session.
        Ok(())
    }
}
