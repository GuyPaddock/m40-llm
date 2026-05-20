#[cfg(feature = "cuda")]
use crate::cuda::KVCache;
#[cfg(feature = "cuda")]
use crate::cuda::{
    CudaContext, CudaEvent, CudaGraphExec, CudaStream, DeviceBuffer, ExactBlockStagingWorkspace,
};
#[cfg(feature = "cuda")]
use crate::infer::LoadedModel;
#[cfg(feature = "cuda")]
use crate::timing;
#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use std::ffi::c_void;

#[cfg(feature = "cuda")]
fn decode_session_log_enabled() -> bool {
    std::env::var("M40LLM_DECODE_SESSION_LOG").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
fn prefill_sync_diag_enabled() -> bool {
    std::env::var("M40LLM_PREFILL_SYNC_DIAG").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
fn forward_sync_diag_enabled() -> bool {
    std::env::var("M40LLM_FORWARD_SYNC_DIAG").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
struct PrefillSyncDiag {
    label: String,
    wall_start: std::time::Instant,
    decode_start: CudaEvent,
    prefill_start: CudaEvent,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct PrefillSyncDiagTimings {
    pub wall_ms: u128,
    pub decode_gpu_ms: f32,
    pub prefill_gpu_ms: f32,
}

#[cfg(feature = "cuda")]
impl PrefillSyncDiag {
    fn start(cuda: &CudaContext, label: impl Into<String>) -> Result<Option<Self>> {
        if !prefill_sync_diag_enabled() {
            return Ok(None);
        }
        let decode_start = cuda.create_event()?;
        let prefill_start = cuda.create_event()?;
        decode_start.record(CudaStream::Decode)?;
        prefill_start.record(CudaStream::Prefill)?;
        Ok(Some(Self {
            label: label.into(),
            wall_start: std::time::Instant::now(),
            decode_start,
            prefill_start,
        }))
    }

    fn finish(self, cuda: &CudaContext) -> Result<PrefillSyncDiagTimings> {
        let decode_stop = cuda.create_event()?;
        let prefill_stop = cuda.create_event()?;
        decode_stop.record(CudaStream::Decode)?;
        prefill_stop.record(CudaStream::Prefill)?;
        let decode_gpu_ms = self
            .decode_start
            .elapsed_sync(&decode_stop, "packed_prefill_sync_diag_decode")?;
        let prefill_gpu_ms = self
            .prefill_start
            .elapsed_sync(&prefill_stop, "packed_prefill_sync_diag_prefill")?;
        let wall = self.wall_start.elapsed();
        timing::log(format_args!("{}.sync_diag.wall", self.label), wall);
        if timing::enabled() {
            eprintln!(
                "[timing] {}.sync_diag.decode_gpu {:.3} ms",
                self.label, decode_gpu_ms
            );
            eprintln!(
                "[timing] {}.sync_diag.prefill_gpu {:.3} ms",
                self.label, prefill_gpu_ms
            );
        }
        Ok(PrefillSyncDiagTimings {
            wall_ms: wall.as_millis(),
            decode_gpu_ms,
            prefill_gpu_ms,
        })
    }
}

#[cfg(feature = "cuda")]
pub struct DecodeSession {
    model: *const LoadedModel,
    sequence_id: u32,
    processed_len: usize,
    can_forward: bool,
    d_x: DeviceBuffer,
    d_out: Option<DeviceBuffer>,
    d_logits: Option<DeviceBuffer>,
    d_norm_hidden: Option<DeviceBuffer>,
    d_graph_position: Option<DeviceBuffer>,
    d_graph_seq_len: Option<DeviceBuffer>,
    exact_block_staging: Option<ExactBlockStagingWorkspace>,
    decode_graph: Option<CudaGraphExec>,
    graph_disabled: bool,
    last_forward_used_graph: bool,
    log_prefix: &'static str,
    step: usize,
    logged_full_forward: bool,
    last_prefill_sync_diag: Option<PrefillSyncDiagTimings>,
    last_forward_sync_diag: Option<PrefillSyncDiagTimings>,
}

#[cfg(feature = "cuda")]
unsafe impl Send for DecodeSession {}

#[cfg(feature = "cuda")]
impl DecodeSession {
    pub fn new(
        model: &LoadedModel,
        d_model: usize,
        can_forward: bool,
        log_prefix: &'static str,
        d_x_tag: &'static str,
        d_out_tag: &'static str,
    ) -> Result<Self> {
        Self::new_for_sequence(
            model,
            0,
            d_model,
            can_forward,
            log_prefix,
            d_x_tag,
            d_out_tag,
        )
    }

    pub fn new_for_sequence(
        model: &LoadedModel,
        sequence_id: u32,
        d_model: usize,
        can_forward: bool,
        log_prefix: &'static str,
        d_x_tag: &'static str,
        d_out_tag: &'static str,
    ) -> Result<Self> {
        if can_forward && !model.kv_cache_can_address_layer_sequence(0, sequence_id) {
            anyhow::bail!(
                "decode session sequence_id {sequence_id} is not addressable by KV cache"
            );
        }
        let bytes = d_model
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("decode session d_model byte size overflow"))?;
        let d_x = DeviceBuffer::new_tagged(&model.cuda, bytes, d_x_tag)?;
        let d_out = if can_forward {
            Some(DeviceBuffer::new_tagged(&model.cuda, bytes, d_out_tag)?)
        } else {
            None
        };
        let (d_logits, d_norm_hidden) = match model.map_lm_head() {
            Ok((_name, _lm, lm_d_model, vocab, _tied)) => {
                if lm_d_model != d_model {
                    anyhow::bail!(
                        "decode session d_model {} != lm_head d_model {}",
                        d_model,
                        lm_d_model
                    );
                }
                let logits_bytes = vocab
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| anyhow::anyhow!("decode session logits byte size overflow"))?;
                let d_logits = DeviceBuffer::new_tagged(
                    &model.cuda,
                    logits_bytes,
                    "decode_session:logits_f32",
                )?;
                let d_norm_hidden = if model.map_output_norm()?.is_some() {
                    Some(DeviceBuffer::new_tagged(
                        &model.cuda,
                        bytes,
                        "decode_session:logits_norm_hidden_f32",
                    )?)
                } else {
                    None
                };
                (Some(d_logits), d_norm_hidden)
            }
            Err(_) => (None, None),
        };
        let graph_params = if can_forward && decode_graph_enabled() {
            Some((
                DeviceBuffer::new_tagged(
                    &model.cuda,
                    std::mem::size_of::<u32>(),
                    "decode_graph:position_u32",
                )?,
                DeviceBuffer::new_tagged(
                    &model.cuda,
                    std::mem::size_of::<u32>(),
                    "decode_graph:seq_len_u32",
                )?,
            ))
        } else {
            None
        };
        let (d_graph_position, d_graph_seq_len) = match graph_params {
            Some((position, seq_len)) => (Some(position), Some(seq_len)),
            None => (None, None),
        };
        let exact_block_staging = if can_forward && exact_block_staging_enabled() {
            let kv_cfg = crate::kv_compression::runtime_config();
            if kv_cfg.mode == crate::kv_compression::KvCompressMode::BlockSelectExact {
                let requested_tokens = kv_cfg
                    .recent_window
                    .saturating_add(kv_cfg.top_blocks.saturating_mul(kv_cfg.block_size))
                    .max(1);
                let capacity_tokens = requested_tokens.min(model.model_config.context_length);
                Some(ExactBlockStagingWorkspace::new(
                    &model.cuda,
                    model.model_config.attention_head_count,
                    model.model_config.attention_key_length,
                    capacity_tokens,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        Ok(Self {
            model: model as *const LoadedModel,
            sequence_id,
            processed_len: 0,
            can_forward,
            d_x,
            d_out,
            d_logits,
            d_norm_hidden,
            d_graph_position,
            d_graph_seq_len,
            exact_block_staging,
            decode_graph: None,
            graph_disabled: false,
            last_forward_used_graph: false,
            log_prefix,
            step: 0,
            logged_full_forward: false,
            last_prefill_sync_diag: None,
            last_forward_sync_diag: None,
        })
    }

    fn model(&self) -> &LoadedModel {
        // DecodeSession does not own the model. CLI/server callers keep the
        // model alive for the whole session; the raw pointer avoids infecting
        // streaming response futures with non-'static borrows.
        unsafe { &*self.model }
    }

    pub fn processed_len(&self) -> usize {
        self.processed_len
    }

    pub fn sequence_id(&self) -> u32 {
        self.sequence_id
    }

    pub fn can_forward(&self) -> bool {
        self.can_forward
    }

    pub fn forward_batch_item(&self, seq_len: u32) -> Result<crate::infer::ForwardBatchItem> {
        let d_out = self
            .d_out
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("d_out is not allocated for this decode session"))?;
        Ok(crate::infer::ForwardBatchItem {
            d_x_f32: self.d_x.as_mut_ptr(),
            sequence_id: self.sequence_id,
            seq_len,
            d_out_f32: d_out.as_mut_ptr(),
        })
    }

    pub fn d_out_ptr(&self) -> Result<*mut std::ffi::c_void> {
        self.d_out
            .as_ref()
            .map(DeviceBuffer::as_mut_ptr)
            .ok_or_else(|| anyhow::anyhow!("d_out is not allocated for this decode session"))
    }

    pub fn exact_block_staging_workspace_bytes(&self) -> Option<usize> {
        self.exact_block_staging
            .as_ref()
            .map(ExactBlockStagingWorkspace::bytes)
    }

    pub fn exact_block_staging_capacity_tokens(&self) -> Option<u32> {
        self.exact_block_staging
            .as_ref()
            .map(ExactBlockStagingWorkspace::capacity_tokens)
    }

    pub fn exact_block_staging_allocations(&self) -> usize {
        usize::from(self.exact_block_staging.is_some())
    }

    pub fn exact_block_staging_reused(&self) -> bool {
        self.exact_block_staging.is_some()
    }

    fn with_exact_block_staging<R>(&self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        crate::infer::with_exact_block_staging(
            self.exact_block_staging
                .as_ref()
                .map(ExactBlockStagingWorkspace::ptrs),
            f,
        )
    }

    pub unsafe fn load_next_unprocessed_token(&mut self, ids: &[u32]) -> Result<Option<usize>> {
        if self.processed_len > ids.len() {
            self.processed_len = 0;
            if self.model().kv_cache.is_some() {
                self.model().reset_kv_cache()?;
            }
        }
        if self.processed_len >= ids.len() {
            return Ok(None);
        }
        let token_idx = self.processed_len;
        let tok_id = ids[token_idx] as u64;
        let embed_start = std::time::Instant::now();
        (*self.model).load_token_embedding_to_f32(tok_id, self.d_x.as_mut_ptr())?;
        timing::timing_log!(
            embed_start.elapsed(),
            "{}.token.{token_idx}.embedding_load",
            self.log_prefix
        );
        Ok(Some(token_idx))
    }

    pub unsafe fn logits_after_batched_forward(&mut self, token_idx: usize) -> Result<Vec<f32>> {
        let d_out = self
            .d_out
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("d_out is not allocated for this decode session"))?
            .as_mut_ptr();
        let logits_start = std::time::Instant::now();
        let logits = match self.d_logits.as_ref() {
            Some(d_logits) => (*self.model).logits_from_hidden_into(
                d_out as *const _,
                d_logits.as_mut_ptr(),
                self.d_norm_hidden.as_ref().map(DeviceBuffer::as_mut_ptr),
            ),
            None => (*self.model).logits_from_hidden(d_out as *const _),
        };
        timing::timing_log!(
            logits_start.elapsed(),
            "{}.token.{token_idx}.logits",
            self.log_prefix
        );
        logits
    }

    pub fn mark_processed_through(&mut self, processed_len: usize) {
        self.processed_len = processed_len;
    }

    pub fn logits_for_packed_prefix_then_ids(
        &mut self,
        ids: &[u32],
        mut on_token_logits: impl FnMut(&[f32]),
    ) -> Result<Vec<f32>> {
        self.last_prefill_sync_diag = None;
        if !self.can_forward {
            anyhow::bail!("packed prefill requires full-layer forward");
        }
        if ids.is_empty() {
            anyhow::bail!("packed prefill requires non-empty ids");
        }
        if self.processed_len != 0 {
            anyhow::bail!(
                "packed prefill can only run before decode starts; processed_len={}",
                self.processed_len
            );
        }
        if ids.len() == 1 {
            return self.logits_for_ids(ids, on_token_logits);
        }
        let d_out = self
            .d_out
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("d_out is not allocated for this decode session"))?
            .as_mut_ptr();
        let prefix_len = ids.len() - 1;
        let prefill_start = std::time::Instant::now();
        let sync_diag = PrefillSyncDiag::start(
            &self.model().cuda,
            format!("{}.packed_prefill.ids_len_{}", self.log_prefix, ids.len()),
        )?;
        let layers = unsafe {
            (*self.model).forward_prefill_all_layers_varlen_for_sequences(&[
                crate::infer::ForwardPrefillSequence {
                    token_ids: &ids[..prefix_len],
                    sequence_id: self.sequence_id,
                    d_out_f32: d_out,
                },
            ])
        }?;
        if let Some(sync_diag) = sync_diag {
            self.last_prefill_sync_diag = Some(sync_diag.finish(&self.model().cuda)?);
        }
        timing::timing_log!(
            prefill_start.elapsed(),
            "{}.packed_prefill.ids_len_{}",
            self.log_prefix,
            ids.len()
        );
        if !self.logged_full_forward {
            eprintln!(
                "[{}] packed prefix prefill enabled layers={layers}",
                self.log_prefix
            );
            self.logged_full_forward = true;
        }
        self.processed_len = prefix_len;
        self.logits_for_ids(ids, |logits| on_token_logits(logits))
    }

    pub fn logits_for_compressed_chunked_prefill_ids(
        &mut self,
        ids: &[u32],
        chunk_size: usize,
        mut on_token_logits: impl FnMut(&[f32]),
    ) -> Result<Vec<f32>> {
        if !self.can_forward {
            anyhow::bail!("compressed chunked prefill requires full-layer forward");
        }
        if ids.is_empty() {
            anyhow::bail!("compressed chunked prefill requires non-empty ids");
        }
        if chunk_size == 0 {
            anyhow::bail!("compressed chunked prefill chunk_size must be > 0");
        }
        if self.processed_len != 0 {
            anyhow::bail!(
                "compressed chunked prefill can only run before decode starts; processed_len={}",
                self.processed_len
            );
        }
        if ids.len() == 1 {
            return self.logits_for_ids(ids, on_token_logits);
        }

        let prefix_len = ids.len() - 1;
        let prefill_start = std::time::Instant::now();
        for chunk_start in (0..prefix_len).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(prefix_len);
            let chunk_timer = std::time::Instant::now();
            for (token_idx, &tok_id_u32) in ids
                .iter()
                .enumerate()
                .skip(chunk_start)
                .take(chunk_end - chunk_start)
            {
                unsafe {
                    self.forward_prefill_token_without_logits(tok_id_u32 as u64, token_idx)?;
                }
                self.processed_len = token_idx + 1;
            }
            timing::timing_log!(
                chunk_timer.elapsed(),
                "{}.compressed_chunked_prefill.chunk_{}_{}",
                self.log_prefix,
                chunk_start,
                chunk_end
            );
        }
        timing::timing_log!(
            prefill_start.elapsed(),
            "{}.compressed_chunked_prefill.ids_len_{}_chunk_{}",
            self.log_prefix,
            ids.len(),
            chunk_size
        );
        if !self.logged_full_forward {
            eprintln!(
                "[{}] compressed chunked prefill enabled chunk_size={chunk_size}",
                self.log_prefix
            );
            self.logged_full_forward = true;
        }
        self.logits_for_ids(ids, |logits| on_token_logits(logits))
    }

    pub fn logits_for_packed_then_compress_prefill_ids(
        &mut self,
        ids: &[u32],
        mut on_token_logits: impl FnMut(&[f32]),
    ) -> Result<(Vec<f32>, usize)> {
        self.last_prefill_sync_diag = None;
        if !self.can_forward {
            anyhow::bail!("packed-then-compress prefill requires full-layer forward");
        }
        if ids.is_empty() {
            anyhow::bail!("packed-then-compress prefill requires non-empty ids");
        }
        if self.processed_len != 0 {
            anyhow::bail!(
                "packed-then-compress prefill can only run before decode starts; processed_len={}",
                self.processed_len
            );
        }
        if ids.len() == 1 {
            let logits = self.logits_for_ids(ids, on_token_logits)?;
            return Ok((logits, 0));
        }
        let active_kv = self
            .model()
            .kv_cache
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("kv_cache is not allocated"))?;
        if !active_kv.is_compressed() {
            anyhow::bail!("packed-then-compress requires active compressed KV cache");
        }
        let prefix_len = ids.len() - 1;
        let temp_dense_kv = KVCache::new_with_context(
            &self.model().cuda,
            prefix_len as u32,
            active_kv.max_batch_size(),
            active_kv.num_heads(),
            active_kv.head_dim(),
        )?;
        let temp_dense_bytes = temp_dense_kv.actual_bytes();
        let d_out = self
            .d_out
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("d_out is not allocated for this decode session"))?
            .as_mut_ptr();
        let prefill_start = std::time::Instant::now();
        let sync_diag = PrefillSyncDiag::start(
            &self.model().cuda,
            format!(
                "{}.packed_then_compress_prefill.forward.ids_len_{}",
                self.log_prefix,
                ids.len()
            ),
        )?;
        let layers = unsafe {
            (*self.model).forward_prefill_all_layers_varlen_for_sequences_with_kv(
                &[crate::infer::ForwardPrefillSequence {
                    token_ids: &ids[..prefix_len],
                    sequence_id: self.sequence_id,
                    d_out_f32: d_out,
                }],
                &temp_dense_kv,
            )
        }?;
        let sync_diag_timings = sync_diag
            .map(|sync_diag| sync_diag.finish(&self.model().cuda))
            .transpose()?;
        let compress_start = std::time::Instant::now();
        active_kv.build_compressed_from_dense(
            &self.model().cuda,
            &temp_dense_kv,
            prefix_len as u32,
        )?;
        self.last_prefill_sync_diag = sync_diag_timings;
        timing::timing_log!(
            compress_start.elapsed(),
            "{}.packed_then_compress_prefill.compress.ids_len_{}",
            self.log_prefix,
            ids.len()
        );
        timing::timing_log!(
            prefill_start.elapsed(),
            "{}.packed_then_compress_prefill.ids_len_{}",
            self.log_prefix,
            ids.len()
        );
        if !self.logged_full_forward {
            eprintln!(
                "[{}] packed-then-compress prefill enabled layers={layers} temp_dense_kv_bytes={temp_dense_bytes}",
                self.log_prefix
            );
            self.logged_full_forward = true;
        }
        self.processed_len = prefix_len;
        let logits = self.logits_for_ids(ids, |logits| on_token_logits(logits))?;
        Ok((logits, temp_dense_bytes))
    }

    pub fn prefill_sync_diag_timings(&self) -> Option<PrefillSyncDiagTimings> {
        self.last_prefill_sync_diag
    }

    pub fn forward_sync_diag_timings(&self) -> Option<PrefillSyncDiagTimings> {
        self.last_forward_sync_diag
    }

    pub fn logits_for_ids(
        &mut self,
        ids: &[u32],
        mut on_token_logits: impl FnMut(&[f32]),
    ) -> Result<Vec<f32>> {
        let logits_fn_start = std::time::Instant::now();
        self.step += 1;
        if decode_session_log_enabled() {
            eprintln!(
                "[{}] logits_fn called with {} tokens",
                self.log_prefix,
                ids.len()
            );
            eprintln!(
                "[mem] (token) step={} pid={} device_id={} TOTAL_DEVICE_BYTES={}",
                self.step,
                std::process::id(),
                self.model().cuda.device_id(),
                CudaContext::total_device_bytes()
            );
        }

        if ids.is_empty() {
            anyhow::bail!("empty ids");
        }
        if self.processed_len > ids.len() {
            self.processed_len = 0;
            if self.model().kv_cache.is_some() {
                self.model().reset_kv_cache()?;
            }
        }

        let start = if self.can_forward {
            self.processed_len
        } else {
            ids.len().saturating_sub(1)
        };
        let mut logits: Option<Vec<f32>> = None;
        for (token_idx, &tok_id_u32) in ids.iter().enumerate().skip(start) {
            let token_start = std::time::Instant::now();
            let tok_id = tok_id_u32 as u64;
            if decode_session_log_enabled() {
                eprintln!("[{}] token id {}", self.log_prefix, tok_id);
            }

            let token_logits = unsafe { self.logits_for_token(tok_id, token_idx, start)? };
            on_token_logits(&token_logits);
            timing::timing_log!(
                token_start.elapsed(),
                "{}.token.{token_idx}.total",
                self.log_prefix
            );
            logits = Some(token_logits);
        }
        if self.can_forward {
            self.processed_len = ids.len();
        }

        let result = logits.ok_or_else(|| anyhow::anyhow!("no token processed for logits"));
        timing::timing_log!(
            logits_fn_start.elapsed(),
            "{}.logits_fn.ids_len_{}",
            self.log_prefix,
            ids.len()
        );
        result
    }

    unsafe fn logits_for_token(
        &mut self,
        tok_id: u64,
        token_idx: usize,
        start: usize,
    ) -> Result<Vec<f32>> {
        let embed_start = std::time::Instant::now();
        (*self.model).load_token_embedding_to_f32(tok_id, self.d_x.as_mut_ptr())?;
        timing::timing_log!(
            embed_start.elapsed(),
            "{}.token.{token_idx}.embedding_load",
            self.log_prefix
        );

        if self.can_forward {
            self.last_forward_sync_diag = None;
            let forward_start = std::time::Instant::now();
            let forward_sync_diag = if forward_sync_diag_enabled() {
                Some(
                    PrefillSyncDiag::start(
                        &self.model().cuda,
                        format!("{}.token.{token_idx}.forward_sync_diag", self.log_prefix),
                    )?
                    .expect("forward sync diagnostic is enabled"),
                )
            } else {
                None
            };
            let d_out = self
                .d_out
                .as_ref()
                .expect("d_out allocated when full forward is enabled")
                .as_mut_ptr();
            let layers = self.forward_with_optional_graph(token_idx, d_out)?;
            timing::timing_log!(
                forward_start.elapsed(),
                "{}.token.{token_idx}.forward_all_layers",
                self.log_prefix
            );
            if token_idx == start && !self.logged_full_forward {
                eprintln!(
                    "[{}] full-layer forward enabled layers={layers}",
                    self.log_prefix
                );
                self.logged_full_forward = true;
            }
            let logits_start = std::time::Instant::now();
            if self.last_forward_used_graph {
                self.model().cuda.stream_wait_for_stream(
                    CudaStream::Prefill,
                    CudaStream::Decode,
                    "logits_wait_graph_replay",
                )?;
            }
            if let Some(sync_diag) = forward_sync_diag {
                self.last_forward_sync_diag = Some(sync_diag.finish(&self.model().cuda)?);
            }
            let logits = match self.d_logits.as_ref() {
                Some(d_logits) => (*self.model).logits_from_hidden_into(
                    d_out as *const _,
                    d_logits.as_mut_ptr(),
                    self.d_norm_hidden.as_ref().map(DeviceBuffer::as_mut_ptr),
                ),
                None => (*self.model).logits_from_hidden(d_out as *const _),
            };
            timing::timing_log!(
                logits_start.elapsed(),
                "{}.token.{token_idx}.logits",
                self.log_prefix
            );
            logits
        } else {
            let logits_start = std::time::Instant::now();
            let logits = match self.d_logits.as_ref() {
                Some(d_logits) => (*self.model).logits_from_hidden_into(
                    self.d_x.as_ptr(),
                    d_logits.as_mut_ptr(),
                    self.d_norm_hidden.as_ref().map(DeviceBuffer::as_mut_ptr),
                ),
                None => (*self.model).logits_from_hidden(self.d_x.as_ptr()),
            };
            timing::timing_log!(
                logits_start.elapsed(),
                "{}.token.{token_idx}.logits",
                self.log_prefix
            );
            logits
        }
    }

    unsafe fn forward_prefill_token_without_logits(
        &mut self,
        tok_id: u64,
        token_idx: usize,
    ) -> Result<()> {
        let token_start = std::time::Instant::now();
        let embed_start = std::time::Instant::now();
        (*self.model).load_token_embedding_to_f32(tok_id, self.d_x.as_mut_ptr())?;
        timing::timing_log!(
            embed_start.elapsed(),
            "{}.token.{token_idx}.embedding_load",
            self.log_prefix
        );
        let forward_start = std::time::Instant::now();
        let d_out = self
            .d_out
            .as_ref()
            .expect("d_out allocated when full forward is enabled")
            .as_mut_ptr();
        let _layers = self.forward_with_optional_graph(token_idx, d_out)?;
        timing::timing_log!(
            forward_start.elapsed(),
            "{}.token.{token_idx}.forward_all_layers",
            self.log_prefix
        );
        timing::timing_log!(
            token_start.elapsed(),
            "{}.token.{token_idx}.prefill_without_logits_total",
            self.log_prefix
        );
        Ok(())
    }

    unsafe fn forward_with_optional_graph(
        &mut self,
        token_idx: usize,
        d_out: *mut c_void,
    ) -> Result<usize> {
        let seq_len = (token_idx + 1) as u32;
        self.last_forward_used_graph = false;
        if self.graph_disabled
            || !decode_graph_enabled()
            || crate::kv_compression::runtime_config().mode.is_enabled()
        {
            return self.with_exact_block_staging(|| {
                (*self.model).forward_one_token_all_layers_for_sequence(
                    self.d_x.as_ptr(),
                    self.sequence_id,
                    seq_len,
                    d_out,
                )
            });
        }
        let position = token_idx as u32;
        let d_position = self
            .d_graph_position
            .as_ref()
            .expect("graph position allocated when graph is enabled");
        let d_seq_len = self
            .d_graph_seq_len
            .as_ref()
            .expect("graph seq_len allocated when graph is enabled");
        self.upload_graph_params(position, seq_len)?;

        if self.decode_graph.is_none() {
            // Warm once through the normal path so lazy workspace and FP32 weight
            // materialization do not occur inside CUDA stream capture.
            let warmed_layers = self.with_exact_block_staging(|| {
                (*self.model).forward_one_token_all_layers_for_sequence(
                    self.d_x.as_ptr(),
                    self.sequence_id,
                    seq_len,
                    d_out,
                )
            })?;
            let captured = (*self.model).cuda.capture_graph(CudaStream::Decode, || {
                (*self.model)
                    .forward_one_token_all_layers_for_sequence_graph_params(
                        self.d_x.as_ptr(),
                        self.sequence_id,
                        d_position.as_ptr() as *const u32,
                        d_seq_len.as_ptr() as *const u32,
                        d_out,
                    )
                    .map(|_| ())
            });
            match captured {
                Ok(graph) => {
                    eprintln!(
                        "[{}] captured full-token decode CUDA graph layers={warmed_layers}",
                        self.log_prefix
                    );
                    self.decode_graph = Some(graph);
                }
                Err(err) => {
                    eprintln!(
                        "[{}] disabling full-token decode CUDA graph after capture failure: {err:#}",
                        self.log_prefix
                    );
                    self.graph_disabled = true;
                    return Ok(warmed_layers);
                }
            }
        }

        if let Some(graph) = &self.decode_graph {
            let launch_result = if decode_graph_diagnostic_sync_enabled() {
                graph
                    .launch_timed_sync(CudaStream::Decode)
                    .map(|elapsed_ms| {
                        eprintln!(
                            "[{}] decode graph replay gpu_elapsed_ms={elapsed_ms:.3}",
                            self.log_prefix
                        );
                        if let Some(max_ms) = decode_graph_diag_max_ms() {
                            if elapsed_ms > max_ms {
                                eprintln!(
                                    "[{}] decode graph replay {:.3}ms exceeds diagnostic max {:.3}ms; disabling graph",
                                    self.log_prefix, elapsed_ms, max_ms
                                );
                                self.graph_disabled = true;
                            }
                        }
                    })
            } else {
                graph.launch(CudaStream::Decode)
            };
            if let Err(err) = launch_result {
                eprintln!(
                    "[{}] disabling full-token decode CUDA graph after launch failure: {err:#}",
                    self.log_prefix
                );
                self.graph_disabled = true;
                return self.with_exact_block_staging(|| {
                    (*self.model).forward_one_token_all_layers_for_sequence(
                        self.d_x.as_ptr(),
                        self.sequence_id,
                        seq_len,
                        d_out,
                    )
                });
            }
            self.last_forward_used_graph = true;
            Ok((*self.model).model_config.block_count as usize)
        } else {
            self.with_exact_block_staging(|| {
                (*self.model).forward_one_token_all_layers_for_sequence(
                    self.d_x.as_ptr(),
                    self.sequence_id,
                    seq_len,
                    d_out,
                )
            })
        }
    }

    fn upload_graph_params(&self, position: u32, seq_len: u32) -> Result<()> {
        let d_position = self
            .d_graph_position
            .as_ref()
            .expect("graph position allocated when graph is enabled");
        let d_seq_len = self
            .d_graph_seq_len
            .as_ref()
            .expect("graph seq_len allocated when graph is enabled");
        unsafe {
            self.model().cuda.memcpy_h2d(
                d_position.as_mut_ptr(),
                (&position as *const u32).cast::<c_void>(),
                std::mem::size_of::<u32>(),
            )?;
            self.model().cuda.memcpy_h2d(
                d_seq_len.as_mut_ptr(),
                (&seq_len as *const u32).cast::<c_void>(),
                std::mem::size_of::<u32>(),
            )?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn decode_graph_enabled() -> bool {
    std::env::var("M40LLM_DECODE_GRAPH")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn exact_block_staging_enabled() -> bool {
    std::env::var("M40LLM_KV_EXACT_BLOCK_STAGING")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn decode_graph_diagnostic_sync_enabled() -> bool {
    std::env::var("M40LLM_DECODE_GRAPH_DIAG_SYNC")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn decode_graph_diag_max_ms() -> Option<f32> {
    std::env::var("M40LLM_DECODE_GRAPH_DIAG_MAX_MS")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| *value > 0.0)
}
