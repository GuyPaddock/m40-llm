use super::LoadedModel;
#[cfg(feature = "cuda")]
use crate::cuda::ExactBlockStagingPtrs;
use crate::cuda::{ExactOldBacking, KVCache};
use crate::kv_compression::{runtime_config, KvCompressMode, KvCompressionConfig};
use crate::kv_selection;
use anyhow::{anyhow, Result};
#[cfg(feature = "cuda")]
use std::cell::Cell;
use std::ffi::c_void;

#[cfg(feature = "cuda")]
thread_local! {
    static EXACT_BLOCK_STAGING_PTRS: Cell<Option<ExactBlockStagingPtrs>> = const { Cell::new(None) };
}

fn exact_block_staging_enabled() -> bool {
    std::env::var("M40LLM_KV_EXACT_BLOCK_STAGING")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn q8_exact_old_backing_enabled() -> bool {
    std::env::var("M40LLM_KV_EXACT_OLD_BACKING")
        .map(|value| matches!(value.as_str(), "q8" | "Q8"))
        .unwrap_or(false)
}

fn fp16_k_q4_v_exact_old_backing_enabled() -> bool {
    std::env::var("M40LLM_KV_EXACT_OLD_BACKING")
        .map(|value| matches!(value.as_str(), "fp16-k-q4-v" | "FP16-K-Q4-V"))
        .unwrap_or(false)
}

fn q8_direct_exact_old_attention_enabled() -> bool {
    std::env::var("M40LLM_KV_EXACT_OLD_ATTENTION")
        .map(|value| matches!(value.as_str(), "q8-direct" | "Q8-DIRECT" | "direct-q8"))
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
pub fn with_exact_block_staging<R>(
    staging: Option<ExactBlockStagingPtrs>,
    f: impl FnOnce() -> Result<R>,
) -> Result<R> {
    struct Reset(Option<ExactBlockStagingPtrs>);
    impl Drop for Reset {
        fn drop(&mut self) {
            EXACT_BLOCK_STAGING_PTRS.with(|slot| slot.set(self.0));
        }
    }

    let previous = EXACT_BLOCK_STAGING_PTRS.with(|slot| {
        let previous = slot.get();
        slot.set(staging);
        previous
    });
    let _reset = Reset(previous);
    f()
}

#[cfg(feature = "cuda")]
fn current_exact_block_staging() -> Option<ExactBlockStagingPtrs> {
    EXACT_BLOCK_STAGING_PTRS.with(Cell::get)
}

impl LoadedModel {
    pub(super) fn kv_physical_slot_for_layer_sequence(
        &self,
        layer_id: u32,
        sequence_id: u32,
    ) -> Result<u32> {
        let kv = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
        if layer_id >= self.model_config.block_count {
            anyhow::bail!(
                "KV layer_id {} out of range for {} layers",
                layer_id,
                self.model_config.block_count
            );
        }
        let layer_count = self.model_config.block_count;
        let sequence_capacity = kv.max_batch_size() / layer_count;
        if sequence_id >= sequence_capacity {
            anyhow::bail!(
                "KV sequence_id {} out of range for {} logical sequences",
                sequence_id,
                sequence_capacity
            );
        }
        if kv.max_batch_size() < layer_count {
            anyhow::bail!(
                "KV cache has {} physical slots, but layer-addressed decode needs {}",
                kv.max_batch_size(),
                layer_count
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

    pub fn kv_cache_can_address_layer_sequence(&self, layer_id: u32, sequence_id: u32) -> bool {
        self.kv_physical_slot_for_layer_sequence(layer_id, sequence_id)
            .is_ok()
    }

    pub fn kv_cache_logical_sequence_capacity(&self) -> u32 {
        self.kv_cache
            .as_ref()
            .and_then(|kv| {
                let layer_count = self.model_config.block_count;
                (layer_count > 0).then_some(kv.max_batch_size() / layer_count)
            })
            .unwrap_or(0)
    }

    pub fn kv_cache_physical_slot_for_layer_sequence(
        &self,
        layer_id: u32,
        sequence_id: u32,
    ) -> Result<u32> {
        self.kv_physical_slot_for_layer_sequence(layer_id, sequence_id)
    }

    // Convenience: Append K/V from host FP32 slices. Copies to device then calls append.
    pub fn append_kv_token_f32_from_host(
        &self,
        seq_id: u32,
        k_host: &[f32],
        v_host: &[f32],
    ) -> Result<()> {
        let kv = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
        let elems = kv.elems_per_token();
        if k_host.len() != elems || v_host.len() != elems {
            anyhow::bail!(
                "append_kv_token_f32_from_host: expected {} elems per token, got k={}, v={}",
                elems,
                k_host.len(),
                v_host.len()
            );
        }
        #[cfg(feature = "cuda")]
        {
            let bytes = elems * std::mem::size_of::<f32>();
            let d_k = self.cuda.device_malloc(bytes)?;
            let d_v = self.cuda.device_malloc(bytes)?;
            unsafe {
                self.cuda
                    .memcpy_h2d(d_k, k_host.as_ptr() as *const c_void, bytes)?;
                self.cuda
                    .memcpy_h2d(d_v, v_host.as_ptr() as *const c_void, bytes)?;
            }
            let res = unsafe {
                self.append_kv_token_f32(seq_id, d_k as *const c_void, d_v as *const c_void)
            };
            // best-effort free even if append errs
            unsafe {
                let _ = self.cuda.device_free(d_k);
                let _ = self.cuda.device_free(d_v);
            }
            res
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Directly write into host-side KVCache
            kv.append_token_f32(
                &self.cuda,
                seq_id,
                k_host.as_ptr() as *const c_void,
                v_host.as_ptr() as *const c_void,
            )
        }
    }

    pub fn append_kv_token_f32_from_host_for_layer(
        &self,
        layer_id: u32,
        sequence_id: u32,
        position: u32,
        k_host: &[f32],
        v_host: &[f32],
    ) -> Result<()> {
        let kv = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
        if position >= kv.max_seq_len() {
            anyhow::bail!(
                "KV position {} out of range for max_seq_len {}",
                position,
                kv.max_seq_len()
            );
        }
        let physical_slot = self.kv_physical_slot_for_layer_sequence(layer_id, sequence_id)?;
        self.append_kv_token_f32_from_host(physical_slot, k_host, v_host)
    }

    pub fn allocate_kv_cache(&mut self, max_seq_len: u32, max_batch_size: u32) -> Result<()> {
        let num_heads = self.model_config.attention_head_count_kv;
        let d_model = self.model_config.embedding_length;
        if self.model_config.attention_head_count == 0
            || num_heads == 0
            || d_model == 0
            || !d_model.is_multiple_of(self.model_config.attention_head_count)
        {
            anyhow::bail!(
                "model_config invalid for KV layout: d_model {} head_count {} kv_head_count {}",
                d_model,
                self.model_config.attention_head_count,
                num_heads
            );
        }
        let head_dim = self.model_config.attention_key_length;
        if head_dim == 0 {
            anyhow::bail!("attention_key_length must be > 0");
        }
        self.kv_cache = None;
        let kv = KVCache::new_with_context(
            &self.cuda,
            max_seq_len,
            max_batch_size,
            num_heads,
            head_dim,
        )?;
        self.kv_cache = Some(kv);
        Ok(())
    }

    pub fn allocate_kv_cache_for_layers(&mut self, max_seq_len: u32) -> Result<()> {
        self.allocate_kv_cache_for_layer_sequences(max_seq_len, 1)
    }

    pub fn allocate_kv_cache_for_layer_sequences(
        &mut self,
        max_seq_len: u32,
        max_sequences: u32,
    ) -> Result<()> {
        let layer_count = self.model_config.block_count;
        if layer_count == 0 {
            anyhow::bail!("model_config.block_count must be > 0");
        }
        if max_sequences == 0 {
            anyhow::bail!("max_sequences must be > 0");
        }
        let physical_slots = layer_count
            .checked_mul(max_sequences)
            .ok_or_else(|| anyhow!("KV physical slot count overflow"))?;
        self.allocate_kv_cache(max_seq_len, physical_slots)
    }

    pub fn allocate_compressed_kv_cache_for_layers(
        &mut self,
        max_seq_len: u32,
        config: &KvCompressionConfig,
    ) -> Result<()> {
        let layer_count = self.model_config.block_count;
        if layer_count == 0 {
            anyhow::bail!("model_config.block_count must be > 0");
        }
        let exact_old_backing =
            if config.mode == KvCompressMode::BlockSelectExact && q8_exact_old_backing_enabled() {
                ExactOldBacking::Q8
            } else if config.mode == KvCompressMode::BlockSelectExact
                && fp16_k_q4_v_exact_old_backing_enabled()
            {
                ExactOldBacking::Fp16KQ4V
            } else {
                ExactOldBacking::Dense
            };
        if !matches!(
            config.mode,
            KvCompressMode::RecentOnly
                | KvCompressMode::BlockSummary
                | KvCompressMode::BlockSelectLossy
        ) && matches!(exact_old_backing, ExactOldBacking::Dense)
        {
            anyhow::bail!(
                "allocate_compressed_kv_cache_for_layers requires a compressed sidecar mode"
            );
        }
        config.validate()?;
        let num_heads = self.model_config.attention_head_count_kv;
        let head_dim = self.model_config.attention_key_length;
        if head_dim != 64 {
            anyhow::bail!("compressed KV cache currently requires head_dim=64");
        }
        let recent_window = config.recent_window.min(max_seq_len);
        self.kv_cache = None;
        let kv = KVCache::new_compressed_with_context(
            &self.cuda,
            max_seq_len,
            layer_count,
            num_heads,
            head_dim,
            recent_window,
            config.block_size,
            config.top_blocks,
            config.representatives,
            config.representative_policy,
            exact_old_backing,
        )?;
        let dense = kv.dense_equivalent_bytes();
        let actual = kv.actual_bytes();
        eprintln!(
            "[kv-compress] mode={:?} dense_equivalent_bytes={} actual_allocated_bytes={} compression_ratio={:.3}",
            config.mode,
            dense,
            actual,
            if dense > 0 { actual as f64 / dense as f64 } else { 1.0 }
        );
        self.kv_cache = Some(kv);
        Ok(())
    }

    pub fn kv_cache_can_address_layers(&self) -> bool {
        self.kv_cache
            .as_ref()
            .map(|kv| kv.max_batch_size() >= self.model_config.block_count)
            .unwrap_or(false)
    }

    pub fn allocate_kv_cache_with_layout(
        &mut self,
        max_seq_len: u32,
        max_batch_size: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        if num_heads == 0 || head_dim == 0 {
            anyhow::bail!("allocate_kv_cache_with_layout: invalid layout");
        }
        self.kv_cache = None;
        let kv = KVCache::new_with_context(
            &self.cuda,
            max_seq_len,
            max_batch_size,
            num_heads,
            head_dim,
        )?;
        self.kv_cache = Some(kv);
        Ok(())
    }

    pub fn reset_kv_cache(&self) -> Result<()> {
        let kv = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
        kv.reset(&self.cuda)
    }

    /// # Safety
    /// `d_a`, `d_b`, and `d_c` must be valid pointers to device buffers sized for GEMM with (m, n, k).
    pub unsafe fn run_attention(
        &self,
        d_q: *const c_void,
        d_out: *mut c_void,
        seq_id: u32,
        seq_len: u32,
        dim: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        // Validate layout: dim must equal num_heads * head_dim
        if dim != num_heads.saturating_mul(head_dim) {
            anyhow::bail!(
                "run_attention: dim {} != num_heads {} * head_dim {}",
                dim,
                num_heads,
                head_dim
            );
        }
        let kv = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
        if kv.num_heads() > num_heads || !num_heads.is_multiple_of(kv.num_heads()) {
            anyhow::bail!(
                "run_attention: query heads {} must be a multiple of kv heads {}",
                num_heads,
                kv.num_heads()
            );
        }
        if kv.head_dim() != head_dim {
            anyhow::bail!(
                "KVCache layout mismatch: kv has (heads={}, dim={}), requested query heads {} dim {}",
                kv.num_heads(),
                kv.head_dim(),
                num_heads,
                head_dim
            );
        }
        #[cfg(feature = "cuda")]
        unsafe {
            kv.attention_last_token_f32_gqa(&self.cuda, seq_id, d_q, num_heads, seq_len, d_out)
        }
        #[cfg(not(feature = "cuda"))]
        {
            kv.attention_last_token_f32_gqa(&self.cuda, seq_id, d_q, num_heads, seq_len, d_out)
        }
    }

    /// # Safety
    /// `d_q` and `d_out` must be valid device pointers sized for one-token GQA attention.
    pub unsafe fn run_attention_async(
        &self,
        d_q: *const c_void,
        d_out: *mut c_void,
        seq_id: u32,
        seq_len: u32,
        dim: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        if dim != num_heads.saturating_mul(head_dim) {
            anyhow::bail!(
                "run_attention: dim {} != num_heads {} * head_dim {}",
                dim,
                num_heads,
                head_dim
            );
        }
        let kv = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
        if kv.num_heads() > num_heads || !num_heads.is_multiple_of(kv.num_heads()) {
            anyhow::bail!(
                "run_attention: query heads {} must be a multiple of kv heads {}",
                num_heads,
                kv.num_heads()
            );
        }
        if kv.head_dim() != head_dim {
            anyhow::bail!(
                "KVCache layout mismatch: kv has (heads={}, dim={}), requested query heads {} dim {}",
                kv.num_heads(),
                kv.head_dim(),
                num_heads,
                head_dim
            );
        }
        let compression = runtime_config();
        if compression.mode == KvCompressMode::DenseRecentOnly {
            if head_dim != 64 {
                anyhow::bail!("dense-recent-only requires head_dim=64");
            }
            #[cfg(feature = "cuda")]
            unsafe {
                return kv.attention_last_token_f32_gqa_dense_recent_async(
                    &self.cuda,
                    seq_id,
                    d_q,
                    num_heads,
                    seq_len,
                    compression.recent_window,
                    d_out,
                );
            }
        } else if compression.mode == KvCompressMode::BlockSelectExact {
            if head_dim != 64 {
                anyhow::bail!("block-select-exact requires head_dim=64");
            }
            #[cfg(feature = "cuda")]
            unsafe {
                if kv_selection::enabled() {
                    if kv_selection::should_capture_attention() {
                        let attention_mode = if q8_exact_old_backing_enabled()
                            || fp16_k_q4_v_exact_old_backing_enabled()
                        {
                            5
                        } else {
                            1
                        };
                        if let Ok(attention) = kv.debug_attention_telemetry(
                            &self.cuda,
                            attention_mode,
                            seq_id,
                            d_q,
                            num_heads,
                            seq_len,
                            compression.recent_window,
                            compression.block_size,
                            compression.top_blocks,
                            kv_selection::needle_block(),
                        ) {
                            kv_selection::record_attention(attention);
                        }
                    }
                    if let Ok((blocks, total_old_blocks)) = kv.debug_select_old_blocks(
                        &self.cuda,
                        seq_id,
                        d_q,
                        num_heads,
                        seq_len,
                        compression.recent_window,
                        compression.block_size,
                        compression.top_blocks,
                    ) {
                        kv_selection::record_scored(
                            blocks,
                            total_old_blocks,
                            compression.top_blocks,
                        );
                    }
                }
                if q8_exact_old_backing_enabled() && q8_direct_exact_old_attention_enabled() {
                    if !kv.is_compressed() {
                        kv.build_q8_old_from_dense(
                            &self.cuda,
                            seq_id,
                            seq_len,
                            compression.recent_window,
                        )?;
                    }
                    return kv.attention_last_token_f32_gqa_block_select_exact_q8_old_direct_async(
                        &self.cuda,
                        seq_id,
                        d_q,
                        num_heads,
                        seq_len,
                        compression.recent_window,
                        compression.block_size,
                        compression.top_blocks,
                        d_out,
                    );
                }
                if exact_block_staging_enabled() {
                    if let Some(staging) = current_exact_block_staging() {
                        if q8_exact_old_backing_enabled() {
                            if !kv.is_compressed() {
                                kv.build_q8_old_from_dense(
                                    &self.cuda,
                                    seq_id,
                                    seq_len,
                                    compression.recent_window,
                                )?;
                            }
                            return kv.attention_last_token_f32_gqa_block_select_exact_staged_q8_old_with_buffers_async(
                                &self.cuda,
                                seq_id,
                                d_q,
                                num_heads,
                                seq_len,
                                compression.recent_window,
                                compression.block_size,
                                compression.top_blocks,
                                staging,
                                d_out,
                            );
                        }
                        if fp16_k_q4_v_exact_old_backing_enabled() {
                            return kv.attention_last_token_f32_gqa_block_select_exact_staged_fp16_k_q4_v_old_with_buffers_async(
                                &self.cuda,
                                seq_id,
                                d_q,
                                num_heads,
                                seq_len,
                                compression.recent_window,
                                compression.block_size,
                                compression.top_blocks,
                                staging,
                                d_out,
                            );
                        }
                        return kv.attention_last_token_f32_gqa_block_select_exact_staged_with_buffers_async(
                            &self.cuda,
                            seq_id,
                            d_q,
                            num_heads,
                            seq_len,
                            compression.recent_window,
                            compression.block_size,
                            compression.top_blocks,
                            staging,
                            d_out,
                        );
                    }
                    return kv.attention_last_token_f32_gqa_block_select_exact_staged_async(
                        &self.cuda,
                        seq_id,
                        d_q,
                        num_heads,
                        seq_len,
                        compression.recent_window,
                        compression.block_size,
                        compression.top_blocks,
                        d_out,
                    );
                }
                if kv.is_compressed() {
                    anyhow::bail!(
                        "compressed block-select-exact KV requires M40LLM_KV_EXACT_BLOCK_STAGING=1"
                    );
                }
                return kv.attention_last_token_f32_gqa_block_select_exact_async(
                    &self.cuda,
                    seq_id,
                    d_q,
                    num_heads,
                    seq_len,
                    compression.recent_window,
                    compression.block_size,
                    compression.top_blocks,
                    d_out,
                );
            }
        } else if compression.mode == KvCompressMode::RecentOnly {
            if head_dim != 64 {
                anyhow::bail!("recent-only requires head_dim=64");
            }
            #[cfg(feature = "cuda")]
            unsafe {
                if kv_selection::should_capture_attention() {
                    if let Ok(attention) = kv.debug_attention_telemetry(
                        &self.cuda,
                        2,
                        seq_id,
                        d_q,
                        num_heads,
                        seq_len,
                        compression.recent_window,
                        compression.block_size,
                        0,
                        kv_selection::needle_block(),
                    ) {
                        kv_selection::record_attention(attention);
                    }
                }
                return kv.attention_last_token_f32_gqa_compressed_recent_only_async(
                    &self.cuda, seq_id, d_q, num_heads, seq_len, d_out,
                );
            }
        } else if matches!(
            compression.mode,
            KvCompressMode::BlockSummary | KvCompressMode::BlockSelectLossy
        ) {
            if head_dim != 64 {
                anyhow::bail!("{:?} requires head_dim=64", compression.mode);
            }
            #[cfg(feature = "cuda")]
            unsafe {
                let top_blocks = if compression.mode == KvCompressMode::BlockSummary {
                    0
                } else {
                    compression.top_blocks
                };
                if kv_selection::enabled() {
                    if kv_selection::should_capture_attention() {
                        let mode_code = if compression.mode == KvCompressMode::BlockSummary {
                            3
                        } else {
                            4
                        };
                        if let Ok(attention) = kv.debug_attention_telemetry(
                            &self.cuda,
                            mode_code,
                            seq_id,
                            d_q,
                            num_heads,
                            seq_len,
                            compression.recent_window,
                            compression.block_size,
                            top_blocks,
                            kv_selection::needle_block(),
                        ) {
                            kv_selection::record_attention(attention);
                        }
                    }
                    if let Ok((blocks, total_old_blocks)) = kv.debug_select_old_blocks(
                        &self.cuda,
                        seq_id,
                        d_q,
                        num_heads,
                        seq_len,
                        compression.recent_window,
                        compression.block_size,
                        top_blocks,
                    ) {
                        kv_selection::record_scored(blocks, total_old_blocks, top_blocks);
                    }
                }
                return kv.attention_last_token_f32_gqa_block_summary_lossy_async(
                    &self.cuda,
                    seq_id,
                    d_q,
                    num_heads,
                    seq_len,
                    compression.recent_window,
                    compression.block_size,
                    top_blocks,
                    d_out,
                );
            }
        } else if compression.mode.is_enabled() {
            anyhow::bail!(
                "KV compression mode {:?} is not implemented in decode attention yet",
                compression.mode
            );
        }
        #[cfg(feature = "cuda")]
        unsafe {
            kv.attention_last_token_f32_gqa_async(
                &self.cuda, seq_id, d_q, num_heads, seq_len, d_out,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            kv.attention_last_token_f32_gqa(&self.cuda, seq_id, d_q, num_heads, seq_len, d_out)
        }
    }

    /// # Safety
    /// `d_q` and `d_out` must be valid device pointers sized for one-token GQA
    /// attention. `d_seq_len` must point to one device-resident u32 sequence length.
    pub unsafe fn run_attention_seq_len_dev_async(
        &self,
        d_q: *const c_void,
        d_out: *mut c_void,
        seq_id: u32,
        d_seq_len: *const u32,
        dim: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        if d_seq_len.is_null() {
            anyhow::bail!("run_attention_seq_len_dev_async: d_seq_len is null");
        }
        if dim != num_heads.saturating_mul(head_dim) {
            anyhow::bail!(
                "run_attention: dim {} != num_heads {} * head_dim {}",
                dim,
                num_heads,
                head_dim
            );
        }
        let kv = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
        if kv.num_heads() > num_heads || !num_heads.is_multiple_of(kv.num_heads()) {
            anyhow::bail!(
                "run_attention: query heads {} must be a multiple of kv heads {}",
                num_heads,
                kv.num_heads()
            );
        }
        if kv.head_dim() != head_dim {
            anyhow::bail!(
                "KVCache layout mismatch: kv has (heads={}, dim={}), requested query heads {} dim {}",
                kv.num_heads(),
                kv.head_dim(),
                num_heads,
                head_dim
            );
        }
        #[cfg(feature = "cuda")]
        unsafe {
            kv.attention_last_token_f32_gqa_seq_len_dev_async(
                &self.cuda, seq_id, d_q, num_heads, d_seq_len, d_out,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_q, d_out, seq_id, d_seq_len, dim, num_heads, head_dim);
            Ok(())
        }
    }

    /// # Safety
    /// `d_q` and `d_out` must be valid device pointers sized for one-token GQA attention.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_attention_for_layer(
        &self,
        d_q: *const c_void,
        d_out: *mut c_void,
        layer_id: u32,
        sequence_id: u32,
        seq_len: u32,
        dim: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        let physical_slot = self.kv_physical_slot_for_layer_sequence(layer_id, sequence_id)?;
        self.run_attention(d_q, d_out, physical_slot, seq_len, dim, num_heads, head_dim)
    }

    /// # Safety
    /// `d_q` and `d_out` must be valid device pointers sized for one-token GQA
    /// attention. This experimental path keeps exact old KV and sparsifies only
    /// the read set.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_attention_block_select_exact_for_layer_async(
        &self,
        d_q: *const c_void,
        d_out: *mut c_void,
        layer_id: u32,
        sequence_id: u32,
        seq_len: u32,
        dim: u32,
        num_heads: u32,
        head_dim: u32,
        recent_window: u32,
        block_size: u32,
        top_blocks: u32,
    ) -> Result<()> {
        if dim != num_heads.saturating_mul(head_dim) {
            anyhow::bail!(
                "run_attention_block_select_exact: dim {} != num_heads {} * head_dim {}",
                dim,
                num_heads,
                head_dim
            );
        }
        let kv = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
        if kv.num_heads() > num_heads || !num_heads.is_multiple_of(kv.num_heads()) {
            anyhow::bail!(
                "run_attention_block_select_exact: query heads {} must be a multiple of kv heads {}",
                num_heads,
                kv.num_heads()
            );
        }
        if kv.head_dim() != head_dim {
            anyhow::bail!(
                "KVCache layout mismatch: kv has (heads={}, dim={}), requested query heads {} dim {}",
                kv.num_heads(),
                kv.head_dim(),
                num_heads,
                head_dim
            );
        }
        let physical_slot = self.kv_physical_slot_for_layer_sequence(layer_id, sequence_id)?;
        #[cfg(feature = "cuda")]
        unsafe {
            if q8_exact_old_backing_enabled() && q8_direct_exact_old_attention_enabled() {
                if !kv.is_compressed() {
                    kv.build_q8_old_from_dense(&self.cuda, physical_slot, seq_len, recent_window)?;
                }
                return kv.attention_last_token_f32_gqa_block_select_exact_q8_old_direct_async(
                    &self.cuda,
                    physical_slot,
                    d_q,
                    num_heads,
                    seq_len,
                    recent_window,
                    block_size,
                    top_blocks,
                    d_out,
                );
            }
            if exact_block_staging_enabled() {
                if let Some(staging) = current_exact_block_staging() {
                    if q8_exact_old_backing_enabled() {
                        if !kv.is_compressed() {
                            kv.build_q8_old_from_dense(
                                &self.cuda,
                                physical_slot,
                                seq_len,
                                recent_window,
                            )?;
                        }
                        return kv
                            .attention_last_token_f32_gqa_block_select_exact_staged_q8_old_with_buffers_async(
                                &self.cuda,
                                physical_slot,
                                d_q,
                                num_heads,
                                seq_len,
                                recent_window,
                                block_size,
                                top_blocks,
                                staging,
                                d_out,
                            );
                    }
                    if fp16_k_q4_v_exact_old_backing_enabled() {
                        return kv
                            .attention_last_token_f32_gqa_block_select_exact_staged_fp16_k_q4_v_old_with_buffers_async(
                                &self.cuda,
                                physical_slot,
                                d_q,
                                num_heads,
                                seq_len,
                                recent_window,
                                block_size,
                                top_blocks,
                                staging,
                                d_out,
                            );
                    }
                    return kv
                        .attention_last_token_f32_gqa_block_select_exact_staged_with_buffers_async(
                            &self.cuda,
                            physical_slot,
                            d_q,
                            num_heads,
                            seq_len,
                            recent_window,
                            block_size,
                            top_blocks,
                            staging,
                            d_out,
                        );
                }
                return kv.attention_last_token_f32_gqa_block_select_exact_staged_async(
                    &self.cuda,
                    physical_slot,
                    d_q,
                    num_heads,
                    seq_len,
                    recent_window,
                    block_size,
                    top_blocks,
                    d_out,
                );
            }
            if kv.is_compressed() {
                anyhow::bail!(
                    "compressed block-select-exact KV requires M40LLM_KV_EXACT_BLOCK_STAGING=1"
                );
            }
            kv.attention_last_token_f32_gqa_block_select_exact_async(
                &self.cuda,
                physical_slot,
                d_q,
                num_heads,
                seq_len,
                recent_window,
                block_size,
                top_blocks,
                d_out,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                d_q,
                d_out,
                physical_slot,
                seq_len,
                recent_window,
                block_size,
                top_blocks,
            );
            Ok(())
        }
    }

    /// # Safety
    /// `d_k_f32` and `d_v_f32` must be valid pointers to device buffers containing one token's K/V in f32 layout.
    pub unsafe fn append_kv_token_f32(
        &self,
        seq_id: u32,
        d_k_f32: *const c_void,
        d_v_f32: *const c_void,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let kv = self
                .kv_cache
                .as_ref()
                .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
            #[cfg(feature = "cuda")]
            unsafe {
                kv.append_token_f32(&self.cuda, seq_id, d_k_f32, d_v_f32)
            }
            #[cfg(not(feature = "cuda"))]
            {
                kv.append_token_f32(&self.cuda, seq_id, d_k_f32, d_v_f32)
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (seq_id, d_k_f32, d_v_f32);
            Ok(())
        }
    }

    /// # Safety
    /// `d_k_f32` and `d_v_f32` must be valid pointers to device buffers
    /// containing one token's K/V in f32 layout. K is RoPE-rotated for
    /// `past_len` while appending to the FP16 KV cache.
    pub unsafe fn append_kv_token_f32_rope_k(
        &self,
        seq_id: u32,
        d_k_f32: *const c_void,
        d_v_f32: *const c_void,
        past_len: u32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let kv = self
                .kv_cache
                .as_ref()
                .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
            unsafe {
                kv.append_token_f32_rope_k(
                    &self.cuda, seq_id, d_k_f32, d_v_f32, past_len, freq_base, freq_scale,
                )
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (seq_id, d_k_f32, d_v_f32, past_len, freq_base, freq_scale);
            Ok(())
        }
    }

    /// # Safety
    /// `d_k_f32` and `d_v_f32` must be valid pointers to device buffers
    /// containing one token's K/V in f32 layout. K is RoPE-rotated for
    /// `past_len` while appending to the FP16 KV cache.
    pub unsafe fn append_kv_token_f32_rope_k_async(
        &self,
        seq_id: u32,
        d_k_f32: *const c_void,
        d_v_f32: *const c_void,
        past_len: u32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let kv = self
                .kv_cache
                .as_ref()
                .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
            unsafe {
                kv.append_token_f32_rope_k_async(
                    &self.cuda, seq_id, d_k_f32, d_v_f32, past_len, freq_base, freq_scale,
                )
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (seq_id, d_k_f32, d_v_f32, past_len, freq_base, freq_scale);
            Ok(())
        }
    }

    /// # Safety
    /// `d_k_f32` and `d_v_f32` must be valid pointers to device buffers
    /// containing one token's K/V in f32 layout. K is RoPE-rotated for
    /// `past_len` and appended at explicit `position`, avoiding a host-side KV
    /// length read.
    pub unsafe fn append_kv_token_f32_rope_k_at_async(
        &self,
        seq_id: u32,
        d_k_f32: *const c_void,
        d_v_f32: *const c_void,
        position: u32,
        past_len: u32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let kv = self
                .kv_cache
                .as_ref()
                .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
            unsafe {
                kv.append_token_f32_rope_k_at_async(
                    &self.cuda, seq_id, d_k_f32, d_v_f32, position, past_len, freq_base, freq_scale,
                )
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                seq_id, d_k_f32, d_v_f32, position, past_len, freq_base, freq_scale,
            );
            Ok(())
        }
    }

    /// # Safety
    /// `d_k_f32` and `d_v_f32` must be valid pointers to device buffers
    /// containing one token's K/V in f32 layout. `position_dev` must point to
    /// one device-resident u32 used both as KV position and RoPE position.
    pub unsafe fn append_kv_token_f32_rope_k_position_dev_async(
        &self,
        seq_id: u32,
        d_k_f32: *const c_void,
        d_v_f32: *const c_void,
        position_dev: *const u32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<()> {
        if position_dev.is_null() {
            anyhow::bail!("append_kv_token_f32_rope_k_position_dev_async: position_dev is null");
        }
        #[cfg(feature = "cuda")]
        {
            let kv = self
                .kv_cache
                .as_ref()
                .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
            unsafe {
                kv.append_token_f32_rope_k_position_dev_async(
                    &self.cuda,
                    seq_id,
                    d_k_f32,
                    d_v_f32,
                    position_dev,
                    freq_base,
                    freq_scale,
                )
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                seq_id,
                d_k_f32,
                d_v_f32,
                position_dev,
                freq_base,
                freq_scale,
            );
            Ok(())
        }
    }

    /// # Safety
    /// `d_k_f32` and `d_v_f32` must be valid device pointers containing one token's
    /// K/V vectors in f32 layout for `layer_id`, `sequence_id`, and `position`.
    pub unsafe fn append_kv_token_f32_for_layer(
        &self,
        layer_id: u32,
        sequence_id: u32,
        position: u32,
        d_k_f32: *const c_void,
        d_v_f32: *const c_void,
    ) -> Result<()> {
        let kv = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
        if position >= kv.max_seq_len() {
            anyhow::bail!(
                "KV position {} out of range for max_seq_len {}",
                position,
                kv.max_seq_len()
            );
        }
        let physical_slot = self.kv_physical_slot_for_layer_sequence(layer_id, sequence_id)?;
        self.append_kv_token_f32(physical_slot, d_k_f32, d_v_f32)
    }

    pub fn run_mlp(
        &self,
        _d_in: *const c_void,
        _d_out: *mut c_void,
        _batch_seq: u32,
        _dim: u32,
        _hidden_dim: u32,
    ) -> Result<()> {
        // Stub: no-op (activation wiring handled in forward_one_token_minimal for now)
        Ok(())
    }
}
