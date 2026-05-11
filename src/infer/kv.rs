use super::LoadedModel;
use crate::cuda::KVCache;
use anyhow::{anyhow, Result};
use std::ffi::c_void;

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
        if sequence_id != 0 {
            anyhow::bail!(
                "KV layer/sequence addressing currently supports sequence_id=0 only; got {}",
                sequence_id
            );
        }
        if kv.max_batch_size() < self.model_config.block_count {
            anyhow::bail!(
                "KV cache has {} physical slots, but layer-addressed decode needs {}",
                kv.max_batch_size(),
                self.model_config.block_count
            );
        }
        Ok(layer_id)
    }

    pub fn kv_cache_can_address_layer_sequence(&self, layer_id: u32, sequence_id: u32) -> bool {
        self.kv_physical_slot_for_layer_sequence(layer_id, sequence_id)
            .is_ok()
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
        let layer_count = self.model_config.block_count;
        if layer_count == 0 {
            anyhow::bail!("model_config.block_count must be > 0");
        }
        self.allocate_kv_cache(max_seq_len, layer_count)
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
