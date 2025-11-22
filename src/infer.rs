// src/infer.rs
#![cfg_attr(not(feature = "server"), allow(dead_code))]

use crate::cuda::{CudaContext, KVCache};
use crate::gguf::GgufModel;
use anyhow::{anyhow, Result};
use std::ffi::c_void;

pub struct LoadedModel {
    pub gguf: GgufModel,
    pub d_data_base: *mut c_void,
    pub cuda: CudaContext,
    pub kv_cache: Option<KVCache>,
}

impl LoadedModel {
    pub fn from_gguf(gguf: GgufModel, gguf_bytes: Vec<u8>, device_id: i32) -> Result<Self> {
        let cuda = CudaContext::new(device_id)?;
        let data_off = gguf.data_offset as usize;
        if data_off > gguf_bytes.len() {
            anyhow::bail!(
                "GGUF data_offset {} beyond file size {}",
                data_off,
                gguf_bytes.len()
            );
        }
        let weights_bytes = &gguf_bytes[data_off..];
        let d_data_base = cuda.upload_weights(weights_bytes)?;
        Ok(Self {
            gguf,
            d_data_base,
            cuda,
            kv_cache: None,
        })
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
        let bytes = elems * std::mem::size_of::<f32>();
        let d_k = self.cuda.device_malloc(bytes)?;
        let d_v = self.cuda.device_malloc(bytes)?;
        self.cuda
            .memcpy_h2d(d_k, k_host.as_ptr() as *const c_void, bytes)?;
        self.cuda
            .memcpy_h2d(d_v, v_host.as_ptr() as *const c_void, bytes)?;
        let res = self.append_kv_token_f32(seq_id, d_k as *const c_void, d_v as *const c_void);
        // best-effort free even if append errs
        let _ = self.cuda.device_free(d_k);
        let _ = self.cuda.device_free(d_v);
        res
    }

    pub fn run_gemm(
        &self,
        d_a: *const c_void,
        d_b: *const c_void,
        d_c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        self.cuda.gemm_f16_f32(d_a, d_b, d_c, m, n, k)
    }

    pub fn allocate_kv_cache(&mut self, max_seq_len: u32, max_batch_size: u32) -> Result<()> {
        let kv = KVCache::new_with_context(&self.cuda, max_seq_len, max_batch_size, 8, 64)?;
        self.kv_cache = Some(kv);
        Ok(())
    }

    pub fn run_attention(
        &self,
        _d_q: *const c_void,
        _d_out: *mut c_void,
        _seq_id: u32,
        _seq_len: u32,
        _dim: u32,
        _num_heads: u32,
        _head_dim: u32,
    ) -> Result<()> {
        // Stub: no-op
        Ok(())
    }

    pub fn append_kv_token_f32(
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
            kv.append_token_f32(&self.cuda, seq_id, d_k_f32, d_v_f32)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (seq_id, d_k_f32, d_v_f32);
            Ok(())
        }
    }

    pub fn run_mlp(
        &self,
        _d_in: *const c_void,
        _d_out: *mut c_void,
        _batch_seq: u32,
        _dim: u32,
        _hidden_dim: u32,
    ) -> Result<()> {
        // Stub: no-op
        Ok(())
    }

    pub fn run_rms_norm(
        &self,
        _d_in: *const c_void,
        _d_out: *mut c_void,
        _seq_len: u32,
        _dim: u32,
        _eps: f32,
    ) -> Result<()> {
        // Stub: no-op
        Ok(())
    }

    pub fn forward_one_token(
        &self,
        _d_input_f16: *const c_void,
        _m: i32,
        _n: i32,
        _k: i32,
        _d_output_f16: *mut c_void,
    ) -> Result<()> {
        // Stub: no-op
        Ok(())
    }
}
