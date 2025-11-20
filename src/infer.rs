// src/infer.rs
use crate::gguf::GgufModel;
use crate::cuda::{CudaContext, KVCache};
use anyhow::Result;
use std::ffi::c_void;

pub struct LoadedModel {
    pub gguf: GgufModel,
    /// Base device pointer corresponding to file offset `gguf.data_offset`.
    pub d_data_base: *mut c_void,
    pub cuda: CudaContext,
    /// Optional KV cache for attention
    pub kv_cache: Option<KVCache>,
    /// Device pointers to weights - wrapped in safe structs for Send trait
    pub d_q_weights: DevicePointer,
    pub d_kv_weights: DevicePointer,
    pub d_gate_weights: DevicePointer,
    pub d_up_weights: DevicePointer,
    pub d_down_weights: DevicePointer,
    pub d_seq_map: DevicePointerMut,
}

pub struct DevicePointer(*const c_void);
pub struct DevicePointerMut(*mut u32);

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

        // Upload only the tensor data region to GPU
        let weights_bytes = &gguf_bytes[data_off..];
        let d_data_base = cuda.upload_weights(weights_bytes)?;

        Ok(Self {
            gguf,
            d_data_base,
            cuda,
        })
    }
    // Later, when we want a specific tensor, we'll compute:
    // let gg = &self.gguf.tensors[idx];
    // let device_ptr_for_tensor = unsafe {
    //     (self.d_data_base as *mut u8).add(gg.offset as usize) as *mut c_void
    // };

    // Run a GEMM operation using model weights
    pub fn run_gemm(
        &self,
        d_A: *const c_void,
        d_B: *const c_void,
        d_C: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        self.cuda.gemm_f16_f32(
            d_A,
            d_B,
            d_C,
            m,
            n,
            k,
        )
    }

    // Allocate and initialize KV cache
    pub fn allocate_kv_cache(&self, max_seq_len: u32, max_batch_size: u32) -> Result<()> {
        let kv = KVCache::new(self.cuda.device_id, max_seq_len, max_batch_size, 8, 64)?;
        self.kv_cache = Some(kv);
        Ok(())
    }

    // Run attention operation
    pub fn run_attention(
        &self,
        d_q: *const c_void,
        d_out: *mut c_void,
        seq_id: u32,
        seq_len: u32,
        dim: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        if self.kv_cache.is_none() {
            anyhow::bail!("KV cache not allocated");
        }

        // Launch attention kernel
        let kv = self.kv_cache.as_ref().unwrap();
        cuda::attention_f16(
            d_q as *const half,
            d_out as *mut half,
            kv,
            self.d_q_weights.0,
            self.d_kv_weights.0,
            seq_id,
            seq_len,
            dim,
            num_heads,
            head_dim,
        );
        Ok(())
    }

    // Run MLP operation
    pub fn run_mlp(
        &self,
        d_in: *const c_void,
        d_out: *mut c_void,
        batch_seq: u32,
        dim: u32,
        hidden_dim: u32,
    ) -> Result<()> {
        // Launch MLP kernel
        cuda::mlp_swiglu_f16(
            d_in as *const half,
            d_out as *mut half,
            self.d_gate_weights.0 as *const float,
            self.d_up_weights.0 as *const float,
            self.d_down_weights.0 as *const float,
            batch_seq,
            dim,
            hidden_dim,
        );
        Ok(())
    }

    // Run RMSNorm operation
    pub fn run_rms_norm(
        &self,
        d_in: *const c_void,
        d_out: *mut c_void,
        seq_len: u32,
        dim: u32,
        eps: f32,
    ) -> Result<()> {
        // Launch RMSNorm kernel
        cuda::rms_norm_f32(
            d_in as *const float,
            d_out as *mut float,
            self.d_seq_map.0 as *const u32,
            seq_len,
            dim,
            eps,
        );
        Ok(())
    }

    // Run a GEMM operation using model weights
    pub fn forward_one_token(
        &self,
        d_input_f16: *const c_void,
        m: i32,
        n: i32,
        k: i32,
        d_output_f16: *mut c_void,
    ) -> Result<()> {
        self.run_gemm(
            self.d_q_weights.0 as *const c_void, // Updated to use d_q_weights as this seems more appropriate based on context
            d_input_f16,
            d_output_f16,
            m,
            n,
            k,
        )
    }
}
