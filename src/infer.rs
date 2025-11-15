// src/infer.rs
use crate::cuda::CudaContext;
use crate::gguf::{GgufModel};
use anyhow::Result;
use std::ffi::c_void;

pub struct LoadedModel {
    pub gguf: GgufModel,
    pub d_weights: *mut c_void, // for now: one big blob
    pub cuda: CudaContext,
}

impl LoadedModel {
    pub fn from_gguf(gguf: GgufModel, gguf_bytes: Vec<u8>, device_id: i32) -> Result<Self> {
        let cuda = CudaContext::new(device_id)?;
        let d_weights = cuda.upload_weights(&gguf_bytes)?;
        Ok(Self {
            gguf,
            d_weights,
            cuda,
        })
    }

    // Super toy: given a token embedding vector (f16 on device),
    // run a single linear layer using weights from GGUF and return logits.
    pub fn forward_one_token(
        &self,
        d_input_f16: *const c_void,
        m: i32,
        n: i32,
        k: i32,
        d_output_f16: *mut c_void,
    ) -> Result<()> {
        self.cuda.gemm_f16_f32(
            self.d_weights as *const c_void,
            d_input_f16,
            d_output_f16,
            m,
            n,
            k,
        )
    }
}
