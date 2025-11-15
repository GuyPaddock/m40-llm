// src/infer.rs
use crate::gguf::GgufModel;
use crate::cuda::CudaContext;
use anyhow::Result;
use std::ffi::c_void;

pub struct LoadedModel {
    pub gguf: GgufModel,
    /// Base device pointer corresponding to file offset `gguf.data_offset`.
    pub d_data_base: *mut c_void,
    pub cuda: CudaContext,
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
