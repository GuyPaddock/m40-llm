// src/cuda.rs
use anyhow::{Result};
use std::ffi::c_void;

#[repr(C)]
pub struct FastllmCudaContext {
    _private: [u8; 0],
}

extern "C" {
    fn fastllm_create_context(device_id: i32) -> *mut FastllmCudaContext;
    fn fastllm_destroy_context(ctx: *mut FastllmCudaContext);

    // Upload weights into device memory
    fn fastllm_upload_weights(
        ctx: *mut FastllmCudaContext,
        host_ptr: *const c_void,
        num_bytes: usize,
        out_device_ptr: *mut *mut c_void,
    ) -> i32;

    // FP16 storage / FP32 compute GEMM: C = A (M×K, f16) * B (K×N, f16)
    fn fastllm_gemm_f16_storage_f32_compute(
        ctx: *mut FastllmCudaContext,
        d_A: *const c_void,
        d_B: *const c_void,
        d_C: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
    ) -> i32;

    // Placeholder for persistent decode kernel control
    fn fastllm_start_persistent_decode(ctx: *mut FastllmCudaContext) -> i32;
    fn fastllm_stop_persistent_decode(ctx: *mut FastllmCudaContext) -> i32;
}

pub struct CudaContext {
    raw: *mut FastllmCudaContext,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new(device_id: i32) -> Result<Self> {
        let raw = unsafe { fastllm_create_context(device_id) };
        if raw.is_null() {
            anyhow::bail!("failed to create CUDA context");
        }
        Ok(Self { raw })
    }

    pub fn upload_weights(&self, data: &[u8]) -> Result<*mut c_void> {
        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            fastllm_upload_weights(
                self.raw,
                data.as_ptr() as *const c_void,
                data.len(),
                &mut dev_ptr as *mut *mut c_void,
            )
        };
        if rc != 0 {
            anyhow::bail!("upload_weights failed with code {rc}");
        }
        Ok(dev_ptr)
    }

    pub fn gemm_f16_f32(
        &self,
        d_a: *const c_void,
        d_b: *const c_void,
        d_c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        let rc = unsafe {
            fastllm_gemm_f16_storage_f32_compute(self.raw, d_a, d_b, d_c, m, n, k)
        };
        if rc != 0 {
            anyhow::bail!("gemm failed: {rc}");
        }
        Ok(())
    }

    pub fn start_persistent_decode(&self) -> Result<()> {
        let rc = unsafe { fastllm_start_persistent_decode(self.raw) };
        if rc != 0 {
            anyhow::bail!("start_persistent_decode failed: {rc}");
        }
        Ok(())
    }

    pub fn stop_persistent_decode(&self) -> Result<()> {
        let rc = unsafe { fastllm_stop_persistent_decode(self.raw) };
        if rc != 0 {
            anyhow::bail!("stop_persistent_decode failed: {rc}");
        }
        Ok(())
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe { fastllm_destroy_context(self.raw) };
    }
}
