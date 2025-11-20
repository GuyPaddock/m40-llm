// src/cuda.rs
use anyhow::{Result};
use std::ffi::c_void;

#[repr(C)]
pub struct FastllmCudaContext {
    _private: [u8; 0],
}

extern "C" {
    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;

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

    // Allocate and initialize KV cache
    fn fastllm_allocate_kv_cache(
        kv: *mut KVCache,
        max_seq_len: u32,
        max_batch_size: u32,
        num_heads: u32,
        head_dim: u32,
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

pub struct KVCache {
    pub d_k: *mut half,
    pub d_v: *mut half,
    pub d_seq_map: *mut u32,
    pub max_seq_len: u32,
    pub max_batch_size: u32,
    pub num_heads: u32,
    pub head_dim: u32,
}

impl KVCache {
    pub fn new(device_id: i32, max_seq_len: u32, max_batch_size: u32, num_heads: u32, head_dim: u32) -> Result<Self> {
        let mut kv = KVCache {
            d_k: std::ptr::null_mut(),
            d_v: std::ptr::null_mut(),
            d_seq_map: std::ptr::null_mut(),
            max_seq_len,
            max_batch_size,
            num_heads,
            head_dim,
        };

        let rc = unsafe {
            fastllm_allocate_kv_cache(
                &mut kv as *mut KVCache as *mut c_void,
                max_seq_len,
                max_batch_size,
                num_heads,
                head_dim,
            )
        };

        if rc != 0 {
            anyhow::bail!("KV cache allocation failed with code {rc}");
        }

        Ok(kv)
    }
}

pub struct SharedRing<T> {
    pub ptr: *mut T,
    pub len: usize,
}

impl<T> SharedRing<T> {
    pub fn new(count: usize) -> Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            cudaHostAlloc(&mut ptr, count * std::mem::size_of::<T>(), 0)?;
        }
        Ok(Self {
            ptr: ptr as *mut T,
            len: count,
        })
    }
}