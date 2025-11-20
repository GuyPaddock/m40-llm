// src/cuda.rs
use anyhow::Result;
use std::ffi::c_void;

// Minimal CPU-side stub of the CUDA context so the crate builds and tests run
// in environments without CUDA/nvcc. Later this can be swapped with real FFI.
#[derive(Clone, Copy, Debug)]
pub struct CudaContext {
    pub device_id: i32,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new(device_id: i32) -> Result<Self> {
        Ok(Self { device_id })
    }

    pub fn upload_weights(&self, _data: &[u8]) -> Result<*mut c_void> {
        // In the stub, we don't actually upload to device; just return null.
        Ok(std::ptr::null_mut())
    }

    pub fn gemm_f16_f32(
        &self,
        _d_a: *const c_void,
        _d_b: *const c_void,
        _d_c: *mut c_void,
        _m: i32,
        _n: i32,
        _k: i32,
    ) -> Result<()> {
        Ok(())
    }

    pub fn start_persistent_decode(&self) -> Result<()> { Ok(()) }
    pub fn stop_persistent_decode(&self) -> Result<()> { Ok(()) }
}

#[derive(Debug, Clone)]
pub struct KVCache {
    pub max_seq_len: u32,
    pub max_batch_size: u32,
    pub num_heads: u32,
    pub head_dim: u32,
}

impl KVCache {
    pub fn new(_device_id: i32, max_seq_len: u32, max_batch_size: u32, num_heads: u32, head_dim: u32) -> Result<Self> {
        Ok(KVCache { max_seq_len, max_batch_size, num_heads, head_dim })
    }
}

// Host-pinned ring buffer stub. In non-CUDA environments, we just heap-allocate.
pub struct SharedRing<T> {
    pub ptr: *mut T,
    pub len: usize,
}

impl<T> SharedRing<T> {
    pub fn new(count: usize) -> Result<Self> {
        let mut v: Vec<T> = Vec::with_capacity(count);
        let ptr = v.as_mut_ptr();
        std::mem::forget(v); // leak capacity; fine for test stubs
        Ok(Self { ptr, len: count })
    }
}