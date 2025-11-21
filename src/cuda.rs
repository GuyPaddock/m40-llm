// src/cuda.rs
use anyhow::{anyhow, Result};
use std::ffi::c_void;

#[cfg(feature = "cuda")]
mod ffi {
    use super::*;
    #[repr(C)]
    pub struct M40llmCudaContext {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct M40llmKVCache {
        _private: [u8; 0],
    }

    extern "C" {
        pub fn m40llm_device_malloc(
            ctx: *mut M40llmCudaContext,
            bytes: usize,
            out_ptr: *mut *mut c_void,
        ) -> i32;
        pub fn m40llm_device_free(ctx: *mut M40llmCudaContext, ptr: *mut c_void) -> i32;
        pub fn m40llm_memcpy_h2d(
            ctx: *mut M40llmCudaContext,
            dst_device: *mut c_void,
            src_host: *const c_void,
            bytes: usize,
        ) -> i32;
        pub fn m40llm_memcpy_d2h(
            ctx: *mut M40llmCudaContext,
            dst_host: *mut c_void,
            src_device: *const c_void,
            bytes: usize,
        ) -> i32;

        pub fn m40llm_create_context(device_id: i32) -> *mut M40llmCudaContext;
        pub fn m40llm_destroy_context(ctx: *mut M40llmCudaContext);

        pub fn m40llm_upload_weights(
            ctx: *mut M40llmCudaContext,
            host_ptr: *const c_void,
            num_bytes: usize,
            out_device_ptr: *mut *mut c_void,
        ) -> i32;

        pub fn m40llm_gemm_f16_storage_f32_compute(
            ctx: *mut M40llmCudaContext,
            d_A: *const c_void,
            d_B: *const c_void,
            d_C: *mut c_void,
            M: i32,
            N: i32,
            K: i32,
        ) -> i32;

        pub fn m40llm_kvcache_create(
            ctx: *mut M40llmCudaContext,
            max_seq_len: u32,
            max_batch_size: u32,
            num_heads: u32,
            head_dim: u32,
        ) -> *mut M40llmKVCache;
        pub fn m40llm_kvcache_append_token(
            ctx: *mut M40llmCudaContext,
            kv: *mut M40llmKVCache,
            seq_id: u32,
            k_dev: *const c_void,
            v_dev: *const c_void,
        ) -> i32;

        pub fn m40llm_kvcache_destroy(kv: *mut M40llmKVCache);

        pub fn m40llm_start_persistent_decode(ctx: *mut M40llmCudaContext) -> i32;
        pub fn m40llm_stop_persistent_decode(ctx: *mut M40llmCudaContext) -> i32;
    }
}

// Public-safe wrapper types usable in both CUDA and non-CUDA builds
#[derive(Clone, Copy, Debug)]
pub struct CudaContext {
    pub device_id: i32,
    #[cfg(feature = "cuda")]
    raw: *mut ffi::M40llmCudaContext,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new(device_id: i32) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let ptr = unsafe { ffi::m40llm_create_context(device_id) };
            if ptr.is_null() {
                return Err(anyhow!("m40llm_create_context returned null"));
            }
            Ok(Self {
                device_id,
                raw: ptr,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self { device_id })
        }
    }
}

#[cfg(feature = "cuda")]
impl CudaContext {
    pub fn device_malloc(&self, bytes: usize) -> Result<*mut c_void> {
        let mut out: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { ffi::m40llm_device_malloc(self.raw, bytes, &mut out as *mut _) };
        if rc != 0 {
            return Err(anyhow!("m40llm_device_malloc failed: {rc}"));
        }
        Ok(out)
    }
    pub fn device_free(&self, ptr: *mut c_void) -> Result<()> {
        let rc = unsafe { ffi::m40llm_device_free(self.raw, ptr) };
        if rc != 0 {
            return Err(anyhow!("m40llm_device_free failed: {rc}"));
        }
        Ok(())
    }
    pub fn memcpy_h2d(
        &self,
        dst_device: *mut c_void,
        src_host: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        let rc = unsafe { ffi::m40llm_memcpy_h2d(self.raw, dst_device, src_host, bytes) };
        if rc != 0 {
            return Err(anyhow!("m40llm_memcpy_h2d failed: {rc}"));
        }
        Ok(())
    }
    pub fn memcpy_d2h(
        &self,
        dst_host: *mut c_void,
        src_device: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        let rc = unsafe { ffi::m40llm_memcpy_d2h(self.raw, dst_host, src_device, bytes) };
        if rc != 0 {
            return Err(anyhow!("m40llm_memcpy_d2h failed: {rc}"));
        }
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
impl CudaContext {
    pub fn device_malloc(&self, _bytes: usize) -> Result<*mut c_void> {
        Ok(std::ptr::null_mut())
    }
    pub fn device_free(&self, _ptr: *mut c_void) -> Result<()> {
        Ok(())
    }
    pub fn memcpy_h2d(
        &self,
        _dst_device: *mut c_void,
        _src_host: *const c_void,
        _bytes: usize,
    ) -> Result<()> {
        Ok(())
    }
    pub fn memcpy_d2h(
        &self,
        _dst_host: *mut c_void,
        _src_device: *const c_void,
        _bytes: usize,
    ) -> Result<()> {
        Ok(())
    }

    pub fn upload_weights(&self, _data: &[u8]) -> Result<*mut c_void> {
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
    pub fn start_persistent_decode(&self) -> Result<()> {
        Ok(())
    }
    pub fn stop_persistent_decode(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl CudaContext {
    pub fn create_kvcache(
        &self,
        max_seq_len: u32,
        max_batch_size: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<*mut ffi::M40llmKVCache> {
        let kv = unsafe {
            ffi::m40llm_kvcache_create(self.raw, max_seq_len, max_batch_size, num_heads, head_dim)
        };
        if kv.is_null() {
            return Err(anyhow!("m40llm_kvcache_create returned null"));
        }
        Ok(kv)
    }

    pub fn upload_weights(&self, data: &[u8]) -> Result<*mut c_void> {
        #[cfg(feature = "cuda")]
        {
            let mut d_ptr: *mut c_void = std::ptr::null_mut();
            let rc = unsafe {
                ffi::m40llm_upload_weights(
                    self.raw,
                    data.as_ptr() as *const _,
                    data.len(),
                    &mut d_ptr as *mut _,
                )
            };
            if rc != 0 {
                return Err(anyhow!("m40llm_upload_weights failed: {rc}"));
            }
            Ok(d_ptr)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(std::ptr::null_mut())
        }
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
        #[cfg(feature = "cuda")]
        {
            let rc = unsafe {
                ffi::m40llm_gemm_f16_storage_f32_compute(self.raw, d_a, d_b, d_c, m, n, k)
            };
            if rc != 0 {
                return Err(anyhow!("m40llm_gemm_f16_storage_f32_compute failed: {rc}"));
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_a, d_b, d_c, m, n, k);
            Ok(())
        }
    }

    pub fn start_persistent_decode(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let rc = unsafe { ffi::m40llm_start_persistent_decode(self.raw) };
            if rc != 0 {
                return Err(anyhow!("m40llm_start_persistent_decode failed: {rc}"));
            }
        }
        Ok(())
    }
    pub fn stop_persistent_decode(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let rc = unsafe { ffi::m40llm_stop_persistent_decode(self.raw) };
            if rc != 0 {
                return Err(anyhow!("m40llm_stop_persistent_decode failed: {rc}"));
            }
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe { ffi::m40llm_destroy_context(self.raw) };
    }
}

#[derive(Debug, Clone)]
pub struct KVCache {
    pub max_seq_len: u32,
    pub max_batch_size: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    #[cfg(feature = "cuda")]
    raw: *mut ffi::M40llmKVCache,
}

impl KVCache {
    pub fn new_with_context(
        ctx: &CudaContext,
        max_seq_len: u32,
        max_batch_size: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let raw = ctx.create_kvcache(max_seq_len, max_batch_size, num_heads, head_dim)?;
            Ok(KVCache {
                max_seq_len,
                max_batch_size,
                num_heads,
                head_dim,
                raw,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ctx;
            Ok(KVCache {
                max_seq_len,
                max_batch_size,
                num_heads,
                head_dim,
            })
        }
    }
}

#[cfg(feature = "cuda")]
impl KVCache {
    pub fn append_token(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        k_dev: *const c_void,
        v_dev: *const c_void,
    ) -> Result<()> {
        let rc =
            unsafe { ffi::m40llm_kvcache_append_token(ctx.raw, self.raw, seq_id, k_dev, v_dev) };
        if rc != 0 {
            return Err(anyhow!("m40llm_kvcache_append_token failed: {rc}"));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for KVCache {
    fn drop(&mut self) {
        unsafe { ffi::m40llm_kvcache_destroy(self.raw) };
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
