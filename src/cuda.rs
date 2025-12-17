// src/cuda.rs
#![allow(clippy::missing_safety_doc)]

use anyhow::{bail, Result};
use std::ffi::c_void;

mod kvcache;
pub use kvcache::KVCache;

// -------------------------------------------------------------------------------------------------
// FFI (always declared; implementation comes from either kernels.cu or stub.c)
// -------------------------------------------------------------------------------------------------

#[allow(non_camel_case_types)]
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
        pub fn m40llm_current_device_props(
            name_buf: *mut ::std::os::raw::c_char,
            buf_len: usize,
            major: *mut i32,
            minor: *mut i32,
            device_id: *mut i32,
        ) -> i32;

        pub fn m40llm_create_context(device_id: i32) -> *mut M40llmCudaContext;
        pub fn m40llm_destroy_context(ctx: *mut M40llmCudaContext);

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

        // Optional helpers (may be stubbed on non-CUDA builds)
        pub fn m40llm_f16_to_f32(
            ctx: *mut M40llmCudaContext,
            d_in_f16: *const c_void,
            d_out_f32: *mut c_void,
            n: usize,
        ) -> i32;
        pub fn m40llm_q80_to_f32(
            ctx: *mut M40llmCudaContext,
            d_in_q80: *const c_void,
            d_out_f32: *mut c_void,
            n: usize,
        ) -> i32;

        // KV cache
        pub fn m40llm_kvcache_create(
            ctx: *mut M40llmCudaContext,
            max_seq_len: u32,
            max_batch_size: u32,
            num_heads: u32,
            head_dim: u32,
        ) -> *mut M40llmKVCache;
        pub fn m40llm_kvcache_destroy(kv: *mut M40llmKVCache);
        pub fn m40llm_kvcache_append_token(
            ctx: *mut M40llmCudaContext,
            kv: *mut M40llmKVCache,
            seq_id: u32,
            k_dev: *const c_void,
            v_dev: *const c_void,
        ) -> i32;
        pub fn m40llm_kvcache_append_token_f32(
            ctx: *mut M40llmCudaContext,
            kv: *mut M40llmKVCache,
            seq_id: u32,
            k_dev_f32: *const c_void,
            v_dev_f32: *const c_void,
        ) -> i32;
        pub fn m40llm_kvcache_debug_read_token(
            ctx: *mut M40llmCudaContext,
            kv: *mut M40llmKVCache,
            seq_id: u32,
            token: u32,
            out_k_host: *mut c_void,
            out_v_host: *mut c_void,
        ) -> i32;

        pub fn m40llm_attention_last_token_f32(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            seq_id: u32,
            q_dev_f32: *const c_void,
            seq_len: u32,
            out_dev_f32: *mut c_void,
        ) -> i32;

        // Persistent decode (optional)
        pub fn m40llm_start_persistent_decode(ctx: *mut M40llmCudaContext) -> i32;
        pub fn m40llm_stop_persistent_decode(ctx: *mut M40llmCudaContext) -> i32;
    }

    pub const CUDA_SUCCESS: i32 = 0;
}

// -------------------------------------------------------------------------------------------------
// CudaContext
// -------------------------------------------------------------------------------------------------

/// A handle to the CUDA-side context. When the `cuda` feature is disabled, this
/// is still constructible, but most methods return an error.
#[derive(Debug, Clone)]
pub struct CudaContext {
    inner: std::sync::Arc<CudaContextInner>,
}

#[derive(Debug)]
struct CudaContextInner {
    device_id: i32,
    #[cfg(feature = "cuda")]
    raw: std::ptr::NonNull<ffi::M40llmCudaContext>,
}

impl CudaContext {
    pub fn new(device_id: i32) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let raw = unsafe { ffi::m40llm_create_context(device_id) };
            let raw = std::ptr::NonNull::new(raw)
                .ok_or_else(|| anyhow::anyhow!("m40llm_create_context returned null"))?;
            Ok(Self {
                inner: std::sync::Arc::new(CudaContextInner { device_id, raw }),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Non-CUDA builds still create a context so the rest of the code can run.
            // The C stub returns a non-null sentinel; we do not store it.
            let _ = unsafe { ffi::m40llm_create_context(device_id) };
            Ok(Self {
                inner: std::sync::Arc::new(CudaContextInner { device_id }),
            })
        }
    }

    pub fn device_id(&self) -> i32 {
        self.inner.device_id
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn raw_ptr(&self) -> *mut ffi::M40llmCudaContext {
        self.inner.raw.as_ptr()
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn raw_ptr(&self) -> *mut ffi::M40llmCudaContext {
        // Stub implementation doesn't require a real pointer.
        std::ptr::null_mut()
    }

    #[cfg(feature = "cuda")]
    pub fn upload_weights(&self, weights: &[u8]) -> Result<*mut c_void> {
        unsafe {
            let mut out: *mut c_void = std::ptr::null_mut();
            let rc = ffi::m40llm_upload_weights(self.raw_ptr(), weights.as_ptr() as _, weights.len(), &mut out);
            if rc != ffi::CUDA_SUCCESS || out.is_null() {
                bail!("m40llm_upload_weights failed (rc={})", rc);
            }
            Ok(out)
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn upload_weights(&self, _weights: &[u8]) -> Result<*mut c_void> {
        bail!("CUDA support not enabled (build without --features cuda)")
    }

    #[cfg(feature = "cuda")]
    pub fn device_malloc(&self, bytes: usize) -> Result<*mut c_void> {
        unsafe {
            let mut out: *mut c_void = std::ptr::null_mut();
            let rc = ffi::m40llm_device_malloc(self.raw_ptr(), bytes, &mut out);
            if rc != ffi::CUDA_SUCCESS || out.is_null() {
                bail!("m40llm_device_malloc({} bytes) failed (rc={})", bytes, rc);
            }
            Ok(out)
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn device_malloc(&self, _bytes: usize) -> Result<*mut c_void> {
        bail!("CUDA support not enabled (build without --features cuda)")
    }

    #[cfg(feature = "cuda")]
    pub unsafe fn device_free(&self, ptr: *mut c_void) -> Result<()> {
        let rc = ffi::m40llm_device_free(self.raw_ptr(), ptr);
        if rc != ffi::CUDA_SUCCESS {
            bail!("m40llm_device_free failed (rc={})", rc);
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub unsafe fn device_free(&self, _ptr: *mut c_void) -> Result<()> {
        bail!("CUDA support not enabled (build without --features cuda)")
    }

    #[cfg(feature = "cuda")]
    pub fn memcpy_h2d(
        &self,
        dst_device: *mut c_void,
        src_host: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        // SAFETY: the caller must ensure pointers are valid for `bytes`.
        let rc = unsafe { ffi::m40llm_memcpy_h2d(self.raw_ptr(), dst_device, src_host, bytes) };
        if rc != ffi::CUDA_SUCCESS {
            bail!("m40llm_memcpy_h2d failed (rc={})", rc);
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn memcpy_h2d(
        &self,
        _dst_device: *mut c_void,
        _src_host: *const c_void,
        _bytes: usize,
    ) -> Result<()> {
        bail!("CUDA support not enabled (build without --features cuda)")
    }

    #[cfg(feature = "cuda")]
    pub fn memcpy_d2h(
        &self,
        dst_host: *mut c_void,
        src_device: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        // SAFETY: the caller must ensure pointers are valid for `bytes`.
        let rc = unsafe { ffi::m40llm_memcpy_d2h(self.raw_ptr(), dst_host, src_device, bytes) };
        if rc != ffi::CUDA_SUCCESS {
            bail!("m40llm_memcpy_d2h failed (rc={})", rc);
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn memcpy_d2h(
        &self,
        _dst_host: *mut c_void,
        _src_device: *const c_void,
        _bytes: usize,
    ) -> Result<()> {
        bail!("CUDA support not enabled (build without --features cuda)")
    }

    /// Convert an f16 buffer to f32 on device.
    ///
    /// This exists primarily for debugging / reference paths.
    pub fn f16_to_f32(&self, d_in_f16: *const c_void, d_out_f32: *mut c_void, n: usize) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let rc = unsafe { ffi::m40llm_f16_to_f32(self.raw_ptr(), d_in_f16, d_out_f32, n) };
            if rc != ffi::CUDA_SUCCESS {
                bail!("m40llm_f16_to_f32 failed (rc={})", rc);
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_in_f16, d_out_f32, n);
            bail!("CUDA support not enabled (build without --features cuda)")
        }
    }

    /// Convert a Q8_0 row (GGUF/ggml format) to f32 on device.
    pub fn q80_to_f32(&self, d_in_q80: *const c_void, d_out_f32: *mut c_void, n: usize) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let rc = unsafe { ffi::m40llm_q80_to_f32(self.raw_ptr(), d_in_q80, d_out_f32, n) };
            if rc != ffi::CUDA_SUCCESS {
                bail!("m40llm_q80_to_f32 failed (rc={})", rc);
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_in_q80, d_out_f32, n);
            bail!("CUDA support not enabled (build without --features cuda)")
        }
    }

    /// GEMM: C = A * B with A/B stored as f16 and accumulate into f32 C.
    ///
    /// # Safety
    /// Device pointers must be valid and sized for (M x K), (K x N), (M x N).
    #[cfg(feature = "cuda")]
    pub unsafe fn gemm_f16_f32(
        &self,
        d_a_f16: *const c_void,
        d_b_f16: *const c_void,
        d_c_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        let rc = ffi::m40llm_gemm_f16_storage_f32_compute(self.raw_ptr(), d_a_f16, d_b_f16, d_c_f32, m, n, k);
        if rc != ffi::CUDA_SUCCESS {
            bail!("m40llm_gemm_f16_storage_f32_compute failed (rc={})", rc);
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub unsafe fn gemm_f16_f32(
        &self,
        _d_a_f16: *const c_void,
        _d_b_f16: *const c_void,
        _d_c_f32: *mut c_void,
        _m: i32,
        _n: i32,
        _k: i32,
    ) -> Result<()> {
        bail!("CUDA support not enabled (build without --features cuda)")
    }

    #[cfg(feature = "cuda")]
    pub fn start_persistent_decode(&self) -> Result<()> {
        let rc = unsafe { ffi::m40llm_start_persistent_decode(self.raw_ptr()) };
        if rc != ffi::CUDA_SUCCESS {
            bail!("m40llm_start_persistent_decode failed (rc={})", rc);
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn start_persistent_decode(&self) -> Result<()> {
        bail!("CUDA support not enabled (build without --features cuda)")
    }

    #[cfg(feature = "cuda")]
    pub fn stop_persistent_decode(&self) -> Result<()> {
        let rc = unsafe { ffi::m40llm_stop_persistent_decode(self.raw_ptr()) };
        if rc != ffi::CUDA_SUCCESS {
            bail!("m40llm_stop_persistent_decode failed (rc={})", rc);
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn stop_persistent_decode(&self) -> Result<()> {
        bail!("CUDA support not enabled (build without --features cuda)")
    }

    /// Convert a contiguous f16 vector to f32 on device.
    ///
    /// # Safety
    /// `d_in_f16` and `d_out_f32` must be valid device pointers.
    #[cfg(feature = "cuda")]
    pub fn f16_to_f32(&self, d_in_f16: *const c_void, d_out_f32: *mut c_void, n: usize) -> Result<()> {
        let rc = unsafe { ffi::m40llm_f16_to_f32(self.raw_ptr(), d_in_f16, d_out_f32, n) };
        if rc != ffi::CUDA_SUCCESS {
            bail!("m40llm_f16_to_f32 failed (rc={})", rc);
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn f16_to_f32(&self, _d_in_f16: *const c_void, _d_out_f32: *mut c_void, _n: usize) -> Result<()> {
        bail!("CUDA support not enabled (build without --features cuda)")
    }

    /// Convert a contiguous Q8_0 row to f32 on device (if implemented by CUDA side).
    #[cfg(feature = "cuda")]
    pub fn q80_to_f32(&self, d_in_q80: *const c_void, d_out_f32: *mut c_void, n: usize) -> Result<()> {
        let rc = unsafe { ffi::m40llm_q80_to_f32(self.raw_ptr(), d_in_q80, d_out_f32, n) };
        if rc != ffi::CUDA_SUCCESS {
            bail!("m40llm_q80_to_f32 failed (rc={})", rc);
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn q80_to_f32(&self, _d_in_q80: *const c_void, _d_out_f32: *mut c_void, _n: usize) -> Result<()> {
        bail!("CUDA support not enabled (build without --features cuda)")
    }

    // Expose a small subset of FFI for KVCache.
    pub(crate) unsafe fn kvcache_create(
        &self,
        max_seq_len: u32,
        max_batch_size: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> *mut ffi::M40llmKVCache {
        ffi::m40llm_kvcache_create(self.raw_ptr(), max_seq_len, max_batch_size, num_heads, head_dim)
    }

    pub(crate) unsafe fn kvcache_destroy(&self, kv: *mut ffi::M40llmKVCache) {
        let _ = self; // keep signature symmetric
        ffi::m40llm_kvcache_destroy(kv)
    }

    pub(crate) unsafe fn kvcache_append_token_f32(
        &self,
        kv: *mut ffi::M40llmKVCache,
        seq_id: u32,
        k_dev_f32: *const c_void,
        v_dev_f32: *const c_void,
    ) -> i32 {
        ffi::m40llm_kvcache_append_token_f32(self.raw_ptr(), kv, seq_id, k_dev_f32, v_dev_f32)
    }

    pub(crate) unsafe fn attention_last_token_f32(
        &self,
        kv: *const ffi::M40llmKVCache,
        seq_id: u32,
        q_dev_f32: *const c_void,
        seq_len: u32,
        out_dev_f32: *mut c_void,
    ) -> i32 {
        ffi::m40llm_attention_last_token_f32(self.raw_ptr(), kv, seq_id, q_dev_f32, seq_len, out_dev_f32)
    }

    pub(crate) unsafe fn kvcache_debug_read_token(
        &self,
        kv: *mut ffi::M40llmKVCache,
        seq_id: u32,
        token: u32,
        out_k_host: *mut c_void,
        out_v_host: *mut c_void,
    ) -> i32 {
        ffi::m40llm_kvcache_debug_read_token(self.raw_ptr(), kv, seq_id, token, out_k_host, out_v_host)
    }
}

impl Drop for CudaContextInner {
    fn drop(&mut self) {
        // Only destroy the context when built with CUDA enabled.
        #[cfg(feature = "cuda")]
        unsafe {
            ffi::m40llm_destroy_context(self.raw.as_ptr());
        }
        #[cfg(not(feature = "cuda"))]
        unsafe {
            // stub: nothing to do (the stub returns a sentinel pointer)
            let _ = self;
        }
    }
}

