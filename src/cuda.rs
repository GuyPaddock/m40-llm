// src/cuda.rs
#[cfg(feature = "cuda")]
use anyhow::anyhow;
use anyhow::Result;
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

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
            out_k_f16: *mut c_void,
            out_v_f16: *mut c_void,
        ) -> i32;

        pub fn m40llm_kvcache_destroy(kv: *mut M40llmKVCache);

        pub fn m40llm_attention_last_token_f32(
            ctx: *mut M40llmCudaContext,
            kv: *const M40llmKVCache,
            seq_id: u32,
            q_dev_f32: *const c_void,
            seq_len: u32,
            out_dev_f32: *mut c_void,
        ) -> i32;

        pub fn m40llm_start_persistent_decode(ctx: *mut M40llmCudaContext) -> i32;
        pub fn m40llm_stop_persistent_decode(ctx: *mut M40llmCudaContext) -> i32;
    }
}

// Public-safe wrapper types usable in both CUDA and non-CUDA builds
#[derive(Debug, Clone)]
pub struct CudaContext {
    #[allow(dead_code)]
    inner: Arc<CudaContextInner>,
}

#[derive(Debug)]
struct CudaContextInner {
    device_id: i32,
    #[allow(dead_code)]
    lock: Mutex<()>,
    #[cfg(feature = "cuda")]
    raw: NonNull<ffi::M40llmCudaContext>,
    #[cfg(feature = "cuda")]
    weights_ptr: Mutex<Option<NonNull<c_void>>>,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaContextInner {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaContextInner {}

impl CudaContext {
    pub fn new(device_id: i32) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let ptr = unsafe { ffi::m40llm_create_context(device_id) };
            let raw =
                NonNull::new(ptr).ok_or_else(|| anyhow!("m40llm_create_context returned null"))?;
            Ok(Self {
                inner: Arc::new(CudaContextInner {
                    device_id,
                    lock: Mutex::new(()),
                    raw,
                    weights_ptr: Mutex::new(None),
                }),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self {
                inner: Arc::new(CudaContextInner {
                    device_id,
                    lock: Mutex::new(()),
                    #[cfg(feature = "cuda")]
                    raw: NonNull::dangling(),
                    #[cfg(feature = "cuda")]
                    weights_ptr: Mutex::new(None),
                }),
            })
        }
    }

    #[allow(dead_code)]
    pub fn device_id(&self) -> i32 {
        self.inner.device_id
    }
}

#[cfg(feature = "cuda")]
impl CudaContext {
    pub fn device_malloc(&self, bytes: usize) -> Result<*mut c_void> {
        let _g = self.inner.lock.lock().unwrap();
        let mut out: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            ffi::m40llm_device_malloc(self.inner.raw.as_ptr(), bytes, &mut out as *mut _)
        };
        if rc != 0 {
            return Err(anyhow!("m40llm_device_malloc failed: {rc}"));
        }
        Ok(out)
    }
    /// # Safety
    /// `ptr` must be a valid device pointer previously allocated by `device_malloc` or the CUDA runtime.
    /// The memory must not be used after this call and must belong to this context/device.
    pub unsafe fn device_free(&self, ptr: *mut c_void) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        let rc = unsafe { ffi::m40llm_device_free(self.inner.raw.as_ptr(), ptr) };
        if rc != 0 {
            return Err(anyhow!("m40llm_device_free failed: {rc}"));
        }
        Ok(())
    }
    /// # Safety
    /// `dst_device` must be a valid, writable device pointer to at least `bytes` bytes on this context's device.
    /// `src_host` must be a valid, readable host pointer to at least `bytes` bytes.
    pub unsafe fn memcpy_h2d(
        &self,
        dst_device: *mut c_void,
        src_host: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        let rc =
            unsafe { ffi::m40llm_memcpy_h2d(self.inner.raw.as_ptr(), dst_device, src_host, bytes) };
        if rc != 0 {
            return Err(anyhow!("m40llm_memcpy_h2d failed: {rc}"));
        }
        Ok(())
    }
    /// # Safety
    /// `dst_host` must be a valid, writable host pointer to at least `bytes` bytes.
    /// `src_device` must be a valid, readable device pointer to at least `bytes` bytes on this context's device.
    pub unsafe fn memcpy_d2h(
        &self,
        dst_host: *mut c_void,
        src_device: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        let rc =
            unsafe { ffi::m40llm_memcpy_d2h(self.inner.raw.as_ptr(), dst_host, src_device, bytes) };
        if rc != 0 {
            return Err(anyhow!("m40llm_memcpy_d2h failed: {rc}"));
        }
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
impl CudaContext {
    #[allow(dead_code)]
    pub fn device_malloc(&self, _bytes: usize) -> Result<*mut c_void> {
        let _g = self.inner.lock.lock().unwrap();
        let _ = self.inner.device_id;
        Ok(std::ptr::null_mut())
    }
    #[allow(dead_code)]
    pub fn device_free(&self, _ptr: *mut c_void) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        let _ = self.inner.device_id;
        Ok(())
    }
    #[allow(dead_code)]
    pub fn memcpy_h2d(
        &self,
        _dst_device: *mut c_void,
        _src_host: *const c_void,
        _bytes: usize,
    ) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        Ok(())
    }
    #[allow(dead_code)]
    pub fn memcpy_d2h(
        &self,
        _dst_host: *mut c_void,
        _src_device: *const c_void,
        _bytes: usize,
    ) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        Ok(())
    }
    #[allow(dead_code)]
    pub fn upload_weights(&self, _data: &[u8]) -> Result<*mut c_void> {
        let _g = self.inner.lock.lock().unwrap();
        Ok(std::ptr::null_mut())
    }
    #[allow(dead_code)]
    pub fn gemm_f16_f32(
        &self,
        _d_a: *const c_void,
        _d_b: *const c_void,
        _d_c: *mut c_void,
        _m: i32,
        _n: i32,
        _k: i32,
    ) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        Ok(())
    }
    #[allow(dead_code)]
    pub fn start_persistent_decode(&self) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
        Ok(())
    }
    #[allow(dead_code)]
    pub fn stop_persistent_decode(&self) -> Result<()> {
        let _g = self.inner.lock.lock().unwrap();
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
        let _g = self.inner.lock.lock().unwrap();
        let kv = unsafe {
            ffi::m40llm_kvcache_create(
                self.inner.raw.as_ptr(),
                max_seq_len,
                max_batch_size,
                num_heads,
                head_dim,
            )
        };
        if kv.is_null() {
            return Err(anyhow!("m40llm_kvcache_create returned null"));
        }
        Ok(kv)
    }

    pub fn upload_weights(&self, data: &[u8]) -> Result<*mut c_void> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            // Free any previously uploaded weights to avoid leaks on re-upload
            if let Some(prev) = self.inner.weights_ptr.lock().unwrap().take() {
                unsafe {
                    let _ = ffi::m40llm_device_free(self.inner.raw.as_ptr(), prev.as_ptr());
                }
            }
            let mut d_ptr: *mut c_void = std::ptr::null_mut();
            let rc = unsafe {
                ffi::m40llm_upload_weights(
                    self.inner.raw.as_ptr(),
                    data.as_ptr() as *const _,
                    data.len(),
                    &mut d_ptr as *mut _,
                )
            };
            if rc != 0 || d_ptr.is_null() {
                return Err(anyhow!("m40llm_upload_weights failed: {rc}"));
            }
            // Track ownership inside the context so it can be freed on drop
            let mut slot = self.inner.weights_ptr.lock().unwrap();
            *slot = NonNull::new(d_ptr);
            Ok(d_ptr)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(std::ptr::null_mut())
        }
    }

    /// # Safety
    /// `d_a`, `d_b`, and `d_c` must be valid device pointers on this context's device.
    /// Dimensions m, n, k must match the underlying buffer shapes.
    pub unsafe fn gemm_f16_f32(
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
            let _g = self.inner.lock.lock().unwrap();
            let rc = unsafe {
                ffi::m40llm_gemm_f16_storage_f32_compute(
                    self.inner.raw.as_ptr(),
                    d_a,
                    d_b,
                    d_c,
                    m,
                    n,
                    k,
                )
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
            let _g = self.inner.lock.lock().unwrap();
            let rc = unsafe { ffi::m40llm_start_persistent_decode(self.inner.raw.as_ptr()) };
            if rc != 0 {
                return Err(anyhow!("m40llm_start_persistent_decode failed: {rc}"));
            }
        }
        Ok(())
    }
    pub fn stop_persistent_decode(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let _g = self.inner.lock.lock().unwrap();
            let rc = unsafe { ffi::m40llm_stop_persistent_decode(self.inner.raw.as_ptr()) };
            if rc != 0 {
                return Err(anyhow!("m40llm_stop_persistent_decode failed: {rc}"));
            }
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaContextInner {
    fn drop(&mut self) {
        // Free tracked weights if present to avoid device memory leak
        if let Ok(inner) = self.weights_ptr.get_mut() {
            if let Some(ptr) = inner.take() {
                unsafe {
                    let _ = ffi::m40llm_device_free(self.raw.as_ptr(), ptr.as_ptr());
                }
            }
        }
        // SAFETY: raw was constructed from a non-null FFI pointer and is only freed here when Arc count drops to 0
        unsafe { ffi::m40llm_destroy_context(self.raw.as_ptr()) };
    }
}

#[derive(Debug, Clone)]
pub struct KVCache {
    inner: Arc<KVCacheInner>,
}

impl KVCache {
    pub fn num_heads(&self) -> u32 {
        self.inner.num_heads
    }
    pub fn head_dim(&self) -> u32 {
        self.inner.head_dim
    }
}

#[derive(Debug)]
struct KVCacheInner {
    // Layout: [seq][token][head][head_dim]
    // - seq in [0, max_batch_size)
    // - token in [0, max_seq_len)
    // - head in [0, num_heads)
    // - head_dim in [0, head_dim)
    // Strides (elements):
    //   elems_per_token = num_heads * head_dim
    //   base(seq, token) = (seq * max_seq_len + token) * elems_per_token
    #[allow(dead_code)]
    //   index(seq, token, head, dim) = base + head * head_dim + dim
    max_seq_len: u32,
    _max_batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    #[cfg(feature = "cuda")]
    raw: NonNull<ffi::M40llmKVCache>,
    #[cfg(not(feature = "cuda"))]
    k: Mutex<Vec<half::f16>>, // length = max_seq_len * max_batch_size * elems_per_token
    #[cfg(not(feature = "cuda"))]
    v: Mutex<Vec<half::f16>>, // same length as k
    #[cfg(not(feature = "cuda"))]
    len_by_seq: Mutex<Vec<u32>>, // current length per sequence
}

#[cfg(feature = "cuda")]
unsafe impl Send for KVCacheInner {}
#[cfg(feature = "cuda")]
unsafe impl Sync for KVCacheInner {}

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
                inner: Arc::new(KVCacheInner {
                    max_seq_len,
                    _max_batch_size: max_batch_size,
                    num_heads,
                    head_dim,
                    raw: NonNull::new(raw).expect("non-null kv from ffi"),
                    #[cfg(not(feature = "cuda"))]
                    k: Mutex::new(Vec::new()),
                    #[cfg(not(feature = "cuda"))]
                    v: Mutex::new(Vec::new()),
                    #[cfg(not(feature = "cuda"))]
                    len_by_seq: Mutex::new(Vec::new()),
                }),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ctx;
            let elems_per_token = (num_heads as usize) * (head_dim as usize);
            let total_tokens = (max_seq_len as usize) * (max_batch_size as usize);
            let cap = total_tokens * elems_per_token;
            Ok(KVCache {
                inner: Arc::new(KVCacheInner {
                    max_seq_len,
                    _max_batch_size: max_batch_size,
                    num_heads,
                    head_dim,
                    k: Mutex::new(vec![half::f16::from_f32(0.0); cap]),
                    v: Mutex::new(vec![half::f16::from_f32(0.0); cap]),
                    len_by_seq: Mutex::new(vec![0u32; max_batch_size as usize]),
                    #[cfg(feature = "cuda")]
                    raw: NonNull::dangling(),
                }),
            })
        }
    }

    #[inline]
    pub fn elems_per_token(&self) -> usize {
        (self.inner.num_heads as usize) * (self.inner.head_dim as usize)
    }

    #[inline]
    pub fn base_offset_elems(&self, seq: u32, token: u32) -> usize {
        ((seq as usize) * (self.inner.max_seq_len as usize) + (token as usize))
            * self.elems_per_token()
    }

    #[cfg(not(feature = "cuda"))]
    fn append_token_host(
        &self,
        seq_id: u32,
        k_f32: *const c_void,
        v_f32: *const c_void,
    ) -> Result<()> {
        let elems = self.elems_per_token();
        // Determine token index from len_by_seq
        let mut lens = self.inner.len_by_seq.lock().unwrap();
        let token = lens[seq_id as usize];
        if token >= self.inner.max_seq_len {
            return Err(anyhow::anyhow!("append_token_host: seq {} full", seq_id));
        }
        // Safety: caller promises k_f32/v_f32 are valid pointers to elems f32 entries
        let k_slice = unsafe { std::slice::from_raw_parts(k_f32 as *const f32, elems) };
        let v_slice = unsafe { std::slice::from_raw_parts(v_f32 as *const f32, elems) };
        let base = self.base_offset_elems(seq_id, token);
        let mut k_lock = self.inner.k.lock().unwrap();
        let mut v_lock = self.inner.v.lock().unwrap();
        for i in 0..elems {
            k_lock[base + i] = half::f16::from_f32(k_slice[i]);
            v_lock[base + i] = half::f16::from_f32(v_slice[i]);
        }
        lens[seq_id as usize] += 1;
        Ok(())
    }

    #[inline]
    #[allow(dead_code)]
    pub fn index_elems(&self, seq: u32, token: u32, head: u32, dim: u32) -> usize {
        self.base_offset_elems(seq, token)
            + (head as usize) * (self.inner.head_dim as usize)
            + (dim as usize)
    }
}

#[cfg(feature = "cuda")]
impl KVCache {
    /// # Safety
    /// `k_dev` and `v_dev` must be valid device pointers containing one token's worth of K/V in f16 layout.
    /// `seq_id` must be in range. Context must target same device as this cache.
    pub unsafe fn append_token(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        k_dev: *const c_void,
        v_dev: *const c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_kvcache_append_token(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                k_dev,
                v_dev,
            )
        };
        if rc != 0 {
            return Err(anyhow!("m40llm_kvcache_append_token failed: {rc}"));
        }
        Ok(())
    }

    /// # Safety
    /// `k_dev_f32` and `v_dev_f32` must be valid device pointers containing one token's worth of K/V in f32 layout.
    /// They will be converted to f16 in-place in the cache. Context/device must match.
    pub unsafe fn append_token_f32(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        k_dev_f32: *const c_void,
        v_dev_f32: *const c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_kvcache_append_token_f32(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                k_dev_f32,
                v_dev_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!("m40llm_kvcache_append_token_f32 failed: {rc}"));
        }
        Ok(())
    }

    /// # Safety
    /// `d_q_f32` and `d_out_f32` must be valid device pointers. `seq_len` must not exceed already appended tokens.
    /// Context/device must match. Shapes must align with KV cache configuration.
    pub unsafe fn attention_last_token_f32(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        seq_len: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let _g = ctx.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_attention_last_token_f32(
                ctx.inner.raw.as_ptr(),
                self.inner.raw.as_ptr(),
                seq_id,
                d_q_f32,
                seq_len,
                d_out_f32,
            )
        };
        if rc != 0 {
            return Err(anyhow!("m40llm_attention_last_token_f32 failed: {}", rc));
        }
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
impl KVCache {
    pub fn append_token(
        &self,
        _ctx: &CudaContext,
        seq_id: u32,
        k_dev: *const c_void,
        v_dev: *const c_void,
    ) -> Result<()> {
        self.append_token_host(seq_id, k_dev, v_dev)
    }
    pub fn append_token_f32(
        &self,
        _ctx: &CudaContext,
        seq_id: u32,
        k_dev_f32: *const c_void,
        v_dev_f32: *const c_void,
    ) -> Result<()> {
        self.append_token_host(seq_id, k_dev_f32, v_dev_f32)
    }
    pub fn attention_last_token_f32(
        &self,
        _ctx: &CudaContext,
        seq_id: u32,
        d_q_f32: *const c_void,
        seq_len: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        // Pure CPU reference implementation operating on host f16 K/V, f32 compute
        let elems = self.elems_per_token();
        let num_heads = self.inner.num_heads as usize;
        let head_dim = self.inner.head_dim as usize;
        let q = unsafe { std::slice::from_raw_parts(d_q_f32 as *const f32, elems) };
        let k_lock = self.inner.k.lock().unwrap();
        let v_lock = self.inner.v.lock().unwrap();
        let mut out = vec![0f32; elems];
        let inv_sqrt = 1.0f32 / (head_dim as f32).sqrt();
        for h in 0..num_heads {
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut max_s = f32::NEG_INFINITY;
            for t in 0..(seq_len as usize) {
                let base = self.base_offset_elems(seq_id, t as u32) + h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let kf = k_lock[base + d].to_f32();
                    dot += qh[d] * kf;
                }
                let s = dot * inv_sqrt;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut denom = 0.0f32;
            let mut scores = vec![0.0f32; seq_len as usize];
            for (t, score) in scores.iter_mut().enumerate().take(seq_len as usize) {
                let base = self.base_offset_elems(seq_id, t as u32) + h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let kf = k_lock[base + d].to_f32();
                    dot += qh[d] * kf;
                }
                let s = dot * inv_sqrt;
                let e = (s - max_s).exp();
                *score = e;
                denom += e;
            }
            if denom == 0.0 {
                denom = 1.0;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, prob) in scores
                    .iter()
                    .map(|s| s / denom)
                    .enumerate()
                    .take(seq_len as usize)
                {
                    let vbase = self.base_offset_elems(seq_id, t as u32) + h * head_dim;
                    acc += prob * v_lock[vbase + d].to_f32();
                }
                out[h * head_dim + d] = acc;
            }
        }
        // Write back to out_dev
        let out_slice = unsafe { std::slice::from_raw_parts_mut(d_out_f32 as *mut f32, elems) };
        out_slice.copy_from_slice(&out);
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for KVCacheInner {
    fn drop(&mut self) {
        unsafe { ffi::m40llm_kvcache_destroy(self.raw.as_ptr()) };
    }
}

// Public test/debug helper to read back one KV token (FP16) via FFI.
// Only available when the CUDA feature is enabled.
#[cfg(feature = "cuda")]
/// # Safety
/// `out_k_f16` and `out_v_f16` must be valid pointers to write one token's K and V (num_heads*head_dim f16 each).
/// `seq_id`/`token` must be within appended ranges; the context and cache must be on the same device.
pub unsafe fn ffi_debug_read_kv_token(
    ctx: &CudaContext,
    kv: &KVCache,
    seq_id: u32,
    token: u32,
    out_k_f16: *mut u8,
    out_v_f16: *mut u8,
) -> i32 {
    let _g = ctx.inner.lock.lock().unwrap();
    ffi::m40llm_kvcache_debug_read_token(
        ctx.inner.raw.as_ptr(),
        kv.inner.raw.as_ptr(),
        seq_id,
        token,
        out_k_f16 as *mut c_void,
        out_v_f16 as *mut c_void,
    )
}

// Host-pinned ring buffer stub. In non-CUDA environments, we just heap-allocate.
pub struct SharedRing<T> {
    pub ptr: *mut T,
    pub len: usize,
}

impl<T> SharedRing<T> {
    #[allow(dead_code)]
    pub fn new(count: usize) -> Result<Self> {
        let mut v: Vec<T> = Vec::with_capacity(count);
        let ptr = v.as_mut_ptr();
        std::mem::forget(v); // leak capacity; fine for test stubs
        Ok(Self { ptr, len: count })
    }
}
