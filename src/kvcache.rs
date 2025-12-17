// src/cuda/kvcache.rs

use super::ffi;
use crate::cuda::CudaContext;
use anyhow::{bail, Result};
use half::f16;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// Key/value cache storing K and V for each sequence and token.
///
/// Layout (elements): `[seq][token][head][head_dim]`.
///
/// Index formula:
/// `idx = (((seq * max_seq_len + token) * num_heads + head) * head_dim + dim)`.
#[derive(Debug, Clone)]
pub struct KVCache {
    inner: Arc<KVCacheInner>,
}

#[derive(Debug)]
struct KVCacheInner {
    max_seq_len: u32,
    max_batch_size: u32,
    num_heads: u32,
    head_dim: u32,

    #[cfg(feature = "cuda")]
    raw: NonNull<ffi::M40llmKVCache>,

    #[cfg(not(feature = "cuda"))]
    k: Mutex<Vec<f16>>,
    #[cfg(not(feature = "cuda"))]
    v: Mutex<Vec<f16>>,
    #[cfg(not(feature = "cuda"))]
    seq_lens: Mutex<Vec<u32>>,
}

impl KVCache {
    pub fn new_with_context(
        cuda: &CudaContext,
        max_seq_len: u32,
        max_batch_size: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<Self> {
        if max_seq_len == 0 || max_batch_size == 0 || num_heads == 0 || head_dim == 0 {
            bail!("KVCache::new_with_context: all dims must be non-zero");
        }

        #[cfg(feature = "cuda")]
        {
            let raw = unsafe { cuda.kvcache_create(max_seq_len, max_batch_size, num_heads, head_dim) };
            let raw = NonNull::new(raw)
                .ok_or_else(|| anyhow::anyhow!("m40llm_kvcache_create returned null"))?;
            Ok(Self {
                inner: Arc::new(KVCacheInner {
                    max_seq_len,
                    max_batch_size,
                    num_heads,
                    head_dim,
                    raw,
                }),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            let elems = (max_seq_len as usize)
                .saturating_mul(max_batch_size as usize)
                .saturating_mul(num_heads as usize)
                .saturating_mul(head_dim as usize);
            Ok(Self {
                inner: Arc::new(KVCacheInner {
                    max_seq_len,
                    max_batch_size,
                    num_heads,
                    head_dim,
                    k: Mutex::new(vec![f16::from_f32(0.0); elems]),
                    v: Mutex::new(vec![f16::from_f32(0.0); elems]),
                    seq_lens: Mutex::new(vec![0u32; max_batch_size as usize]),
                }),
            })
        }
    }

    #[inline]
    pub fn max_seq_len(&self) -> u32 {
        self.inner.max_seq_len
    }

    #[inline]
    pub fn max_batch_size(&self) -> u32 {
        self.inner.max_batch_size
    }

    #[inline]
    pub fn num_heads(&self) -> u32 {
        self.inner.num_heads
    }

    #[inline]
    pub fn head_dim(&self) -> u32 {
        self.inner.head_dim
    }

    #[inline]
    pub fn elems_per_token(&self) -> usize {
        (self.inner.num_heads as usize) * (self.inner.head_dim as usize)
    }

    /// Append one token's K/V (provided as f32) for `seq_id`.
    ///
    /// # Safety
    /// `k_dev_f32` and `v_dev_f32` must point to at least `elems_per_token()` contiguous `f32`.
    pub unsafe fn append_token_f32(
        &self,
        cuda: &CudaContext,
        seq_id: u32,
        k_dev_f32: *const c_void,
        v_dev_f32: *const c_void,
    ) -> Result<()> {
        if seq_id >= self.inner.max_batch_size {
            bail!("append_token_f32: seq_id {} out of range", seq_id);
        }
        if k_dev_f32.is_null() || v_dev_f32.is_null() {
            bail!("append_token_f32: null k/v pointer");
        }

        #[cfg(feature = "cuda")]
        {
            let rc = cuda.kvcache_append_token_f32(
                self.inner.raw.as_ptr(),
                seq_id,
                k_dev_f32,
                v_dev_f32,
            );
            if rc != ffi::CUDA_SUCCESS {
                bail!("m40llm_kvcache_append_token_f32 failed (rc={})", rc);
            }
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            let elems = self.elems_per_token();
            let k_in = std::slice::from_raw_parts(k_dev_f32 as *const f32, elems);
            let v_in = std::slice::from_raw_parts(v_dev_f32 as *const f32, elems);

            let mut lens = self.inner.seq_lens.lock().unwrap();
            let cur_len = lens[seq_id as usize] as usize;
            if cur_len >= self.inner.max_seq_len as usize {
                bail!("append_token_f32: seq {} exceeds max_seq_len", seq_id);
            }

            let base = (((seq_id as usize) * (self.inner.max_seq_len as usize) + cur_len) * elems) as usize;

            let mut k = self.inner.k.lock().unwrap();
            let mut v = self.inner.v.lock().unwrap();
            for i in 0..elems {
                k[base + i] = f16::from_f32(k_in[i]);
                v[base + i] = f16::from_f32(v_in[i]);
            }

            lens[seq_id as usize] = (cur_len as u32) + 1;
            Ok(())
        }
    }

    /// Compute attention for the *last token* using stored K/V and input Q.
    ///
    /// CUDA build: delegates to the GPU kernel.
    /// Non-CUDA build: computes a slow reference implementation on CPU.
    ///
    /// # Safety
    /// `q_dev_f32` must point to `elems_per_token()` f32 values.
    /// `out_dev_f32` must point to `elems_per_token()` writable f32 values.
    pub unsafe fn attention_last_token_f32(
        &self,
        cuda: &CudaContext,
        seq_id: u32,
        q_dev_f32: *const c_void,
        seq_len: u32,
        out_dev_f32: *mut c_void,
    ) -> Result<()> {
        if seq_id >= self.inner.max_batch_size {
            bail!("attention_last_token_f32: seq_id {} out of range", seq_id);
        }
        if q_dev_f32.is_null() || out_dev_f32.is_null() {
            bail!("attention_last_token_f32: null q/out pointer");
        }
        if seq_len == 0 {
            bail!("attention_last_token_f32: seq_len must be > 0");
        }
        if seq_len > self.inner.max_seq_len {
            bail!("attention_last_token_f32: seq_len {} > max_seq_len {}", seq_len, self.inner.max_seq_len);
        }

        #[cfg(feature = "cuda")]
        {
            let rc = cuda.attention_last_token_f32(
                self.inner.raw.as_ptr(),
                seq_id,
                q_dev_f32,
                seq_len,
                out_dev_f32,
            );
            if rc != ffi::CUDA_SUCCESS {
                bail!("m40llm_attention_last_token_f32 failed (rc={})", rc);
            }
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Reference CPU attention: per-head softmax(q·k/sqrt(d)) @ v
            let elems = self.elems_per_token();
            let q = std::slice::from_raw_parts(q_dev_f32 as *const f32, elems);
            let out = std::slice::from_raw_parts_mut(out_dev_f32 as *mut f32, elems);

            let num_heads = self.inner.num_heads as usize;
            let head_dim = self.inner.head_dim as usize;
            let t_max = seq_len as usize;

            // Zero output
            for o in out.iter_mut() {
                *o = 0.0;
            }

            let k = self.inner.k.lock().unwrap();
            let v = self.inner.v.lock().unwrap();

            let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

            // For each head, compute attention weights over tokens
            for h in 0..num_heads {
                let q_off = h * head_dim;

                // 1) scores
                let mut scores: Vec<f32> = Vec::with_capacity(t_max);
                let mut max_score = f32::NEG_INFINITY;
                for t in 0..t_max {
                    let base = (((seq_id as usize) * (self.inner.max_seq_len as usize) + t) * elems)
                        + q_off;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * k[base + d].to_f32();
                    }
                    let s = dot * inv_sqrt_d;
                    if s > max_score {
                        max_score = s;
                    }
                    scores.push(s);
                }

                // 2) softmax
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_score).exp();
                    sum += *s;
                }
                if sum == 0.0 {
                    continue;
                }
                let inv_sum = 1.0f32 / sum;

                // 3) weighted sum of V
                for t in 0..t_max {
                    let w = scores[t] * inv_sum;
                    let base = (((seq_id as usize) * (self.inner.max_seq_len as usize) + t) * elems)
                        + q_off;
                    for d in 0..head_dim {
                        out[q_off + d] += w * v[base + d].to_f32();
                    }
                }
            }
            Ok(())
        }
    }

    /// Debug helper: read one token's K/V into host-side f16 buffers.
    ///
    /// CUDA build: uses `m40llm_kvcache_debug_read_token`.
    /// Non-CUDA build: copies directly from the host vectors.
    pub fn debug_read_token_f16(&self, cuda: &CudaContext, seq_id: u32, token: u32) -> Result<(Vec<f16>, Vec<f16>)> {
        if seq_id >= self.inner.max_batch_size {
            bail!("debug_read_token_f16: seq_id out of range");
        }
        if token >= self.inner.max_seq_len {
            bail!("debug_read_token_f16: token out of range");
        }
        let elems = self.elems_per_token();

        #[cfg(feature = "cuda")]
        {
            let mut k = vec![f16::from_f32(0.0); elems];
            let mut v = vec![f16::from_f32(0.0); elems];
            let rc = unsafe {
                cuda.kvcache_debug_read_token(
                    self.inner.raw.as_ptr(),
                    seq_id,
                    token,
                    k.as_mut_ptr() as *mut c_void,
                    v.as_mut_ptr() as *mut c_void,
                )
            };
            if rc != ffi::CUDA_SUCCESS {
                bail!("m40llm_kvcache_debug_read_token failed (rc={})", rc);
            }
            Ok((k, v))
        }

        #[cfg(not(feature = "cuda"))]
        {
            let elems_per_token = elems;
            let base = (((seq_id as usize) * (self.inner.max_seq_len as usize) + token as usize) * elems_per_token) as usize;
            let k_src = self.inner.k.lock().unwrap();
            let v_src = self.inner.v.lock().unwrap();
            Ok((k_src[base..base + elems_per_token].to_vec(), v_src[base..base + elems_per_token].to_vec()))
        }
    }
}

impl Drop for KVCacheInner {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            ffi::m40llm_kvcache_destroy(self.raw.as_ptr());
        }
    }
}
