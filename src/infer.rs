// src/infer.rs
#![allow(dead_code)]

use crate::cuda::{CudaContext, KVCache};
use crate::gguf::{GgmlDType, GgufModel, GgufTensor};
use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::ffi::c_void;

#[derive(Debug, Clone)]
pub struct DeviceTensorView {
    pub dtype: GgmlDType,
    pub shape: Vec<u64>,
    pub byte_offset: u64, // from start of tensor data region in file
    pub nbytes: usize,    // 0 if unknown dtype sizing
    #[cfg(feature = "cuda")]
    pub dptr: *mut c_void, // base + byte_offset (null in non-CUDA builds)
}

pub struct LoadedModel {
    pub gguf: GgufModel,
    pub cuda: CudaContext,
    pub kv_cache: Option<KVCache>,
    pub device_tensors: HashMap<String, DeviceTensorView>,
    #[cfg(feature = "cuda")]
    pub d_weights_base: *mut c_void,
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
        let weights_bytes = &gguf_bytes[data_off..];
        let d_base = cuda.upload_weights(weights_bytes)?;
        // Validate that all known-sized tensors fit within weights_bytes
        for t in &gguf.tensors {
            if let Some(esize) = dtype_size_bytes(t.dtype) {
                let n_elems: u64 = t.shape.iter().copied().product::<u64>();
                let need = (n_elems as usize)
                    .checked_mul(esize)
                    .context("tensor size overflow")?;
                let start = t.offset as usize;
                let end = start.saturating_add(need);
                if end > weights_bytes.len() {
                    anyhow::bail!(
                        "tensor '{}' overflows weights blob: [{}..{}) > {}",
                        t.name,
                        start,
                        end,
                        weights_bytes.len()
                    );
                }
            }
        }
        let device_tensors = build_device_tensor_views(&gguf.tensors, d_base);
        Ok(Self {
            gguf,
            cuda,
            kv_cache: None,
            device_tensors,
            #[cfg(feature = "cuda")]
            d_weights_base: d_base,
        })
    }
}

fn dtype_size_bytes(dt: GgmlDType) -> Option<usize> {
    match dt {
        GgmlDType::F32 => Some(4),
        GgmlDType::F16 => Some(2),
        _ => None,
    }
}

#[allow(clippy::collapsible_if)]
fn build_device_tensor_views(
    tensors: &[GgufTensor],
    #[allow(unused_variables)] d_base: *mut c_void,
) -> HashMap<String, DeviceTensorView> {
    let mut map = HashMap::with_capacity(tensors.len());
    for t in tensors {
        let n_elems: u64 = t.shape.iter().copied().product::<u64>();
        let sz_opt = dtype_size_bytes(t.dtype).map(|s| (n_elems as usize) * s);
        let view = DeviceTensorView {
            dtype: t.dtype,
            shape: t.shape.clone(),
            byte_offset: t.offset,
            nbytes: sz_opt.unwrap_or(0),
            #[cfg(feature = "cuda")]
            dptr: (d_base as usize + t.offset as usize) as *mut c_void,
        };
        map.insert(t.name.clone(), view);
    }
    map
}

impl LoadedModel {
    // Convenience: Append K/V from host FP32 slices. Copies to device then calls append.
    pub fn append_kv_token_f32_from_host(
        &self,
        seq_id: u32,
        k_host: &[f32],
        v_host: &[f32],
    ) -> Result<()> {
        let kv = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
        let elems = kv.elems_per_token();
        if k_host.len() != elems || v_host.len() != elems {
            anyhow::bail!(
                "append_kv_token_f32_from_host: expected {} elems per token, got k={}, v={}",
                elems,
                k_host.len(),
                v_host.len()
            );
        }
        #[cfg(feature = "cuda")]
        {
            let bytes = elems * std::mem::size_of::<f32>();
            let d_k = self.cuda.device_malloc(bytes)?;
            let d_v = self.cuda.device_malloc(bytes)?;
            unsafe {
                self.cuda
                    .memcpy_h2d(d_k, k_host.as_ptr() as *const c_void, bytes)?;
                self.cuda
                    .memcpy_h2d(d_v, v_host.as_ptr() as *const c_void, bytes)?;
            }
            let res = unsafe {
                self.append_kv_token_f32(seq_id, d_k as *const c_void, d_v as *const c_void)
            };
            // best-effort free even if append errs
            unsafe {
                let _ = self.cuda.device_free(d_k);
                let _ = self.cuda.device_free(d_v);
            }
            res
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Directly write into host-side KVCache
            kv.append_token_f32(
                &self.cuda,
                seq_id,
                k_host.as_ptr() as *const c_void,
                v_host.as_ptr() as *const c_void,
            )
        }
    }

    /// # Safety
    /// `d_a`, `d_b`, and `d_c` must be valid pointers to device buffers sized for GEMM with (m, n, k).
    pub unsafe fn run_gemm(
        &self,
        d_a: *const c_void,
        d_b: *const c_void,
        d_c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        self.cuda.gemm_f16_f32(d_a, d_b, d_c, m, n, k)
    }

    pub fn allocate_kv_cache(&mut self, max_seq_len: u32, max_batch_size: u32) -> Result<()> {
        let kv = KVCache::new_with_context(&self.cuda, max_seq_len, max_batch_size, 8, 64)?;
        self.kv_cache = Some(kv);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// # Safety
    /// `d_q` and `d_out` must be valid pointers to device buffers matching KV cache layout.
    pub unsafe fn run_attention(
        &self,
        d_q: *const c_void,
        d_out: *mut c_void,
        seq_id: u32,
        seq_len: u32,
        dim: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        // Validate layout: dim must equal num_heads * head_dim
        if dim != num_heads.saturating_mul(head_dim) {
            anyhow::bail!(
                "run_attention: dim {} != num_heads {} * head_dim {}",
                dim,
                num_heads,
                head_dim
            );
        }
        let kv = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
        if kv.num_heads() != num_heads || kv.head_dim() != head_dim {
            anyhow::bail!(
                "KVCache layout mismatch: kv has (heads={}, dim={}), requested ({},{})",
                kv.num_heads(),
                kv.head_dim(),
                num_heads,
                head_dim
            );
        }
        #[cfg(feature = "cuda")]
        unsafe {
            kv.attention_last_token_f32(&self.cuda, seq_id, d_q, seq_len, d_out)
        }
        #[cfg(not(feature = "cuda"))]
        {
            kv.attention_last_token_f32(&self.cuda, seq_id, d_q, seq_len, d_out)
        }
    }

    /// # Safety
    /// `d_k_f32` and `d_v_f32` must be valid pointers to device buffers containing one token's K/V in f32 layout.
    pub unsafe fn append_kv_token_f32(
        &self,
        seq_id: u32,
        d_k_f32: *const c_void,
        d_v_f32: *const c_void,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let kv = self
                .kv_cache
                .as_ref()
                .ok_or_else(|| anyhow!("kv_cache not allocated; call allocate_kv_cache first"))?;
            #[cfg(feature = "cuda")]
            unsafe {
                kv.append_token_f32(&self.cuda, seq_id, d_k_f32, d_v_f32)
            }
            #[cfg(not(feature = "cuda"))]
            {
                kv.append_token_f32(&self.cuda, seq_id, d_k_f32, d_v_f32)
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (seq_id, d_k_f32, d_v_f32);
            Ok(())
        }
    }

    pub fn run_mlp(
        &self,
        _d_in: *const c_void,
        _d_out: *mut c_void,
        _batch_seq: u32,
        _dim: u32,
        _hidden_dim: u32,
    ) -> Result<()> {
        // Stub: no-op (activation wiring handled in forward_one_token_minimal for now)
        Ok(())
    }

    /// Minimal forward pass for one token through attention + MLP using existing primitives.
    /// Assumptions:
    /// - Batch size = 1
    /// - Q/K/V and out-proj output dims equal d_model
    /// - Activations and residual adds are performed on host as a temporary fallback
    /// - KV cache must be pre-allocated with matching num_heads/head_dim
    ///
    /// Safety: All device pointers must be valid on this context's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_one_token_minimal(
        &self,
        d_x_f32: *const c_void,
        d_model: i32,
        d_wq_f16: *const c_void,
        d_wk_f16: *const c_void,
        d_wv_f16: *const c_void,
        d_wo_f16: *const c_void,
        d_w_gate_f16: *const c_void,
        d_w_up_f16: *const c_void,
        d_w_down_f16: *const c_void,
        hidden_dim: i32,
        seq_id: u32,
        seq_len: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        if d_model <= 0 || hidden_dim <= 0 {
            anyhow::bail!("forward_one_token_minimal: invalid dims");
        }

        #[cfg(feature = "cuda")]
        {
            // Allocate scratch buffers
            let bytes_d = (d_model as usize) * 4;
            let bytes_h = (hidden_dim as usize) * 4;
            let dq = self.cuda.device_malloc(bytes_d)?;
            let dk = self.cuda.device_malloc(bytes_d)?;
            let dv = self.cuda.device_malloc(bytes_d)?;
            let datt = self.cuda.device_malloc(bytes_d)?; // attention output
            let dy_attn = self.cuda.device_malloc(bytes_d)?; // after out-proj
            let dgate = self.cuda.device_malloc(bytes_h)?;
            let dup = self.cuda.device_malloc(bytes_h)?;
            let dhid = self.cuda.device_malloc(bytes_h)?;
            let dy_mlp = self.cuda.device_malloc(bytes_d)?;

            let res = (|| -> Result<()> {
                // Q/K/V projections (m=1, k=d_model, n=d_model)
                self.qkv_project_f32xf16_f32(
                    d_x_f32, 1, d_model, d_wq_f16, d_model, d_wk_f16, d_model, d_wv_f16, d_model,
                    dq, dk, dv,
                )?;

                // Append K/V for this token, then attention over last token
                self.append_kv_token_f32(seq_id, dk as *const c_void, dv as *const c_void)?;

                // Use KV cache layout to validate/run attention
                let kv = self.kv_cache.as_ref().ok_or_else(|| {
                    anyhow!("kv_cache not allocated; call allocate_kv_cache first")
                })?;
                let num_heads = kv.num_heads();
                let head_dim = kv.head_dim();
                self.run_attention(
                    dq as *const c_void,
                    datt,
                    seq_id,
                    seq_len,
                    d_model as u32,
                    num_heads,
                    head_dim,
                )?;

                // Output projection of attention
                self.out_proj_f32xf16_f32(
                    datt as *const c_void,
                    d_wo_f16,
                    dy_attn,
                    1,
                    d_model,
                    d_model,
                )?;

                // MLP gates and up
                self.mlp_gates_f32xf16_f32(
                    d_x_f32,
                    1,
                    d_model,
                    d_w_gate_f16,
                    d_w_up_f16,
                    hidden_dim,
                    dgate,
                    dup,
                )?;

                // Host fallback for elementwise: hidden = sigmoid(gate) * up
                let mut h_gate = vec![0u8; bytes_h];
                let mut h_up = vec![0u8; bytes_h];
                self.cuda.memcpy_d2h(
                    h_gate.as_mut_ptr() as *mut c_void,
                    dgate as *const c_void,
                    bytes_h,
                )?;
                self.cuda.memcpy_d2h(
                    h_up.as_mut_ptr() as *mut c_void,
                    dup as *const c_void,
                    bytes_h,
                )?;
                let mut gate_f = Vec::with_capacity(hidden_dim as usize);
                let mut up_f = Vec::with_capacity(hidden_dim as usize);
                for ch in h_gate.chunks_exact(4) {
                    gate_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
                }
                for ch in h_up.chunks_exact(4) {
                    up_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
                }
                fn sigmoid(v: f32) -> f32 {
                    1.0 / (1.0 + (-v).exp())
                }
                let mut hid_f = Vec::with_capacity(hidden_dim as usize);
                for i in 0..(hidden_dim as usize) {
                    hid_f.push(sigmoid(gate_f[i]) * up_f[i]);
                }
                // Copy hidden back to device
                let mut h_hid_bytes = Vec::with_capacity(bytes_h);
                for v in &hid_f {
                    h_hid_bytes.extend_from_slice(&v.to_le_bytes());
                }
                self.cuda
                    .memcpy_h2d(dhid, h_hid_bytes.as_ptr() as *const c_void, bytes_h)?;

                // Down projection
                self.mlp_down_proj_f32xf16_f32(
                    dhid as *const c_void,
                    1,
                    hidden_dim,
                    d_w_down_f16,
                    d_model,
                    dy_mlp,
                )?;

                // Residual add on host: out = x + y_attn + y_mlp
                let mut h_x = vec![0u8; bytes_d];
                let mut h_attn = vec![0u8; bytes_d];
                let mut h_mlp = vec![0u8; bytes_d];
                self.cuda
                    .memcpy_d2h(h_x.as_mut_ptr() as *mut c_void, d_x_f32, bytes_d)?;
                self.cuda.memcpy_d2h(
                    h_attn.as_mut_ptr() as *mut c_void,
                    dy_attn as *const c_void,
                    bytes_d,
                )?;
                self.cuda.memcpy_d2h(
                    h_mlp.as_mut_ptr() as *mut c_void,
                    dy_mlp as *const c_void,
                    bytes_d,
                )?;
                let mut x_f = Vec::with_capacity(d_model as usize);
                let mut attn_f = Vec::with_capacity(d_model as usize);
                let mut mlp_f = Vec::with_capacity(d_model as usize);
                for ch in h_x.chunks_exact(4) {
                    x_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
                }
                for ch in h_attn.chunks_exact(4) {
                    attn_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
                }
                for ch in h_mlp.chunks_exact(4) {
                    mlp_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
                }
                let mut y_f = Vec::with_capacity(d_model as usize);
                for i in 0..(d_model as usize) {
                    y_f.push(x_f[i] + attn_f[i] + mlp_f[i]);
                }
                let mut y_bytes = Vec::with_capacity(bytes_d);
                for v in &y_f {
                    y_bytes.extend_from_slice(&v.to_le_bytes());
                }
                self.cuda
                    .memcpy_h2d(d_out_f32, y_bytes.as_ptr() as *const c_void, bytes_d)?;

                Ok(())
            })();

            // Free scratch (best-effort)
            let _ = self.cuda.device_free(dq);
            let _ = self.cuda.device_free(dk);
            let _ = self.cuda.device_free(dv);
            let _ = self.cuda.device_free(datt);
            let _ = self.cuda.device_free(dy_attn);
            let _ = self.cuda.device_free(dgate);
            let _ = self.cuda.device_free(dup);
            let _ = self.cuda.device_free(dhid);
            let _ = self.cuda.device_free(dy_mlp);

            return res;
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                d_x_f32,
                d_model,
                d_wq_f16,
                d_wk_f16,
                d_wv_f16,
                d_wo_f16,
                d_w_gate_f16,
                d_w_up_f16,
                d_w_down_f16,
                hidden_dim,
                seq_id,
                seq_len,
                d_out_f32,
            );
            Ok(())
        }
    }

    /// # Safety
    /// MLP projections (no activation): computes gate = X·W_gate and up = X·W_up.
    /// All matrices are row-major; X is f32 (MxK), W_* are f16 (KxH), outputs are f32 (MxH).
    pub unsafe fn mlp_gates_f32xf16_f32(
        &self,
        d_x_f32: *const c_void,
        m: i32,
        k: i32,
        d_w_gate_f16: *const c_void,
        d_w_up_f16: *const c_void,
        h: i32,
        d_gate_out_f32: *mut c_void,
        d_up_out_f32: *mut c_void,
    ) -> Result<()> {
        if m <= 0 || k <= 0 || h <= 0 {
            anyhow::bail!("mlp_gates: invalid dims");
        }
        self.matmul_f32xf16_f32(d_x_f32, d_w_gate_f16, d_gate_out_f32, m, h, k)?;
        self.matmul_f32xf16_f32(d_x_f32, d_w_up_f16, d_up_out_f32, m, h, k)
    }

    /// # Safety
    /// Down projection: Y = H · W_down, where H is hidden f32 (MxH), W_down is f16 (H x N), output f32 (MxN)
    pub unsafe fn mlp_down_proj_f32xf16_f32(
        &self,
        d_hidden_f32: *const c_void,
        m: i32,
        h: i32,
        d_w_down_f16: *const c_void,
        n: i32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        if m <= 0 || h <= 0 || n <= 0 {
            anyhow::bail!("mlp_down_proj: invalid dims");
        }
        self.matmul_f32xf16_f32(d_hidden_f32, d_w_down_f16, d_out_f32, m, n, h)
    }

    pub fn run_rms_norm(
        &self,
        _d_in: *const c_void,
        _d_out: *mut c_void,
        _seq_len: u32,
        _dim: u32,
        _eps: f32,
    ) -> Result<()> {
        // Stub: no-op
        Ok(())
    }

    pub fn forward_one_token(
        &self,
        _d_input_f16: *const c_void,
        _m: i32,
        _n: i32,
        _k: i32,
        _d_output_f16: *mut c_void,
    ) -> Result<()> {
        // Stub: no-op
        Ok(())
    }

    // Start integrating GEMM matmul sites: generic wrappers usable for Q/K/V, MLP, and output projection
    /// # Safety
    /// f16 × f16 → f32 row-major GEMM. Device pointers must be valid on the same device as the context.
    pub unsafe fn matmul_f16xf16_f32(
        &self,
        d_a_f16: *const c_void,
        d_b_f16: *const c_void,
        d_c_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        self.cuda
            .gemm_f16xf16_f32(d_a_f16, d_b_f16, d_c_f32, m, n, k)
    }

    /// # Safety
    /// f32 × f16 → f32 row-major GEMM, useful when input activations are f32 and weights are f16.
    pub unsafe fn matmul_f32xf16_f32(
        &self,
        d_a_f32: *const c_void,
        d_b_f16: *const c_void,
        d_c_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        self.cuda
            .gemm_f32xf16_f32(d_a_f32, d_b_f16, d_c_f32, m, n, k)
    }
}

impl LoadedModel {
    /// # Safety
    /// A f32 (MxK) × Wq/Wk/Wv f16 (KxNq/KxNk/KxNv) → Q/K/V f32 (MxN*)
    pub unsafe fn qkv_project_f32xf16_f32(
        &self,
        d_x_f32: *const c_void,
        m: i32,
        k: i32,
        d_wq_f16: *const c_void,
        n_q: i32,
        d_wk_f16: *const c_void,
        n_k: i32,
        d_wv_f16: *const c_void,
        n_v: i32,
        d_q_out_f32: *mut c_void,
        d_k_out_f32: *mut c_void,
        d_v_out_f32: *mut c_void,
    ) -> Result<()> {
        if m <= 0 || k <= 0 || n_q <= 0 || n_k <= 0 || n_v <= 0 {
            anyhow::bail!("qkv_project: invalid dims");
        }
        self.matmul_f32xf16_f32(d_x_f32, d_wq_f16, d_q_out_f32, m, n_q, k)?;
        self.matmul_f32xf16_f32(d_x_f32, d_wk_f16, d_k_out_f32, m, n_k, k)?;
        self.matmul_f32xf16_f32(d_x_f32, d_wv_f16, d_v_out_f32, m, n_v, k)
    }

    /// # Safety
    /// A f32 (MxK) × W f16 (KxN) → Out f32 (MxN)
    pub unsafe fn out_proj_f32xf16_f32(
        &self,
        d_in_f32: *const c_void,
        d_w_out_f16: *const c_void,
        d_out_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        if m <= 0 || n <= 0 || k <= 0 {
            anyhow::bail!("out_proj: invalid dims");
        }
        self.matmul_f32xf16_f32(d_in_f32, d_w_out_f16, d_out_f32, m, n, k)
    }

    /// # Safety
    /// Generic projection helper A f32 (MxK) × W f16 (KxN) → C f32 (MxN)
    pub unsafe fn project_f32xf16_f32(
        &self,
        d_a_f32: *const c_void,
        d_b_f16: *const c_void,
        d_c_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        self.matmul_f32xf16_f32(d_a_f32, d_b_f16, d_c_f32, m, n, k)
    }
}
