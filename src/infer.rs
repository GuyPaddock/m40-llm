// src/infer.rs
#![allow(dead_code)]

use crate::cuda::{CudaContext, KVCache};
use crate::gguf::{GgmlDType, GgufModel, GgufTensor};
use anyhow::{anyhow, Context, Result};
#[cfg(feature = "cuda")]
use half::f16;
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

#[derive(Debug, Clone)]
pub struct StandardLayerWeights {
    pub d_model: usize,
    pub hidden_dim: usize,
    pub tok_embeddings: DeviceTensorView,
    pub wq: DeviceTensorView,
    pub wk: DeviceTensorView,
    pub wv: DeviceTensorView,
    pub wo: DeviceTensorView,
    pub w_gate: DeviceTensorView,
    pub w_up: DeviceTensorView,
    pub w_down: DeviceTensorView,
}

impl LoadedModel {
    fn get_u32_meta(&self, key: &str) -> Option<u32> {
        use crate::gguf::{GgufScalar, GgufValue};
        self.gguf.metadata.get(key).and_then(|v| match v {
            GgufValue::Scalar(GgufScalar::U32(x)) => Some(*x),
            _ => None,
        })
    }

    fn device_tensor(&self, name: &str) -> Option<&DeviceTensorView> {
        self.device_tensors.get(name)
    }

    fn find_tensor_any<'a>(&'a self, candidates: &[String]) -> Result<&'a DeviceTensorView> {
        for c in candidates {
            if let Some(v) = self.device_tensor(c) {
                return Ok(v);
            }
        }
        anyhow::bail!("missing required tensor; tried: {}", candidates.join(", "))
    }

    /// Map standard LLaMA-style GGUF tensor names for a single layer.
    /// Supports both layers.N.* and blk.N.* naming variants.
    pub fn map_standard_layer(&self, layer: usize) -> Result<StandardLayerWeights> {
        use crate::gguf::GgmlDType;
        // Determine d_model from metadata or embeddings tensor shape
        let d_model_meta = self
            .get_u32_meta("llama.embedding_length")
            .map(|x| x as usize);
        let tok_name = "tok_embeddings.weight".to_string();
        let tok = self
            .device_tensor(&tok_name)
            .ok_or_else(|| anyhow!("missing tensor: {}", tok_name))?;
        let d_model =
            d_model_meta.unwrap_or_else(|| tok.shape.get(1).copied().unwrap_or(0) as usize);
        if d_model == 0 {
            anyhow::bail!("could not determine d_model")
        }
        // Candidates for layer names
        let wq_names = vec![
            format!("layers.{layer}.attention.wq.weight"),
            format!("blk.{layer}.attn_q.weight"),
        ];
        let wk_names = vec![
            format!("layers.{layer}.attention.wk.weight"),
            format!("blk.{layer}.attn_k.weight"),
        ];
        let wv_names = vec![
            format!("layers.{layer}.attention.wv.weight"),
            format!("blk.{layer}.attn_v.weight"),
        ];
        let wo_names = vec![
            format!("layers.{layer}.attention.wo.weight"),
            format!("blk.{layer}.attn_output.weight"),
        ];
        // LLaMA convention: w1=up, w3=gate, w2=down
        let w_gate_names = vec![
            format!("layers.{layer}.feed_forward.w3.weight"),
            format!("blk.{layer}.ffn_gate.weight"),
        ];
        let w_up_names = vec![
            format!("layers.{layer}.feed_forward.w1.weight"),
            format!("blk.{layer}.ffn_up.weight"),
        ];
        let w_down_names = vec![
            format!("layers.{layer}.feed_forward.w2.weight"),
            format!("blk.{layer}.ffn_down.weight"),
        ];

        let wq = self.find_tensor_any(&wq_names)?.clone();
        let wk = self.find_tensor_any(&wk_names)?.clone();
        let wv = self.find_tensor_any(&wv_names)?.clone();
        let wo = self.find_tensor_any(&wo_names)?.clone();
        let w_gate = self.find_tensor_any(&w_gate_names)?.clone();
        let w_up = self.find_tensor_any(&w_up_names)?.clone();
        let w_down = self.find_tensor_any(&w_down_names)?.clone();

        // DType checks: we currently require FP16 weights for GEMM paths
        for (name, t) in [
            ("wq", &wq),
            ("wk", &wk),
            ("wv", &wv),
            ("wo", &wo),
            ("w_gate", &w_gate),
            ("w_up", &w_up),
            ("w_down", &w_down),
        ] {
            if t.dtype != GgmlDType::F16 {
                anyhow::bail!("tensor {} expected F16, got {:?}", name, t.dtype);
            }
        }
        // Shape checks (row-major): X [1 x d_model] · W[K=d_model x N] => out [1 x N]
        let k_q = *wq.shape.first().unwrap_or(&0) as usize;
        let n_q = *wq.shape.get(1).unwrap_or(&0) as usize;
        if k_q != d_model || n_q == 0 {
            anyhow::bail!(
                "wq shape invalid: expected [d_model, N], got {:?}",
                wq.shape
            );
        }
        // Infer hidden_dim from up/gate
        let k_up = *w_up.shape.first().unwrap_or(&0) as usize;
        let h_up = *w_up.shape.get(1).unwrap_or(&0) as usize;
        if k_up != d_model || h_up == 0 {
            anyhow::bail!(
                "w_up shape invalid: expected [d_model, H], got {:?}",
                w_up.shape
            );
        }
        let hidden_dim = h_up;
        // Down must be [hidden_dim, d_model]
        let h_down = *w_down.shape.first().unwrap_or(&0) as usize;
        let n_down = *w_down.shape.get(1).unwrap_or(&0) as usize;
        if h_down != hidden_dim || n_down != d_model {
            anyhow::bail!(
                "w_down shape invalid: expected [hidden_dim, d_model]=[{}, {}], got {:?}",
                hidden_dim,
                d_model,
                w_down.shape
            );
        }

        Ok(StandardLayerWeights {
            d_model,
            hidden_dim,
            tok_embeddings: tok.clone(),
            wq,
            wk,
            wv,
            wo,
            w_gate,
            w_up,
            w_down,
        })
    }
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
        // Backward-compatible default layout
        let kv = KVCache::new_with_context(&self.cuda, max_seq_len, max_batch_size, 8, 64)?;
        self.kv_cache = Some(kv);
        Ok(())
    }

    pub fn allocate_kv_cache_with_layout(
        &mut self,
        max_seq_len: u32,
        max_batch_size: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        if num_heads == 0 || head_dim == 0 {
            anyhow::bail!("allocate_kv_cache_with_layout: invalid layout");
        }
        let kv = KVCache::new_with_context(
            &self.cuda,
            max_seq_len,
            max_batch_size,
            num_heads,
            head_dim,
        )?;
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
    /// # Safety
    /// All device pointers must be valid on this context's device.
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
            let d_xn = self.cuda.device_malloc(bytes_d)?; // pre-attn norm(x)
            let d_x1 = self.cuda.device_malloc(bytes_d)?; // x + attn(xn)
            let d_x1n = self.cuda.device_malloc(bytes_d)?; // norm(x1)

            let res = (|| -> Result<()> {
                // Pre-norm (RMSNorm) on x -> xn
                self.run_rms_norm(d_x_f32, d_xn, 1, d_model as u32, 1e-6)?;

                // Q/K/V projections (m=1, k=d_model, n=d_model)
                self.qkv_project_f32xf16_f32(
                    d_xn, 1, d_model, d_wq_f16, d_model, d_wk_f16, d_model, d_wv_f16, d_model, dq,
                    dk, dv,
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

                // Residual add y_attn: x1 = x + y_attn (host fallback)
                {
                    let mut h_x = vec![0u8; bytes_d];
                    let mut h_y = vec![0u8; bytes_d];
                    self.cuda
                        .memcpy_d2h(h_x.as_mut_ptr() as *mut c_void, d_x_f32, bytes_d)?;
                    self.cuda.memcpy_d2h(
                        h_y.as_mut_ptr() as *mut c_void,
                        dy_attn as *const c_void,
                        bytes_d,
                    )?;
                    let mut x = Vec::with_capacity(d_model as usize);
                    let mut y = Vec::with_capacity(d_model as usize);
                    for ch in h_x.chunks_exact(4) {
                        x.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
                    }
                    for ch in h_y.chunks_exact(4) {
                        y.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
                    }
                    let mut out = Vec::with_capacity(d_model as usize);
                    for i in 0..(d_model as usize) {
                        out.push(x[i] + y[i]);
                    }
                    let mut out_bytes = Vec::with_capacity(bytes_d);
                    for v in &out {
                        out_bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    self.cuda
                        .memcpy_h2d(d_x1, out_bytes.as_ptr() as *const c_void, bytes_d)?;
                }

                // Post-attention norm: x1n = norm(x1)
                self.run_rms_norm(d_x1, d_x1n, 1, d_model as u32, 1e-6)?;

                // MLP gates and up (now feed post-attn normalized x1n)
                self.mlp_gates_f32xf16_f32(
                    d_x1n,
                    1,
                    d_model,
                    d_w_gate_f16,
                    d_w_up_f16,
                    hidden_dim,
                    dgate,
                    dup,
                )?;

                // Host fallback for elementwise: hidden = SiLU(gate) * up, where SiLU(x) = x * sigmoid(x)
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
                    let g = gate_f[i];
                    let silu = g * sigmoid(g);
                    hid_f.push(silu * up_f[i]);
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

                // Final residual add on host per pre-norm layout: out = x1 + y_mlp, where x1 = x + y_attn
                let mut h_x1 = vec![0u8; bytes_d];
                let mut h_mlp = vec![0u8; bytes_d];
                self.cuda.memcpy_d2h(
                    h_x1.as_mut_ptr() as *mut c_void,
                    d_x1 as *const c_void,
                    bytes_d,
                )?;
                self.cuda.memcpy_d2h(
                    h_mlp.as_mut_ptr() as *mut c_void,
                    dy_mlp as *const c_void,
                    bytes_d,
                )?;
                let mut x1_f = Vec::with_capacity(d_model as usize);
                let mut mlp_f = Vec::with_capacity(d_model as usize);
                for ch in h_x1.chunks_exact(4) {
                    x1_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
                }
                for ch in h_mlp.chunks_exact(4) {
                    mlp_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
                }
                let mut y_f = Vec::with_capacity(d_model as usize);
                for i in 0..(d_model as usize) {
                    y_f.push(x1_f[i] + mlp_f[i]);
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
            let _ = self.cuda.device_free(d_xn);
            let _ = self.cuda.device_free(d_x1);
            let _ = self.cuda.device_free(d_x1n);

            res
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
    #[allow(clippy::too_many_arguments)]
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

    /// # Safety
    /// d_in and d_out must be valid device pointers to buffers sized for seq_len*dim f32 values on this context's device.
    /// The memory regions must not overlap improperly and must be accessible for the duration of the call.
    pub unsafe fn run_rms_norm(
        &self,
        d_in: *const c_void,
        d_out: *mut c_void,
        seq_len: u32,
        dim: u32,
        eps: f32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // Host fallback operating on device buffers: copy MxD slice, normalize per-row, copy back
            let m = seq_len as usize;
            let d = dim as usize;
            let bytes = m * d * std::mem::size_of::<f32>();

            let mut host = vec![0u8; bytes];
            unsafe {
                self.cuda
                    .memcpy_d2h(host.as_mut_ptr() as *mut c_void, d_in, bytes)?;
            }
            let mut out = vec![0f32; m * d];
            let inp: Vec<f32> = host
                .chunks_exact(4)
                .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
                .collect();
            for row in 0..m {
                let start = row * d;
                let x = &inp[start..start + d];
                let mut mean_sq = 0.0f32;
                for &v in x {
                    mean_sq += v * v;
                }
                mean_sq /= d as f32;
                let scale = 1.0f32 / (mean_sq + eps).sqrt();
                for i in 0..d {
                    out[start + i] = x[i] * scale;
                }
            }
            let mut out_bytes = Vec::with_capacity(bytes);
            for v in &out {
                out_bytes.extend_from_slice(&v.to_le_bytes());
            }
            unsafe {
                self.cuda
                    .memcpy_h2d(d_out, out_bytes.as_ptr() as *const c_void, bytes)?;
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (d_in, d_out, seq_len, dim, eps);
            Ok(())
        }
    }
}
impl LoadedModel {
    /// Load embedding vector for a token (row from tok_embeddings F16 matrix) into a device f32 buffer.
    /// CUDA-only helper used for minimal forward smoke until a full embedding kernel exists.
    ///
    /// # Safety
    /// - d_out_f32 must be a valid device pointer to a buffer of size d_model * sizeof(f32)
    /// - token_id must be < vocab size (tok_embeddings.shape[0])
    pub unsafe fn load_token_embedding_f16_to_f32(
        &self,
        token_id: u64,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (token_id, d_out_f32);
            anyhow::bail!("load_token_embedding_f16_to_f32 requires CUDA feature");
        }
        #[cfg(feature = "cuda")]
        {
            let tok = self
                .device_tensors
                .get("tok_embeddings.weight")
                .ok_or_else(|| anyhow!("missing tok_embeddings.weight"))?;
            if tok.dtype != GgmlDType::F16 || tok.shape.len() != 2 {
                anyhow::bail!("tok_embeddings must be F16 [vocab, d_model]");
            }
            let vocab = tok.shape[0] as usize;
            let d_model = tok.shape[1] as usize;
            if token_id as usize >= vocab {
                anyhow::bail!("token_id {} out of range (vocab={})", token_id, vocab);
            }
            let row_bytes_f16 = d_model * 2;
            let offset = tok.byte_offset as usize + (token_id as usize) * row_bytes_f16;
            // Copy F16 row from device to host
            let mut row_f16 = vec![0u8; row_bytes_f16];
            self.cuda.memcpy_d2h(
                row_f16.as_mut_ptr() as *mut c_void,
                (self.d_weights_base as usize + offset) as *const c_void,
                row_bytes_f16,
            )?;
            // Convert to f32 on host
            let mut row_f32_bytes = Vec::with_capacity(d_model * 4);
            for i in 0..d_model {
                let lo = row_f16[2 * i] as u16;
                let hi = row_f16[2 * i + 1] as u16;
                let bits = lo | (hi << 8);
                let v = f16::from_bits(bits).to_f32();
                row_f32_bytes.extend_from_slice(&v.to_le_bytes());
            }
            // Upload to device output buffer
            self.cuda.memcpy_h2d(
                d_out_f32,
                row_f32_bytes.as_ptr() as *const c_void,
                d_model * 4,
            )?;
            Ok(())
        }
    }
}

impl LoadedModel {
    /// Helper to run forward_one_token_minimal using mapped layer weights.
    /// Assumes embeddings have already produced x (f32) for the token index.
    /// # Safety
    /// - d_x_f32 and d_out_f32 must be valid device pointers on this model's CUDA context
    /// - Pointers must reference buffers sized for the provided d_model and hidden dims of the mapped layer
    pub unsafe fn forward_one_token_with_layer(
        &self,
        d_x_f32: *const c_void,
        layer: usize,
        seq_id: u32,
        seq_len: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        let w = self.map_standard_layer(layer)?;
        #[cfg(feature = "cuda")]
        let wq_ptr = (self.d_weights_base as usize + w.wq.byte_offset as usize) as *const c_void;
        #[cfg(not(feature = "cuda"))]
        let wq_ptr = std::ptr::null();
        #[cfg(feature = "cuda")]
        let wk_ptr = (self.d_weights_base as usize + w.wk.byte_offset as usize) as *const c_void;
        #[cfg(not(feature = "cuda"))]
        let wk_ptr = std::ptr::null();
        #[cfg(feature = "cuda")]
        let wv_ptr = (self.d_weights_base as usize + w.wv.byte_offset as usize) as *const c_void;
        #[cfg(not(feature = "cuda"))]
        let wv_ptr = std::ptr::null();
        #[cfg(feature = "cuda")]
        let wo_ptr = (self.d_weights_base as usize + w.wo.byte_offset as usize) as *const c_void;
        #[cfg(not(feature = "cuda"))]
        let wo_ptr = std::ptr::null();
        #[cfg(feature = "cuda")]
        let w_gate_ptr =
            (self.d_weights_base as usize + w.w_gate.byte_offset as usize) as *const c_void;
        #[cfg(not(feature = "cuda"))]
        let w_gate_ptr = std::ptr::null();
        #[cfg(feature = "cuda")]
        let w_up_ptr =
            (self.d_weights_base as usize + w.w_up.byte_offset as usize) as *const c_void;
        #[cfg(not(feature = "cuda"))]
        let w_up_ptr = std::ptr::null();
        #[cfg(feature = "cuda")]
        let w_down_ptr =
            (self.d_weights_base as usize + w.w_down.byte_offset as usize) as *const c_void;
        #[cfg(not(feature = "cuda"))]
        let w_down_ptr = std::ptr::null();

        self.forward_one_token_minimal(
            d_x_f32,
            w.d_model as i32,
            wq_ptr,
            wk_ptr,
            wv_ptr,
            wo_ptr,
            w_gate_ptr,
            w_up_ptr,
            w_down_ptr,
            w.hidden_dim as i32,
            seq_id,
            seq_len,
            d_out_f32,
        )
    }
}

impl LoadedModel {
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
    #[allow(clippy::too_many_arguments)]
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
