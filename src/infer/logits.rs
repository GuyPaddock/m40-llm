use super::LoadedModel;
use anyhow::{Context, Result};
use std::ffi::c_void;

impl LoadedModel {
    /// # Safety
    /// Compute logits = H (1xD f32) × W^T (D×V f16) -> (1xV f32) using device GEMM and return host Vec<f32>.
    pub unsafe fn logits_from_hidden_gpu(&self, d_hidden_f32: *const c_void) -> Result<Vec<f32>> {
        let (name, lm, _d_model, _vocab, _) = self.map_lm_head()?;
        #[cfg(not(feature = "cuda"))]
        let _ = &name;
        let d_model = self.model_config.embedding_length as usize;
        let vocab = lm.shape[1] as usize;
        #[cfg(feature = "cuda")]
        let d_w = self.tensor_device_ptr(&name, &lm)?;
        #[cfg(not(feature = "cuda"))]
        let d_w: *const c_void = std::ptr::null();
        #[cfg(not(feature = "cuda"))]
        let _ = &lm;
        let bytes_logits = vocab * std::mem::size_of::<f32>();
        let d_logits = self.cuda.device_malloc(bytes_logits)?;
        self.matmul_f32xf16_gguf_f32(d_hidden_f32, d_w, d_logits, 1, vocab as i32, d_model as i32)
            .with_context(|| {
                format!("lm_head GEMM failed: m=1 n={vocab} k={d_model} tensor={name}")
            })?;
        let mut host = vec![0u8; bytes_logits];
        self.cuda
            .memcpy_d2h(host.as_mut_ptr() as *mut c_void, d_logits, bytes_logits)?;
        let mut logits = Vec::with_capacity(vocab);
        for ch in host.chunks_exact(4) {
            logits.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
        }
        let _ = self.cuda.device_free(d_logits);
        Ok(logits)
    }
}

impl LoadedModel {
    /// # Safety
    /// Compute logits from a device hidden state. Prefer GPU GEMM with lm_head when present; otherwise fallback
    /// to host-side dot with per-row copies from tok_embeddings.
    pub unsafe fn logits_from_hidden(&self, d_hidden_f32: *const c_void) -> Result<Vec<f32>> {
        if let Ok((name, lm, d_model, vocab, _)) = self.map_lm_head() {
            #[cfg(not(feature = "cuda"))]
            let _ = &name;
            #[cfg(feature = "cuda")]
            {
                let d_w = self.tensor_device_ptr(&name, &lm)?;
                let bytes_hidden = d_model * std::mem::size_of::<f32>();
                let bytes_logits = vocab * std::mem::size_of::<f32>();
                let mut d_norm_hidden = std::ptr::null_mut();
                let d_logits = self.cuda.device_malloc(bytes_logits)?;
                let result = (|| -> Result<Vec<f32>> {
                    let hidden_for_logits =
                        if let Some((norm_name, norm)) = self.map_output_norm()? {
                            let d_norm_w = self.tensor_device_ptr(&norm_name, &norm)?;
                            d_norm_hidden = self
                                .cuda
                                .device_malloc_tagged(bytes_hidden, "logits:output_norm_f32")?;
                            self.run_rms_norm_weighted(
                                d_hidden_f32,
                                d_norm_w,
                                norm.dtype,
                                d_norm_hidden,
                                1,
                                d_model as u32,
                                self.model_config.layer_norm_epsilon,
                            )?;
                            d_norm_hidden as *const c_void
                        } else {
                            d_hidden_f32
                        };
                    self.matmul_f32xf16_gguf_f32(
                        hidden_for_logits,
                        d_w,
                        d_logits,
                        1,
                        vocab as i32,
                        d_model as i32,
                    )
                    .with_context(|| {
                        format!("lm_head GEMM failed: m=1 n={vocab} k={d_model} tensor={name}")
                    })?;
                    let mut host = vec![0u8; bytes_logits];
                    self.cuda.memcpy_d2h(
                        host.as_mut_ptr() as *mut c_void,
                        d_logits,
                        bytes_logits,
                    )?;
                    let mut logits = Vec::with_capacity(vocab);
                    for ch in host.chunks_exact(4) {
                        logits.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
                    }
                    Ok(logits)
                })();
                if !d_norm_hidden.is_null() {
                    let _ = self.cuda.device_free(d_norm_hidden);
                }
                let _ = self.cuda.device_free(d_logits);
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            {
                // Host path: compute logits = hidden (1xD f32) × W (D×V f16) on CPU using host-stored weights
                // SAFETY: treat d_hidden_f32 as host pointer to D f32s in non-CUDA builds
                let hidden_bytes =
                    unsafe { std::slice::from_raw_parts(d_hidden_f32 as *const u8, d_model * 4) };
                let mut hidden = vec![0f32; d_model];
                for (i, ch) in hidden_bytes.chunks_exact(4).enumerate() {
                    hidden[i] = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
                }
                // Weights are row-major [D, V] in f16
                let off = lm.byte_offset as usize;
                let w_bytes = &self.host_weights[off..off + d_model * vocab * 2];
                let mut logits = vec![0f32; vocab];
                for (col, logit_ref) in logits.iter_mut().enumerate().take(vocab) {
                    let mut acc = 0f32;
                    let mut idx = col * d_model * 2;
                    for &hidden_val in hidden.iter().take(d_model) {
                        let lo = w_bytes[idx] as u16;
                        let hi = w_bytes[idx + 1] as u16;
                        let bits = lo | (hi << 8);
                        let w = half::f16::from_bits(bits).to_f32();
                        acc += hidden_val * w;
                        idx += 2;
                    }
                    *logit_ref = acc;
                }
                return Ok(logits);
            }
        }
        // Fallback: use tok_embeddings^T as lm_head on host
        // Resolve embeddings for fallback when lm_head absent
        let tok = self.find_tensor_any(&[
            "tok_embeddings.weight".to_string(),
            "token_embd.weight".to_string(),
            "token_embd".to_string(),
            "token_embeddings.weight".to_string(),
        ])?;
        if tok.shape.len() != 2 {
            anyhow::bail!("embeddings must be rank-2 [vocab, d_model]");
        }
        let vocab = tok.shape[0] as usize;
        let d_model = tok.shape[1] as usize;
        let bytes_d = d_model * std::mem::size_of::<f32>();
        // Copy hidden to host
        let mut out_h = vec![0u8; bytes_d];
        self.cuda
            .memcpy_d2h(out_h.as_mut_ptr() as *mut c_void, d_hidden_f32, bytes_d)?;
        let mut out_f = vec![0f32; d_model];
        for (i, ch) in out_h.chunks_exact(4).enumerate() {
            out_f[i] = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
        }
        use half::f16;
        let row_bytes = d_model * 2;
        let mut logits = vec![0f32; vocab];
        for (v, logit_ref) in logits.iter_mut().enumerate().take(vocab) {
            #[cfg(feature = "cuda")]
            {
                let mut row_h = vec![0u8; row_bytes];
                let row_dev = self
                    .tensor_device_ptr("tok_embeddings", tok)?
                    .cast::<u8>()
                    .wrapping_add(v * row_bytes) as *const c_void;
                self.cuda
                    .memcpy_d2h(row_h.as_mut_ptr() as *mut c_void, row_dev, row_bytes)?;
                let mut acc = 0f32;
                for i in 0..d_model {
                    let lo = row_h[i * 2];
                    let hi = row_h[i * 2 + 1];
                    let val = f16::from_bits(u16::from_le_bytes([lo, hi])).to_f32();
                    acc += out_f[i] * val;
                }
                *logit_ref = acc;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let off = tok.byte_offset as usize + v * row_bytes;
                let row_slice = &self.host_weights[off..off + row_bytes];
                let mut acc = 0f32;
                for i in 0..d_model {
                    let lo = row_slice[i * 2];
                    let hi = row_slice[i * 2 + 1];
                    let val = f16::from_bits(u16::from_le_bytes([lo, hi])).to_f32();
                    acc += out_f[i] * val;
                }
                *logit_ref = acc;
            }
        }
        Ok(logits)
    }
}
