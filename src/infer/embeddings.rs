use super::LoadedModel;
#[cfg(feature = "cuda")]
use crate::gguf::GgmlDType;
use anyhow::Result;
use std::ffi::c_void;

impl LoadedModel {
    /// Load embedding vector for a token (row from tok_embeddings F16 matrix) into a device f32 buffer.
    /// CUDA-only helper used for minimal forward smoke until a full embedding kernel exists.
    ///
    /// # Safety
    /// - d_out_f32 must be a valid device pointer to a buffer of size d_model * sizeof(f32)
    /// - token_id must be < vocab size (tok_embeddings.shape[0])
    pub unsafe fn load_token_embedding_to_f32(
        &self,
        token_id: u64,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (token_id, d_out_f32);
            anyhow::bail!("load_token_embedding_to_f32 requires CUDA feature");
        }
        #[cfg(feature = "cuda")]
        {
            // Resolve embeddings tensor (supports Qwen names)
            let tok = self.find_tensor_any(&[
                "tok_embeddings.weight".to_string(),
                "token_embd.weight".to_string(),
                "token_embd".to_string(),
                "token_embeddings.weight".to_string(),
            ])?;
            if tok.shape.len() != 2 {
                anyhow::bail!("embeddings must be [vocab, d_model]");
            }
            let r0 = tok.shape[0] as usize;
            let r1 = tok.shape[1] as usize;
            // Determine d_model and which dimension is vocab vs d_model
            let d_model = self.model_config.embedding_length as usize;
            // rows_are_vocab = true when shape is [vocab, d_model]
            let rows_are_vocab = {
                let vsz = self.model_config.vocab_size as usize;
                if vsz == r0 {
                    true
                } else if vsz == r1 {
                    false
                } else if r1 == d_model {
                    true
                } else if r0 == d_model {
                    false
                } else {
                    r1 == d_model
                }
            };
            let vocab = if rows_are_vocab { r0 } else { r1 };
            if (token_id as usize) >= vocab {
                anyhow::bail!(
                    "token_id {} out of range for vocab size {}",
                    token_id,
                    vocab
                );
            }
            match tok.dtype {
                GgmlDType::F16 => {
                    if rows_are_vocab {
                        // Contiguous row: [vocab, d_model]
                        let row_bytes = d_model * 2;
                        let d_row =
                            (tok.dptr as usize + (token_id as usize) * row_bytes) as *const c_void;
                        self.cuda.f16_to_f32(d_row, d_out_f32, d_model)?;
                    } else {
                        // GGUF-native [d_model, vocab]: dimension 0 is fastest, so each token
                        // vector is contiguous even though vocab is shape dimension 1.
                        let row_bytes = d_model * 2;
                        let d_row =
                            (tok.dptr as usize + (token_id as usize) * row_bytes) as *const c_void;
                        self.cuda.f16_to_f32(d_row, d_out_f32, d_model)?;
                    }
                }
                GgmlDType::Q8_0 => {
                    const Q8_0_BLOCK: usize = 32;
                    const Q8_0_BLOCK_BYTES: usize = 34;
                    if rows_are_vocab {
                        // Row dequant: shape [vocab, d_model], blocks along d_model
                        let blocks = d_model.div_ceil(Q8_0_BLOCK);
                        let row_bytes = blocks * Q8_0_BLOCK_BYTES;
                        let d_row =
                            (tok.dptr as usize + (token_id as usize) * row_bytes) as *const c_void;
                        // Pull the quantized row bytes to host
                        let mut qrow = vec![0u8; row_bytes];
                        self.cuda
                            .memcpy_d2h(qrow.as_mut_ptr() as *mut c_void, d_row, row_bytes)?;
                        // Dequantize to f32 on host
                        let mut out = vec![0f32; d_model];
                        for blk in 0..blocks {
                            let base = blk * Q8_0_BLOCK_BYTES;
                            let d_bits = u16::from_le_bytes([qrow[base], qrow[base + 1]]);
                            let d = half::f16::from_bits(d_bits).to_f32();
                            for idx in 0..Q8_0_BLOCK {
                                let i = blk * Q8_0_BLOCK + idx;
                                if i >= d_model {
                                    break;
                                }
                                let q = qrow[base + 2 + idx] as i8 as f32;
                                out[i] = d * q;
                            }
                        }
                        let out_bytes = d_model * std::mem::size_of::<f32>();
                        self.cuda.memcpy_h2d(
                            d_out_f32,
                            out.as_ptr() as *const c_void,
                            out_bytes,
                        )?;
                    } else {
                        // GGUF-native [d_model, vocab]: blocks are along d_model for each token.
                        let blocks = d_model.div_ceil(Q8_0_BLOCK);
                        let row_bytes = blocks * Q8_0_BLOCK_BYTES;
                        let d_row =
                            (tok.dptr as usize + (token_id as usize) * row_bytes) as *const c_void;
                        let mut qrow = vec![0u8; row_bytes];
                        self.cuda
                            .memcpy_d2h(qrow.as_mut_ptr() as *mut c_void, d_row, row_bytes)?;
                        let mut out = vec![0f32; d_model];
                        for blk in 0..blocks {
                            let base = blk * Q8_0_BLOCK_BYTES;
                            let d_bits = u16::from_le_bytes([qrow[base], qrow[base + 1]]);
                            let d = half::f16::from_bits(d_bits).to_f32();
                            for idx in 0..Q8_0_BLOCK {
                                let i = blk * Q8_0_BLOCK + idx;
                                if i >= d_model {
                                    break;
                                }
                                let q = qrow[base + 2 + idx] as i8 as f32;
                                out[i] = d * q;
                            }
                        }
                        let out_bytes = d_model * std::mem::size_of::<f32>();
                        self.cuda.memcpy_h2d(
                            d_out_f32,
                            out.as_ptr() as *const c_void,
                            out_bytes,
                        )?;
                    }
                }
                GgmlDType::Q5_1 => {
                    // Q5_1 dequantization: shape [K, N], blocks along N
                    const Q5_1_BLOCK: usize = 32;
                    const Q5_1_BLOCK_BYTES: usize = 36; // 4-byte scale + 32 quantized bytes
                    let blocks = (r1 + Q5_1_BLOCK - 1) / Q5_1_BLOCK;
                    let row_stride = blocks * Q5_1_BLOCK_BYTES;
                    let b = (token_id as usize) / 32;
                    let idx = (token_id as usize) % 32;
                    let mut out = vec![0f32; d_model];
                    let mut block_buf = [0u8; Q5_1_BLOCK_BYTES];
                    for i in 0..d_model {
                        let row_base = (tok.dptr as usize) + i * row_stride;
                        let blk_off = row_base + b * Q5_1_BLOCK_BYTES;
                        self.cuda.memcpy_d2h(
                            block_buf.as_mut_ptr() as *mut c_void,
                            blk_off as *const c_void,
                            Q5_1_BLOCK_BYTES,
                        )?;
                        let d = f32::from_le_bytes([
                            block_buf[0],
                            block_buf[1],
                            block_buf[2],
                            block_buf[3],
                        ]);
                        let q = block_buf[4 + idx] as i8 as f32;
                        out[i] = d * q;
                    }
                    let out_bytes = d_model * std::mem::size_of::<f32>();
                    self.cuda
                        .memcpy_h2d(d_out_f32, out.as_ptr() as *const c_void, out_bytes)?;
                }
                other => anyhow::bail!("unsupported embedding dtype: {:?}", other),
            }
            Ok(())
        }
    }
}

#[cfg(feature = "cuda")]
impl LoadedModel {
    /// Backward-compat shim for tests until they are updated
    /// Safety: same as load_token_embedding_to_f32
    pub unsafe fn load_token_embedding_f16_to_f32(
        &self,
        token_id: u64,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        self.load_token_embedding_to_f32(token_id, d_out_f32)
    }
}
