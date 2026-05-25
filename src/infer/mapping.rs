use super::{DeviceTensorView, LoadedModel, StandardLayerWeights};
use crate::gguf::GgmlDType;
use anyhow::{Context, Result};
#[cfg(feature = "cuda")]
use std::ffi::c_void;

impl LoadedModel {
    pub(super) fn device_tensor(&self, name: &str) -> Option<&DeviceTensorView> {
        self.device_tensors.get(name)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn tensor_device_ptr(
        &self,
        name: &str,
        view: &DeviceTensorView,
    ) -> Result<*const c_void> {
        if view.dptr.is_null() {
            anyhow::bail!("tensor '{}' device pointer is null", name);
        }
        let off: usize = view
            .byte_offset
            .try_into()
            .context("tensor offset does not fit in usize")?;
        let end = off
            .checked_add(view.nbytes)
            .context("tensor span overflow")?;
        if end > self.weights_len {
            anyhow::bail!(
                "tensor '{}' device span {}..{} exceeds weights length {}",
                name,
                off,
                end,
                self.weights_len
            );
        }
        let expected = (self.d_weights_base as usize)
            .checked_add(off)
            .context("device pointer offset overflow")?;
        if view.dptr as usize != expected {
            anyhow::bail!(
                "tensor '{}' device pointer {} != base {} + offset {}",
                name,
                view.dptr as usize,
                self.d_weights_base as usize,
                off
            );
        }
        Ok(view.dptr as *const c_void)
    }

    pub(super) fn find_tensor_any<'a>(
        &'a self,
        candidates: &[String],
    ) -> Result<&'a DeviceTensorView> {
        for c in candidates {
            if let Some(v) = self.device_tensor(c) {
                return Ok(v);
            }
        }
        anyhow::bail!("missing required tensor; tried: {}", candidates.join(", "))
    }

    pub(super) fn find_tensor_any_optional(
        &self,
        candidates: &[String],
    ) -> Option<DeviceTensorView> {
        candidates
            .iter()
            .find_map(|name| self.device_tensor(name))
            .cloned()
    }

    pub(super) fn validate_norm_weight(
        &self,
        label: &str,
        view: &DeviceTensorView,
        dim: usize,
    ) -> Result<()> {
        if view.dtype != GgmlDType::F16 && view.dtype != GgmlDType::F32 {
            anyhow::bail!("{label} expected F16 or F32, got {:?}", view.dtype);
        }
        if view.shape.len() != 1 || view.shape[0] as usize != dim {
            anyhow::bail!(
                "{label} shape invalid: expected [{dim}], got {:?}",
                view.shape
            );
        }
        Ok(())
    }

    /// Map standard LLaMA-style GGUF tensor names for a single layer.
    /// Supports both layers.N.* and blk.N.* naming variants.
    pub fn map_standard_layer(&self, layer: usize) -> Result<StandardLayerWeights> {
        use crate::gguf::GgmlDType;
        // Determine d_model from parsed config and support embeddings layout
        // in either [vocab, d_model] or [d_model, vocab] (e.g., Qwen).
        // Try common embedding tensor names across families (LLaMA/Qwen)
        let tok = self.find_tensor_any(&[
            "tok_embeddings.weight".to_string(),
            "token_embd.weight".to_string(),
            "token_embd".to_string(),
            "token_embeddings.weight".to_string(),
        ])?;
        // Enforce embeddings dtype policy: F16 or Q8_0
        if tok.dtype != GgmlDType::F16 && tok.dtype != GgmlDType::Q8_0 {
            anyhow::bail!(
                "tok_embeddings.weight expected F16 or Q8_0 [*, *], got {:?}",
                tok.dtype
            )
        }
        if tok.shape.len() != 2 {
            anyhow::bail!(
                "embeddings shape invalid: expected rank-2, got {:?}",
                tok.shape
            );
        }
        let r0 = tok.shape[0] as usize;
        let r1 = tok.shape[1] as usize;
        let d_model = self.model_config.embedding_length as usize;
        // Validate that one embedding dimension equals d_model and the other matches vocab if present
        let rows_are_vocab = if r1 == d_model {
            true // [vocab, d_model]
        } else if r0 == d_model {
            false // [d_model, vocab]
        } else {
            // Neither dimension equals d_model -> invalid
            anyhow::bail!(
                "embeddings dims {:?} do not contain d_model {}",
                tok.shape,
                d_model
            );
        };
        let vocab_dim = if rows_are_vocab { r0 } else { r1 };
        let v_meta = self.model_config.vocab_size as usize;
        if v_meta != vocab_dim {
            anyhow::bail!(
                "vocab_size meta {} != embeddings vocab dim {} (shape={:?})",
                v_meta,
                vocab_dim,
                tok.shape
            );
        }
        // If metadata has block_count, ensure requested layer is in-range
        if layer as u32 >= self.model_config.block_count {
            anyhow::bail!(
                "layer {} out of range (block_count={})",
                layer,
                self.model_config.block_count
            );
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
        let bq_names = vec![
            format!("layers.{layer}.attention.wq.bias"),
            format!("blk.{layer}.attn_q.bias"),
        ];
        let bk_names = vec![
            format!("layers.{layer}.attention.wk.bias"),
            format!("blk.{layer}.attn_k.bias"),
        ];
        let bv_names = vec![
            format!("layers.{layer}.attention.wv.bias"),
            format!("blk.{layer}.attn_v.bias"),
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
        let attn_norm_names = vec![
            format!("layers.{layer}.attention_norm.weight"),
            format!("layers.{layer}.attn_norm.weight"),
            format!("blk.{layer}.attn_norm.weight"),
        ];
        let ffn_norm_names = vec![
            format!("layers.{layer}.ffn_norm.weight"),
            format!("layers.{layer}.feed_forward_norm.weight"),
            format!("blk.{layer}.ffn_norm.weight"),
        ];

        let wq = self.find_tensor_any(&wq_names)?.clone();
        let wk = self.find_tensor_any(&wk_names)?.clone();
        let wv = self.find_tensor_any(&wv_names)?.clone();
        let bq = self.find_tensor_any_optional(&bq_names);
        let bk = self.find_tensor_any_optional(&bk_names);
        let bv = self.find_tensor_any_optional(&bv_names);
        let wo = self.find_tensor_any(&wo_names)?.clone();
        let w_gate = self.find_tensor_any(&w_gate_names)?.clone();
        let w_up = self.find_tensor_any(&w_up_names)?.clone();
        let w_down = self.find_tensor_any(&w_down_names)?.clone();
        let attn_norm = self.find_tensor_any_optional(&attn_norm_names);
        let ffn_norm = self.find_tensor_any_optional(&ffn_norm_names);

        // DType checks: projection paths support F16 plus selected quantized
        // GGUF layouts when a fused dequant backend is available.
        for (name, t) in [
            ("wq", &wq),
            ("wk", &wk),
            ("wv", &wv),
            ("wo", &wo),
            ("w_gate", &w_gate),
            ("w_up", &w_up),
            ("w_down", &w_down),
        ] {
            if t.dtype != GgmlDType::F16 && t.dtype != GgmlDType::Q5_1 && t.dtype != GgmlDType::Q8_0
            {
                anyhow::bail!(
                    "tensor {} expected F16, Q5_1, or Q8_0, got {:?}",
                    name,
                    t.dtype
                );
            }
        }
        // Shape checks (row-major): X [1 x d_model] · W[K=d_model x N] => out [1 x N]
        for (name, t) in [("wq", &wq), ("wk", &wk), ("wv", &wv), ("wo", &wo)] {
            let k = *t.shape.first().unwrap_or(&0) as usize;
            let n = *t.shape.get(1).unwrap_or(&0) as usize;
            if k != d_model || n == 0 {
                anyhow::bail!(
                    "{} shape invalid: expected [d_model, N], got {:?}",
                    name,
                    t.shape
                );
            }
        }
        // Validate Q/K/V/WO N dims using head counts from parsed config
        let n_head = self.model_config.attention_head_count as usize;
        let head_dim = self.model_config.attention_key_length as usize;
        let n_kv = self.model_config.attention_head_count_kv as usize;
        let expect_nq = n_head * head_dim;
        let expect_nk = n_kv * head_dim;
        let expect_nv = n_kv * head_dim;
        let expect_no = d_model;
        let qn = *wq.shape.get(1).unwrap_or(&0) as usize;
        let kn = *wk.shape.get(1).unwrap_or(&0) as usize;
        let vn = *wv.shape.get(1).unwrap_or(&0) as usize;
        let on = *wo.shape.get(1).unwrap_or(&0) as usize;
        if qn != expect_nq {
            anyhow::bail!("wq second dim {} != expected {}", qn, expect_nq);
        }
        if kn != expect_nk {
            anyhow::bail!("wk second dim {} != expected {}", kn, expect_nk);
        }
        if vn != expect_nv {
            anyhow::bail!("wv second dim {} != expected {}", vn, expect_nv);
        }
        if on != expect_no {
            anyhow::bail!("wo second dim {} != expected {}", on, expect_no);
        }
        for (name, bias, expected) in [
            ("bq", &bq, expect_nq),
            ("bk", &bk, expect_nk),
            ("bv", &bv, expect_nv),
        ] {
            if let Some(t) = bias {
                if t.dtype != GgmlDType::F32 {
                    anyhow::bail!("{name} expected F32 bias, got {:?}", t.dtype);
                }
                if t.shape.len() != 1 || t.shape[0] as usize != expected {
                    anyhow::bail!(
                        "{name} shape invalid: expected [{expected}], got {:?}",
                        t.shape
                    );
                }
            }
        }
        // Infer hidden_dim from up/gate
        for (name, t) in [("w_gate", &w_gate), ("w_up", &w_up)] {
            let k = *t.shape.first().unwrap_or(&0) as usize;
            let h = *t.shape.get(1).unwrap_or(&0) as usize;
            if k != d_model || h == 0 {
                anyhow::bail!(
                    "{} shape invalid: expected [d_model, H], got {:?}",
                    name,
                    t.shape
                );
            }
        }
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
        // If parsed config has feed_forward_length, enforce it matches hidden_dim
        let h_cfg = self.model_config.feed_forward_length as usize;
        if h_cfg != hidden_dim {
            anyhow::bail!(
                "model_config.feed_forward_length {} != inferred hidden_dim {}",
                h_cfg,
                hidden_dim
            );
        }
        if let Some(t) = &attn_norm {
            self.validate_norm_weight("attention norm", t, d_model)?;
        }
        if let Some(t) = &ffn_norm {
            self.validate_norm_weight("ffn norm", t, d_model)?;
        }

        Ok(StandardLayerWeights {
            d_model,
            hidden_dim,
            tok_embeddings: tok.clone(),
            wq,
            wk,
            wv,
            bq,
            bk,
            bv,
            wo,
            w_gate,
            w_up,
            w_down,
            attn_norm,
            ffn_norm,
        })
    }
    /// Map the language modeling head (output projection) tensor.
    /// Prefers a dedicated output.weight/lm_head.weight [d_model, vocab] in F16.
    /// If absent, accepts tied F16 embeddings only when they already have the
    /// same GGUF-native [d_model, vocab] layout expected by the projection path.
    /// Returns (tensor name, tensor view, d_model, vocab, tied_to_embeddings).
    pub fn map_lm_head(&self) -> Result<(String, DeviceTensorView, usize, usize, bool)> {
        use crate::gguf::GgmlDType;
        let candidates = vec![
            "output.weight".to_string(),
            "lm_head.weight".to_string(),
            "output".to_string(),
        ];
        if let Ok(t) = self.find_tensor_any(&candidates) {
            let name = candidates
                .iter()
                .find(|candidate| self.device_tensor(candidate).is_some())
                .context("lm_head tensor name not found after candidate match")?
                .clone();
            if t.dtype != GgmlDType::F16 {
                anyhow::bail!("lm_head expected F16, got {:?}", t.dtype);
            }
            if t.shape.len() != 2 {
                anyhow::bail!(
                    "lm_head shape invalid: expected [d_model, vocab], got {:?}",
                    t.shape
                );
            }
            let d_model = t.shape[0] as usize;
            let vocab = t.shape[1] as usize;
            if d_model == 0 || vocab == 0 {
                anyhow::bail!(
                    "lm_head dims invalid: non-zero [d_model, vocab] required: {:?}",
                    t.shape
                );
            }
            return Ok((name, t.clone(), d_model, vocab, false));
        }
        let embedding_candidates = vec![
            "tok_embeddings.weight".to_string(),
            "token_embd.weight".to_string(),
            "token_embd".to_string(),
            "token_embeddings.weight".to_string(),
        ];
        if let Ok(t) = self.find_tensor_any(&embedding_candidates) {
            let name = embedding_candidates
                .iter()
                .find(|candidate| self.device_tensor(candidate).is_some())
                .context("embedding tensor name not found after candidate match")?
                .clone();
            if t.dtype != GgmlDType::F16 {
                anyhow::bail!(
                    "tied lm_head requires F16 embeddings when output head is absent; tensor {name} has {:?}",
                    t.dtype
                );
            }
            if t.shape.len() != 2 {
                anyhow::bail!(
                    "tied lm_head embedding shape invalid: expected [d_model, vocab], got {:?}",
                    t.shape
                );
            }
            let d_model = self.model_config.embedding_length as usize;
            let vocab = self.model_config.vocab_size as usize;
            if t.shape[0] as usize != d_model || t.shape[1] as usize != vocab {
                anyhow::bail!(
                    "tied lm_head embedding layout unsupported: expected [d_model, vocab]=[{}, {}], got {:?}",
                    d_model,
                    vocab,
                    t.shape
                );
            }
            return Ok((name, t.clone(), d_model, vocab, true));
        }
        anyhow::bail!(
            "output projection tensor not found; expected one of {} or tied F16 embeddings in [d_model, vocab] layout",
            candidates.join(", ")
        )
    }

    pub fn map_output_norm(&self) -> Result<Option<(String, DeviceTensorView)>> {
        let candidates = vec![
            "output_norm.weight".to_string(),
            "norm.weight".to_string(),
            "model.norm.weight".to_string(),
        ];
        let Some(view) = self.find_tensor_any_optional(&candidates) else {
            return Ok(None);
        };
        let name = candidates
            .iter()
            .find(|candidate| self.device_tensor(candidate).is_some())
            .context("output norm tensor name not found after candidate match")?
            .clone();
        self.validate_norm_weight(
            "output norm",
            &view,
            self.model_config.embedding_length as usize,
        )?;
        Ok(Some((name, view)))
    }

    pub fn validate_standard_layers(&self) -> Result<(usize, usize)> {
        let layer_count = self.model_config.block_count as usize;
        if layer_count == 0 {
            anyhow::bail!("model_config.block_count must be > 0");
        }

        let mut d_model = None;
        let mut hidden_dim = None;
        for layer in 0..layer_count {
            let w = self
                .map_standard_layer(layer)
                .with_context(|| format!("mapping standard layer {layer}"))?;
            match d_model {
                Some(expected) if expected != w.d_model => anyhow::bail!(
                    "layer {layer} d_model mismatch: expected {expected}, got {}",
                    w.d_model
                ),
                None => d_model = Some(w.d_model),
                _ => {}
            }
            match hidden_dim {
                Some(expected) if expected != w.hidden_dim => anyhow::bail!(
                    "layer {layer} hidden_dim mismatch: expected {expected}, got {}",
                    w.hidden_dim
                ),
                None => hidden_dim = Some(w.hidden_dim),
                _ => {}
            }
        }

        Ok((d_model.unwrap_or(0), hidden_dim.unwrap_or(0)))
    }

    pub fn validate_full_layer_decode(&self) -> Result<(usize, usize)> {
        let (d_model, hidden_dim) = self.validate_standard_layers()?;
        let (_lm_name, _lm_head, lm_d_model, _vocab, _tied) = self.map_lm_head()?;
        if lm_d_model != d_model {
            anyhow::bail!(
                "lm_head d_model {} != layer d_model {}",
                lm_d_model,
                d_model
            );
        }
        if !self.kv_cache_can_address_layers() {
            let slots = self
                .kv_cache
                .as_ref()
                .map(|kv| kv.max_batch_size())
                .unwrap_or(0);
            anyhow::bail!(
                "KV cache has {} slots, but full-layer decode needs {} layer slots",
                slots,
                self.model_config.block_count
            );
        }

        #[cfg(feature = "cuda")]
        {
            self.tensor_device_ptr(&_lm_name, &_lm_head)
                .with_context(|| format!("validating lm_head tensor '{_lm_name}'"))?;
            for layer in 0..self.model_config.block_count as usize {
                let w = self.map_standard_layer(layer)?;
                for (name, view) in [
                    ("tok_embeddings", &w.tok_embeddings),
                    ("wq", &w.wq),
                    ("wk", &w.wk),
                    ("wv", &w.wv),
                    ("wo", &w.wo),
                    ("w_gate", &w.w_gate),
                    ("w_up", &w.w_up),
                    ("w_down", &w.w_down),
                ] {
                    self.tensor_device_ptr(name, view)
                        .with_context(|| format!("validating layer {layer} tensor '{name}'"))?;
                }
                if let Some(view) = &w.attn_norm {
                    self.tensor_device_ptr("attn_norm", view)
                        .with_context(|| format!("validating layer {layer} attn_norm"))?;
                }
                if let Some(view) = &w.ffn_norm {
                    self.tensor_device_ptr("ffn_norm", view)
                        .with_context(|| format!("validating layer {layer} ffn_norm"))?;
                }
            }
        }

        Ok((d_model, hidden_dim))
    }
}
