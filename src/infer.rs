// src/infer.rs
#![allow(dead_code)]

use crate::cuda::{CudaContext, KVCache};
use crate::gguf::{GgmlDType, GgufModel, GgufScalar, GgufTensor, GgufValue};
use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::ffi::c_void;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub block_count: u32,
    pub context_length: u32,
    pub embedding_length: u32,
    pub feed_forward_length: u32,
    pub attention_head_count: u32,
    pub attention_head_count_kv: u32,
    pub attention_key_length: u32,
    pub layer_norm_epsilon: f32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub vocab_size: u32,
}

impl ModelConfig {
    #[cfg(feature = "gguf_ext")]
    fn from_gguf_ext_bytes(
        bytes: &[u8],
        metadata: &HashMap<String, GgufValue>,
        tensors: &[GgufTensor],
    ) -> Result<(Self, gguf_llms::model::ModelConfig)> {
        let typed = crate::gguf_ext::extract_model_config_from_bytes(bytes)?;
        let model_cfg = Self::from_typed(&typed, metadata, tensors)?;
        Ok((model_cfg, typed))
    }

    #[cfg(feature = "gguf_ext")]
    pub fn from_typed(
        typed: &gguf_llms::model::ModelConfig,
        metadata: &HashMap<String, GgufValue>,
        tensors: &[GgufTensor],
    ) -> Result<Self> {
        let d_model = typed.embedding_length;
        let vocab_size =
            derive_vocab_size(metadata, tensors, d_model).context("derive vocab_size from gguf")?;
        let head_kv = typed
            .attention_head_count_kv
            .unwrap_or(typed.attention_head_count);
        let head_dim = typed
            .attention_key_length
            .unwrap_or_else(|| typed.embedding_length / typed.attention_head_count);
        let layer_norm_epsilon = typed.layer_norm_epsilon.unwrap_or(1e-5);
        let rope_freq_base = typed.rope_freq_base.unwrap_or(10_000.0);
        let rope_freq_scale = get_f32_meta(metadata, "llama.rope.freq_scale").unwrap_or(1.0);
        let feed_forward_length = typed.feed_forward_length;
        let cfg = Self {
            architecture: typed.architecture.clone(),
            block_count: typed.block_count,
            context_length: typed.context_length,
            embedding_length: d_model,
            feed_forward_length,
            attention_head_count: typed.attention_head_count,
            attention_head_count_kv: head_kv,
            attention_key_length: head_dim,
            layer_norm_epsilon,
            rope_freq_base,
            rope_freq_scale,
            vocab_size,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_metadata(
        metadata: &HashMap<String, GgufValue>,
        tensors: &[GgufTensor],
    ) -> Result<Self> {
        let architecture = get_str_meta(metadata, "general.architecture")
            .ok_or_else(|| anyhow!("missing general.architecture"))?
            .to_string();
        let embedding_length = get_u32_meta(metadata, "llama.embedding_length")
            .ok_or_else(|| anyhow!("missing llama.embedding_length"))?;
        let attention_head_count = get_u32_meta(metadata, "llama.attention.head_count")
            .ok_or_else(|| anyhow!("missing llama.attention.head_count"))?;
        let block_count = get_u32_meta(metadata, "llama.block_count")
            .ok_or_else(|| anyhow!("missing llama.block_count"))?;
        let context_length = get_u32_meta(metadata, "llama.context_length")
            .ok_or_else(|| anyhow!("missing llama.context_length"))?;
        let feed_forward_length = derive_feed_forward_length(metadata, tensors, embedding_length)?;
        let head_kv =
            get_u32_meta(metadata, "llama.attention.head_count_kv").unwrap_or(attention_head_count);
        let head_dim = get_u32_meta(metadata, "llama.attention.key_length")
            .unwrap_or_else(|| embedding_length / attention_head_count);
        let layer_norm_epsilon = get_f32_meta(metadata, "llama.layer_norm_epsilon").unwrap_or(1e-5);
        let rope_freq_base = get_f32_meta(metadata, "llama.rope.freq_base").unwrap_or(10_000.0);
        let rope_freq_scale = get_f32_meta(metadata, "llama.rope.freq_scale").unwrap_or(1.0);
        let vocab_size = derive_vocab_size(metadata, tensors, embedding_length)
            .context("derive vocab_size from gguf metadata")?;
        let cfg = Self {
            architecture,
            block_count,
            context_length,
            embedding_length,
            feed_forward_length,
            attention_head_count,
            attention_head_count_kv: head_kv,
            attention_key_length: head_dim,
            layer_norm_epsilon,
            rope_freq_base,
            rope_freq_scale,
            vocab_size,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.embedding_length == 0 {
            anyhow::bail!("embedding_length must be > 0");
        }
        if self.attention_head_count == 0 {
            anyhow::bail!("attention_head_count must be > 0");
        }
        if self.block_count == 0 {
            anyhow::bail!("block_count must be > 0");
        }
        if self.context_length == 0 {
            anyhow::bail!("context_length must be > 0");
        }
        if self.feed_forward_length == 0 {
            anyhow::bail!("feed_forward_length must be > 0");
        }
        if self.vocab_size == 0 {
            anyhow::bail!("vocab_size must be > 0");
        }
        if self.attention_head_count_kv == 0 {
            anyhow::bail!("attention_head_count_kv must be > 0");
        }
        if self.attention_key_length == 0 {
            anyhow::bail!("attention_key_length must be > 0");
        }
        if self.embedding_length != self.attention_head_count * self.attention_key_length {
            anyhow::bail!(
                "embedding_length {} must equal attention_head_count {} * attention_key_length {}",
                self.embedding_length,
                self.attention_head_count,
                self.attention_key_length
            );
        }
        if self.embedding_length % self.attention_head_count != 0 {
            anyhow::bail!(
                "embedding_length {} not divisible by attention_head_count {}",
                self.embedding_length,
                self.attention_head_count
            );
        }
        if self.attention_head_count % self.attention_head_count_kv != 0 {
            anyhow::bail!(
                "attention_head_count {} not divisible by attention_head_count_kv {}",
                self.attention_head_count,
                self.attention_head_count_kv
            );
        }
        if !self.rope_freq_base.is_finite() || self.rope_freq_base <= 0.0 {
            anyhow::bail!("rope_freq_base must be finite and > 0");
        }
        if !self.rope_freq_scale.is_finite() || self.rope_freq_scale <= 0.0 {
            anyhow::bail!("rope_freq_scale must be finite and > 0");
        }
        Ok(())
    }
}

fn get_u32_meta(metadata: &HashMap<String, GgufValue>, key: &str) -> Option<u32> {
    metadata.get(key).and_then(|v| match v {
        GgufValue::Scalar(GgufScalar::U32(x)) => Some(*x),
        GgufValue::Scalar(GgufScalar::I32(x)) => u32::try_from(*x).ok(),
        _ => None,
    })
}

fn derive_vocab_size(
    metadata: &HashMap<String, GgufValue>,
    tensors: &[GgufTensor],
    d_model: u32,
) -> Result<u32> {
    if let Some(v) = get_u32_meta(metadata, "llama.vocab_size") {
        if v == 0 {
            anyhow::bail!("llama.vocab_size must be > 0");
        }
        return Ok(v);
    }
    let candidates = [
        "tok_embeddings.weight",
        "token_embd.weight",
        "token_embd",
        "token_embeddings.weight",
    ];
    for name in candidates {
        if let Some(t) = tensors.iter().find(|t| t.name == name) {
            if t.shape.len() != 2 {
                anyhow::bail!("embedding tensor {name} must be rank-2");
            }
            let r0 = t.shape[0] as u32;
            let r1 = t.shape[1] as u32;
            if r0 == d_model && r1 > 0 {
                return Ok(r1);
            }
            if r1 == d_model && r0 > 0 {
                return Ok(r0);
            }
        }
    }
    anyhow::bail!("vocab_size missing; add llama.vocab_size or embeddings tensor")
}

fn derive_feed_forward_length(
    metadata: &HashMap<String, GgufValue>,
    tensors: &[GgufTensor],
    d_model: u32,
) -> Result<u32> {
    if let Some(v) = get_u32_meta(metadata, "llama.feed_forward_length") {
        if v == 0 {
            anyhow::bail!("llama.feed_forward_length must be > 0");
        }
        return Ok(v);
    }
    let candidates = [
        "layers.0.feed_forward.w1.weight",
        "layers.0.feed_forward.w3.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "layers.0.ffn_gate.weight",
        "layers.0.ffn_up.weight",
    ];
    for name in candidates {
        if let Some(t) = tensors.iter().find(|t| t.name == name) {
            if t.shape.len() != 2 {
                anyhow::bail!("feed-forward tensor {name} must be rank-2");
            }
            let k = t.shape[0] as u32;
            let n = t.shape[1] as u32;
            if k == d_model && n > 0 {
                return Ok(n);
            }
        }
    }
    anyhow::bail!(
        "feed_forward_length missing; add llama.feed_forward_length or a feed-forward weight tensor"
    )
}

fn get_f32_meta(metadata: &HashMap<String, GgufValue>, key: &str) -> Option<f32> {
    metadata.get(key).and_then(|v| match v {
        GgufValue::Scalar(GgufScalar::F32(x)) => Some(*x),
        _ => None,
    })
}

fn get_str_meta<'a>(metadata: &'a HashMap<String, GgufValue>, key: &str) -> Option<&'a str> {
    metadata.get(key).and_then(|v| v.as_str())
}

#[derive(Debug, Clone)]
pub struct DeviceTensorView {
    pub dtype: GgmlDType,
    pub shape: Vec<u64>,
    /// Row-major contiguous strides in elements (dense view).
    /// For quantized tensors, these reflect logical element strides.
    pub strides: Vec<usize>,
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
    #[cfg(not(feature = "cuda"))]
    pub host_weights: Vec<u8>,
    pub model_config: ModelConfig,
    #[cfg(feature = "gguf_ext")]
    pub typed_config: gguf_llms::model::ModelConfig,
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

        // DType checks: we support FP16 or Q5_1 weights for GEMM paths
        for (name, t) in [
            ("wq", &wq),
            ("wk", &wk),
            ("wv", &wv),
            ("wo", &wo),
            ("w_gate", &w_gate),
            ("w_up", &w_up),
            ("w_down", &w_down),
        ] {
            if t.dtype != GgmlDType::F16 && t.dtype != GgmlDType::Q5_1 {
                anyhow::bail!("tensor {} expected F16 or Q5_1, got {:?}", name, t.dtype);
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
    /// Map the language modeling head (output projection) tensor if present.
    /// Prefers a dedicated output.weight/lm_head.weight [d_model, vocab] in F16.
    /// Returns (tensor view, d_model, vocab, tied_to_embeddings=false).
    pub fn map_lm_head(&self) -> Result<(DeviceTensorView, usize, usize, bool)> {
        use crate::gguf::GgmlDType;
        let candidates = vec![
            "output.weight".to_string(),
            "lm_head.weight".to_string(),
            "output".to_string(),
        ];
        if let Ok(t) = self.find_tensor_any(&candidates) {
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
            return Ok((t.clone(), d_model, vocab, false));
        }
        anyhow::bail!(
            "lm_head tensor not found; expected one of {}",
            candidates.join(", ")
        )
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
        #[cfg(feature = "cuda")]
        let d_base = cuda.upload_weights(weights_bytes)?;
        // Validate that all known-sized tensors fit within weights_bytes
        for t in &gguf.tensors {
            if let Some(layout) = dtype_size_bytes(t.dtype) {
                let n_elems: u64 = t.shape.iter().copied().product::<u64>();
                let n_elems: usize = usize::try_from(n_elems)
                    .context("tensor element count does not fit in usize")?;
                let n_blocks = (n_elems + layout.block_elems - 1) / layout.block_elems;
                let need = n_blocks
                    .checked_mul(layout.block_bytes)
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
        #[cfg(feature = "cuda")]
        let device_tensors = build_device_tensor_views(&gguf.tensors, d_base, weights_bytes.len())?;
        #[cfg(not(feature = "cuda"))]
        let device_tensors =
            build_device_tensor_views(&gguf.tensors, std::ptr::null_mut(), weights_bytes.len())?;
        #[cfg(feature = "gguf_ext")]
        let (model_config, typed_cfg) =
            ModelConfig::from_gguf_ext_bytes(&gguf_bytes, &gguf.metadata, &gguf.tensors)?;
        #[cfg(not(feature = "gguf_ext"))]
        let model_config = ModelConfig::from_metadata(&gguf.metadata, &gguf.tensors)?;

        Ok(Self {
            gguf,
            cuda,
            kv_cache: None,
            device_tensors,
            #[cfg(feature = "cuda")]
            d_weights_base: d_base,
            #[cfg(not(feature = "cuda"))]
            host_weights: weights_bytes.to_vec(),
            model_config,
            #[cfg(feature = "gguf_ext")]
            typed_config: typed_cfg,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DTypeLayout {
    block_elems: usize,
    block_bytes: usize,
}

fn dtype_size_bytes(dt: GgmlDType) -> Option<DTypeLayout> {
    match dt {
        GgmlDType::F32 => Some(DTypeLayout {
            block_elems: 1,
            block_bytes: 4,
        }),
        GgmlDType::F16 => Some(DTypeLayout {
            block_elems: 1,
            block_bytes: 2,
        }),
        // GGUF/ggml quantization block sizes (see ggml-common.h)
        GgmlDType::Q4_0 => Some(DTypeLayout {
            block_elems: 32,
            block_bytes: 18,
        }),
        GgmlDType::Q4_1 => Some(DTypeLayout {
            block_elems: 32,
            block_bytes: 20,
        }),
        GgmlDType::Q5_0 => Some(DTypeLayout {
            block_elems: 32,
            block_bytes: 22,
        }),
        GgmlDType::Q5_1 => Some(DTypeLayout {
            block_elems: 32,
            block_bytes: 36,
        }),
        GgmlDType::Q8_0 => Some(DTypeLayout {
            block_elems: 32,
            block_bytes: 34,
        }),
        GgmlDType::Q8_1 => Some(DTypeLayout {
            block_elems: 32,
            block_bytes: 36,
        }),
        GgmlDType::Q2K => Some(DTypeLayout {
            block_elems: 256,
            block_bytes: 84,
        }),
        GgmlDType::Q3K => Some(DTypeLayout {
            block_elems: 256,
            block_bytes: 110,
        }),
        GgmlDType::Q4K => Some(DTypeLayout {
            block_elems: 256,
            block_bytes: 144,
        }),
        GgmlDType::Q5K => Some(DTypeLayout {
            block_elems: 256,
            block_bytes: 176,
        }),
        GgmlDType::Q6K => Some(DTypeLayout {
            block_elems: 256,
            block_bytes: 210,
        }),
        GgmlDType::Q8K => Some(DTypeLayout {
            block_elems: 256,
            block_bytes: 292,
        }),
        _ => None,
    }
}

fn contiguous_strides(shape: &[u64]) -> Result<Vec<usize>> {
    let mut strides_rev = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for (dim_idx, &dim) in shape.iter().enumerate().rev() {
        strides_rev.push(stride);
        let dim_usize =
            usize::try_from(dim).with_context(|| format!("dim {dim_idx} does not fit in usize"))?;
        stride = stride
            .checked_mul(dim_usize)
            .context("stride overflow while building tensor view")?;
    }
    strides_rev.reverse();
    Ok(strides_rev)
}

#[allow(clippy::collapsible_if)]
fn build_device_tensor_views(
    tensors: &[GgufTensor],
    #[allow(unused_variables)] d_base: *mut c_void,
    weights_len: usize,
) -> Result<HashMap<String, DeviceTensorView>> {
    let mut map = HashMap::with_capacity(tensors.len());
    for t in tensors {
        // Compute size and perform bounds/alignment checks
        let offset_usize: usize = t
            .offset
            .try_into()
            .context("tensor offset does not fit in usize")?;
        // Alignment by dtype
        let align = match t.dtype {
            GgmlDType::F16 => 2usize,
            GgmlDType::F32 => 4usize,
            _ => 1usize,
        };
        if align > 1 && !offset_usize.is_multiple_of(align) {
            anyhow::bail!(
                "tensor '{}' offset {} misaligned for {:?} (align {})",
                t.name,
                offset_usize,
                t.dtype,
                align
            );
        }
        let layout = dtype_size_bytes(t.dtype)
            .with_context(|| format!("tensor '{}' unsupported dtype: {:?}", t.name, t.dtype))?;
        // Known element size: check shape product and bounds within weights_len
        let n_elems_u64: u64 = t.shape.iter().copied().product::<u64>();
        let n_elems: usize =
            usize::try_from(n_elems_u64).context("tensor element count does not fit in usize")?;
        let n_blocks = (n_elems + layout.block_elems - 1) / layout.block_elems;
        let need = n_blocks
            .checked_mul(layout.block_bytes)
            .context("tensor size overflow")?;
        let end = offset_usize
            .checked_add(need)
            .context("tensor end offset overflow")?;
        let nbytes = need;
        let end_ok = end <= weights_len;
        if !end_ok {
            anyhow::bail!(
                "tensor '{}' overflows weights blob or starts beyond end (off={}, nbytes={}, total={})",
                t.name, offset_usize, nbytes, weights_len
            );
        }
        let strides = contiguous_strides(&t.shape)
            .with_context(|| format!("tensor '{}' stride computation failed", t.name))?;
        // Safe device pointer arithmetic
        #[cfg(feature = "cuda")]
        let dptr: *mut c_void = {
            if d_base.is_null() {
                anyhow::bail!("device weights base pointer is null");
            }
            let base = d_base as usize;
            let addr = base
                .checked_add(offset_usize)
                .context("device pointer offset overflow")?;
            addr as *mut c_void
        };
        #[cfg(not(feature = "cuda"))]
        let _dptr: *mut c_void = std::ptr::null_mut();
        let view = DeviceTensorView {
            dtype: t.dtype,
            shape: t.shape.clone(),
            strides,
            byte_offset: t.offset,
            nbytes,
            #[cfg(feature = "cuda")]
            dptr,
        };
        map.insert(t.name.clone(), view);
    }
    Ok(map)
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
        let num_heads = self.model_config.attention_head_count;
        let d_model = self.model_config.embedding_length;
        if num_heads == 0 || d_model == 0 || !d_model.is_multiple_of(num_heads) {
            anyhow::bail!(
                "model_config invalid for KV layout: d_model {} head_count {}",
                d_model,
                num_heads
            );
        }
        let head_dim = self.model_config.attention_key_length;
        if head_dim == 0 {
            anyhow::bail!("attention_key_length must be > 0");
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
            let bytes_d = d_model * 4;
            let bytes_h = (hidden_dim as usize) * 4;
            let dq = self.cuda.device_malloc_tagged(bytes_d, "fwd:dq_f32")?;
            let dk = self.cuda.device_malloc_tagged(bytes_d, "fwd:dk_f32")?;
            let dv = self.cuda.device_malloc_tagged(bytes_d, "fwd:dv_f32")?;
            let datt = self.cuda.device_malloc_tagged(bytes_d, "fwd:datt_f32")?; // attention output
            let dy_attn = self.cuda.device_malloc_tagged(bytes_d, "fwd:dy_attn_f32")?; // after out-proj
            let dgate = self.cuda.device_malloc_tagged(bytes_h, "fwd:dgate_f32")?;
            let dup = self.cuda.device_malloc_tagged(bytes_h, "fwd:dup_f32")?;
            let dhid = self.cuda.device_malloc_tagged(bytes_h, "fwd:dhid_f32")?;
            let dy_mlp = self.cuda.device_malloc_tagged(bytes_d, "fwd:dy_mlp_f32")?;
            let d_xn = self.cuda.device_malloc_tagged(bytes_d, "fwd:d_xn_f32")?; // pre-attn norm(x)
            let mut d_x1 = self.cuda.device_malloc_tagged(bytes_d, "fwd:d_x1_f32")?; // x + attn(xn)
            let d_x1n = self.cuda.device_malloc_tagged(bytes_d, "fwd:d_x1n_f32")?; // norm(x1)

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
            let n_vocab = tok.shape[0];
            if token_id >= n_vocab {
                anyhow::bail!(
                    "token_id {} out of range for vocab size {}",
                    token_id,
                    n_vocab
                );
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
                        // Column gather: shape [d_model, vocab], take column = token_id
                        use half::f16;
                        let row_stride = r1 * 2; // vocab elements per row (f16)
                        let mut out = vec![0f32; d_model];
                        for i in 0..d_model {
                            let elem_off =
                                (tok.dptr as usize) + i * row_stride + (token_id as usize) * 2;
                            let mut bytes = [0u8; 2];
                            self.cuda.memcpy_d2h(
                                bytes.as_mut_ptr() as *mut c_void,
                                elem_off as *const c_void,
                                2,
                            )?;
                            let bits = (bytes[0] as u16) | ((bytes[1] as u16) << 8);
                            out[i] = f16::from_bits(bits).to_f32();
                        }
                        let out_bytes = d_model * std::mem::size_of::<f32>();
                        self.cuda.memcpy_h2d(
                            d_out_f32,
                            out.as_ptr() as *const c_void,
                            out_bytes,
                        )?;
                    }
                }
                GgmlDType::Q8_0 => {
                    if rows_are_vocab {
                        // Row dequant: shape [vocab, d_model], blocks along d_model
                        let blocks = (d_model + 31) / 32;
                        let row_bytes = blocks * 36; // 4 + 32 bytes per block
                        let d_row =
                            (tok.dptr as usize + (token_id as usize) * row_bytes) as *const c_void;
                        // Pull the quantized row bytes to host
                        let mut qrow = vec![0u8; row_bytes];
                        self.cuda
                            .memcpy_d2h(qrow.as_mut_ptr() as *mut c_void, d_row, row_bytes)?;
                        // Dequantize to f32 on host
                        let mut out = vec![0f32; d_model];
                        for blk in 0..blocks {
                            let base = blk * 36;
                            let d = f32::from_le_bytes([
                                qrow[base + 0],
                                qrow[base + 1],
                                qrow[base + 2],
                                qrow[base + 3],
                            ]);
                            for idx in 0..32 {
                                let i = blk * 32 + idx;
                                if i >= d_model {
                                    break;
                                }
                                let q = qrow[base + 4 + idx] as i8 as f32;
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
                        // Column dequant: shape [d_model, vocab], blocks along vocab
                        let vocab = r1;
                        let blocks = (vocab + 31) / 32;
                        let row_stride = blocks * 36; // bytes per row
                        let b = (token_id as usize) / 32;
                        let idx = (token_id as usize) % 32;
                        let mut out = vec![0f32; d_model];
                        let mut block_buf = [0u8; 36];
                        for i in 0..d_model {
                            let row_base = (tok.dptr as usize) + i * row_stride;
                            let blk_off = row_base + b * 36;
                            self.cuda.memcpy_d2h(
                                block_buf.as_mut_ptr() as *mut c_void,
                                blk_off as *const c_void,
                                36,
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
        {
            let wq_ptr =
                (self.d_weights_base as usize + w.wq.byte_offset as usize) as *const c_void;
            let wk_ptr =
                (self.d_weights_base as usize + w.wk.byte_offset as usize) as *const c_void;
            let wv_ptr =
                (self.d_weights_base as usize + w.wv.byte_offset as usize) as *const c_void;
            let wo_ptr =
                (self.d_weights_base as usize + w.wo.byte_offset as usize) as *const c_void;
            let w_gate_ptr =
                (self.d_weights_base as usize + w.w_gate.byte_offset as usize) as *const c_void;
            let w_up_ptr =
                (self.d_weights_base as usize + w.w_up.byte_offset as usize) as *const c_void;
            let w_down_ptr =
                (self.d_weights_base as usize + w.w_down.byte_offset as usize) as *const c_void;

            #[cfg(feature = "cuda")]
            {
                // Allocate scratch buffers for attention output and intermediate results
                let bytes_d = (w.d_model as usize) * 4;
                let d_y_attn = self
                    .cuda
                    .device_malloc_tagged(bytes_d, "fwd:d_y_attn_f32")?;
                let d_x1 = self.cuda.device_malloc_tagged(bytes_d, "fwd:d_x1_f32")?;

                self.forward_one_token_minimal(
                    d_x1,
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
                )?;

                // Free scratch buffers
                self.cuda.device_free(d_y_attn)?;
                self.cuda.device_free(d_x1)?;
            }

            // Residual add on host: x1 = x + y_attn (fallback)
            let bytes_d = (w.d_model as usize) * 4;
            let mut h_x = vec![0u8; bytes_d];
            let mut h_y_attn = vec![0u8; bytes_d];
            self.cuda
                .memcpy_d2h(h_x.as_mut_ptr() as *mut c_void, d_x_f32, bytes_d)?;
            self.cuda.memcpy_d2h(
                h_y_attn.as_mut_ptr() as *mut c_void,
                d_y_attn as *const c_void,
                bytes_d,
            )?;
            let mut x_f = Vec::with_capacity(d_model);
            let mut y_attn_f = Vec::with_capacity(d_model);
            for ch in h_x.chunks_exact(4) {
                x_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
            }
            for ch in h_y_attn.chunks_exact(4) {
                y_attn_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
            }
            let mut x1_f = Vec::with_capacity(d_model);
            for i in 0..d_model {
                x1_f.push(x_f[i] + y_attn_f[i]);
            }
            let mut x1_bytes = Vec::with_capacity(bytes_d);
            for v in &x1_f {
                x1_bytes.extend_from_slice(&v.to_le_bytes());
            }
            self.cuda.memcpy_h2d(
                d_x_f32 as *mut c_void,
                x1_bytes.as_ptr() as *const c_void,
                bytes_d,
            )?;

            // Post-attention norm
            let mut h_x1 = vec![0u8; bytes_d];
            self.cuda.memcpy_d2h(
                h_x1.as_mut_ptr() as *mut c_void,
                d_x1 as *const c_void,
                bytes_d,
            )?;
            let mut x1_f = Vec::with_capacity(d_model);
            for ch in h_x1.chunks_exact(4) {
                x1_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
            }
            let mut mean_sq2 = 0f32;
            for v in &x1_f {
                mean_sq2 += v * v;
            }
            mean_sq2 /= d_model as f32;
            let scale2 = 1.0f32 / (mean_sq2 + eps).sqrt();
            let mut x1n = vec![0f32; d_model];
            for i in 0..d_model {
                x1n[i] = x1_f[i] * scale2;
            }

            // Free scratch tensors
            let _ = self.cuda.device_free(d_y_attn);
        }
        #[cfg(not(feature = "cuda"))]
        {
            use half::f16;
            use std::slice;
            let d_model = w.d_model;
            let hidden_dim = w.hidden_dim;

            // Read input x (f32) from host pointer
            let x_slice = slice::from_raw_parts(d_x_f32 as *const f32, d_model);

            // RMSNorm
            let eps = 1e-6f32;
            let mut mean_sq = 0f32;
            for &v in x_slice {
                mean_sq += v * v;
            }
            mean_sq /= d_model as f32;
            let scale = 1.0f32 / (mean_sq + eps).sqrt();
            let mut x_n = vec![0f32; d_model];
            for i in 0..d_model {
                x_n[i] = x_slice[i] * scale;
            }

            // Helper to multiply f32 row (1xK) by f16 matrix (KxN) into f32 (1xN)
            let dot_row = |row: &[f32], t: &crate::infer::DeviceTensorView, n: usize| -> Vec<f32> {
                let k = t.shape[0] as usize;
                debug_assert_eq!(k, row.len());
                let off = t.byte_offset as usize;
                let bytes = &self.host_weights[off..off + k * n * 2];
                let mut out = vec![0f32; n];
                for (j, out_j) in out.iter_mut().enumerate().take(n) {
                    let mut acc = 0f32;
                    let mut idx = j * 2; // column-major stride over rows in row-major [k x n]
                    for &row_val in row.iter().take(k) {
                        let lo = bytes[idx] as u16;
                        let hi = bytes[idx + 1] as u16;
                        let w = f16::from_bits(lo | (hi << 8)).to_f32();
                        acc += row_val * w;
                        idx += n * 2;
                    }
                    *out_j = acc;
                }
                out
            };

            // Helper to multiply f32 row (1xK) by Q5_1 matrix (KxN) into f32 (1xN)
            let dot_row_q5_1 =
                |row: &[f32], t: &crate::infer::DeviceTensorView, n: usize| -> Vec<f32> {
                    let k = t.shape[0] as usize;
                    debug_assert_eq!(k, row.len());
                    let off = t.byte_offset as usize;
                    let mut out = vec![0f32; n];

                    let blocks = k.div_ceil(32);
                    for (j, out_j) in out.iter_mut().enumerate().take(n) {
                        let mut acc = 0f32;
                        for (r, &row_val) in row.iter().enumerate().take(k) {
                            let b = r / 32;
                            let idx = r % 32;
                            let block_base = off + j * blocks * 6 + b * 6;
                            let scale_bytes = &self.host_weights[block_base..block_base + 4];
                            let scale = f32::from_le_bytes([
                                scale_bytes[0],
                                scale_bytes[1],
                                scale_bytes[2],
                                scale_bytes[3],
                            ]);
                            let q = self.host_weights[block_base + 4 + idx] as i8 as f32;
                            acc += row_val * scale * q;
                        }
                        *out_j = acc;
                    }

                    out
                };

            // Q, K, V
            let nq = w.wq.shape[1] as usize;
            let nk = w.wk.shape[1] as usize;
            let nv = w.wv.shape[1] as usize;
            let q = if w.wq.dtype == GgmlDType::Q5_1 {
                dot_row_q5_1(&x_n, &w.wq, nq)
            } else {
                dot_row(&x_n, &w.wq, nq)
            };
            let k_vec = if w.wk.dtype == GgmlDType::Q5_1 {
                dot_row_q5_1(&x_n, &w.wk, nk)
            } else {
                dot_row(&x_n, &w.wk, nk)
            };
            let v_vec = if w.wv.dtype == GgmlDType::Q5_1 {
                dot_row_q5_1(&x_n, &w.wv, nv)
            } else {
                dot_row(&x_n, &w.wv, nv)
            };

            // Append KV (host path)
            self.append_kv_token_f32_from_host(seq_id, &k_vec, &v_vec)?;

            // Attention over last token using KV cache helper
            let kv = self
                .kv_cache
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("kv_cache not allocated"))?;
            let num_heads = kv.num_heads();
            let head_dim = kv.head_dim();
            let mut attn_out = vec![0u8; d_model * 4];
            kv.attention_last_token_f32(
                &self.cuda,
                seq_id,
                q.as_ptr() as *const c_void,
                seq_len,
                attn_out.as_mut_ptr() as *mut c_void,
            )?;
            let attn: Vec<f32> = attn_out
                .chunks_exact(4)
                .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
                .collect();
            let _ = (num_heads, head_dim); // already validated during allocation

            // Out projection and residual add: x1 = x + attn·Wo
            let no = w.wo.shape[1] as usize;
            debug_assert_eq!(no, d_model);
            let y_attn = if w.wo.dtype == GgmlDType::Q5_1 {
                dot_row_q5_1(&attn, &w.wo, no)
            } else {
                dot_row(&attn, &w.wo, no)
            };
            // Residual add on host: x1 = x + y_attn (fallback)
            let bytes_d = d_model * 4;
            let mut h_x = vec![0u8; bytes_d];
            let mut h_y_attn = vec![0u8; bytes_d];
            self.cuda
                .memcpy_d2h(h_x.as_mut_ptr() as *mut c_void, d_x_f32, bytes_d)?;
            // In non-CUDA path, we don't have d_y_attn, so we use y_attn directly
            for i in 0..y_attn.len() {
                let bytes = y_attn[i].to_le_bytes();
                h_y_attn[i * 4..i * 4 + 4].copy_from_slice(&bytes);
            }
            let mut x_f = Vec::with_capacity(d_model);
            let mut y_attn_f = Vec::with_capacity(d_model);
            for ch in h_x.chunks_exact(4) {
                x_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
            }
            for ch in h_y_attn.chunks_exact(4) {
                y_attn_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
            }
            let mut x1_f = Vec::with_capacity(d_model);
            for i in 0..d_model {
                x1_f.push(x_f[i] + y_attn_f[i]);
            }
            let mut x1_bytes = Vec::with_capacity(bytes_d);
            for v in &x1_f {
                x1_bytes.extend_from_slice(&v.to_le_bytes());
            }
            self.cuda.memcpy_h2d(
                d_x_f32 as *mut c_void,
                x1_bytes.as_ptr() as *const c_void,
                bytes_d,
            )?;

            // Post-attention norm
            let mut h_x1 = vec![0u8; bytes_d];
            self.cuda.memcpy_d2h(
                h_x1.as_mut_ptr() as *mut c_void,
                h_x1.as_ptr() as *const c_void,
                bytes_d,
            )?;
            let mut x1_f = Vec::with_capacity(d_model);
            for ch in h_x1.chunks_exact(4) {
                x1_f.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
            }
            let mut mean_sq2 = 0f32;
            for v in &x1_f {
                mean_sq2 += v * v;
            }
            mean_sq2 /= d_model as f32;
            let scale2 = 1.0f32 / (mean_sq2 + eps).sqrt();
            let mut x1n = vec![0f32; d_model];
            for i in 0..d_model {
                x1n[i] = x1_f[i] * scale2;
            }

            // MLP: gate/up -> SiLU(gate)*up -> down -> residual add with x1n
            let h = w.w_up.shape[1] as usize;
            debug_assert_eq!(h, hidden_dim);
            let gate = if w.w_gate.dtype == GgmlDType::Q5_1 {
                dot_row_q5_1(&x1n, &w.w_gate, h)
            } else {
                dot_row(&x1n, &w.w_gate, h)
            };
            let up = if w.w_up.dtype == GgmlDType::Q5_1 {
                dot_row_q5_1(&x1n, &w.w_up, h)
            } else {
                dot_row(&x1n, &w.w_up, h)
            };
            fn sigmoid(v: f32) -> f32 {
                1.0 / (1.0 + (-v).exp())
            }
            let mut hidden = vec![0f32; h];
            for i in 0..h {
                let g = gate[i];
                hidden[i] = (g * sigmoid(g)) * up[i];
            }
            let y_mlp = if w.w_down.dtype == GgmlDType::Q5_1 {
                dot_row_q5_1(&hidden, &w.w_down, d_model)
            } else {
                dot_row(&hidden, &w.w_down, d_model)
            };
            let mut y = vec![0f32; d_model];
            for i in 0..d_model {
                y[i] = x1n[i] + y_mlp[i];
            }

            // Write to output pointer
            let out_slice = slice::from_raw_parts_mut(d_out_f32 as *mut f32, d_model);
            out_slice.copy_from_slice(&y);

            // Free scratch buffers

            Ok(())
        }
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

impl LoadedModel {
    /// # Safety
    /// Compute logits = H (1xD f32) × W^T (D×V f16) -> (1xV f32) using device GEMM and return host Vec<f32>.
    pub unsafe fn logits_from_hidden_gpu(&self, d_hidden_f32: *const c_void) -> Result<Vec<f32>> {
        let lm = self.map_lm_head()?.0;
        let d_model = self.model_config.embedding_length as usize;
        let vocab = lm.shape[1] as usize;
        #[cfg(feature = "cuda")]
        let d_w = (self.d_weights_base as usize + lm.byte_offset as usize) as *const c_void;
        #[cfg(not(feature = "cuda"))]
        let d_w: *const c_void = std::ptr::null();
        #[cfg(not(feature = "cuda"))]
        let _ = &lm;
        let bytes_logits = vocab * std::mem::size_of::<f32>();
        let d_logits = self.cuda.device_malloc(bytes_logits)?;
        self.matmul_f32xf16_f32(d_hidden_f32, d_w, d_logits, 1, vocab as i32, d_model as i32)?;
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
        if let Ok((lm, d_model, vocab, _)) = self.map_lm_head() {
            #[cfg(feature = "cuda")]
            {
                let d_w = (self.d_weights_base as usize + lm.byte_offset as usize) as *const c_void;
                let bytes_logits = vocab * std::mem::size_of::<f32>();
                let d_logits = self.cuda.device_malloc(bytes_logits)?;
                self.matmul_f32xf16_f32(
                    d_hidden_f32,
                    d_w,
                    d_logits,
                    1,
                    vocab as i32,
                    d_model as i32,
                )?;
                let mut host = vec![0u8; bytes_logits];
                self.cuda
                    .memcpy_d2h(host.as_mut_ptr() as *mut c_void, d_logits, bytes_logits)?;
                let mut logits = Vec::with_capacity(vocab);
                for ch in host.chunks_exact(4) {
                    logits.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
                }
                let _ = self.cuda.device_free(d_logits);
                return Ok(logits);
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
                    let mut idx = col * 2; // element (row=0,col) byte index within the 2-byte stream
                    for &hidden_val in hidden.iter().take(d_model) {
                        let lo = w_bytes[idx] as u16;
                        let hi = w_bytes[idx + 1] as u16;
                        let bits = lo | (hi << 8);
                        let w = half::f16::from_bits(bits).to_f32();
                        acc += hidden_val * w;
                        idx += vocab * 2; // move to next row at same column
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
                let row_dev = (self.d_weights_base as usize
                    + tok.byte_offset as usize
                    + v * row_bytes) as *const c_void;
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
