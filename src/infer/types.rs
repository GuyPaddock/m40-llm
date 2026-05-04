use super::meta::{
    derive_attention_head_count_kv, derive_feed_forward_length, derive_vocab_size, get_f32_meta,
    get_str_meta, get_u32_meta,
};
use crate::cuda::{CudaContext, KVCache};
use crate::gguf::{GgmlDType, GgufModel, GgufTensor, GgufValue};
use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
#[cfg(feature = "cuda")]
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
    pub(super) fn from_gguf_ext_bytes(
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
        if d_model % typed.attention_head_count != 0 {
            anyhow::bail!(
                "embedding_length {} not divisible by attention_head_count {}",
                d_model,
                typed.attention_head_count
            );
        }
        let head_kv = typed
            .attention_head_count_kv
            .unwrap_or(typed.attention_head_count);
        if typed.attention_head_count % head_kv != 0 {
            anyhow::bail!(
                "attention_head_count {} not divisible by attention_head_count_kv {}",
                typed.attention_head_count,
                head_kv
            );
        }
        let head_dim = typed
            .attention_key_length
            .unwrap_or_else(|| d_model / typed.attention_head_count);
        if d_model != typed.attention_head_count * head_dim {
            anyhow::bail!(
                "embedding_length {} must equal attention_head_count {} * attention_key_length {}",
                d_model,
                typed.attention_head_count,
                head_dim
            );
        }
        let vocab_size =
            derive_vocab_size(metadata, tensors, d_model).context("derive vocab_size from gguf")?;
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
        let head_dim = get_u32_meta(metadata, "llama.attention.key_length")
            .unwrap_or_else(|| embedding_length / attention_head_count);
        let head_kv = derive_attention_head_count_kv(
            metadata,
            tensors,
            embedding_length,
            attention_head_count,
            head_dim,
        )?;
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

// Functions moved to src/gguf_utils.rs

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
    pub weights_len: usize,
    #[cfg(feature = "cuda")]
    pub d_weights_base: *mut c_void,
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
    pub attn_norm: Option<DeviceTensorView>,
    pub ffn_norm: Option<DeviceTensorView>,
}
