use crate::gguf::{GgmlDType, GgufScalar, GgufTensor, GgufValue};
use anyhow::Result;
use std::collections::HashMap;

pub fn get_u32_meta(metadata: &HashMap<String, GgufValue>, key: &str) -> Option<u32> {
    metadata.get(key).and_then(|v| match v {
        GgufValue::Scalar(GgufScalar::U32(x)) => Some(*x),
        GgufValue::Scalar(GgufScalar::I32(s)) => u32::try_from(*s).ok(),
        GgufValue::Scalar(GgufScalar::U64(x)) => u32::try_from(*x).ok(),
        GgufValue::Scalar(GgufScalar::I64(s)) => u32::try_from(*s).ok(),
        _ => None,
    })
}

pub fn get_u32_meta_any(metadata: &HashMap<String, GgufValue>, keys: &[&str]) -> Option<u32> {
    keys.iter().find_map(|key| get_u32_meta(metadata, key))
}

pub fn arch_key(architecture: &str, suffix: &str) -> String {
    format!("{architecture}.{suffix}")
}

pub fn derive_attention_head_count_kv(
    metadata: &HashMap<String, GgufValue>,
    tensors: &[GgufTensor],
    architecture: &str,
    d_model: u32,
    attention_head_count: u32,
    head_dim: u32,
) -> Result<u32> {
    let kv_key = arch_key(architecture, "attention.head_count_kv");
    if let Some(v) = get_u32_meta_any(metadata, &[&kv_key, "llama.attention.head_count_kv"]) {
        return Ok(v);
    }

    let candidates = [
        "layers.0.attention.wk.weight",
        "blk.0.attn_k.weight",
        "layers.0.attention.wv.weight",
        "blk.0.attn_v.weight",
    ];
    for name in candidates {
        if let Some(t) = tensors.iter().find(|t| t.name == name) {
            if t.shape.len() != 2 {
                anyhow::bail!("attention tensor {name} must be rank-2");
            }
            let k = t.shape[0] as u32;
            let n = t.shape[1] as u32;
            if k == d_model && n > 0 && n.is_multiple_of(head_dim) {
                return Ok(n / head_dim);
            }
        }
    }

    Ok(attention_head_count)
}

pub fn derive_vocab_size(
    metadata: &HashMap<String, GgufValue>,
    tensors: &[GgufTensor],
    architecture: &str,
    d_model: u32,
) -> Result<u32> {
    let vocab_key = arch_key(architecture, "vocab_size");
    if let Some(v) = get_u32_meta_any(metadata, &[&vocab_key, "llama.vocab_size"]) {
        if v == 0 {
            anyhow::bail!("vocab_size must be > 0");
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
    anyhow::bail!("vocab_size missing; add architecture vocab_size or embeddings tensor")
}

pub fn derive_feed_forward_length(
    metadata: &HashMap<String, GgufValue>,
    tensors: &[GgufTensor],
    architecture: &str,
    d_model: u32,
) -> Result<u32> {
    let ff_key = arch_key(architecture, "feed_forward_length");
    if let Some(v) = get_u32_meta_any(metadata, &[&ff_key, "llama.feed_forward_length"]) {
        if v == 0 {
            anyhow::bail!("feed_forward_length must be > 0");
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
        "feed_forward_length missing; add architecture feed_forward_length or a feed-forward weight tensor"
    )
}

pub fn get_f32_meta(metadata: &HashMap<String, GgufValue>, key: &str) -> Option<f32> {
    metadata.get(key).and_then(|v| match v {
        GgufValue::Scalar(GgufScalar::F32(x)) => Some(*x),
        _ => None,
    })
}

pub fn get_f32_meta_any(metadata: &HashMap<String, GgufValue>, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|key| get_f32_meta(metadata, key))
}

pub fn get_str_meta<'a>(metadata: &'a HashMap<String, GgufValue>, key: &str) -> Option<&'a str> {
    metadata.get(key).and_then(|v| v.as_str())
}

#[derive(Debug, Clone)]
pub struct DTypeLayout {
    pub block_elems: usize,
    pub block_bytes: usize,
}

pub fn dtype_size_bytes(dt: GgmlDType) -> Option<DTypeLayout> {
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

pub fn norm_weight_dtype_code(dtype: GgmlDType) -> Result<u32> {
    match dtype {
        GgmlDType::F16 => Ok(0),
        GgmlDType::F32 => Ok(1),
        other => anyhow::bail!("unsupported norm weight dtype: {:?}", other),
    }
}
