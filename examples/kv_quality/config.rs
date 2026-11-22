use anyhow::{ensure, Result};
use m40_llm::kv_compression::KvCompressionConfig;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheArm {
    Dense,
    QuantizedAll,
    QuantizedTop8,
}

impl CacheArm {
    pub const ALL: [Self; 3] = [Self::Dense, Self::QuantizedAll, Self::QuantizedTop8];

    pub fn create_config(self) -> KvCompressionConfig {
        match self {
            Self::Dense => KvCompressionConfig::dense_reference(),
            Self::QuantizedAll => KvCompressionConfig {
                top_blocks: 64,
                ..Default::default()
            },
            Self::QuantizedTop8 => KvCompressionConfig::default(),
        }
    }

    #[cfg(all(feature = "cuda", nvcc))]
    pub fn get_attention_operation(self) -> &'static str {
        match self {
            Self::Dense => "attention_last_token_f32_gqa",
            _ => "attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct",
        }
    }
}

pub const ENVIRONMENT: &[(&str, &str)] = &[
    ("M40LLM_ENABLE_NVCC", "1"),
    ("M40LLM_ENABLE_CUBLAS", "1"),
    ("M40LLM_PROJECTION_BACKEND", "large-model"),
    ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
    ("M40LLM_Q8_EXECUTION_WEIGHTS", "off"),
    ("M40LLM_DECODE_GRAPH", "0"),
    ("M40LLM_Q8_FFN_KERNEL", "native"),
    ("M40LLM_Q8_PREFILL_BACKEND", "dequant-sgemm"),
    ("M40LLM_ATTN_HEAD128_KERNEL", "native"),
    ("M40LLM_Q8_DECODE_STREAM", "decode"),
    ("M40LLM_FUSED_Q8_QKV", "1"),
    ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
    ("M40LLM_Q8_DECODE_COL_TILE", "4"),
    ("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4"),
    ("M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK", "4"),
    ("M40LLM_CUDA_GREEDY_ARGMAX", "1"),
    ("M40LLM_Q8_LM_HEAD_ARGMAX", "1"),
    ("M40LLM_Q8_LM_HEAD_ARGMAX_SHARED_A", "1"),
];

pub fn validate_environment(environment: &BTreeMap<String, String>) -> Result<()> {
    for &(key, value) in ENVIRONMENT {
        ensure!(
            environment.get(key).map(String::as_str) == Some(value),
            "quality run requires {key}={value}"
        );
    }
    for key in environment.keys().filter(|key| key.starts_with("M40LLM_")) {
        ensure!(
            key == "M40LLM_SYSROOT" || ENVIRONMENT.iter().any(|&(name, _)| name == key),
            "unfrozen runtime setting {key}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use m40_llm::kv_compression::{KvCompressMode, KvExactOldAttention, KvExactOldBacking};

    /// Every arm has explicit attention semantics; only top-k coverage changes
    /// between the two mixed-cache arms, and no environment policy can override it.
    #[test]
    fn test_cache_arms_preserve_the_frozen_comparison() -> Result<()> {
        let dense = CacheArm::Dense.create_config();
        assert_eq!(dense.mode, KvCompressMode::Off);
        let all = CacheArm::QuantizedAll.create_config();
        let mut top8 = CacheArm::QuantizedTop8.create_config();
        assert_eq!(all.top_blocks, 64);
        assert_eq!(top8.top_blocks, 8);
        assert_eq!(top8.exact_old_backing, KvExactOldBacking::Fp16KQ4V);
        assert_eq!(
            top8.exact_old_attention,
            KvExactOldAttention::Fp16KQ4VDirect
        );
        top8.top_blocks = 64;
        assert_eq!(all, top8);
        for arm in CacheArm::ALL {
            arm.create_config().validate_runtime_support(3072, 128)?;
        }
        let mut environment = ENVIRONMENT
            .iter()
            .map(|&(k, v)| (k.into(), v.into()))
            .collect();
        validate_environment(&environment)?;
        environment.insert("M40LLM_KV_TOPK_POLICY".into(), "score-cluster".into());
        assert!(validate_environment(&environment).is_err());
        Ok(())
    }
}
