#![cfg_attr(not(all(feature = "cuda", nvcc)), allow(dead_code))]

#[cfg(all(feature = "cuda", nvcc))]
use anyhow::ensure;
use anyhow::Result;
use clap::ValueEnum;
use m40_llm::{
    generate::PromptPrefill,
    infer::{q8_prefill::Q8PrefillBackend, Q8AttentionProjectionBackend, Q8FfnConfig},
};
use serde::Serialize;

#[cfg(all(feature = "cuda", nvcc))]
#[path = "q8_prepared_control.rs"]
mod prepared_control;
#[cfg(all(feature = "cuda", nvcc))]
pub use prepared_control::validate_prepared_control;

/// Reject quality evidence collected before no-logits prefix input reuse was
/// ordered. Historical reports remain readable, but cannot qualify a new run.
pub const PREFILL_EXECUTION_CONTRACT: &str = "prefix-decode-completed-v1";

/// Only the bounded M03/M04 candidates that passed the complete-FFN screen.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FfnChoice {
    #[default]
    Native,
    PackedF32_256,
    PackedF32_128,
    #[value(name = "split-f32x4-128")]
    #[serde(rename = "split-f32x4-128")]
    SplitF32x4,
    #[value(name = "q8-1-128")]
    Q81_128,
    #[value(name = "q8-1-256")]
    Q81_256,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrefillChoice {
    #[default]
    Sequential,
    NativePacked,
    DequantSgemm,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AttentionProjectionChoice {
    #[default]
    Native,
    #[value(name = "warp-256")]
    #[serde(rename = "warp-256")]
    Warp256,
    #[value(name = "packed-warp-256")]
    #[serde(rename = "packed-warp-256")]
    PackedWarp256,
}

impl AttentionProjectionChoice {
    pub fn backend(self) -> Q8AttentionProjectionBackend {
        match self {
            Self::Native => Q8AttentionProjectionBackend::Native,
            Self::Warp256 => Q8AttentionProjectionBackend::Warp256,
            Self::PackedWarp256 => Q8AttentionProjectionBackend::PackedWarp256,
        }
    }
}

/// Historical controls remain explicit. New projection work names the current
/// split-FFN/SGEMM control instead of silently substituting an older baseline.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ControlChoice {
    #[default]
    NativeSequential,
    QualifiedPacked128Sgemm,
    QualifiedSplitSgemm,
    QualifiedWarpSgemm,
}

impl ControlChoice {
    pub fn config(self) -> RuntimeConfig {
        match self {
            Self::NativeSequential => RuntimeConfig::default(),
            Self::QualifiedPacked128Sgemm => RuntimeConfig {
                ffn: FfnChoice::PackedF32_128,
                prefill: PrefillChoice::DequantSgemm,
                attention_projection: AttentionProjectionChoice::Native,
            },
            Self::QualifiedSplitSgemm => RuntimeConfig {
                ffn: FfnChoice::SplitF32x4,
                prefill: PrefillChoice::DequantSgemm,
                attention_projection: AttentionProjectionChoice::Native,
            },
            Self::QualifiedWarpSgemm => RuntimeConfig {
                ffn: FfnChoice::SplitF32x4,
                prefill: PrefillChoice::DequantSgemm,
                attention_projection: AttentionProjectionChoice::Warp256,
            },
        }
    }
}

/// Typed decode/prefill/projection choices, with pinned existing preparation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RuntimeConfig {
    pub ffn: FfnChoice,
    pub prefill: PrefillChoice,
    pub attention_projection: AttentionProjectionChoice,
}

impl RuntimeConfig {
    pub fn validate_comparison(self, control: Self) -> Result<()> {
        anyhow::ensure!(
            match self.attention_projection {
                AttentionProjectionChoice::Native => true,
                AttentionProjectionChoice::Warp256 => control == ControlChoice::QualifiedSplitSgemm.config(),
                AttentionProjectionChoice::PackedWarp256 => control == ControlChoice::QualifiedWarpSgemm.config(),
            },
            "projection comparison requires qualified-split-sgemm for warp-256 or qualified-warp-sgemm for packed-warp-256 as the explicit control"
        );
        Ok(())
    }

    pub fn ffn_config(self) -> Result<Q8FfnConfig> {
        match self.ffn {
            FfnChoice::Native => Ok(Q8FfnConfig::native()),
            FfnChoice::PackedF32_256 => Q8FfnConfig::packed_f32(256),
            FfnChoice::PackedF32_128 => Q8FfnConfig::packed_f32(128),
            FfnChoice::SplitF32x4 => Ok(Q8FfnConfig::split_f32x4()),
            FfnChoice::Q81_128 => Q8FfnConfig::q8_1(128),
            FfnChoice::Q81_256 => Q8FfnConfig::q8_1(256),
        }
    }

    pub fn prefill_backend(self) -> Q8PrefillBackend {
        match self.prefill {
            PrefillChoice::DequantSgemm => Q8PrefillBackend::DequantSgemm,
            _ => Q8PrefillBackend::Native,
        }
    }

    pub fn prompt_prefill(self) -> PromptPrefill {
        match self.prefill {
            PrefillChoice::Sequential => PromptPrefill::Sequential,
            _ => PromptPrefill::PackedPrefix {
                max_prompt_tokens: 128,
            },
        }
    }

    pub fn is_numerical(self) -> bool {
        matches!(self.ffn, FfnChoice::Q81_128 | FfnChoice::Q81_256)
    }

    pub fn execution_config(self) -> Result<m40_llm::infer::execution_weights::Q8ExecutionConfig> {
        anyhow::ensure!(
            self.attention_projection == AttentionProjectionChoice::Native
                || (self.ffn == FfnChoice::SplitF32x4
                    && self.prefill == PrefillChoice::DequantSgemm),
            "warp runtime comparison requires qualified split-f32x4-128 plus dequant-sgemm"
        );
        m40_llm::infer::execution_weights::Q8ExecutionConfig::parse(
            "split",
            Some("6144"),
            Some(if self.ffn == FfnChoice::SplitF32x4 {
                "gate-up,down,vocabulary"
            } else {
                "down,vocabulary"
            }),
        )
    }

    #[cfg(all(feature = "cuda", nvcc))]
    pub fn validate_loaded_model(self, model: &m40_llm::infer::LoadedModel) -> Result<()> {
        ensure!(
            model.q8_ffn == self.ffn_config()?
                && model.q8_prefill == self.prefill_backend()
                && model.q8_attention_projection.backend() == self.attention_projection.backend(),
            "loaded runtime backends differ from the requested FFN/prefill/projection configuration"
        );
        Ok(())
    }

    /// Shared by primary and candidate loaders. Keep every execution choice at
    /// this boundary so a newly added axis cannot be omitted by one caller.
    #[cfg(all(feature = "cuda", nvcc))]
    pub fn load_weights(
        self,
        gguf: m40_llm::gguf::GgufModel,
        bytes: Vec<u8>,
    ) -> Result<m40_llm::infer::LoadedModel> {
        let model = m40_llm::infer::LoadedModel::from_gguf_with_q8_runtime(
            gguf,
            bytes,
            -1,
            self.execution_config()?,
            self.ffn_config()?,
            self.prefill_backend(),
            self.attention_projection.backend(),
        )?;
        self.validate_loaded_model(&model)?;
        Ok(model)
    }

    #[cfg(all(feature = "cuda", nvcc))]
    pub fn load_model(
        self,
        path: &std::path::Path,
        context: u32,
    ) -> Result<(m40_llm::infer::LoadedModel, m40_llm::tokenizer::Tokenizer)> {
        use m40_llm::{gguf, tokenizer::Tokenizer};
        let gguf = gguf::load_gguf(path)?;
        let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)?;
        ensure!(
            tokenizer.is_qwen2(),
            "Maxwell runtime comparison requires Qwen2 ChatML"
        );
        let mut model = self.load_weights(gguf, std::fs::read(path)?)?;
        let props = model.cuda.current_device_props()?;
        ensure!(
            props.major == 5 && props.minor == 2,
            "runtime comparison requires sm_52"
        );
        ensure!(
            context <= model.model_config.context_length,
            "context exceeds model capacity"
        );
        model.allocate_kv_cache_for_layers(context)?;
        model.validate_full_layer_decode()?;
        Ok((model, tokenizer))
    }
}

/// Frozen scientific control; reject hidden scheduling or head changes.
#[cfg(all(feature = "cuda", nvcc))]
pub fn validate_environment(
    environment: &std::collections::BTreeMap<String, String>,
) -> Result<()> {
    for (key, expected) in [
        ("M40LLM_ENABLE_NVCC", "1"),
        ("M40LLM_ENABLE_CUBLAS", "1"),
        ("M40LLM_PROJECTION_BACKEND", "large-model"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_Q8_EXECUTION_WEIGHTS", "off"),
        ("M40LLM_DECODE_GRAPH", "0"),
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_Q8_DECODE_COL_TILE", "4"),
        ("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4"),
        ("M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK", "4"),
        ("M40LLM_FUSED_Q8_QKV", "1"),
        ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
        ("M40LLM_CUDA_GREEDY_ARGMAX", "1"),
        ("M40LLM_Q8_LM_HEAD_ARGMAX", "1"),
        ("M40LLM_Q8_LM_HEAD_ARGMAX_SHARED_A", "1"),
        ("M40LLM_ATTN_HEAD128_KERNEL", "parallel"),
        ("M40LLM_ATTN_HEAD128_PARTITIONS", "8"),
        ("M40LLM_ATTN_HEAD128_THREADS", "256"),
    ] {
        ensure!(
            environment.get(key).map(String::as_str) == Some(expected),
            "runtime comparison requires qualified control {key}={expected}"
        );
    }
    for (key, value) in environment {
        ensure!(
            !key.starts_with("M40LLM_KV_"),
            "unset KV diagnostics for dense comparison"
        );
        ensure!(
            !key.starts_with("M40LLM_SERVER_BATCH_") || value == "0",
            "unset scheduler experiments for runtime comparison"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode and prefill selections remain independent, typed and immutable;
    /// the reference never inherits experimental activation quantization.
    #[test]
    fn test_runtime_choices_keep_reference_and_axes_separate() -> Result<()> {
        let control = RuntimeConfig::default();
        assert!(control.ffn_config()?.is_native());
        assert!(!control.is_numerical());
        assert_eq!(control.prompt_prefill(), PromptPrefill::Sequential);
        for ffn in [
            FfnChoice::Native,
            FfnChoice::PackedF32_256,
            FfnChoice::PackedF32_128,
            FfnChoice::SplitF32x4,
            FfnChoice::Q81_128,
            FfnChoice::Q81_256,
        ] {
            for prefill in [
                PrefillChoice::Sequential,
                PrefillChoice::NativePacked,
                PrefillChoice::DequantSgemm,
            ] {
                let config = RuntimeConfig {
                    ffn,
                    prefill,
                    ..RuntimeConfig::default()
                };
                assert_eq!(
                    config.ffn_config()?.has_quantization(),
                    config.is_numerical()
                );
                assert_eq!(
                    config.prefill_backend().is_native(),
                    prefill != PrefillChoice::DequantSgemm
                );
                assert_eq!(
                    config.prompt_prefill() == PromptPrefill::Sequential,
                    prefill == PrefillChoice::Sequential
                );
            }
        }
        Ok(())
    }

    /// The next incremental control is exactly split FFN/SGEMM. Projection
    /// changes keep all preparation and reject unrelated experiment mixtures.
    #[test]
    fn test_projection_candidate_pins_current_split_control() -> Result<()> {
        let control = ControlChoice::QualifiedSplitSgemm.config();
        assert_eq!(control.ffn, FfnChoice::SplitF32x4);
        assert_eq!(control.prefill, PrefillChoice::DequantSgemm);
        assert_eq!(
            control.attention_projection,
            AttentionProjectionChoice::Native
        );
        let candidate = RuntimeConfig {
            attention_projection: AttentionProjectionChoice::Warp256,
            ..control
        };
        assert_eq!(
            serde_json::to_value(candidate.execution_config()?)?,
            serde_json::to_value(control.execution_config()?)?
        );
        assert!(!candidate.is_numerical());
        candidate.validate_comparison(control)?;
        assert!(candidate
            .validate_comparison(ControlChoice::QualifiedPacked128Sgemm.config())
            .is_err());
        for config in [
            RuntimeConfig {
                ffn: FfnChoice::Native,
                ..candidate
            },
            RuntimeConfig {
                prefill: PrefillChoice::Sequential,
                ..candidate
            },
        ] {
            assert!(config.execution_config().is_err());
        }
        let warp_control = ControlChoice::QualifiedWarpSgemm.config();
        assert_eq!(warp_control, candidate);
        let packed = RuntimeConfig {
            attention_projection: AttentionProjectionChoice::PackedWarp256,
            ..warp_control
        };
        packed.validate_comparison(warp_control)?;
        assert!(packed.validate_comparison(control).is_err());
        assert!(!packed.is_numerical());
        assert_eq!(
            serde_json::to_value(packed.execution_config()?)?,
            serde_json::to_value(warp_control.execution_config()?)?
        );
        assert_eq!(
            packed.attention_projection.backend(),
            Q8AttentionProjectionBackend::PackedWarp256
        );
        Ok(())
    }

    /// The incremental control keeps both qualified axes, while split decode
    /// adds only gate/up preparation and no activation quantization.
    #[test]
    fn test_incremental_control_and_split_weight_selection_are_explicit() -> Result<()> {
        use m40_llm::infer::execution_weights::Q8ProjectionGroup;
        let control = ControlChoice::QualifiedPacked128Sgemm.config();
        assert_eq!(control.ffn_config()?, Q8FfnConfig::packed_f32(128)?);
        assert_eq!(control.prefill_backend(), Q8PrefillBackend::DequantSgemm);
        let split = RuntimeConfig {
            ffn: FfnChoice::SplitF32x4,
            ..control
        };
        assert_eq!(split.ffn_config()?, Q8FfnConfig::split_f32x4());
        assert_eq!(split.prompt_prefill(), control.prompt_prefill());
        assert!(!split.is_numerical());
        assert!(!control
            .execution_config()?
            .families
            .contains(Q8ProjectionGroup::GateUp));
        for group in [
            Q8ProjectionGroup::GateUp,
            Q8ProjectionGroup::Down,
            Q8ProjectionGroup::Vocabulary,
        ] {
            assert!(split.execution_config()?.families.contains(group));
        }
        assert!(!split
            .execution_config()?
            .families
            .contains(Q8ProjectionGroup::Qkv));
        Ok(())
    }
}
