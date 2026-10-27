//! Load-time selection for the bounded Maxwell single-row Q8 FFN candidates.
//! Weight preparation remains an independent model-local configuration.

use super::q8_activation::Q8ActivationShape;
use anyhow::{ensure, Result};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Q8FfnKernel {
    #[serde(rename = "native")]
    Native,
    #[serde(rename = "packed-f32")]
    PackedF32,
    #[serde(rename = "split-f32x4-128")]
    SplitF32x4,
    #[serde(rename = "joint-shared-scales-128")]
    JointSharedScales,
    #[serde(rename = "q8-1")]
    Q81,
}

/// Captured when loading the model. No per-token environment lookup or mutable
/// process-global selection participates in dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Q8FfnConfig {
    kernel: Q8FfnKernel,
    threads: Option<u32>,
}

impl Default for Q8FfnConfig {
    fn default() -> Self {
        Self::native()
    }
}

impl Q8FfnConfig {
    pub const fn native() -> Self {
        Self {
            kernel: Q8FfnKernel::Native,
            threads: None,
        }
    }

    pub fn packed_f32(threads: u32) -> Result<Self> {
        Self::candidate(Q8FfnKernel::PackedF32, threads)
    }

    pub fn q8_1(threads: u32) -> Result<Self> {
        Self::candidate(Q8FfnKernel::Q81, threads)
    }

    /// The split-plane screen qualifies one fixed FP32-activation geometry.
    pub const fn split_f32x4() -> Self {
        Self {
            kernel: Q8FfnKernel::SplitF32x4,
            threads: Some(128),
        }
    }

    /// The single compound consumer qualified by the four-arm FFN screen.
    pub const fn joint_shared_scales() -> Self {
        Self {
            kernel: Q8FfnKernel::JointSharedScales,
            threads: Some(128),
        }
    }

    pub(crate) fn validate_execution(
        self,
        execution: super::execution_weights::Q8ExecutionConfig,
    ) -> Result<()> {
        if matches!(
            self.kernel,
            Q8FfnKernel::SplitF32x4 | Q8FfnKernel::JointSharedScales
        ) {
            ensure!(
                execution.mode == super::execution_weights::Q8ExecutionMode::Split
                    && execution
                        .families
                        .contains(super::execution_weights::Q8ProjectionGroup::GateUp),
                "split/compound FFN requires M40LLM_Q8_EXECUTION_WEIGHTS=split and \
                 M40LLM_Q8_EXECUTION_FAMILIES including gate-up; \
                 use M40LLM_Q8_FFN_KERNEL=native to disable"
            );
        }
        Ok(())
    }

    fn candidate(kernel: Q8FfnKernel, threads: u32) -> Result<Self> {
        ensure!(
            matches!(threads, 128 | 256),
            "Q8 FFN candidates support only 128/256 threads"
        );
        Ok(Self {
            kernel,
            threads: Some(threads),
        })
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "native" => Ok(Self::native()),
            "packed-f32" => Self::packed_f32(256),
            "split-f32x4-128" => Ok(Self::split_f32x4()),
            "joint-shared-scales-128" => Ok(Self::joint_shared_scales()),
            "q8-1" => Self::q8_1(128),
            _ => anyhow::bail!(
                "M40LLM_Q8_FFN_KERNEL must be native, packed-f32, split-f32x4-128, joint-shared-scales-128, or q8-1"
            ),
        }
    }

    pub fn from_env() -> Result<Self> {
        match std::env::var("M40LLM_Q8_FFN_KERNEL") {
            Ok(value) => Self::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(Self::native()),
            Err(error) => Err(error.into()),
        }
    }

    pub fn kernel(self) -> Q8FfnKernel {
        self.kernel
    }
    pub fn threads(self) -> Option<u32> {
        self.threads
    }
    pub fn is_native(self) -> bool {
        self.kernel == Q8FfnKernel::Native
    }
    pub fn has_quantization(self) -> bool {
        self.kernel == Q8FfnKernel::Q81
    }

    /// Single-row decode scratch, shared across layers under workspace ownership.
    pub fn activation_bytes(self, d_model: usize) -> Result<usize> {
        if self.has_quantization() {
            Ok(Q8ActivationShape::new(1, d_model, d_model)?.bytes())
        } else {
            Ok(0)
        }
    }
}

/// Caller-owned optional scratch for the unsafe fused-FFN wrapper. Workspace
/// views report the actual allocation capacity, not inferred pointer validity.
#[derive(Debug, Clone, Copy)]
pub struct Q8FfnScratch {
    pub pointer: *mut std::ffi::c_void,
    pub bytes: usize,
}

#[cfg(test)]
mod compound_tests {
    use super::*;
    use crate::infer::execution_weights::Q8ExecutionConfig;

    /// The compound is explicit, fixed at 128 threads, non-quantizing and
    /// cannot inherit missing or mixed gate/up preparation as a fallback.
    #[test]
    fn test_compound_selection_requires_explicit_split_owners() -> Result<()> {
        assert!(Q8FfnConfig::default().is_native());
        let compound = Q8FfnConfig::parse("joint-shared-scales-128")?;
        assert_eq!(compound, Q8FfnConfig::joint_shared_scales());
        assert_eq!(compound.threads(), Some(128));
        assert!(!compound.has_quantization());
        assert_eq!(compound.activation_bytes(2048)?, 0);
        for config in [
            Q8ExecutionConfig::disabled(),
            Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
            Q8ExecutionConfig::parse("mixed", Some("6144"), Some("gate-up,down,vocabulary"))?,
        ] {
            assert!(compound.validate_execution(config).is_err());
        }
        compound.validate_execution(Q8ExecutionConfig::parse(
            "split",
            Some("6144"),
            Some("gate-up,down,vocabulary"),
        )?)?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl super::LoadedModel {
    pub(crate) fn validate_q8_ffn_backend(&self) -> Result<()> {
        if self.q8_ffn.is_native() {
            return Ok(());
        }
        let props = self.cuda.current_device_props()?;
        ensure!(
            props.major == 5 && props.minor == 2,
            "experimental Q8 FFN requires sm_52; use M40LLM_Q8_FFN_KERNEL=native"
        );
        ensure!(
            !self.d_weights_base.is_null(),
            "experimental Q8 FFN requires uploaded device weights; use M40LLM_Q8_FFN_KERNEL=native"
        );
        let (d_model, hidden) = self.validate_standard_layers().map_err(|error| {
            anyhow::anyhow!(
                "experimental Q8 FFN requires standard layers: {error}; use M40LLM_Q8_FFN_KERNEL=native"
            )
        })?;
        ensure!(
            self.q8_ffn.kernel() != Q8FfnKernel::JointSharedScales || d_model <= 4096,
            "joint-shared-scales-128 requires K <= 4096; use M40LLM_Q8_FFN_KERNEL=native"
        );
        ensure!(
            d_model > 0
                && d_model <= i32::MAX as usize - 255
                && d_model.is_multiple_of(32)
                && hidden > 0
                && hidden <= i32::MAX as usize,
            "experimental Q8 FFN requires bounded complete Q8_0 blocks; use M40LLM_Q8_FFN_KERNEL=native"
        );
        for layer in 0..self.model_config.block_count as usize {
            let mapped = self.map_standard_layer(layer)?;
            ensure!(
                mapped.w_gate.dtype == crate::gguf::GgmlDType::Q8_0
                    && mapped.w_up.dtype == crate::gguf::GgmlDType::Q8_0,
                "experimental Q8 FFN requires Q8_0 gate/up weights in layer {layer}; use M40LLM_Q8_FFN_KERNEL=native"
            );
            self.tensor_device_ptr("experimental FFN gate", &mapped.w_gate)?;
            self.tensor_device_ptr("experimental FFN up", &mapped.w_up)?;
        }
        Ok(())
    }

    pub(crate) fn validate_q8_ffn_prepared_weights(&self) -> Result<()> {
        if !matches!(
            self.q8_ffn.kernel(),
            Q8FfnKernel::SplitF32x4 | Q8FfnKernel::JointSharedScales
        ) {
            return Ok(());
        }
        let (k, hidden) = self.validate_standard_layers()?;
        for layer in 0..self.model_config.block_count as usize {
            let mapped = self.map_standard_layer(layer)?;
            let sources = [
                mapped.w_gate.dptr.cast_const(),
                mapped.w_up.dptr.cast_const(),
            ];
            self.q8_execution.ffn_planes(
                [
                    mapped.w_gate.dptr.cast_const(),
                    mapped.w_up.dptr.cast_const(),
                ],
                hidden as i32,
                k as i32,
            )?;
            if self.q8_ffn.kernel() == Q8FfnKernel::JointSharedScales {
                self.q8_execution
                    .compound_pair(sources, hidden as i32, k as i32)?;
            }
        }
        Ok(())
    }
}
