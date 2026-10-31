//! Model-local selection of screened native Q8 QKV/output implementations.
use super::execution_weights::{Q8ExecutionConfig, Q8ExecutionMode, Q8ProjectionGroup};
use anyhow::{ensure, Result};
use serde::Serialize;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Q8AttentionProjectionBackend {
    #[default]
    Native,
    #[serde(rename = "warp-256")]
    Warp256,
    #[serde(rename = "packed-warp-256")]
    PackedWarp256,
}

impl Q8AttentionProjectionBackend {
    pub fn is_native(self) -> bool {
        self == Self::Native
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "native" => Ok(Self::Native),
            "warp-256" => Ok(Self::Warp256),
            "packed-warp-256" => Ok(Self::PackedWarp256),
            _ => anyhow::bail!(
                "M40LLM_Q8_ATTN_PROJECTIONS must be native, warp-256 or packed-warp-256"
            ),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Warp256 => "warp-256",
            Self::PackedWarp256 => "packed-warp-256",
        }
    }

    /// Exact enqueue identities for explicit backends; native dispatch still
    /// depends on the existing independently selected projection paths.
    pub fn get_operations(self) -> Option<[&'static str; 2]> {
        match self {
            Self::Native => None,
            Self::Warp256 => Some(["qkv_f32xq8_0_gguf_warp", "gemm_f32xq8_0_gguf_coltile4_warp"]),
            Self::PackedWarp256 => Some([
                "qkv_f32xq8_0_gguf_packed_warp",
                "gemm_f32xq8_0_gguf_coltile4_packed_warp",
            ]),
        }
    }

    pub fn from_env() -> Result<Self> {
        match std::env::var("M40LLM_Q8_ATTN_PROJECTIONS") {
            Ok(value) => Self::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(Self::Native),
            Err(error) => Err(error.into()),
        }
    }

    pub(crate) fn validate_execution(self, execution: Q8ExecutionConfig) -> Result<()> {
        ensure!(
            self.is_native()
                || execution.mode == Q8ExecutionMode::Off
                || (!execution.families.contains(Q8ProjectionGroup::Qkv)
                    && !execution.families.contains(Q8ProjectionGroup::Output)),
            "experimental attention projections require raw Q8_0 QKV/output weights; \
             exclude qkv/output from prepared families or use M40LLM_Q8_ATTN_PROJECTIONS=native"
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct QkvSpec {
    widths: [i32; 3],
    k: i32,
    biases: [Option<usize>; 3],
}

/// Owned, load-time configuration and validated raw device identities. In-memory
/// references use `native()`; never replace this state during workspace/session use.
/// Candidate construction is restricted to the validated model loader.
pub struct Q8AttentionProjectionState {
    backend: Q8AttentionProjectionBackend,
    #[cfg(feature = "cuda")]
    qkv: std::collections::HashMap<[usize; 3], QkvSpec>,
    #[cfg(feature = "cuda")]
    output: std::collections::HashMap<usize, (i32, i32)>,
}

impl Q8AttentionProjectionState {
    pub fn native() -> Self {
        Self {
            backend: Q8AttentionProjectionBackend::Native,
            #[cfg(feature = "cuda")]
            qkv: Default::default(),
            #[cfg(feature = "cuda")]
            output: Default::default(),
        }
    }

    pub fn backend(&self) -> Q8AttentionProjectionBackend {
        self.backend
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn prepare(
        model: &super::LoadedModel,
        backend: Q8AttentionProjectionBackend,
    ) -> Result<Self> {
        if backend.is_native() {
            return Ok(Self::native());
        }
        let props = model.cuda.current_device_props()?;
        ensure!(
            props.major == 5 && props.minor == 2 && !model.d_weights_base.is_null(),
            "experimental attention projections require sm_52 and uploaded device weights; \
             use M40LLM_Q8_ATTN_PROJECTIONS=native"
        );
        model.validate_standard_layers()?;
        let mut state = Self::native();
        for layer in 0..model.model_config.block_count as usize {
            let w = model.map_standard_layer(layer)?;
            let weights = [&w.wq, &w.wk, &w.wv, &w.wo];
            for view in weights {
                ensure!(
                    view.dtype == crate::gguf::GgmlDType::Q8_0
                        && view.shape.len() == 2
                        && view.shape[0] > 0
                        && view.shape[0].is_multiple_of(32)
                        && view.shape[0] <= i32::MAX as u64
                        && view.shape[1] > 0
                        && view.shape[1] <= i32::MAX as u64,
                    "experimental attention projections require bounded Q8_0 QKV/output in layer {layer}; \
                     use M40LLM_Q8_ATTN_PROJECTIONS=native"
                );
                model.tensor_device_ptr("Q8 attention projection", view)?;
            }
            let widths = std::array::from_fn(|i| weights[i].shape[1] as i32);
            ensure!(
                widths.iter().map(|&v| i64::from(v)).sum::<i64>() <= i64::from(i32::MAX) - 255,
                "Q8 attention projection widths overflow CUDA grid"
            );
            let biases = [w.bq, w.bk, w.bv];
            let mut bias_ptrs = [None; 3];
            for (i, bias) in biases.iter().enumerate() {
                if let Some(view) = bias {
                    ensure!(
                        view.dtype == crate::gguf::GgmlDType::F32
                            && view.nbytes == widths[i] as usize * 4,
                        "Q8 attention projection requires FP32 bias vectors"
                    );
                    bias_ptrs[i] =
                        Some(model.tensor_device_ptr("Q8 projection bias", view)? as usize);
                }
            }
            let key = std::array::from_fn(|i| weights[i].dptr as usize);
            ensure!(
                state
                    .qkv
                    .insert(
                        key,
                        QkvSpec {
                            widths,
                            k: w.d_model as i32,
                            biases: bias_ptrs
                        }
                    )
                    .is_none(),
                "aliased Q8 attention layer QKV identities"
            );
            ensure!(
                state
                    .output
                    .insert(
                        w.wo.dptr as usize,
                        (w.wo.shape[1] as i32, w.wo.shape[0] as i32)
                    )
                    .is_none(),
                "aliased Q8 attention layer output identity"
            );
        }
        state.backend = backend;
        eprintln!("[q8-attn-projections] backend={} scope=single-row qkv_layers={} output_layers={} stream=decode extra_device_bytes=0",backend.as_str(),state.qkv.len(),state.output.len());
        Ok(state)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn validate_qkv(
        &self,
        weights: [*const std::ffi::c_void; 3],
        biases: [Option<*const std::ffi::c_void>; 3],
        widths: [i32; 3],
        k: i32,
    ) -> Result<()> {
        let spec = self.qkv.get(&weights.map(|p| p as usize)).ok_or_else(|| {
            anyhow::anyhow!("Q8 QKV requires this model's original layer weight spans")
        })?;
        ensure!(
            spec.widths == widths
                && spec.k == k
                && spec.biases == biases.map(|b| b.map(|p| p as usize)),
            "Q8 QKV shape/bias does not match the loaded model"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn validate_output(
        &self,
        weight: *const std::ffi::c_void,
        n: i32,
        k: i32,
    ) -> Result<()> {
        ensure!(
            self.output.get(&(weight as usize)) == Some(&(n, k)),
            "Q8 output requires this model's original attention-output weight and shape"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn contains_weight(&self, weight: *const std::ffi::c_void) -> bool {
        self.output.contains_key(&(weight as usize))
            || self
                .qkv
                .keys()
                .any(|weights| weights.contains(&(weight as usize)))
    }
}
