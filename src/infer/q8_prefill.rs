//! Bounded temporary execution storage, not a persistent FP32 weight cache.

use anyhow::{ensure, Context, Result};

pub const Q8_PREFILL_MAX_WEIGHT_BYTES: usize = 128 * 1024 * 1024;

/// Load-time prefill selection, independent of the one-row FFN candidate.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Q8PrefillBackend {
    #[default]
    Native,
    DequantSgemm,
}

impl Q8PrefillBackend {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "native" => Ok(Self::Native),
            "dequant-sgemm" => Ok(Self::DequantSgemm),
            _ => anyhow::bail!("M40LLM_Q8_PREFILL_BACKEND must be native or dequant-sgemm"),
        }
    }

    pub fn from_env() -> Result<Self> {
        match std::env::var("M40LLM_Q8_PREFILL_BACKEND") {
            Ok(value) => Self::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(Self::Native),
            Err(error) => Err(error.into()),
        }
    }

    pub fn is_native(self) -> bool {
        self == Self::Native
    }
}

/// View of the single temporary weight allocation, reused on prefill after
/// each preceding SGEMM. Never alias it with projection input/output.
#[derive(Debug, Clone, Copy)]
pub struct Q8PrefillScratch {
    pub pointer: *mut std::ffi::c_void,
    pub bytes: usize,
}

/// One logical GGUF [input_dim, output_dim] Q8_0 tensor, converted to
/// column-major [output_dim, input_dim] FP32 for the existing SGEMM wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q8PrefillWeightShape {
    input_dim: usize,
    output_dim: usize,
    f32_bytes: usize,
    q8_bytes: usize,
}

impl Q8PrefillWeightShape {
    pub fn new(input_dim: usize, output_dim: usize) -> Result<Self> {
        ensure!(
            input_dim > 0 && input_dim.is_multiple_of(32) && output_dim > 0,
            "Q8 prefill requires positive dimensions and complete 32-value input blocks"
        );
        let elements = input_dim
            .checked_mul(output_dim)
            .context("Q8 prefill weight shape overflow")?;
        let f32_bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .context("Q8 prefill FP32 bytes overflow")?;
        ensure!(
            f32_bytes <= Q8_PREFILL_MAX_WEIGHT_BYTES,
            "Q8 prefill weight needs {f32_bytes} bytes, exceeding the 128 MiB temporary budget"
        );
        let q8_bytes = (elements / 32)
            .checked_mul(34)
            .context("Q8 prefill source bytes overflow")?;
        Ok(Self {
            input_dim,
            output_dim,
            f32_bytes,
            q8_bytes,
        })
    }

    pub fn input_dim(self) -> usize {
        self.input_dim
    }

    pub fn output_dim(self) -> usize {
        self.output_dim
    }

    pub fn f32_bytes(self) -> usize {
        self.f32_bytes
    }

    pub fn q8_bytes(self) -> usize {
        self.q8_bytes
    }
}

#[cfg(feature = "cuda")]
impl super::LoadedModel {
    pub(crate) fn q8_prefill_weight_bytes(&self) -> Result<usize> {
        if self.q8_prefill.is_native() {
            return Ok(0);
        }
        let props = self.cuda.current_device_props()?;
        ensure!(
            props.major == 5 && props.minor == 2,
            "experimental Q8 prefill requires sm_52; use M40LLM_Q8_PREFILL_BACKEND=native"
        );
        ensure!(!self.d_weights_base.is_null(),
            "experimental Q8 prefill requires uploaded weights; use M40LLM_Q8_PREFILL_BACKEND=native");
        self.validate_standard_layers()?;
        ensure!(matches!(self.model_config.attention_key_length, 64 | 128),
            "experimental Q8 prefill requires head_dim=64 or 128; use M40LLM_Q8_PREFILL_BACKEND=native");
        let mut capacity = 0;
        for layer in 0..self.model_config.block_count as usize {
            let w = self.map_standard_layer(layer)?;
            for view in [&w.wq, &w.wk, &w.wv, &w.wo, &w.w_gate, &w.w_up, &w.w_down] {
                ensure!(view.dtype == crate::gguf::GgmlDType::Q8_0 && view.shape.len() == 2,
                    "experimental Q8 prefill requires Q8_0 transformer projections in layer {layer}; use M40LLM_Q8_PREFILL_BACKEND=native");
                let pointer = self.tensor_device_ptr("Q8 prefill projection", view)?;
                ensure!(
                    (pointer as usize).is_multiple_of(2),
                    "Q8 prefill weight requires two-byte alignment"
                );
                let shape = Q8PrefillWeightShape::new(view.shape[0] as usize, view.shape[1] as usize)
                    .context("experimental Q8 prefill weight is unsupported; use M40LLM_Q8_PREFILL_BACKEND=native")?;
                capacity = capacity.max(shape.f32_bytes());
            }
        }
        ensure!(
            capacity > 0,
            "experimental Q8 prefill has no transformer projections"
        );
        Ok(capacity)
    }

    /// # Safety
    /// Input/output and weight hold the stated device matrices and remain alive
    /// through prefill completion. Scratch is disjoint, owned by this forward
    /// workspace, and reused only after preceding prefill consumers enqueue.
    pub(crate) unsafe fn matmul_prefill_f32_gguf_async(
        &self,
        input: *const std::ffi::c_void,
        weight: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        rows: i32,
        output_dim: i32,
        input_dim: i32,
        scratch: Option<Q8PrefillScratch>,
    ) -> Result<()> {
        if self.q8_prefill.is_native() {
            return unsafe {
                self.matmul_f32xf16_gguf_f32_async(
                    input, weight, output, rows, output_dim, input_dim,
                )
            };
        }
        ensure!(
            rows > 0 && input_dim > 0 && output_dim > 0,
            "invalid Q8 prefill projection dimensions"
        );
        let shape = Q8PrefillWeightShape::new(input_dim as usize, output_dim as usize)?;
        let view = self
            .device_tensors
            .values()
            .find(|view| std::ptr::eq(view.dptr.cast_const(), weight))
            .context("missing Q8 prefill projection tensor")?;
        ensure!(
            view.dtype == crate::gguf::GgmlDType::Q8_0
                && view.shape == [input_dim as u64, output_dim as u64],
            "experimental Q8 prefill requires matching Q8_0 projection weights"
        );
        let scratch =
            scratch.context("experimental Q8 prefill requires owned temporary weight scratch")?;
        ensure!(
            !input.is_null()
                && !output.is_null()
                && input != output.cast_const()
                && scratch.pointer.cast_const() != input
                && scratch.pointer != output,
            "invalid/aliased Q8 prefill projection buffers"
        );
        unsafe {
            self.cuda.dequantize_q8_0_to_f32_colmajor_async(
                weight,
                scratch.pointer,
                scratch.bytes,
                shape,
            )?;
            self.cuda.gemm_f32xf32_f32_stream_async(
                input,
                scratch.pointer,
                output,
                rows,
                output_dim,
                input_dim,
                crate::cuda::CudaStream::Prefill,
            )
        }
    }
}
