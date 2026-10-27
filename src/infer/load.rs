use super::execution_weights::{Q8ExecutionConfig, Q8ExecutionMode};
use super::meta::dtype_size_bytes;
use super::tensor_views::build_device_tensor_views;
use super::{LoadedModel, ModelConfig};
use crate::cuda::CudaContext;
use crate::gguf::GgufModel;
use anyhow::{Context, Result};
#[cfg(not(feature = "cuda"))]
use std::ffi::c_void;

impl LoadedModel {
    pub fn from_gguf(gguf: GgufModel, gguf_bytes: Vec<u8>, device_id: i32) -> Result<Self> {
        Self::from_gguf_with_q8_runtime(
            gguf,
            gguf_bytes,
            device_id,
            Q8ExecutionConfig::from_env()?,
            super::Q8FfnConfig::from_env()?,
            super::q8_prefill::Q8PrefillBackend::from_env()?,
            super::Q8AttentionProjectionBackend::from_env()?,
        )
    }

    pub fn from_gguf_with_q8_execution(
        gguf: GgufModel,
        gguf_bytes: Vec<u8>,
        device_id: i32,
        execution: Q8ExecutionConfig,
    ) -> Result<Self> {
        // Explicit weight-cache controls retain native FFN independently of
        // ambient experimental flags, including frozen benchmark references.
        Self::from_gguf_with_q8_backends(
            gguf,
            gguf_bytes,
            device_id,
            execution,
            super::Q8FfnConfig::native(),
            super::q8_prefill::Q8PrefillBackend::Native,
        )
    }

    pub fn from_gguf_with_q8_backends(
        gguf: GgufModel,
        gguf_bytes: Vec<u8>,
        device_id: i32,
        execution: Q8ExecutionConfig,
        ffn: super::Q8FfnConfig,
        prefill: super::q8_prefill::Q8PrefillBackend,
    ) -> Result<Self> {
        Self::from_gguf_with_q8_runtime(
            gguf,
            gguf_bytes,
            device_id,
            execution,
            ffn,
            prefill,
            super::Q8AttentionProjectionBackend::Native,
        )
    }

    /// Explicit independent model-local choices; ambient experiment flags do
    /// not override them. The old backend constructor remains a native control.
    pub fn from_gguf_with_q8_runtime(
        gguf: GgufModel,
        gguf_bytes: Vec<u8>,
        device_id: i32,
        execution: Q8ExecutionConfig,
        ffn: super::Q8FfnConfig,
        prefill: super::q8_prefill::Q8PrefillBackend,
        attention_projection: super::Q8AttentionProjectionBackend,
    ) -> Result<Self> {
        anyhow::ensure!(
            attention_projection.is_native()
                || (cfg!(feature = "cuda")
                    && cfg!(nvcc)
                    && std::env::var("M40LLM_ENABLE_NVCC").ok().as_deref() != Some("0")),
            "experimental attention projections require compiled CUDA; \
             use M40LLM_Q8_ATTN_PROJECTIONS=native"
        );
        attention_projection.validate_execution(execution)?;
        anyhow::ensure!(
            ffn.is_native()
                || (cfg!(feature = "cuda")
                    && cfg!(nvcc)
                    && std::env::var("M40LLM_ENABLE_NVCC").ok().as_deref() != Some("0")),
            "experimental Q8 FFN requires compiled CUDA kernels; use M40LLM_Q8_FFN_KERNEL=native"
        );
        anyhow::ensure!(
            prefill.is_native()
                || (cfg!(feature = "cuda") && cfg!(nvcc) && cfg!(have_cublas)
                    && std::env::var("M40LLM_ENABLE_NVCC").ok().as_deref() != Some("0")),
            "experimental Q8 prefill requires compiled CUDA/cuBLAS; use M40LLM_Q8_PREFILL_BACKEND=native"
        );
        ffn.validate_execution(execution)?;
        #[cfg(not(feature = "cuda"))]
        anyhow::ensure!(
            execution.mode == Q8ExecutionMode::Off,
            "prepared Q8 weights require CUDA; use M40LLM_Q8_EXECUTION_WEIGHTS=off"
        );
        #[cfg(feature = "cuda")]
        let source_sha256: Option<([u8; 32], f64)> = if execution.mode == Q8ExecutionMode::Off {
            None
        } else {
            let started = std::time::Instant::now();
            let hash = ring::digest::digest(&ring::digest::SHA256, &gguf_bytes)
                .as_ref()
                .try_into()
                .context("SHA256 digest length mismatch")?;
            Some((hash, started.elapsed().as_secs_f64() * 1000.0))
        };
        let cuda = CudaContext::new(device_id)?;
        let data_off = gguf.data_offset as usize;
        if data_off > gguf_bytes.len() {
            anyhow::bail!(
                "GGUF data_offset {} beyond file size {}",
                data_off,
                gguf_bytes.len()
            );
        }
        #[cfg(feature = "gguf_ext")]
        let (model_config, typed_cfg) =
            ModelConfig::from_gguf_ext_bytes(&gguf_bytes, &gguf.metadata, &gguf.tensors)?;
        #[cfg(not(feature = "gguf_ext"))]
        let model_config = ModelConfig::from_metadata(&gguf.metadata, &gguf.tensors)?;

        let weights_bytes = gguf_bytes[data_off..].to_vec();
        let weights_len = weights_bytes.len();
        #[cfg(not(feature = "cuda"))]
        let _host_base = weights_bytes.as_ptr() as *mut c_void;
        #[cfg(feature = "cuda")]
        let use_device_weights = std::env::var("M40LLM_ENABLE_NVCC")
            .map(|v| v != "0")
            .unwrap_or(cfg!(nvcc));
        #[cfg(feature = "cuda")]
        let d_base = if use_device_weights {
            cuda.upload_weights(&weights_bytes).ok()
        } else {
            None
        };
        // Validate that all known-sized tensors fit within weights_bytes
        for t in &gguf.tensors {
            if let Some(layout) = dtype_size_bytes(t.dtype) {
                let n_elems: u64 = t.shape.iter().copied().product::<u64>();
                let n_elems: usize = usize::try_from(n_elems)
                    .context("tensor element count does not fit in usize")?;
                let n_blocks = n_elems.div_ceil(layout.block_elems);
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
        let device_tensors = build_device_tensor_views(
            &gguf.tensors,
            d_base.unwrap_or(std::ptr::null_mut()),
            weights_len,
        )?;
        #[cfg(not(feature = "cuda"))]
        let device_tensors =
            build_device_tensor_views(&gguf.tensors, std::ptr::null_mut(), weights_len)?;

        #[cfg(feature = "cuda")]
        let original_file_bytes = gguf_bytes.len();
        let model = Self {
            gguf,
            cuda,
            kv_cache: None,
            layer_mask: super::LayerMask::default(),
            q8_ffn: ffn,
            q8_prefill: prefill,
            q8_attention_projection: super::Q8AttentionProjectionState::native(),
            #[cfg(feature = "cuda")]
            forward_workspace: std::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            materialized_weights: std::sync::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "cuda")]
            q8_execution: super::execution_weights::Q8ExecutionCache::disabled(),
            #[cfg(feature = "cuda")]
            activation_research: None,
            #[cfg(feature = "cuda")]
            layer_influence: None,
            #[cfg(feature = "cuda")]
            ffn_capture: None,
            device_tensors,
            weights_len,
            #[cfg(feature = "cuda")]
            d_weights_base: d_base.unwrap_or(std::ptr::null_mut()),
            host_weights: weights_bytes,
            model_config,
            #[cfg(feature = "gguf_ext")]
            typed_config: typed_cfg,
        };
        #[cfg(feature = "cuda")]
        let mut model = model;
        #[cfg(feature = "cuda")]
        {
            model.q8_attention_projection =
                super::Q8AttentionProjectionState::prepare(&model, attention_projection)?;
        }
        #[cfg(feature = "cuda")]
        model.validate_q8_ffn_backend()?;
        #[cfg(feature = "cuda")]
        if !model.q8_prefill.is_native() {
            let bytes = model.q8_prefill_weight_bytes()?;
            eprintln!("[q8-prefill] backend=dequant-sgemm temporary_weight_capacity_bytes={bytes} budget_bytes={} scope=single-prompt-packed-prefix decode=unchanged", super::q8_prefill::Q8_PREFILL_MAX_WEIGHT_BYTES);
        }
        #[cfg(feature = "cuda")]
        if let Some((sha256, hash_ms)) = source_sha256 {
            let props = model.cuda.current_device_props()?;
            anyhow::ensure!(
                props.major == 5 && props.minor == 2,
                "prepared Q8 layouts are only validated on sm_52; use M40LLM_Q8_EXECUTION_WEIGHTS=off"
            );
            let (lm_head_name, _, _, _, _) = model.map_lm_head()?;
            model.q8_execution = super::execution_weights::Q8ExecutionCache::prepare(
                &model.cuda,
                &model.gguf.tensors,
                &model.host_weights,
                model.d_weights_base,
                &lm_head_name,
                sha256,
                original_file_bytes,
                execution,
                hash_ms,
                ffn.ownership(),
            )?;
        }
        #[cfg(feature = "cuda")]
        if model.q8_ffn.kernel() == super::Q8FfnKernel::JointSharedScales {
            let sources = (0..model.model_config.block_count as usize)
                .map(|layer| {
                    let mapped = model.map_standard_layer(layer)?;
                    Ok([
                        mapped.w_gate.dptr.cast_const(),
                        mapped.w_up.dptr.cast_const(),
                    ])
                })
                .collect::<Result<Vec<_>>>()?;
            model.q8_execution.prepare_compound_pairs(
                &model.cuda,
                &model.host_weights,
                &sources,
            )?;
        }
        #[cfg(feature = "cuda")]
        model.validate_q8_ffn_prepared_weights()?;
        if !model.q8_ffn.is_native() {
            eprintln!("[q8-ffn] kernel={:?} threads={:?} scope=single-row activation_bytes={} numerical_track={}",
                model.q8_ffn.kernel(),model.q8_ffn.threads(),
                model.q8_ffn.activation_bytes(model.model_config.embedding_length as usize)?,model.q8_ffn.has_quantization());
        }
        Ok(model)
    }
}
