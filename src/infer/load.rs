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
        Self::from_gguf_with_q8_execution(
            gguf,
            gguf_bytes,
            device_id,
            Q8ExecutionConfig::from_env()?,
        )
    }

    pub fn from_gguf_with_q8_execution(
        gguf: GgufModel,
        gguf_bytes: Vec<u8>,
        device_id: i32,
        execution: Q8ExecutionConfig,
    ) -> Result<Self> {
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
            #[cfg(feature = "cuda")]
            forward_workspace: std::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            materialized_weights: std::sync::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "cuda")]
            q8_execution: super::execution_weights::Q8ExecutionCache::disabled(),
            #[cfg(feature = "cuda")]
            activation_research: None,
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
            )?;
        }
        Ok(model)
    }
}
