use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    time::Instant,
};

use anyhow::{ensure, Context, Result};

use super::{
    runtime::{check_budget, ProjectionFamily},
    DeviceQ8ExecutionView, DeviceQ8ExecutionWeight, HostQ8ExecutionWeight, Q8ExecutionConfig,
    Q8ExecutionMode, Q8ExecutionStats, Q8PreparedTensorStats, Q8ProjectionLaunch, Q8SeparateLaunch,
    Q8SourceIdentity, Q8SourceView, Q8TiledAccess, Q8TiledLaunch,
};
use crate::{
    cuda::{CudaContext, CudaStream},
    gguf::{GgmlDType, GgufTensor},
};

struct PreparedTensor {
    family: ProjectionFamily,
    split: DeviceQ8ExecutionWeight,
    tiled: Option<DeviceQ8ExecutionWeight>,
}

/// Immutable model-owned cache. Pointer lookup is local to this model/context;
/// owned execution keys retain semantic model/tensor/layout identity.
pub struct Q8ExecutionCache {
    tensors: HashMap<usize, PreparedTensor>,
    stats: Q8ExecutionStats,
}

impl Q8ExecutionCache {
    pub fn disabled() -> Self {
        Self {
            tensors: HashMap::new(),
            stats: Q8ExecutionStats::empty(),
        }
    }

    pub(crate) fn prepare(
        ctx: &CudaContext,
        tensors: &[GgufTensor],
        host_weights: &[u8],
        source_base: *mut c_void,
        lm_head_name: &str,
        model_sha256: [u8; 32],
        original_file_bytes: usize,
        config: Q8ExecutionConfig,
        hash_elapsed_ms: f64,
    ) -> Result<Self> {
        Self::prepare_with_upload(
            ctx,
            tensors,
            host_weights,
            source_base,
            lm_head_name,
            model_sha256,
            original_file_bytes,
            config,
            hash_elapsed_ms,
            DeviceQ8ExecutionWeight::upload,
        )
    }

    fn prepare_with_upload(
        ctx: &CudaContext,
        tensors: &[GgufTensor],
        host_weights: &[u8],
        source_base: *mut c_void,
        lm_head_name: &str,
        model_sha256: [u8; 32],
        original_file_bytes: usize,
        config: Q8ExecutionConfig,
        hash_elapsed_ms: f64,
        mut upload: impl FnMut(&CudaContext, &HostQ8ExecutionWeight) -> Result<DeviceQ8ExecutionWeight>,
    ) -> Result<Self> {
        ensure!(
            config.mode != Q8ExecutionMode::Off,
            "disabled preparation must not allocate"
        );
        ensure!(
            !source_base.is_null(),
            "Q8 execution preparation requires uploaded device weights"
        );
        let start = Instant::now();
        let mut entries = Vec::new();
        let mut source_pointers = HashSet::new();
        let mut estimated = 0_usize;
        let mut largest_host = 0_usize;
        for tensor in tensors {
            let Some(family) = ProjectionFamily::from_name(&tensor.name, lm_head_name) else {
                continue;
            };
            if tensor.dtype != GgmlDType::Q8_0 {
                continue;
            }
            let identity = Q8SourceIdentity::from_gguf_tensor(model_sha256, tensor)?;
            Q8SourceView::new(&identity, host_weights)?;
            let pointer = (source_base as usize)
                .checked_add(usize::try_from(identity.byte_offset())?)
                .context("Q8 source pointer overflow")?;
            ensure!(
                source_pointers.insert(pointer),
                "aliased Q8 projection offsets are not supported by preparation: {}",
                identity.tensor_name()
            );
            for layout in family.layouts(config.mode) {
                let bytes = layout.allocation_plan(identity.shape())?.resident_bytes();
                estimated = estimated
                    .checked_add(bytes)
                    .context("Q8 preparation estimate overflow")?;
                largest_host = largest_host.max(bytes);
            }
            entries.push((identity, family, pointer));
        }
        ensure!(
            !entries.is_empty(),
            "no supported Q8_0 projections for execution preparation; use M40LLM_Q8_EXECUTION_WEIGHTS=off"
        );
        check_budget(estimated, config.budget_bytes)?;
        let mut stats = Q8ExecutionStats::empty();
        stats.mode = config.mode;
        stats.budget_bytes = config.budget_bytes;
        stats.model_hash_elapsed_ms = hash_elapsed_ms;
        stats.estimated_bytes = estimated;
        stats.source_weights_bytes = host_weights.len();
        stats.peak_setup_host_payload_bytes = original_file_bytes
            .checked_add(host_weights.len())
            .and_then(|v| v.checked_add(largest_host))
            .context("Q8 host payload estimate overflow")?;
        let mut cache = Self {
            tensors: HashMap::new(),
            stats,
        };
        for (identity, family, pointer) in entries {
            let source = Q8SourceView::new(&identity, host_weights)?;
            let split = cache.upload(
                ctx,
                HostQ8ExecutionWeight::separate_scales_values(source)?,
                &mut upload,
            )?;
            let tiled = if family.layouts(config.mode).len() == 2 {
                Some(cache.upload(
                    ctx,
                    HostQ8ExecutionWeight::tile_inputs(source, 16)?,
                    &mut upload,
                )?)
            } else {
                None
            };
            cache.tensors.insert(
                pointer,
                PreparedTensor {
                    family,
                    split,
                    tiled,
                },
            );
        }
        cache.stats.total_weight_payload_bytes = host_weights
            .len()
            .checked_add(cache.stats.resident_bytes)
            .context("Q8 resident payload overflow")?;
        cache.stats.preparation_elapsed_ms =
            start.elapsed().as_secs_f64() * 1000.0 + hash_elapsed_ms;
        eprintln!("[q8-execution] mode={:?} tensors={} layouts={} prepared_bytes={} source_bytes={} total_weight_payload_bytes={} budget_bytes={} peak_setup_host_payload_bytes={} preparation_ms={:.3}",
            config.mode,cache.tensors.len(),cache.stats.tensors.len(),cache.stats.resident_bytes,
            host_weights.len(),cache.stats.total_weight_payload_bytes,config.budget_bytes,
            cache.stats.peak_setup_host_payload_bytes,cache.stats.preparation_elapsed_ms);
        Ok(cache)
    }

    fn upload(
        &mut self,
        ctx: &CudaContext,
        host: HostQ8ExecutionWeight,
        upload: &mut impl FnMut(&CudaContext, &HostQ8ExecutionWeight) -> Result<DeviceQ8ExecutionWeight>,
    ) -> Result<DeviceQ8ExecutionWeight> {
        let device = upload(ctx, &host)?;
        self.stats.resident_bytes = self
            .stats
            .resident_bytes
            .checked_add(device.plan().resident_bytes())
            .context("Q8 resident bytes overflow")?;
        self.stats.tensors.push(Q8PreparedTensorStats {
            key: device.key().clone(),
            source_bytes: device.plan().source_bytes(),
            resident_bytes: device.plan().resident_bytes(),
        });
        Ok(device)
    }

    pub fn stats(&self) -> Q8ExecutionStats {
        self.stats.clone()
    }

    /// # Safety
    /// Source/input/output/bias pointers have the stated spans in this model's
    /// context. The caller supplies producer dependencies and retains all owners.
    pub(crate) unsafe fn try_project_qkv(
        &self,
        ctx: &CudaContext,
        input: *const c_void,
        sources: [*const c_void; 3],
        biases: [Option<*const c_void>; 3],
        outputs: [*mut c_void; 3],
        widths: [i32; 3],
        k: i32,
        stream: CudaStream,
    ) -> Result<bool> {
        let [Some(q), Some(key), Some(v)] =
            sources.map(|source| self.tensors.get(&(source as usize)))
        else {
            return Ok(false);
        };
        ensure!(k > 0 && !input.is_null(), "invalid prepared Q8 QKV input");
        let mut scales = [std::ptr::null(); 3];
        let mut values = [std::ptr::null(); 3];
        for (i, entry) in [q, key, v].into_iter().enumerate() {
            let shape = entry.split.key().source.shape();
            ensure!(
                widths[i] > 0
                    && shape.input_dim() == k as usize
                    && shape.output_dim() == widths[i] as usize,
                "prepared Q8 QKV shape mismatch"
            );
            ensure!(
                !outputs[i].is_null() && outputs[i].cast_const() != input,
                "invalid/aliased prepared Q8 QKV output"
            );
            for previous in &outputs[..i] {
                ensure!(
                    *previous != outputs[i],
                    "prepared Q8 QKV outputs must not alias"
                );
            }
            let DeviceQ8ExecutionView::SeparateScalesValues {
                scales: s,
                values: v,
                ..
            } = entry.split.view()
            else {
                anyhow::bail!("prepared Q8 QKV requires split planes");
            };
            scales[i] = s.as_ptr();
            values[i] = v.as_ptr();
        }
        ctx.qkv_f32xq8_separate_async(input, scales, values, biases, outputs, widths, k, stream)?;
        Ok(true)
    }

    /// # Safety
    /// Source/activation/output pointers must belong to this model's context,
    /// have the stated spans, and outlive all work on the supplied stream.
    pub(crate) unsafe fn try_project(
        &self,
        source: *const c_void,
        input: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        stream: CudaStream,
    ) -> Result<bool> {
        let Some(entry) = self.tensors.get(&(source as usize)) else {
            return Ok(false);
        };
        let (weight, launch) = if let Some(weight) = entry.tiled.as_ref().filter(|_| m >= 64) {
            (
                weight,
                Q8ProjectionLaunch::Tiled(Q8TiledLaunch::new(16, 128, Q8TiledAccess::Vector)?),
            )
        } else {
            (
                &entry.split,
                Q8ProjectionLaunch::Separate(Q8SeparateLaunch::new(
                    entry.family.split_columns(),
                    128,
                )?),
            )
        };
        weight.enqueue_projection_async(input, output, m, n, k, stream, launch)?;
        Ok(true)
    }
}

impl crate::infer::LoadedModel {
    pub fn q8_execution_stats(&self) -> Q8ExecutionStats {
        let mut stats = self.q8_execution.stats();
        if stats.mode == Q8ExecutionMode::Off {
            stats.source_weights_bytes = if self.d_weights_base.is_null() {
                0
            } else {
                self.weights_len
            };
            stats.total_weight_payload_bytes = stats.source_weights_bytes;
        }
        stats
    }
}

#[cfg(all(test, nvcc))]
mod tests {
    use super::*;
    use crate::profile;

    /// Aliased, out-of-bounds, and invalid-shape plans must be rejected before
    /// any prepared tensor upload is attempted, preserving native storage.
    #[test]
    fn test_invalid_preparation_plans_do_not_upload() -> Result<()> {
        let ctx = CudaContext::new(-1)?;
        let weights = vec![13_u8; 136];
        let base = ctx.upload_weights(&weights)?;
        let query = GgufTensor {
            name: "blk.0.attn_q.weight".into(),
            dtype: GgmlDType::Q8_0,
            shape: vec![32, 2],
            offset: 0,
        };
        for (label, invalid) in [
            (
                "alias",
                GgufTensor {
                    name: "blk.0.attn_k.weight".into(),
                    ..query.clone()
                },
            ),
            (
                "bounds",
                GgufTensor {
                    offset: 136,
                    ..query.clone()
                },
            ),
            (
                "shape",
                GgufTensor {
                    shape: vec![31, 2],
                    ..query.clone()
                },
            ),
        ] {
            let before = profile::snapshot();
            let mut attempts = 0;
            let result = Q8ExecutionCache::prepare_with_upload(
                &ctx,
                &[query.clone(), invalid],
                &weights,
                base,
                "",
                [1; 32],
                136,
                Q8ExecutionConfig {
                    mode: Q8ExecutionMode::Split,
                    budget_bytes: 1024,
                },
                0.0,
                |_, _| {
                    attempts += 1;
                    anyhow::bail!("invalid plan unexpectedly reached upload")
                },
            );
            assert!(result.is_err(), "{label}");
            assert_eq!(attempts, 0, "{label}");
            let delta = profile::counter_delta(&before, &profile::snapshot());
            assert_eq!(delta.device_allocs, 0, "{label}");
            assert_eq!(delta.h2d_copies, 0, "{label}");
        }
        let mut actual = vec![0_u8; weights.len()];
        unsafe {
            ctx.memcpy_d2h(actual.as_mut_ptr().cast(), base, actual.len())?;
        }
        assert_eq!(actual, weights);
        Ok(())
    }

    /// A real allocation error while preparing a later tensor must release
    /// earlier cached planes, preserve native weights, and leave the context usable.
    #[test]
    fn test_partial_cache_preparation_failure_releases_prior_tensors() -> Result<()> {
        let before_context = CudaContext::total_device_bytes();
        {
            let ctx = CudaContext::new(-1)?;
            let props = ctx.current_device_props()?;
            ensure!(
                props.major == 5 && props.minor == 2,
                "cache cleanup test requires M40"
            );
            let weights = vec![13_u8; 136];
            let base = ctx.upload_weights(&weights)?;
            let tensors = [
                GgufTensor {
                    name: "blk.0.attn_q.weight".into(),
                    dtype: GgmlDType::Q8_0,
                    shape: vec![32, 2],
                    offset: 0,
                },
                GgufTensor {
                    name: "blk.0.attn_k.weight".into(),
                    dtype: GgmlDType::Q8_0,
                    shape: vec![32, 2],
                    offset: 68,
                },
            ];
            let initial = CudaContext::total_device_bytes();
            let before = profile::snapshot();
            let mut uploads = 0;
            let result = Q8ExecutionCache::prepare_with_upload(
                &ctx,
                &tensors,
                &weights,
                base,
                "",
                [1; 32],
                136,
                Q8ExecutionConfig {
                    mode: Q8ExecutionMode::Split,
                    budget_bytes: 1024,
                },
                0.0,
                |ctx, host| {
                    uploads += 1;
                    if uploads == 2 {
                        return Err(ctx
                            .device_malloc(usize::MAX)
                            .expect_err("impossible allocation must fail"));
                    }
                    DeviceQ8ExecutionWeight::upload(ctx, host)
                },
            );
            assert!(result.is_err());
            assert_eq!(uploads, 2);
            assert_eq!(CudaContext::total_device_bytes(), initial);
            let delta = profile::counter_delta(&before, &profile::snapshot());
            assert_eq!(delta.device_allocs, 2);
            assert_eq!(delta.device_frees, 2);
            let mut actual = vec![0_u8; weights.len()];
            unsafe {
                ctx.memcpy_d2h(actual.as_mut_ptr().cast(), base, weights.len())?;
            }
            assert_eq!(actual, weights);
        }
        assert_eq!(CudaContext::total_device_bytes(), before_context);
        Ok(())
    }
}
