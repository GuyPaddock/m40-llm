use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    time::Instant,
};

use anyhow::{ensure, Context, Result};

use super::{
    runtime::{check_budget, ProjectionFamily},
    DeviceQ8ExecutionView, DeviceQ8ExecutionWeight, HostQ8ExecutionWeight, Q8ExecutionConfig,
    Q8ExecutionMode, Q8ExecutionStats, Q8PreparedTensorStats, Q8ProjectionLaunch, Q8SourceIdentity,
    Q8SourceView, Q8TiledLaunch,
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

impl PreparedTensor {
    fn split_planes(&self, n: i32, k: i32) -> Result<[*const c_void; 2]> {
        let shape = self.split.key().source.shape();
        ensure!(
            n > 0 && k > 0 && shape.input_dim() == k as usize && shape.output_dim() == n as usize,
            "prepared Q8 projection shape mismatch"
        );
        let DeviceQ8ExecutionView::SeparateScalesValues { scales, values, .. } = self.split.view()
        else {
            anyhow::bail!("prepared Q8 fused projection requires split planes");
        };
        Ok([scales.as_ptr(), values.as_ptr()])
    }
}

/// Immutable model-owned cache. Pointer lookup is local to this model/context;
/// owned execution keys retain semantic model/tensor/layout identity.
pub struct Q8ExecutionCache {
    tensors: HashMap<usize, PreparedTensor>,
    sources: HashMap<usize, (Q8SourceIdentity, ProjectionFamily)>,
    stats: Q8ExecutionStats,
    ffn_pairs: super::ffn_pairs::FfnPairCache,
}

impl Q8ExecutionCache {
    pub fn disabled() -> Self {
        Self {
            tensors: HashMap::new(),
            sources: HashMap::new(),
            stats: Q8ExecutionStats::empty(),
            ffn_pairs: super::ffn_pairs::FfnPairCache::default(),
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
        ownership: crate::infer::Q8FfnOwnership,
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
            ownership,
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
        ownership: crate::infer::Q8FfnOwnership,
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
        let mut joint_estimated = 0_usize;
        let mut largest_host = 0_usize;
        for tensor in tensors {
            let Some(family) = ProjectionFamily::from_name(&tensor.name, lm_head_name) else {
                continue;
            };
            if !config.families.contains(family.group()) {
                continue;
            }
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
            let is_joint_source = ownership == crate::infer::Q8FfnOwnership::JointOnly
                && matches!(family, ProjectionFamily::Gate | ProjectionFamily::Up);
            if is_joint_source {
                ensure!(
                    identity.shape().input_dim() <= 4096
                        && identity.shape().input_dim().is_multiple_of(32),
                    "unsupported joint-only source shape"
                );
                joint_estimated = joint_estimated
                    .checked_add(identity.shape().source_bytes()?)
                    .context("joint-only estimate overflow")?;
            }
            for layout in family
                .layouts(config.mode)
                .into_iter()
                .filter(|_| !is_joint_source)
            {
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
        check_budget(
            estimated
                .checked_add(joint_estimated)
                .context("combined preparation overflow")?,
            config.budget_bytes,
        )?;
        let mut stats = Q8ExecutionStats::empty();
        stats.mode = config.mode;
        stats.families = config.families;
        stats.has_prepared_vocabulary = entries
            .iter()
            .any(|(_, family, _)| *family == ProjectionFamily::Vocabulary);
        stats.single_row_down_split_launch = entries
            .iter()
            .any(|(_, family, _)| *family == ProjectionFamily::Down)
            .then(|| ProjectionFamily::Down.select_split_launch(1))
            .transpose()?;
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
            sources: entries
                .iter()
                .map(|(identity, family, pointer)| (*pointer, (identity.clone(), *family)))
                .collect(),
            stats,
            ffn_pairs: super::ffn_pairs::FfnPairCache::default(),
        };
        for (identity, family, pointer) in entries {
            if ownership == crate::infer::Q8FfnOwnership::JointOnly
                && matches!(family, ProjectionFamily::Gate | ProjectionFamily::Up)
            {
                continue;
            }
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
        eprintln!("[q8-execution] mode={:?} families={:?} tensors={} layouts={} prepared_bytes={} source_bytes={} total_weight_payload_bytes={} budget_bytes={} peak_setup_host_payload_bytes={} preparation_ms={:.3}",
            config.mode,config.families.groups(),cache.tensors.len(),cache.stats.tensors.len(),cache.stats.resident_bytes,
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

    /// Logical source coverage is not a count of resident split allocations.
    pub fn logical_sources(&self) -> Vec<Q8SourceIdentity> {
        let mut sources: Vec<_> = self.sources.values().map(|(id, _)| id.clone()).collect();
        sources.sort_by(|a, b| a.tensor_name().cmp(b.tensor_name()));
        sources
    }

    /// Explicit cold/debug readback; never called by ordinary generation.
    pub fn debug_joint_payload_hashes(&self) -> Result<Vec<(super::Q8FfnPairKey, String)>> {
        self.ffn_pairs.debug_payload_hashes()
    }

    pub(crate) fn prepare_compound_pairs(
        &mut self,
        ctx: &CudaContext,
        weights: &[u8],
        sources: &[[*const c_void; 2]],
    ) -> Result<()> {
        ensure!(
            !self.ffn_pairs.stats().is_enabled,
            "compound FFN pairs already prepared"
        );
        let plans = sources
            .iter()
            .map(|sources| {
                let identities = sources
                    .iter()
                    .enumerate()
                    .map(|(index, source)| {
                        let (identity, family) = self
                            .sources
                            .get(&(*source as usize))
                            .context("compound FFN requires validated gate/up sources")?;
                        ensure!(
                            *family == [ProjectionFamily::Gate, ProjectionFamily::Up][index],
                            "compound FFN source role mismatch"
                        );
                        Ok(identity.clone())
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok((
                    sources.map(|p| p as usize),
                    super::Q8FfnPairKey {
                        gate: identities[0].clone(),
                        up: identities[1].clone(),
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        self.ffn_pairs = super::ffn_pairs::FfnPairCache::prepare(
            ctx,
            weights,
            &plans,
            self.stats.resident_bytes,
            self.stats.budget_bytes,
        )?;
        let stats = self.ffn_pairs.stats();
        eprintln!("[q8-ffn-pairs] pairs={} joint_bytes={} combined_prepared_bytes={} budget_bytes={} temporary_host_pair_bytes={} preparation_ms={:.3}",
            stats.pair_count,stats.resident_bytes,stats.combined_prepared_bytes,stats.budget_bytes,
            stats.peak_temporary_host_pair_bytes,stats.preparation_elapsed_ms);
        Ok(())
    }

    pub(crate) fn compound_pair(
        &self,
        sources: [*const c_void; 2],
        hidden: i32,
        k: i32,
    ) -> Result<&super::DeviceQ8FfnPair> {
        self.ffn_pairs.get(sources, hidden, k)
    }

    pub fn ffn_pair_stats(&self) -> super::Q8FfnPairStats {
        let mut stats = self.ffn_pairs.stats();
        stats.combined_prepared_bytes = self.stats.resident_bytes + stats.resident_bytes;
        stats.budget_bytes = self.stats.budget_bytes;
        stats
    }

    /// Borrow this model's existing split owner for explicit diagnostics. This
    /// never prepares weights, changes dispatch or dereferences the lookup key.
    pub fn get_separate_weight(&self, source: *const c_void) -> Option<&DeviceQ8ExecutionWeight> {
        self.tensors
            .get(&(source as usize))
            .map(|entry| &entry.split)
    }

    pub(crate) fn has_prepared_vocabulary(&self) -> bool {
        self.stats.has_prepared_vocabulary
    }

    /// Resolve only this model's gate/up tensors. The returned planes borrow
    /// model-owned allocations; no host packing or upload occurs in decode.
    pub(crate) fn ffn_planes(
        &self,
        sources: [*const c_void; 2],
        hidden: i32,
        k: i32,
    ) -> Result<[[*const c_void; 2]; 2]> {
        ensure!(
            sources[0] != sources[1],
            "split FFN gate/up must be distinct"
        );
        let mut planes = [[std::ptr::null(); 2]; 2];
        for (index, source) in sources.into_iter().enumerate() {
            let entry = self
                .tensors
                .get(&(source as usize))
                .context("split-f32x4-128 requires prepared gate/up weights in this model")?;
            ensure!(
                entry.family == [ProjectionFamily::Gate, ProjectionFamily::Up][index],
                "split FFN requires gate/up projection family"
            );
            planes[index] = entry.split_planes(hidden, k)?;
        }
        Ok(planes)
    }

    /// # Safety
    /// Input/output have the stated device spans in this context. The caller
    /// orders their producers and retains the model until decode completes.
    pub(crate) unsafe fn project_ffn(
        &self,
        ctx: &CudaContext,
        input: *const c_void,
        sources: [*const c_void; 2],
        output: *mut c_void,
        hidden: i32,
        k: i32,
    ) -> Result<()> {
        let [[gate_scale, gate_values], [up_scale, up_values]] =
            self.ffn_planes(sources, hidden, k)?;
        ctx.ffn_q8_split_f32_async(
            input,
            [gate_scale, up_scale],
            [gate_values, up_values],
            output,
            1,
            hidden,
            k,
            None,
        )
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
            [scales[i], values[i]] = entry.split_planes(widths[i], k)?;
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
                Q8ProjectionLaunch::Tiled(Q8TiledLaunch::new(16, 128)?),
            )
        } else {
            (
                &entry.split,
                Q8ProjectionLaunch::Separate(entry.family.select_split_launch(m)?),
            )
        };
        weight.enqueue_projection_async(input, output, m, n, k, stream, launch)?;
        Ok(true)
    }
}

impl crate::infer::LoadedModel {
    pub fn q8_ffn_pair_stats(&self) -> super::Q8FfnPairStats {
        self.q8_execution.ffn_pair_stats()
    }
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
    #[ignore = "requires CUDA hardware; run with --include-ignored --test-threads=1"]
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
                    families: super::super::Q8ExecutionFamilies::all(),
                },
                0.0,
                crate::infer::Q8FfnOwnership::RetainSplit,
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
    #[ignore = "requires CUDA hardware; run with --include-ignored --test-threads=1"]
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
                    families: super::super::Q8ExecutionFamilies::all(),
                },
                0.0,
                crate::infer::Q8FfnOwnership::RetainSplit,
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
