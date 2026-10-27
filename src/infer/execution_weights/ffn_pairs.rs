//! Model-local joint owners, constructed from validated logical source plans.
use super::{DeviceQ8FfnPair, HostQ8FfnPair, Q8FfnPairKey, Q8FfnPairStats, Q8SourceView};
use crate::cuda::CudaContext;
use anyhow::{ensure, Context, Result};
use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    time::Instant,
};

pub(super) type PairPlan = ([usize; 2], Q8FfnPairKey);

#[derive(Default)]
pub(super) struct FfnPairCache {
    pairs: HashMap<[usize; 2], DeviceQ8FfnPair>,
    stats: Q8FfnPairStats,
}

impl FfnPairCache {
    pub fn prepare(
        ctx: &CudaContext,
        weights: &[u8],
        plans: &[PairPlan],
        split_bytes: usize,
        budget: usize,
    ) -> Result<Self> {
        Self::prepare_with_upload(
            ctx,
            weights,
            plans,
            split_bytes,
            budget,
            DeviceQ8FfnPair::upload,
        )
    }

    fn prepare_with_upload(
        ctx: &CudaContext,
        weights: &[u8],
        plans: &[PairPlan],
        split_bytes: usize,
        budget: usize,
        mut upload: impl FnMut(&CudaContext, &HostQ8FfnPair) -> Result<DeviceQ8FfnPair>,
    ) -> Result<Self> {
        let began = Instant::now();
        let mut bytes = 0usize;
        let mut largest = 0;
        let mut sources = HashSet::new();
        ensure!(!plans.is_empty(), "compound FFN requires every layer pair");
        // Validate all plans and the combined budget before any candidate upload.
        for (pointers, key) in plans {
            let shape = key.gate.shape();
            ensure!(
                shape == key.up.shape()
                    && shape.input_dim() <= 4096
                    && shape.input_dim().is_multiple_of(32)
                    && shape.output_dim() <= i32::MAX as usize
                    && key.gate.model_sha256() == key.up.model_sha256()
                    && key.gate.tensor_name() != key.up.tensor_name(),
                "unsupported compound FFN pair identity/shape"
            );
            for (&pointer, id) in pointers.iter().zip([&key.gate, &key.up]) {
                ensure!(
                    pointer != 0 && sources.insert(pointer),
                    "aliased compound FFN source"
                );
                Q8SourceView::new(id, weights)?;
            }
            let pair_bytes = shape
                .source_bytes()?
                .checked_mul(2)
                .context("compound pair bytes overflow")?;
            bytes = bytes
                .checked_add(pair_bytes)
                .context("compound preparation overflow")?;
            largest = largest.max(pair_bytes);
        }
        let combined = split_bytes
            .checked_add(bytes)
            .context("combined Q8 preparation overflow")?;
        super::runtime::check_budget(combined, budget)?;
        let mut cache = Self {
            pairs: HashMap::new(),
            stats: Q8FfnPairStats {
                is_enabled: true,
                pair_count: plans.len(),
                pairs: plans.iter().map(|(_, key)| key.clone()).collect(),
                resident_bytes: bytes,
                combined_prepared_bytes: combined,
                budget_bytes: budget,
                peak_temporary_host_pair_bytes: largest,
                ..Q8FfnPairStats::default()
            },
        };
        for (pointers, key) in plans {
            let start = Instant::now();
            let host = HostQ8FfnPair::pack(
                Q8SourceView::new(&key.gate, weights)?,
                Q8SourceView::new(&key.up, weights)?,
            )?;
            cache.stats.packing_elapsed_ms += start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            let device = upload(ctx, &host)?;
            ensure!(device.key() == key, "compound upload changed pair identity");
            cache.stats.upload_elapsed_ms += start.elapsed().as_secs_f64() * 1000.0;
            cache.pairs.insert(*pointers, device);
        }
        cache.stats.preparation_elapsed_ms = began.elapsed().as_secs_f64() * 1000.0;
        Ok(cache)
    }

    pub fn stats(&self) -> Q8FfnPairStats {
        self.stats.clone()
    }

    pub fn debug_payload_hashes(&self) -> Result<Vec<(Q8FfnPairKey, String)>> {
        let mut hashes = self
            .pairs
            .values()
            .map(|pair| Ok((pair.key().clone(), pair.debug_payload_sha256()?)))
            .collect::<Result<Vec<_>>>()?;
        hashes.sort_by(|a, b| a.0.gate.tensor_name().cmp(b.0.gate.tensor_name()));
        Ok(hashes)
    }
    pub fn get(
        &self,
        sources: [*const c_void; 2],
        hidden: i32,
        k: i32,
    ) -> Result<&DeviceQ8FfnPair> {
        let pair = self
            .pairs
            .get(&sources.map(|p| p as usize))
            .context("missing model-owned compound FFN pair; no fallback")?;
        let shape = pair.key().gate.shape();
        ensure!(
            hidden > 0
                && k > 0
                && shape.input_dim() == k as usize
                && shape.output_dim() == hidden as usize,
            "compound FFN pair shape differs from request"
        );
        Ok(pair)
    }
}

#[cfg(all(test, nvcc))]
mod tests {
    use super::super::{Q8MatrixShape, Q8SourceIdentity};
    use super::*;

    fn plans() -> Result<Vec<PairPlan>> {
        let shape = Q8MatrixShape::new(32, 2)?;
        (0..2)
            .map(|layer| {
                Ok((
                    [1 + layer * 2, 2 + layer * 2],
                    Q8FfnPairKey {
                        gate: Q8SourceIdentity::from_column_blocks(
                            [1; 32],
                            format!("{layer}.gate"),
                            (layer * 136) as u64,
                            shape,
                        )?,
                        up: Q8SourceIdentity::from_column_blocks(
                            [1; 32],
                            format!("{layer}.up"),
                            (layer * 136 + 68) as u64,
                            shape,
                        )?,
                    },
                ))
            })
            .collect()
    }

    /// Budget, duplicate sources and invalid later plans fail before any pair
    /// upload, even though existing split owners may already be resident.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_pair_plans_fail_before_candidate_upload() -> Result<()> {
        let ctx = CudaContext::new_m40()?;
        let weights = vec![13u8; 272];
        let valid = plans()?;
        let mut duplicate = valid.clone();
        duplicate[1].0 = duplicate[0].0;
        let mut invalid = valid.clone();
        invalid[1].1.up = Q8SourceIdentity::from_column_blocks(
            [1; 32],
            "bad.up".into(),
            272,
            Q8MatrixShape::new(32, 2)?,
        )?;
        for (plan, budget) in [(valid, 100usize), (duplicate, 1000), (invalid, 1000)] {
            let mut uploads = 0;
            let before = CudaContext::total_device_bytes();
            assert!(FfnPairCache::prepare_with_upload(
                &ctx,
                &weights,
                &plan,
                512,
                budget,
                |_, _| {
                    uploads += 1;
                    anyhow::bail!("invalid plan reached upload")
                }
            )
            .is_err());
            assert_eq!(uploads, 0);
            assert_eq!(CudaContext::total_device_bytes(), before);
        }
        Ok(())
    }

    /// Failure after a real first upload releases both prior GPU planes and
    /// leaves the context and pre-existing allocations usable.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_partial_pair_upload_failure_releases_owned_planes() -> Result<()> {
        let ctx = CudaContext::new_m40()?;
        let weights = vec![13u8; 272];
        let plans = plans()?;
        let initial = CudaContext::total_device_bytes();
        let mut uploads = 0;
        let result =
            FfnPairCache::prepare_with_upload(&ctx, &weights, &plans, 512, 1024, |ctx, host| {
                uploads += 1;
                if uploads == 2 {
                    anyhow::bail!("injected second pair failure");
                }
                DeviceQ8FfnPair::upload(ctx, host)
            });
        assert!(result.is_err());
        assert_eq!(uploads, 2);
        assert_eq!(CudaContext::total_device_bytes(), initial);
        let cache = FfnPairCache::prepare(&ctx, &weights, &plans, 512, 1024)?;
        assert_eq!(cache.stats().resident_bytes, 272);
        assert_eq!(cache.stats().combined_prepared_bytes, 784);
        drop(cache);
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }
}
