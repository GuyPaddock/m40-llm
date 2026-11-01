//! Immutable packet owners retained equally in all five research arms.
use super::{
    DeviceQ8WarpPacket, HostQ8WarpPacket, Q8PortfolioArm, Q8PortfolioStats, Q8PortfolioTensorStats,
    Q8SeparateLaunch, Q8SourceIdentity, Q8SourceView,
};
use crate::{
    cuda::{CudaContext, CudaStream},
    evaluation::identity::sha256,
};
use anyhow::{ensure, Context, Result};
use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    time::Instant,
};

pub(super) struct PortfolioPlan {
    pub pointer: usize,
    pub source: Q8SourceIdentity,
    pub is_vocabulary: bool,
}

pub(super) struct PortfolioCache {
    ctx: CudaContext,
    packets: HashMap<usize, (DeviceQ8WarpPacket, bool)>,
    stats: Q8PortfolioStats,
}

impl PortfolioCache {
    pub fn prepare(
        ctx: &CudaContext,
        weights: &[u8],
        plans: &[PortfolioPlan],
        original_prepared_bytes: usize,
        budget: usize,
        arm: Q8PortfolioArm,
    ) -> Result<Self> {
        Self::prepare_with_upload(
            ctx,
            weights,
            plans,
            original_prepared_bytes,
            budget,
            arm,
            DeviceQ8WarpPacket::upload,
        )
    }

    fn prepare_with_upload(
        ctx: &CudaContext,
        weights: &[u8],
        plans: &[PortfolioPlan],
        original_prepared_bytes: usize,
        budget: usize,
        arm: Q8PortfolioArm,
        mut upload: impl FnMut(&CudaContext, &HostQ8WarpPacket) -> Result<DeviceQ8WarpPacket>,
    ) -> Result<Self> {
        let began = Instant::now();
        ensure!(
            plans.len() > 1 && plans.iter().filter(|p| p.is_vocabulary).count() == 1,
            "portfolio requires down projections and exactly one vocabulary"
        );
        let mut pointers = HashSet::new();
        let mut resident_bytes = 0usize;
        let mut largest = 0usize;
        for plan in plans {
            ensure!(
                plan.pointer != 0 && pointers.insert(plan.pointer),
                "aliased portfolio source"
            );
            ensure!(
                plan.source.model_sha256() == plans[0].source.model_sha256(),
                "portfolio model identity mismatch"
            );
            let shape = plan.source.shape();
            ensure!(
                shape.input_dim().is_multiple_of(256)
                    && shape.output_dim().is_multiple_of(4)
                    && shape.input_dim() <= i32::MAX as usize
                    && shape.output_dim() <= i32::MAX as usize,
                "unsupported portfolio packet shape"
            );
            let bytes = Q8SourceView::new(&plan.source, weights)?.blocks().len();
            resident_bytes = resident_bytes
                .checked_add(bytes)
                .context("portfolio size overflow")?;
            largest = largest.max(bytes);
        }
        let combined = original_prepared_bytes
            .checked_add(resident_bytes)
            .context("portfolio budget overflow")?;
        super::runtime::check_budget(combined, budget)?;
        let mut cache = Self {
            ctx: ctx.clone(),
            packets: HashMap::new(),
            stats: Q8PortfolioStats {
                arm,
                tensors: Vec::new(),
                resident_bytes,
                original_prepared_bytes,
                combined_prepared_bytes: combined,
                budget_bytes: budget,
                peak_temporary_host_bytes: largest
                    .checked_mul(2)
                    .context("portfolio host size overflow")?,
                packing_and_validation_ms: 0.,
                upload_ms: 0.,
                preparation_ms: 0.,
            },
        };
        for plan in plans {
            let start = Instant::now();
            let source = Q8SourceView::new(&plan.source, weights)?;
            let host = HostQ8WarpPacket::pack(source)?;
            ensure!(
                host.unpack_column_blocks()? == source.blocks(),
                "portfolio inverse changed source bits"
            );
            let hashes = [sha256(host.planes()[0])?, sha256(host.planes()[1])?];
            cache.stats.packing_and_validation_ms += start.elapsed().as_secs_f64() * 1000.;
            let start = Instant::now();
            let device = upload(ctx, &host)?;
            ensure!(
                device.key() == host.key(),
                "portfolio upload changed identity"
            );
            cache.stats.upload_ms += start.elapsed().as_secs_f64() * 1000.;
            cache.stats.tensors.push(Q8PortfolioTensorStats {
                key: host.key().clone(),
                is_vocabulary: plan.is_vocabulary,
                bytes: host.bytes(),
                plane_sha256: hashes,
            });
            cache
                .packets
                .insert(plan.pointer, (device, plan.is_vocabulary));
        }
        cache.stats.preparation_ms = began.elapsed().as_secs_f64() * 1000.;
        Ok(cache)
    }

    pub fn stats(&self) -> Q8PortfolioStats {
        self.stats.clone()
    }
    pub fn arm(&self) -> Q8PortfolioArm {
        self.stats.arm
    }

    /// # Safety
    /// Spans and producers obey the model's projection contract. Packet owners
    /// remain live until the supplied stream completes, just like split owners.
    pub unsafe fn try_project(
        &self,
        source: *const c_void,
        input: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        stream: CudaStream,
    ) -> Result<bool> {
        if m != 1 || (!self.stats.arm.has_packet_down() && !self.stats.arm.has_packet_vocabulary())
        {
            return Ok(false);
        }
        let Some((packet, is_vocabulary)) = self.packets.get(&(source as usize)) else {
            return Ok(false);
        };
        let is_enabled = if *is_vocabulary {
            self.stats.arm.has_packet_vocabulary()
        } else {
            self.stats.arm.has_packet_down()
        };
        if !is_enabled {
            return Ok(false);
        }
        let shape = packet.key().source.shape();
        ensure!(
            n > 0 && k > 0 && shape.input_dim() == k as usize && shape.output_dim() == n as usize,
            "portfolio projection shape mismatch; no fallback"
        );
        packet.enqueue_async(
            input,
            output,
            1,
            Q8SeparateLaunch::new(4, if *is_vocabulary { 128 } else { 256 })?,
            stream,
        )?;
        Ok(true)
    }

    /// Cold audit only. It must not be called while a generation is in flight.
    pub fn verify_payloads(&self) -> Result<()> {
        let ctx = &self.ctx;
        for (packet, _) in self.packets.values() {
            let expected = self
                .stats
                .tensors
                .iter()
                .find(|t| &t.key == packet.key())
                .context("missing packet audit identity")?;
            for (plane, hash) in packet.planes().into_iter().zip(&expected.plane_sha256) {
                let mut bytes = vec![0; plane.bytes()];
                unsafe {
                    ctx.memcpy_d2h(bytes.as_mut_ptr().cast(), plane.as_ptr(), bytes.len())?;
                }
                ensure!(sha256(&bytes)? == *hash, "portfolio payload mutated");
            }
        }
        Ok(())
    }
}

#[cfg(all(test, nvcc))]
mod tests {
    use super::super::Q8MatrixShape;
    use super::*;

    fn plans() -> Result<Vec<PortfolioPlan>> {
        let shape = Q8MatrixShape::new(256, 4)?;
        (0..2)
            .map(|i| {
                Ok(PortfolioPlan {
                    pointer: i + 1,
                    source: Q8SourceIdentity::from_column_blocks(
                        [1; 32],
                        format!("tensor{i}"),
                        (i * 1088) as u64,
                        shape,
                    )?,
                    is_vocabulary: i == 1,
                })
            })
            .collect()
    }

    /// Validate every plan and combined budget before any candidate allocation;
    /// a later upload failure releases all earlier packet planes.
    #[test]
    #[ignore = "requires M40 hardware"]
    fn test_portfolio_preparation_is_transactional() -> Result<()> {
        let ctx = CudaContext::new_m40()?;
        let bytes = vec![13u8; 2176];
        let initial = CudaContext::total_device_bytes();
        let mut calls = 0;
        assert!(PortfolioCache::prepare_with_upload(
            &ctx,
            &bytes,
            &plans()?,
            512,
            1000,
            Q8PortfolioArm::All,
            |_, _| {
                calls += 1;
                anyhow::bail!("unexpected upload")
            }
        )
        .is_err());
        assert_eq!(calls, 0);
        let mut bad = plans()?;
        bad[1].pointer = bad[0].pointer;
        assert!(PortfolioCache::prepare_with_upload(
            &ctx,
            &bytes,
            &bad,
            512,
            4096,
            Q8PortfolioArm::All,
            |_, _| {
                calls += 1;
                anyhow::bail!("unexpected upload")
            }
        )
        .is_err());
        assert_eq!(calls, 0);
        assert!(PortfolioCache::prepare_with_upload(
            &ctx,
            &bytes,
            &plans()?,
            512,
            4096,
            Q8PortfolioArm::All,
            |ctx, host| {
                calls += 1;
                if calls == 2 {
                    anyhow::bail!("injected second upload failure");
                }
                DeviceQ8WarpPacket::upload(ctx, host)
            }
        )
        .is_err());
        assert_eq!(calls, 2);
        assert_eq!(CudaContext::total_device_bytes(), initial);
        for arm in [
            Q8PortfolioArm::Control,
            Q8PortfolioArm::Down,
            Q8PortfolioArm::Vocabulary,
            Q8PortfolioArm::Argmax,
            Q8PortfolioArm::All,
        ] {
            let cache = PortfolioCache::prepare(&ctx, &bytes, &plans()?, 512, 4096, arm)?;
            assert_eq!(cache.stats().resident_bytes, 2176);
            assert_eq!(cache.stats().combined_prepared_bytes, 2688);
            assert_eq!(CudaContext::total_device_bytes() - initial, 2176);
            cache.verify_payloads()?;
            drop(cache);
            assert_eq!(CudaContext::total_device_bytes(), initial);
        }
        Ok(())
    }
}
