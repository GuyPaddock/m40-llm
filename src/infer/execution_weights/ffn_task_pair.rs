//! Original-thread task pairs with unchanged separate gate/up scale planes.
use super::{Q8FfnPairKey, Q8SourceView};
use anyhow::{ensure, Context, Result};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Q8FfnTaskPairKey {
    pub sources: Q8FfnPairKey,
    pub format_version: u32,
}

pub struct HostQ8FfnTaskPair {
    key: Q8FfnTaskPairKey,
    planes: [Vec<u8>; 3],
}

impl HostQ8FfnTaskPair {
    /// Pair the original words for task tid and task tid+128, not adjacent
    /// input values. Both scale planes retain their original column order.
    pub fn pack(gate: Q8SourceView<'_>, up: Q8SourceView<'_>) -> Result<Self> {
        let shape = gate.identity().shape();
        ensure!(
            shape == up.identity().shape()
                && gate.identity().model_sha256() == up.identity().model_sha256()
                && gate.identity().tensor_name() != up.identity().tensor_name(),
            "FFN task pairs require distinct same-model matching sources"
        );
        let (k, n) = (shape.input_dim(), shape.output_dim());
        ensure!(
            k.is_multiple_of(1024),
            "FFN task pairs require K divisible by 1024"
        );
        let expected = shape
            .source_bytes()?
            .checked_mul(2)
            .context("FFN task-pair size overflow")?;
        let mut planes = [
            Vec::with_capacity(shape.block_count()? * 2),
            Vec::with_capacity(shape.block_count()? * 2),
            vec![0u8; k * n * 2],
        ];
        for (role, source) in [gate.blocks(), up.blocks()].into_iter().enumerate() {
            for block in source.chunks_exact(34) {
                planes[role].extend_from_slice(&block[..2]);
            }
        }
        let bundles = k / 1024;
        for column in 0..n {
            for bundle in 0..bundles {
                for tid in 0..128 {
                    for half in 0..2 {
                        let task = bundle * 256 + tid + half * 128;
                        let src = (column * (k / 32) + task / 8) * 34 + 2 + (task % 8) * 4;
                        let dst = (column * bundles + bundle) * 2048 + tid * 16 + half * 8;
                        planes[2][dst..dst + 4].copy_from_slice(&gate.blocks()[src..src + 4]);
                        planes[2][dst + 4..dst + 8].copy_from_slice(&up.blocks()[src..src + 4]);
                    }
                }
            }
        }
        ensure!(
            planes.iter().map(Vec::len).sum::<usize>() == expected,
            "FFN task-pair packing changed byte count"
        );
        Ok(Self {
            key: Q8FfnTaskPairKey {
                sources: Q8FfnPairKey {
                    gate: gate.identity().clone(),
                    up: up.identity().clone(),
                },
                format_version: 1,
            },
            planes,
        })
    }

    pub fn key(&self) -> &Q8FfnTaskPairKey {
        &self.key
    }
    pub fn planes(&self) -> [&[u8]; 3] {
        self.planes.each_ref().map(Vec::as_slice)
    }
    pub fn bytes(&self) -> usize {
        self.planes.iter().map(Vec::len).sum()
    }

    /// Reconstruct original blocks in source column/block/byte order.
    pub fn unpack_column_blocks(&self) -> Result<[Vec<u8>; 2]> {
        let shape = self.key.sources.gate.shape();
        let (k, n) = (shape.input_dim(), shape.output_dim());
        let mut raw = [
            vec![0u8; shape.source_bytes()?],
            vec![0u8; shape.source_bytes()?],
        ];
        for (role, source) in raw.iter_mut().enumerate() {
            for column in 0..n {
                for block in 0..k / 32 {
                    let target = (column * (k / 32) + block) * 34;
                    let scale = (column * (k / 32) + block) * 2;
                    source[target..target + 2]
                        .copy_from_slice(&self.planes[role][scale..scale + 2]);
                    for byte in 0..32 {
                        let task = block * 8 + byte / 4;
                        let index = (column * (k / 1024) + task / 256) * 2048
                            + (task % 128) * 16
                            + ((task % 256) / 128) * 8
                            + role * 4
                            + byte % 4;
                        source[target + 2 + byte] = self.planes[2][index];
                    }
                }
            }
        }
        Ok(raw)
    }
}

#[cfg(any(feature = "cuda", test))]
fn upload_planes<B>(
    planes: [&[u8]; 3],
    mut upload: impl FnMut(&[u8]) -> Result<B>,
) -> Result<[B; 3]> {
    Ok([upload(planes[0])?, upload(planes[1])?, upload(planes[2])?])
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use crate::cuda::{CudaContext, DeviceBuffer};
    use std::ffi::c_void;

    pub struct DeviceQ8FfnTaskPair {
        ctx: CudaContext,
        key: Q8FfnTaskPairKey,
        planes: [DeviceBuffer; 3],
    }
    impl DeviceQ8FfnTaskPair {
        pub fn upload(ctx: &CudaContext, host: &HostQ8FfnTaskPair) -> Result<Self> {
            let planes = upload_planes(host.planes(), |bytes| {
                let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), "ffn_task_pair:weights")?;
                unsafe {
                    ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
                }
                Ok(buffer)
            })?;
            Ok(Self {
                ctx: ctx.clone(),
                key: host.key.clone(),
                planes,
            })
        }
        pub fn key(&self) -> &Q8FfnTaskPairKey {
            &self.key
        }
        pub fn planes(&self) -> [&DeviceBuffer; 3] {
            self.planes.each_ref()
        }

        /// # Safety
        /// Input is a live rows*K f32 span at 16-byte alignment; output and
        /// optional gate/up spans hold rows*N f32 values and are disjoint.
        /// All owners belong to this context and survive ordered decode work.
        pub unsafe fn enqueue_async(
            &self,
            input: *const c_void,
            output: *mut c_void,
            rows: i32,
            intermediates: Option<(*mut c_void, *mut c_void)>,
        ) -> Result<()> {
            let shape = self.key.sources.gate.shape();
            self.ctx.ffn_q8_task_pair_packet_async(
                input,
                self.planes[0].as_ptr(),
                self.planes[1].as_ptr(),
                self.planes[2].as_ptr(),
                output,
                rows,
                i32::try_from(shape.output_dim())?,
                i32::try_from(shape.input_dim())?,
                intermediates,
            )
        }
    }
}
#[cfg(feature = "cuda")]
pub use device::DeviceQ8FfnTaskPair;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::execution_weights::{Q8MatrixShape, Q8SourceIdentity};
    fn identity(model: u8, name: &str, k: usize, n: usize) -> Q8SourceIdentity {
        Q8SourceIdentity::from_column_blocks(
            [model; 32],
            name.into(),
            64,
            Q8MatrixShape::new(k, n).unwrap(),
        )
        .unwrap()
    }

    /// Every original bit survives task-pair packing, including arbitrary
    /// scales, signed payloads, odd outputs and multiple dynamic-K bundles.
    #[test]
    fn test_task_pair_roundtrip_preserves_sources_and_size() -> Result<()> {
        for (k, n) in [(1024, 1), (2048, 37), (4096, 3)] {
            let gate = identity(7, "gate", k, n);
            let up = identity(7, "up", k, n);
            let g: Vec<_> = (0..gate.shape().source_bytes()?)
                .map(|i| (i.wrapping_mul(17) ^ (i / 13)) as u8)
                .collect();
            let u: Vec<_> = g.iter().map(|v| !v).collect();
            let host = HostQ8FfnTaskPair::pack(
                Q8SourceView::from_tensor_bytes(&gate, &g)?,
                Q8SourceView::from_tensor_bytes(&up, &u)?,
            )?;
            assert_eq!(
                host.key(),
                &Q8FfnTaskPairKey {
                    sources: Q8FfnPairKey { gate, up },
                    format_version: 1
                }
            );
            assert_eq!(host.bytes(), g.len() + u.len());
            assert_eq!(host.unpack_column_blocks()?, [g, u]);
        }
        Ok(())
    }

    /// Wrong-model, same-role, different-shape and unsupported-K inputs fail
    /// before allocating packet planes, rather than padding or reinterpreting.
    #[test]
    fn test_task_pair_rejects_incompatible_sources() -> Result<()> {
        let gate = identity(7, "gate", 1024, 3);
        let raw = vec![0; gate.shape().source_bytes()?];
        for up in [
            identity(8, "up", 1024, 3),
            identity(7, "gate", 1024, 3),
            identity(7, "up", 2048, 3),
        ] {
            let other = vec![0; up.shape().source_bytes()?];
            assert!(HostQ8FfnTaskPair::pack(
                Q8SourceView::from_tensor_bytes(&gate, &raw)?,
                Q8SourceView::from_tensor_bytes(&up, &other)?
            )
            .is_err());
        }
        for k in [32, 512, 1025] {
            let g = identity(7, "gate", k, 3);
            let u = identity(7, "up", k, 3);
            let raw = vec![0; g.shape().source_bytes()?];
            assert!(HostQ8FfnTaskPair::pack(
                Q8SourceView::from_tensor_bytes(&g, &raw)?,
                Q8SourceView::from_tensor_bytes(&u, &raw)?
            )
            .is_err());
        }
        Ok(())
    }

    /// Upload failure after allocating any of three planes releases the
    /// current plane and every previous owner; success retains all three.
    #[test]
    fn test_task_pair_upload_failure_releases_all_planes() -> Result<()> {
        use std::{cell::Cell, rc::Rc};
        struct Plane(Rc<Cell<usize>>);
        impl Drop for Plane {
            fn drop(&mut self) {
                self.0.set(self.0.get() - 1);
            }
        }
        for fail_at in 0..=3 {
            let live = Rc::new(Cell::new(0));
            let mut calls = 0;
            let result = upload_planes([&[1], &[2], &[3]], |bytes| {
                assert_eq!(bytes, &[calls as u8 + 1]);
                calls += 1;
                live.set(live.get() + 1);
                let plane = Plane(live.clone());
                anyhow::ensure!(calls - 1 != fail_at, "injected upload failure");
                Ok(plane)
            });
            if fail_at < 3 {
                assert!(result.is_err());
                assert_eq!(live.get(), 0);
            } else {
                assert_eq!(live.get(), 3);
                drop(result);
                assert_eq!(live.get(), 0);
            }
        }
        Ok(())
    }
}
