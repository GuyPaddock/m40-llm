//! Lossless joint gate/up layout and typed ownership. Selection is model-local.
use super::{Q8MatrixShape, Q8SourceIdentity, Q8SourceView};
use anyhow::{ensure, Context, Result};
use serde::Serialize;

/// Pair identity includes both sources; a single-tensor layout key is not enough.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Q8FfnPairKey {
    pub gate: Q8SourceIdentity,
    pub up: Q8SourceIdentity,
}

/// Physical joint owners; original GGUF sources always remain resident.
#[derive(Debug, Default, Clone, Serialize)]
pub struct Q8FfnPairStats {
    pub is_enabled: bool,
    pub pair_count: usize,
    pub pairs: Vec<Q8FfnPairKey>,
    pub resident_bytes: usize,
    pub combined_prepared_bytes: usize,
    pub budget_bytes: usize,
    pub peak_temporary_host_pair_bytes: usize,
    pub retained_host_pair_bytes: usize,
    pub packing_elapsed_ms: f64,
    pub upload_elapsed_ms: f64,
    pub preparation_elapsed_ms: f64,
}

#[derive(Debug)]
pub struct HostQ8FfnPair {
    key: Q8FfnPairKey,
    scales: Vec<u8>,
    values: Vec<u8>,
}

impl HostQ8FfnPair {
    /// Copy original FP16 scale bits and signed payload bytes, without rounding.
    /// Each scale record is [gate:u16,up:u16]; each value record is
    /// [gate:4 bytes,up:4 bytes], column-major with consecutive input tasks.
    pub fn pack(gate: Q8SourceView<'_>, up: Q8SourceView<'_>) -> Result<Self> {
        let shape = gate.identity().shape();
        ensure!(
            shape == up.identity().shape()
                && gate.identity().model_sha256() == up.identity().model_sha256()
                && gate.identity().tensor_name() != up.identity().tensor_name(),
            "joint Q8 FFN requires distinct same-model gate/up with matching shapes"
        );
        ensure!(
            shape.input_dim().is_multiple_of(32),
            "joint Q8 FFN requires complete Q8 blocks"
        );
        let bytes = shape
            .source_bytes()?
            .checked_mul(2)
            .context("joint Q8 FFN bytes overflow")?;
        let mut scales = Vec::with_capacity(shape.block_count()? * 4);
        let mut values = Vec::with_capacity(shape.block_count()? * 64);
        for (g, u) in gate
            .blocks()
            .chunks_exact(34)
            .zip(up.blocks().chunks_exact(34))
        {
            scales.extend_from_slice(&g[..2]);
            scales.extend_from_slice(&u[..2]);
            for (g4, u4) in g[2..].chunks_exact(4).zip(u[2..].chunks_exact(4)) {
                values.extend_from_slice(g4);
                values.extend_from_slice(u4);
            }
        }
        ensure!(
            scales.len() + values.len() == bytes,
            "joint Q8 FFN packed bytes mismatch"
        );
        Ok(Self {
            key: Q8FfnPairKey {
                gate: gate.identity().clone(),
                up: up.identity().clone(),
            },
            scales,
            values,
        })
    }

    pub fn key(&self) -> &Q8FfnPairKey {
        &self.key
    }
    pub fn shape(&self) -> Q8MatrixShape {
        self.key.gate.shape()
    }
    pub fn planes(&self) -> [&[u8]; 2] {
        [&self.scales, &self.values]
    }
    pub fn bytes(&self) -> usize {
        self.scales.len() + self.values.len()
    }
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use crate::cuda::{CudaContext, DeviceBuffer};
    use std::ffi::c_void;

    pub struct DeviceQ8FfnPair {
        ctx: CudaContext,
        key: Q8FfnPairKey,
        planes: [DeviceBuffer; 2],
    }

    impl DeviceQ8FfnPair {
        /// Setup completes each upload. RAII releases earlier allocations on error.
        pub fn upload(ctx: &CudaContext, host: &HostQ8FfnPair) -> Result<Self> {
            let upload = |bytes: &[u8]| -> Result<DeviceBuffer> {
                let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), "q8_ffn_joint:weights")?;
                unsafe {
                    ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
                }
                Ok(buffer)
            };
            Ok(Self {
                ctx: ctx.clone(),
                key: host.key.clone(),
                planes: [upload(&host.scales)?, upload(&host.values)?],
            })
        }

        pub fn key(&self) -> &Q8FfnPairKey {
            &self.key
        }
        pub fn planes(&self) -> [&DeviceBuffer; 2] {
            [&self.planes[0], &self.planes[1]]
        }

        /// Cold audit only. Hash actual resident scale/value bytes in order.
        pub fn debug_payload_sha256(&self) -> Result<String> {
            let mut digest = ring::digest::Context::new(&ring::digest::SHA256);
            for plane in &self.planes {
                let mut bytes = vec![0u8; plane.bytes()];
                unsafe {
                    self.ctx
                        .memcpy_d2h(bytes.as_mut_ptr().cast(), plane.as_ptr(), bytes.len())?;
                }
                digest.update(&bytes);
            }
            Ok(digest
                .finish()
                .as_ref()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect())
        }

        /// # Safety
        /// Input/output and optional intermediates are disjoint spans on this
        /// owner's context. Input holds rows*K f32 at 16-byte alignment; outputs
        /// hold rows*N f32 at four-byte alignment. Order producer streams before
        /// decode, and retain every span and this owner until decode completes.
        pub unsafe fn enqueue_async(
            &self,
            input: *const c_void,
            output: *mut c_void,
            rows: i32,
            intermediates: Option<(*mut c_void, *mut c_void)>,
        ) -> Result<()> {
            let shape = self.key.gate.shape();
            self.ctx.ffn_q8_joint_f32_async(
                input,
                self.planes[0].as_ptr(),
                self.planes[1].as_ptr(),
                output,
                rows,
                i32::try_from(shape.output_dim())?,
                i32::try_from(shape.input_dim())?,
                intermediates,
            )
        }

        /// # Safety
        /// Same spans and lifetime contract as `enqueue_async`; K <= 4096.
        /// Uses the same immutable planes without additional global storage.
        pub unsafe fn enqueue_shared_scales_async(
            &self,
            input: *const c_void,
            output: *mut c_void,
            rows: i32,
            intermediates: Option<(*mut c_void, *mut c_void)>,
        ) -> Result<()> {
            let shape = self.key.gate.shape();
            self.ctx.ffn_q8_joint_shared_scales_f32_async(
                input,
                self.planes[0].as_ptr(),
                self.planes[1].as_ptr(),
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
pub use device::DeviceQ8FfnPair;

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(model: u8, name: &str, k: usize, n: usize) -> Q8SourceIdentity {
        Q8SourceIdentity::from_column_blocks(
            [model; 32],
            name.into(),
            0,
            Q8MatrixShape::new(k, n).unwrap(),
        )
        .unwrap()
    }

    /// Every original bit survives joint packing, including arbitrary scale
    /// encodings, signed extremes and a final odd output column.
    #[test]
    fn test_joint_packing_preserves_both_sources_and_layout() -> Result<()> {
        for (k, n) in [(32, 1), (96, 37), (2048, 3)] {
            let g = identity(1, "gate", k, n);
            let u = identity(1, "up", k, n);
            let gate: Vec<u8> = (0..g.shape().source_bytes()?).map(|i| i as u8).collect();
            let up: Vec<u8> = gate.iter().map(|v| !v).collect();
            let packed = HostQ8FfnPair::pack(
                Q8SourceView::from_tensor_bytes(&g, &gate)?,
                Q8SourceView::from_tensor_bytes(&u, &up)?,
            )?;
            assert_eq!(packed.key(), &Q8FfnPairKey { gate: g, up: u });
            assert_eq!(packed.bytes(), gate.len() + up.len());
            let [scales, values] = packed.planes();
            for (block, (g, u)) in gate.chunks_exact(34).zip(up.chunks_exact(34)).enumerate() {
                assert_eq!(&scales[block * 4..block * 4 + 2], &g[..2]);
                assert_eq!(&scales[block * 4 + 2..block * 4 + 4], &u[..2]);
                for task in 0..8 {
                    let base = block * 64 + task * 8;
                    assert_eq!(&values[base..base + 4], &g[2 + task * 4..6 + task * 4]);
                    assert_eq!(&values[base + 4..base + 8], &u[2 + task * 4..6 + task * 4]);
                }
            }
        }
        Ok(())
    }

    /// Wrong-model, aliased-role, mismatched-shape and ragged-K pairs are
    /// rejected before packing or device allocation rather than reinterpreted.
    #[test]
    fn test_joint_packing_rejects_incompatible_pairs() -> Result<()> {
        let gate = identity(1, "gate", 32, 3);
        let bytes = vec![0; gate.shape().source_bytes()?];
        for up in [
            identity(2, "up", 32, 3),
            identity(1, "gate", 32, 3),
            identity(1, "up", 64, 3),
        ] {
            let other = vec![0; up.shape().source_bytes()?];
            assert!(HostQ8FfnPair::pack(
                Q8SourceView::from_tensor_bytes(&gate, &bytes)?,
                Q8SourceView::from_tensor_bytes(&up, &other)?
            )
            .is_err());
        }
        let gate = identity(1, "gate", 33, 1);
        let up = identity(1, "up", 33, 1);
        let bytes = vec![0; gate.shape().source_bytes()?];
        assert!(HostQ8FfnPair::pack(
            Q8SourceView::from_tensor_bytes(&gate, &bytes)?,
            Q8SourceView::from_tensor_bytes(&up, &bytes)?
        )
        .is_err());
        Ok(())
    }
}
