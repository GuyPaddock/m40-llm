//! Lossless four-column packets retaining the original eight-value task tree.
use super::{Q8SourceIdentity, Q8SourceView};
use anyhow::{ensure, Result};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Q8WarpPacketKey {
    pub source: Q8SourceIdentity,
    pub format_version: u32,
}

#[derive(Debug)]
pub struct HostQ8WarpPacket {
    key: Q8WarpPacketKey,
    scales: Vec<u8>,
    values: Vec<u8>,
}

impl HostQ8WarpPacket {
    /// Copy raw scale bits and payload bytes into aligned full-warp packets.
    /// No padding, quantization or floating-point conversion is performed.
    pub fn pack(source: Q8SourceView<'_>) -> Result<Self> {
        let shape = source.identity().shape();
        let (k, n) = (shape.input_dim(), shape.output_dim());
        ensure!(
            k.is_multiple_of(256) && n.is_multiple_of(4),
            "warp packets require K divisible by 256 and N divisible by four"
        );
        let blocks = k / 32;
        let warp_tiles = k / 256;
        let raw = source.blocks();
        let mut scales = vec![0u8; shape.block_count()? * 2];
        let mut values = vec![0u8; k * n];
        for tile in 0..n / 4 {
            for block in 0..blocks {
                for column in 0..4 {
                    let src = ((tile * 4 + column) * blocks + block) * 34;
                    let dst = ((tile * blocks + block) * 4 + column) * 2;
                    scales[dst..dst + 2].copy_from_slice(&raw[src..src + 2]);
                }
            }
            for warp_tile in 0..warp_tiles {
                for half in 0..2 {
                    for lane in 0..32 {
                        for column in 0..4 {
                            let task = warp_tile * 32 + lane;
                            let src = ((tile * 4 + column) * blocks + task / 4) * 34
                                + 2
                                + (task % 4) * 8
                                + half * 4;
                            let dst = (tile * warp_tiles + warp_tile) * 1024
                                + half * 512
                                + lane * 16
                                + column * 4;
                            values[dst..dst + 4].copy_from_slice(&raw[src..src + 4]);
                        }
                    }
                }
            }
        }
        ensure!(
            scales.len() + values.len() == raw.len(),
            "warp packet payload size changed"
        );
        Ok(Self {
            key: Q8WarpPacketKey {
                source: source.identity().clone(),
                format_version: 1,
            },
            scales,
            values,
        })
    }

    pub fn key(&self) -> &Q8WarpPacketKey {
        &self.key
    }
    pub fn planes(&self) -> [&[u8]; 2] {
        [&self.scales, &self.values]
    }
    pub fn bytes(&self) -> usize {
        self.scales.len() + self.values.len()
    }

    /// Invert by logical source column/block/byte, independently of pack loops.
    pub fn unpack_column_blocks(&self) -> Result<Vec<u8>> {
        let shape = self.key.source.shape();
        let (k, n) = (shape.input_dim(), shape.output_dim());
        let blocks = k / 32;
        let mut raw = vec![0u8; shape.source_bytes()?];
        for col in 0..n {
            for block in 0..blocks {
                let out = (col * blocks + block) * 34;
                let scale = ((col / 4 * blocks + block) * 4 + col % 4) * 2;
                raw[out..out + 2].copy_from_slice(&self.scales[scale..scale + 2]);
                for byte in 0..32 {
                    let task = block * 4 + byte / 8;
                    let src = (col / 4 * (k / 256) + task / 32) * 1024
                        + (byte % 8 / 4) * 512
                        + (task % 32) * 16
                        + (col % 4) * 4
                        + byte % 4;
                    raw[out + 2 + byte] = self.values[src];
                }
            }
        }
        Ok(raw)
    }
}

#[cfg(any(feature = "cuda", test))]
fn upload_planes<B>(
    planes: [&[u8]; 2],
    mut upload: impl FnMut(&[u8]) -> Result<B>,
) -> Result<[B; 2]> {
    Ok([upload(planes[0])?, upload(planes[1])?])
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use crate::cuda::{CudaContext, CudaStream, DeviceBuffer};
    use crate::infer::execution_weights::Q8SeparateLaunch;
    use std::ffi::c_void;

    pub struct DeviceQ8WarpPacket {
        ctx: CudaContext,
        key: Q8WarpPacketKey,
        planes: [DeviceBuffer; 2],
    }

    impl DeviceQ8WarpPacket {
        /// Every plane becomes RAII-owned before the next allocation/upload.
        pub fn upload(ctx: &CudaContext, host: &HostQ8WarpPacket) -> Result<Self> {
            let planes = upload_planes(host.planes(), |bytes| {
                let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), "q8_warp_packet:weights")?;
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

        pub fn key(&self) -> &Q8WarpPacketKey {
            &self.key
        }
        pub fn planes(&self) -> [&DeviceBuffer; 2] {
            [&self.planes[0], &self.planes[1]]
        }

        /// # Safety
        /// Input/output are live disjoint rows*K/N f32 spans on this context.
        /// Order their producers and retain every owner until stream drains.
        pub unsafe fn enqueue_async(
            &self,
            input: *const c_void,
            output: *mut c_void,
            rows: usize,
            launch: Q8SeparateLaunch,
            stream: CudaStream,
        ) -> Result<()> {
            ensure!(
                launch.columns() == 4,
                "warp packets require four output columns"
            );
            let shape = self.key.source.shape();
            self.ctx.gemm_f32xq8_warp_packet_async(
                input,
                self.planes[0].as_ptr(),
                self.planes[1].as_ptr(),
                output,
                i32::try_from(rows)?,
                i32::try_from(shape.output_dim())?,
                i32::try_from(shape.input_dim())?,
                launch.threads() as i32,
                stream,
            )
        }
    }
}
#[cfg(feature = "cuda")]
pub use device::DeviceQ8WarpPacket;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::execution_weights::Q8MatrixShape;

    /// Full-warp packets preserve every original bit and identity with no
    /// padding, across output tiles and long input loops.
    #[test]
    fn test_packet_roundtrip_preserves_source_bits_and_size() -> Result<()> {
        for (k, n) in [(256, 4), (512, 12), (11008, 8)] {
            let source = Q8SourceIdentity::from_column_blocks(
                [7; 32],
                "fixture".into(),
                96,
                Q8MatrixShape::new(k, n)?,
            )?;
            let raw: Vec<_> = (0..source.shape().source_bytes()?)
                .map(|i| (i.wrapping_mul(17) ^ (i / 13)) as u8)
                .collect();
            let packet = HostQ8WarpPacket::pack(Q8SourceView::from_tensor_bytes(&source, &raw)?)?;
            assert_eq!(packet.bytes(), raw.len());
            assert_eq!(
                packet.key(),
                &Q8WarpPacketKey {
                    source,
                    format_version: 1
                }
            );
            assert_eq!(packet.unpack_column_blocks()?, raw);
            let [scales, values] = packet.planes();
            assert_eq!(scales.len(), k * n / 16);
            assert_eq!(values.len(), k * n);
        }
        Ok(())
    }

    /// Unsupported shapes reject before packing rather than silently padding
    /// or interpreting a ragged source as an aligned full-warp packet.
    #[test]
    fn test_packet_rejects_unsupported_shapes() -> Result<()> {
        for (k, n) in [(32, 4), (255, 4), (257, 4), (256, 1), (256, 5)] {
            let source = Q8SourceIdentity::from_column_blocks(
                [1; 32],
                "fixture".into(),
                0,
                Q8MatrixShape::new(k, n)?,
            )?;
            let raw = vec![0; source.shape().source_bytes()?];
            assert!(
                HostQ8WarpPacket::pack(Q8SourceView::from_tensor_bytes(&source, &raw)?).is_err()
            );
        }
        Ok(())
    }

    /// Failure after allocating either plane releases that plane and every
    /// earlier owner; success retains both until their caller drops them.
    #[test]
    fn test_packet_upload_failure_releases_partial_planes() -> Result<()> {
        use std::{cell::Cell, rc::Rc};
        struct Plane {
            live: Rc<Cell<usize>>,
        }
        impl Drop for Plane {
            fn drop(&mut self) {
                self.live.set(self.live.get() - 1);
            }
        }
        for fail_at in [0, 1, 2] {
            let live = Rc::new(Cell::new(0));
            let mut attempts = 0;
            let result = upload_planes([&[1, 2], &[3, 4]], |bytes| {
                assert_eq!(bytes, if attempts == 0 { &[1, 2] } else { &[3, 4] });
                let this = attempts;
                attempts += 1;
                live.set(live.get() + 1);
                let plane = Plane { live: live.clone() };
                anyhow::ensure!(this != fail_at, "injected upload failure");
                Ok(plane)
            });
            if fail_at < 2 {
                assert!(result.is_err());
                assert_eq!(live.get(), 0);
            } else {
                assert_eq!(live.get(), 2);
                drop(result);
                assert_eq!(live.get(), 0);
            }
        }
        Ok(())
    }
}
