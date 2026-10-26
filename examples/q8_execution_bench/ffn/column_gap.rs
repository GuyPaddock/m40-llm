//! Fixed payload placement diagnostic, never inference storage.
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, DeviceBuffer},
    infer::execution_weights::HostQ8ExecutionWeight,
};
use serde_json::{json, Value};
use std::time::Instant;

pub(super) fn pack_columns(source: &[u8], k: usize) -> Result<Vec<u8>> {
    ensure!(
        k > 0 && k.is_multiple_of(128) && !source.is_empty() && source.len().is_multiple_of(k),
        "column gap requires complete 128-byte-aligned columns"
    );
    let stride = k.checked_add(128).context("column gap stride overflow")?;
    let len = (source.len() / k)
        .checked_mul(stride)
        .context("column gap size overflow")?;
    let mut result = vec![0xa5; len];
    for (column, original) in result.chunks_exact_mut(stride).zip(source.chunks_exact(k)) {
        column[..k].copy_from_slice(original);
    }
    Ok(result)
}

pub(super) struct GappedPayloads {
    ctx: CudaContext,
    host: [Vec<u8>; 2],
    device: [DeviceBuffer; 2],
    pub report: Value,
}

impl GappedPayloads {
    pub fn new(
        ctx: &CudaContext,
        original: &[HostQ8ExecutionWeight; 2],
        control_bytes: usize,
    ) -> Result<Self> {
        let shape = original[0].key().source.shape();
        ensure!(
            shape == original[1].key().source.shape(),
            "column gap shape mismatch"
        );
        let k = shape.input_dim();
        let expected = k
            .checked_add(128)
            .and_then(|stride| stride.checked_mul(shape.output_dim()))
            .and_then(|one| one.checked_mul(2))
            .context("column gap allocation overflow")?;
        ensure!(
            expected
                .checked_add(control_bytes)
                .is_some_and(|v| v <= 128 * 1024 * 1024),
            "column gap and control exceed 128 MiB"
        );
        let began = Instant::now();
        let host = [
            pack_columns(&original[0].planes()[1], k)?,
            pack_columns(&original[1].planes()[1], k)?,
        ];
        let bytes: usize = host.iter().map(Vec::len).sum();
        ensure!(bytes == expected, "column gap byte accounting mismatch");
        for (padded, source) in host.iter().zip(original) {
            ensure!(
                padded
                    .chunks_exact(k + 128)
                    .flat_map(|c| c[..k].iter())
                    .eq(source.planes()[1].iter()),
                "column gap changed original payload bits"
            );
        }
        let packing_and_inverse_ms = began.elapsed().as_secs_f64() * 1000.0;
        let before = CudaContext::total_device_bytes();
        let began = Instant::now();
        let upload = |values: &[u8]| -> Result<_> {
            let buffer = DeviceBuffer::new_tagged(ctx, values.len(), "ffn_read:column_gap")?;
            unsafe {
                ctx.memcpy_h2d(buffer.as_mut_ptr(), values.as_ptr().cast(), values.len())?;
            }
            Ok(buffer)
        };
        let device = [upload(&host[0])?, upload(&host[1])?];
        let upload_ms = began.elapsed().as_secs_f64() * 1000.0;
        let after = CudaContext::total_device_bytes();
        ensure!(
            after.checked_sub(before) == Some(bytes),
            "column gap tracked allocation mismatch"
        );
        let logical_bytes: usize = original.iter().map(|w| w.planes()[1].len()).sum();
        let report = json!({"layout":"PayloadColumnGap128V1","keys":original.iter().map(|w|w.key()).collect::<Vec<_>>(),
            "k":k,"n":shape.output_dim(),"column_gap_bytes":128,"payload_stride_bytes":k+128,
            "logical_payload_bytes":logical_bytes,"padding_bytes":bytes-logical_bytes,
            "additional_payload_bytes":bytes,"retained_host_payload_bytes":bytes,"combined_prepared_bytes":bytes+control_bytes,
            "packing_and_inverse_ms":packing_and_inverse_ms,"allocation_upload_ms":upload_ms,
            "source_payload_bits_preserved":true,"tracked_bytes_before":before,"tracked_bytes_after":after});
        Ok(Self {
            ctx: ctx.clone(),
            host,
            device,
            report,
        })
    }

    pub fn pointers(&self) -> [*const std::ffi::c_void; 2] {
        self.device.each_ref().map(DeviceBuffer::as_ptr)
    }

    pub fn verify(&self) -> Result<()> {
        for (device, original) in self.device.iter().zip(&self.host) {
            let mut actual = vec![0u8; original.len()];
            unsafe {
                self.ctx
                    .memcpy_d2h(actual.as_mut_ptr().cast(), device.as_ptr(), actual.len())?;
            }
            ensure!(actual == *original, "column gap payload or padding changed");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every original payload byte survives at odd widths; padding stays
    /// sentinel-filled and malformed columns are rejected rather than truncated.
    #[test]
    fn test_column_gap_preserves_bytes_and_separates_columns() -> Result<()> {
        for (k, n) in [(128, 1), (640, 7), (2048, 17)] {
            let raw: Vec<_> = (0..k * n).map(|i| (i * 17 + i / 11) as u8).collect();
            let packed = pack_columns(&raw, k)?;
            assert_eq!(packed.len(), (k + 128) * n);
            for (padded, original) in packed.chunks_exact(k + 128).zip(raw.chunks_exact(k)) {
                assert_eq!(&padded[..k], original);
                assert!(padded[k..].iter().all(|b| *b == 0xa5));
            }
        }
        for (raw, k) in [(vec![1; 128], 0), (vec![1; 127], 128), (vec![1; 32], 32)] {
            assert!(pack_columns(&raw, k).is_err());
        }
        Ok(())
    }
}
