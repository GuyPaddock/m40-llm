use super::{ffi, record_async_kernel, CudaContext, CudaStream};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

impl CudaContext {
    /// # Safety
    /// Input/output are live rows*K/N f32 spans; scales/packets contain N*K/16
    /// and N*K bytes in the version-1 warp-packet format. Order producers on
    /// the selected stream and retain all owners through completion.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn gemm_f32xq8_warp_packet_async(
        &self,
        input: *const c_void,
        scales: *const c_void,
        packets: *const c_void,
        output: *mut c_void,
        rows: i32,
        n: i32,
        k: i32,
        threads: i32,
        stream: CudaStream,
    ) -> Result<()> {
        ensure!(
            matches!(rows, 1 | 4)
                && matches!(threads, 128 | 256)
                && n > 0
                && n % 4 == 0
                && k > 0
                && k % 256 == 0
                && k <= i32::MAX - 31,
            "warp packet requires rows 1/4, threads 128/256, N%4=0 and K%256=0"
        );
        let weights = (n as usize)
            .checked_mul(k as usize)
            .context("warp packet weight span overflow")?;
        let in_bytes = (rows as usize)
            .checked_mul(k as usize)
            .and_then(|v| v.checked_mul(4))
            .context("warp packet input span overflow")?;
        let out_bytes = (rows as usize)
            .checked_mul(n as usize)
            .and_then(|v| v.checked_mul(4))
            .context("warp packet output span overflow")?;
        let span = |pointer: *const c_void, bytes: usize, alignment: usize| -> Result<_> {
            let start = pointer as usize;
            ensure!(
                start != 0 && start.is_multiple_of(alignment),
                "unaligned/null warp packet span"
            );
            Ok((
                start,
                start
                    .checked_add(bytes)
                    .context("warp packet address overflow")?,
            ))
        };
        let (out_start, out_end) = span(output.cast_const(), out_bytes, 4)?;
        for (pointer, bytes, alignment) in [
            (input, in_bytes, 4),
            (scales, weights / 16, 8),
            (packets, weights, 16),
        ] {
            let (start, end) = span(pointer, bytes, alignment)?;
            ensure!(
                end <= out_start || out_end <= start,
                "warp packet output overlaps a read span"
            );
        }
        let _guard = self.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_gemm_f32xq8_warp_packet_async(
            self.inner.raw.as_ptr(),
            input,
            scales,
            packets,
            output,
            rows,
            n,
            k,
            threads,
            stream.ffi_kind(),
        );
        ensure!(rc == 0, "m40llm_gemm_f32xq8_warp_packet_async failed: {rc}");
        record_async_kernel("gemm_f32xq8_warp_packet");
        Ok(())
    }
}
