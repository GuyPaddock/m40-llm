//! Benchmark-only exact reductions; no generation dispatch is selected here.
use super::{ffi, record_async_kernel, CudaContext, CudaStream};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

impl CudaContext {
    /// Fixed original-tree warp finish for weighted RMSNorm, with two barriers.
    /// The production consumer and its cache experiment are unchanged.
    ///
    /// # Safety
    /// Input/output hold rows*dim FP32 values on this context at four-byte
    /// alignment. Weight holds dim FP16/FP32 values at two/four-byte alignment
    /// for dtype 0/1. Output cannot overlap either read span. Order producers
    /// and retain every span through the selected stream's completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rms_norm_f32_weighted_warp_async(
        &self,
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: u32,
        dim: u32,
        eps: f32,
        weight_dtype: u32,
        stream: CudaStream,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= i32::MAX as u32 && dim > 0 && weight_dtype <= 1,
            "warp weighted RMSNorm requires bounded positive rows/dim and FP16/FP32 weights"
        );
        (rows as usize)
            .checked_mul(dim as usize)
            .and_then(|n| n.checked_mul(4))
            .context("warp weighted RMSNorm size overflow")?;
        ensure!(eps.is_finite() && eps >= 0.0, "invalid RMSNorm epsilon");
        ensure!(
            !input.is_null()
                && !weight.is_null()
                && !output.is_null()
                && input != output.cast_const()
                && weight != output.cast_const()
                && (input as usize).is_multiple_of(4)
                && (output as usize).is_multiple_of(4)
                && (weight as usize).is_multiple_of(if weight_dtype == 0 { 2 } else { 4 }),
            "warp weighted RMSNorm requires aligned disjoint spans"
        );
        let _guard = self.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_rms_norm_f32_weighted_warp_async(
            self.inner.raw.as_ptr(),
            input,
            weight,
            output,
            rows,
            dim,
            eps,
            weight_dtype,
            stream.ffi_kind(),
        );
        ensure!(rc == 0, "warp weighted RMSNorm enqueue failed: {rc}");
        record_async_kernel("rms_norm_f32_weighted_warp");
        Ok(())
    }
}
