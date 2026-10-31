//! Experiment-only SA4_128 projection with FP32 sign-add accumulation.
use super::{ffi, record_async_kernel, CudaContext, CudaStream};
use anyhow::{ensure, Context, Result};
use std::{ffi::c_void, ops::Range};

fn validate_span(pointer: *const c_void, bytes: usize, alignment: usize) -> Result<Range<usize>> {
    let start = pointer as usize;
    ensure!(
        start != 0 && start.is_multiple_of(alignment),
        "invalid ShiftAdd pointer alignment"
    );
    let end = start
        .checked_add(bytes)
        .context("ShiftAdd address overflow")?;
    Ok(start..end)
}

impl CudaContext {
    /// Enqueue independent FP32-input GEMVs with SA4_128 weights. There are no
    /// allocations, activation conversions, synchronizations or runtime fallbacks.
    ///
    /// # Safety
    /// All buffers belong to this context and remain live until `stream`
    /// completes, with ordered producers. Input holds rows*K finite FP32 values
    /// and output is writable for rows*N FP32 values, at four-byte alignment.
    /// Weights span N*(K/256)
    /// complete 138-byte SA4_128 blocks at two-byte alignment, with finite
    /// nonnegative FP16 scales. Output must not overlap either read span.
    /// The caller owns buffers and establishes cross-stream dependencies.
    pub unsafe fn gemv_f32xsa4_128_async(
        &self,
        input: *const c_void,
        weights: *const c_void,
        output: *mut c_void,
        rows: i32,
        output_width: i32,
        input_width: i32,
        stream: CudaStream,
    ) -> Result<()> {
        ensure!(
            matches!(rows, 1 | 2 | 4 | 8)
                && output_width > 0
                && input_width > 0
                && input_width % 256 == 0,
            "ShiftAdd requires rows 1/2/4/8 and positive complete-block dimensions"
        );
        let input_bytes = (rows as usize)
            .checked_mul(input_width as usize)
            .and_then(|v| v.checked_mul(4))
            .context("ShiftAdd input size overflow")?;
        let weight_bytes = (output_width as usize)
            .checked_mul(input_width as usize / 256)
            .and_then(|v| v.checked_mul(138))
            .context("ShiftAdd weight size overflow")?;
        let output_bytes = (rows as usize)
            .checked_mul(output_width as usize)
            .and_then(|v| v.checked_mul(4))
            .context("ShiftAdd output size overflow")?;
        let reads = [
            validate_span(input, input_bytes, 4)?,
            validate_span(weights, weight_bytes, 2)?,
        ];
        let destination = validate_span(output.cast_const(), output_bytes, 4)?;
        ensure!(
            reads
                .iter()
                .all(|read| destination.start >= read.end || read.start >= destination.end),
            "ShiftAdd output overlaps a read span"
        );
        let _guard = self.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_gemv_f32xsa4_128_async(
            self.inner.raw.as_ptr(),
            input,
            weights,
            output,
            rows,
            output_width,
            input_width,
            stream.ffi_kind(),
        );
        ensure!(rc == 0, "ShiftAdd enqueue failed: {rc}");
        record_async_kernel("gemv_f32xsa4_128");
        Ok(())
    }
}
