use super::{ffi, record_async_kernel, CudaContext, CudaStream};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

fn validate_span(pointer: *const c_void, bytes: usize, alignment: usize) -> Result<(usize, usize)> {
    let begin = pointer as usize;
    ensure!(
        begin != 0 && begin.is_multiple_of(alignment),
        "null or unaligned Q4 FFN pointer"
    );
    Ok((
        begin,
        begin
            .checked_add(bytes)
            .context("Q4 FFN pointer overflow")?,
    ))
}

impl CudaContext {
    /// Research-only multi-row gate/up/SwiGLU, preserving single-row reduction
    /// order. The caller owns all buffers; the operation has no device scratch.
    ///
    /// # Safety
    /// Input is rows*K FP32; each native Q4_0 weight matrix is H*K/32*18 bytes;
    /// output is rows*H FP32. All spans must be live on this context, with no
    /// writable overlap. Order producers/consumers on stream and retain owners
    /// until completion. Weights and activations must represent finite values.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn q4_fused_rows_async(
        &self,
        input: *const c_void,
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        rows: usize,
        h: usize,
        k: usize,
        stream: CudaStream,
    ) -> Result<()> {
        ensure!(
            matches!(rows, 1 | 2 | 4 | 8)
                && h > 0
                && h <= i32::MAX as usize
                && k > 0
                && k <= i32::MAX as usize
                && k.is_multiple_of(32),
            "unsupported Q4 FFN shape"
        );
        let weights = h
            .checked_mul(k / 32)
            .and_then(|n| n.checked_mul(18))
            .context("Q4 weight size overflow")?;
        let input_bytes = rows
            .checked_mul(k)
            .and_then(|n| n.checked_mul(4))
            .context("Q4 input size overflow")?;
        let output_bytes = rows
            .checked_mul(h)
            .and_then(|n| n.checked_mul(4))
            .context("Q4 output size overflow")?;
        let output_span = validate_span(output.cast_const(), output_bytes, 4)?;
        for span in [
            validate_span(input, input_bytes, 4)?,
            validate_span(gate, weights, 2)?,
            validate_span(up, weights, 2)?,
        ] {
            ensure!(
                span.1 <= output_span.0 || output_span.1 <= span.0,
                "Q4 output overlaps read-only span"
            );
        }
        let _guard = self.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_q4_fused_rows_async(
            self.inner.raw.as_ptr(),
            input,
            gate,
            up,
            output,
            rows as i32,
            h as i32,
            k as i32,
            stream.ffi_kind(),
        );
        ensure!(rc == 0, "m40llm_q4_fused_rows_async failed: {rc}");
        record_async_kernel("q4_fused_rows");
        Ok(())
    }
}
