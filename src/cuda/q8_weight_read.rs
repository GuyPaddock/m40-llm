//! Checksum-only diagnostics, deliberately separate from inference kernels.
use super::{ffi, record_async_kernel, CudaContext, CudaStream};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

#[allow(clippy::too_many_arguments)]
fn validate_checksum_spans(
    scales: &[*const c_void],
    values: &[*const c_void],
    output: *mut c_void,
    rows: i32,
    n: i32,
    k: i32,
    outputs_per_column: usize,
    value_alignment: usize,
) -> Result<()> {
    ensure!(
        matches!(rows, 1 | 4) && n > 0 && k > 0 && k % 32 == 0 && k <= i32::MAX - 31,
        "weight checksum requires rows 1/4 and complete Q8 blocks"
    );
    let payload_bytes = (n as usize)
        .checked_mul(k as usize)
        .context("weight checksum span overflow")?;
    let output_bytes = (rows as usize)
        .checked_mul(n as usize)
        .and_then(|v| v.checked_mul(outputs_per_column * 4))
        .context("weight checksum output overflow")?;
    let span = |pointer: *const c_void, bytes: usize, alignment: usize| -> Result<_> {
        let start = pointer as usize;
        ensure!(
            start != 0 && start.is_multiple_of(alignment),
            "unaligned/null weight checksum span"
        );
        let end = start
            .checked_add(bytes)
            .context("weight checksum address overflow")?;
        Ok((start, end))
    };
    let (out_start, out_end) = span(output.cast_const(), output_bytes, 4)?;
    for (&scale, &value) in scales.iter().zip(values) {
        for (pointer, bytes, alignment) in [
            (scale, payload_bytes / 16, 2),
            (value, payload_bytes, value_alignment),
        ] {
            let (start, end) = span(pointer, bytes, alignment)?;
            ensure!(
                end <= out_start || out_end <= start,
                "weight checksum output overlaps a plane"
            );
        }
    }
    Ok(())
}

impl CudaContext {
    /// Checksum all gate/up scale and payload bits on the decode stream.
    /// This benchmark diagnostic does not compute inference activations.
    ///
    /// # Safety
    /// Each scale plane holds hidden*(k/32) u16 values; each payload plane
    /// holds hidden*k bytes. Output holds rows*hidden*2 u32 checksums.
    /// All spans belong to this context, survive decode completion and have
    /// producers ordered before decode. Output must not overlap any plane.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_weight_checksum_async(
        &self,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
    ) -> Result<()> {
        validate_checksum_spans(&scales, &values, output, rows, hidden, k, 2, 4)?;
        let _guard = self.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_ffn_q8_weight_checksum_async(
            self.inner.raw.as_ptr(),
            scales.as_ptr(),
            values.as_ptr(),
            output,
            rows,
            hidden,
            k,
        );
        ensure!(rc == 0, "m40llm_ffn_q8_weight_checksum_async failed: {rc}");
        record_async_kernel("ffn_q8_weight_checksum_128");
        Ok(())
    }

    /// Read original planes with the prepared four-column/eight-value mapping.
    /// Output is a modular checksum, not an inference activation.
    ///
    /// # Safety
    /// Scales/values hold n*(k/32) u16 and n*k bytes at two/eight-byte
    /// alignment. Output holds rows*n u32 values and is disjoint from both.
    /// Retain all owners and order producers through stream completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn projection_q8_weight_checksum_async(
        &self,
        scales: *const c_void,
        values: *const c_void,
        output: *mut c_void,
        rows: i32,
        n: i32,
        k: i32,
        threads: i32,
        stream: CudaStream,
    ) -> Result<()> {
        ensure!(
            matches!(threads, 128 | 256),
            "prepared checksum requires 128/256 threads"
        );
        validate_checksum_spans(&[scales], &[values], output, rows, n, k, 1, 8)?;
        let _guard = self.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_q8_separate_weight_checksum_async(
            self.inner.raw.as_ptr(),
            scales,
            values,
            output,
            rows,
            n,
            k,
            threads,
            stream.ffi_kind(),
        );
        ensure!(
            rc == 0,
            "m40llm_q8_separate_weight_checksum_async failed: {rc}"
        );
        record_async_kernel("projection_q8_weight_checksum");
        Ok(())
    }
}
