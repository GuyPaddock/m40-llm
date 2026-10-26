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
    column_gap: usize,
) -> Result<()> {
    ensure!(
        matches!(rows, 1 | 4) && n > 0 && k > 0 && k % 32 == 0 && k <= i32::MAX - 31,
        "weight checksum requires rows 1/4 and complete Q8 blocks"
    );
    let payload_bytes = (n as usize)
        .checked_mul(k as usize)
        .context("weight checksum span overflow")?;
    let value_bytes = (k as usize)
        .checked_add(column_gap)
        .and_then(|pitch| pitch.checked_mul(n as usize))
        .context("pitched checksum span overflow")?;
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
            (value, value_bytes, value_alignment),
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
    /// Checksum original planes in sequential or fixed transposed column order.
    /// Each modular output additionally contains its physical CTA index.
    ///
    /// # Safety
    /// Same live, ordered, disjoint spans as `ffn_q8_weight_checksum_async`.
    /// Hidden is positive and divisible by 32; outputs remain rows*hidden*2 u32.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_weight_checksum_ordered_async(
        &self,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        is_transposed: bool,
    ) -> Result<()> {
        ensure!(
            hidden > 0 && hidden % 32 == 0,
            "ordered checksum requires N divisible by 32"
        );
        validate_checksum_spans(&scales, &values, output, rows, hidden, k, 2, 4, 0)?;
        let _guard = self.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_ffn_q8_weight_checksum_ordered_async(
            self.inner.raw.as_ptr(),
            scales.as_ptr(),
            values.as_ptr(),
            output,
            rows,
            hidden,
            k,
            i32::from(is_transposed),
        );
        ensure!(
            rc == 0,
            "m40llm_ffn_q8_weight_checksum_ordered_async failed: {rc}"
        );
        record_async_kernel("ffn_q8_weight_checksum_ordered_128");
        Ok(())
    }

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
        validate_checksum_spans(&scales, &values, output, rows, hidden, k, 2, 4, 0)?;
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

    /// Read the same logical gate/up words with zero or 128-byte column gaps.
    /// This diagnostic never computes inference activations.
    ///
    /// # Safety
    /// Scales hold hidden*(k/32) u16 values; payloads hold hidden*(k+gap)
    /// bytes at 128-byte alignment, with k logical bytes per column. Output
    /// holds rows*hidden*2 u32 values, disjoint from every full padded span.
    /// Keep context owners alive and producers ordered through decode completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_weight_checksum_pitched_async(
        &self,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        column_gap: i32,
    ) -> Result<()> {
        ensure!(
            matches!(column_gap, 0 | 128) && k > 0 && k % 128 == 0 && k <= i32::MAX - 128,
            "pitched checksum requires K divisible by 128 and gap 0/128"
        );
        validate_checksum_spans(
            &scales,
            &values,
            output,
            rows,
            hidden,
            k,
            2,
            128,
            column_gap as usize,
        )?;
        let _guard = self.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_ffn_q8_weight_checksum_pitched_async(
            self.inner.raw.as_ptr(),
            scales.as_ptr(),
            values.as_ptr(),
            output,
            rows,
            hidden,
            k,
            column_gap,
        );
        ensure!(
            rc == 0,
            "m40llm_ffn_q8_weight_checksum_pitched_async failed: {rc}"
        );
        record_async_kernel("ffn_q8_weight_checksum_pitched_128");
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
        validate_checksum_spans(&[scales], &[values], output, rows, n, k, 1, 8, 0)?;
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
