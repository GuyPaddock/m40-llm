//! Benchmark-only split storage with the qualified packed projection mapping.
use super::{ffi, record_async_kernel, CudaContext, CudaStream};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

fn validate_spans(
    input: *const c_void,
    scales: &[*const c_void],
    values: &[*const c_void],
    biases: &[Option<*const c_void>],
    outputs: &[*mut c_void],
    widths: &[i32],
    k: i32,
) -> Result<()> {
    ensure!(
        k > 0 && k % 32 == 0 && widths.iter().all(|&n| n > 0),
        "split packed projection requires positive widths and complete Q8 blocks"
    );
    widths
        .iter()
        .try_fold(0i32, |sum, &n| sum.checked_add(n))
        .context("split packed projection output width overflow")?;
    ensure!(
        !input.is_null() && (input as usize).is_multiple_of(16),
        "split packed projection requires sixteen-byte input alignment"
    );
    for i in 0..widths.len() {
        ensure!(
            !scales[i].is_null() && (scales[i] as usize).is_multiple_of(2),
            "split packed projection requires FP16 scale planes"
        );
        ensure!(
            !values[i].is_null() && (values[i] as usize).is_multiple_of(8),
            "split packed projection requires eight-byte payload alignment"
        );
        if let Some(bias) = biases[i] {
            ensure!(
                !bias.is_null() && (bias as usize).is_multiple_of(4),
                "split packed projection requires aligned FP32 biases"
            );
        }
        let output = outputs[i].cast_const();
        ensure!(
            !output.is_null()
                && (output as usize).is_multiple_of(4)
                && output != input
                && !scales.contains(&output)
                && !values.contains(&output)
                && !biases.contains(&Some(output))
                && !outputs[..i].contains(&outputs[i]),
            "split packed projection requires disjoint aligned output spans"
        );
    }
    Ok(())
}

impl CudaContext {
    /// Fixed one-column/256-thread QKV on lossless split scale/value planes.
    /// No runtime selection, allocation, copy, synchronization or fallback.
    ///
    /// # Safety
    /// All spans belong to this context, have ordered producers and remain live
    /// through `stream` completion. Input holds K FP32 values at 16-byte
    /// alignment. Each scales/values plane holds widths[i]*(K/32) FP16 scales
    /// and widths[i]*K signed bytes, at two/eight-byte alignment respectively.
    /// Biases and outputs hold widths[i] FP32 values. Outputs are disjoint from
    /// each other and all read spans, including partial overlaps.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn qkv_f32xq8_separate_packed_warp_async(
        &self,
        input: *const c_void,
        scales: [*const c_void; 3],
        values: [*const c_void; 3],
        biases: [Option<*const c_void>; 3],
        outputs: [*mut c_void; 3],
        widths: [i32; 3],
        k: i32,
        stream: CudaStream,
    ) -> Result<()> {
        validate_spans(input, &scales, &values, &biases, &outputs, &widths, k)?;
        let biases = biases.map(|bias| bias.unwrap_or(std::ptr::null()));
        let _guard = self.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_qkv_f32xq8_separate_packed_warp_async(
            self.inner.raw.as_ptr(),
            input,
            scales.as_ptr(),
            values.as_ptr(),
            biases.as_ptr(),
            outputs.as_ptr(),
            widths.as_ptr(),
            k,
            stream.ffi_kind(),
        );
        ensure!(rc == 0, "split packed QKV enqueue failed: {rc}");
        record_async_kernel("qkv_f32xq8_separate_packed_warp");
        Ok(())
    }

    /// Fixed four-column/256-thread output projection on existing split planes.
    ///
    /// # Safety
    /// The span, alignment, disjointness and lifetime requirements of
    /// `qkv_f32xq8_separate_packed_warp_async` apply to one N-column weight.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_f32xq8_separate_coltile4_packed_warp_async(
        &self,
        input: *const c_void,
        scales: *const c_void,
        values: *const c_void,
        output: *mut c_void,
        n: i32,
        k: i32,
        stream: CudaStream,
    ) -> Result<()> {
        validate_spans(input, &[scales], &[values], &[None], &[output], &[n], k)?;
        let _guard = self.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_gemm_f32xq8_separate_coltile4_packed_warp_async(
            self.inner.raw.as_ptr(),
            input,
            scales,
            values,
            output,
            n,
            k,
            stream.ffi_kind(),
        );
        ensure!(rc == 0, "split packed output enqueue failed: {rc}");
        record_async_kernel("gemm_f32xq8_separate_coltile4_packed_warp");
        Ok(())
    }
}
