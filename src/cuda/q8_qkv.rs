use super::{ffi, record_async_kernel, CudaContext, CudaStream};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

impl CudaContext {
    /// Enqueue the fixed 256-thread native split4 QKV with exact-tree warp finish.
    /// This low-level experiment does not select a generation configuration.
    ///
    /// # Safety
    /// All spans belong to this context, have completed/ordered producers, and
    /// remain live through the selected stream's completion. Input holds K f32;
    /// each raw Q8_0 weight holds widths[i]*(K/32)*34 bytes at two-byte alignment.
    /// Outputs and optional biases hold widths[i] f32 at four-byte alignment.
    /// Output spans are mutually disjoint and do not overlap any read span.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn qkv_f32xq8_0_gguf_warp_async(
        &self,
        input: *const c_void,
        weights: [*const c_void; 3],
        biases: [Option<*const c_void>; 3],
        outputs: [*mut c_void; 3],
        widths: [i32; 3],
        k: i32,
        stream: CudaStream,
    ) -> Result<()> {
        ensure!(
            k > 0 && k % 32 == 0 && widths.iter().all(|&n| n > 0),
            "QKV warp requires positive widths and complete Q8 blocks"
        );
        widths
            .iter()
            .try_fold(0i32, |sum, &n| sum.checked_add(n))
            .context("QKV output width overflow")?;
        ensure!(
            !input.is_null() && (input as usize).is_multiple_of(4),
            "QKV input requires aligned FP32 storage"
        );
        for i in 0..3 {
            ensure!(
                !weights[i].is_null() && (weights[i] as usize).is_multiple_of(2),
                "QKV weights require aligned Q8 blocks"
            );
            ensure!(
                !outputs[i].is_null()
                    && (outputs[i] as usize).is_multiple_of(4)
                    && outputs[i].cast_const() != input
                    && !outputs[..i].contains(&outputs[i]),
                "QKV outputs require disjoint aligned FP32 storage"
            );
            if let Some(bias) = biases[i] {
                ensure!(
                    !bias.is_null() && (bias as usize).is_multiple_of(4),
                    "QKV bias requires aligned FP32 storage"
                );
            }
        }
        let biases = biases.map(|p| p.unwrap_or(std::ptr::null()));
        let _guard = self.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_qkv_f32xq8_0_gguf_warp_async(
            self.inner.raw.as_ptr(),
            input,
            weights.as_ptr(),
            biases.as_ptr(),
            outputs.as_ptr(),
            widths.as_ptr(),
            k,
            stream.ffi_kind(),
        );
        ensure!(rc == 0, "m40llm_qkv_f32xq8_0_gguf_warp_async failed: {rc}");
        record_async_kernel("qkv_f32xq8_0_gguf_warp");
        Ok(())
    }
}
