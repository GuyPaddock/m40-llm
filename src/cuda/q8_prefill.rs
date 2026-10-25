use super::{ffi, record_async_kernel, CudaContext};
use crate::infer::q8_prefill::Q8PrefillWeightShape;
use anyhow::{ensure, Result};
use std::ffi::c_void;

impl CudaContext {
    /// Enqueue a bounded temporary FP32 weight on prefill, without retaining a
    /// cache, allocating, copying to/from host or synchronizing. The result is
    /// column-major [output_dim,input_dim], not native GGUF linear order.
    ///
    /// # Safety
    /// Source/output are disjoint device spans on this context, containing at
    /// least shape.q8_bytes() and capacity_bytes bytes at two/four-byte alignment.
    /// Order any source producer before prefill and keep both spans alive until
    /// the final SGEMM using this temporary weight completes on prefill.
    pub unsafe fn dequantize_q8_0_to_f32_colmajor_async(
        &self,
        source: *const c_void,
        output: *mut c_void,
        capacity_bytes: usize,
        shape: Q8PrefillWeightShape,
    ) -> Result<()> {
        ensure!(
            !source.is_null() && !output.is_null() && source != output.cast_const(),
            "invalid/aliased Q8 prefill dequantization buffers"
        );
        ensure!(
            (source as usize).is_multiple_of(2) && (output as usize).is_multiple_of(4),
            "Q8 prefill source/output require two/four-byte alignment"
        );
        ensure!(
            capacity_bytes >= shape.f32_bytes(),
            "Q8 prefill temporary weight capacity is too small"
        );
        let _guard = self.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_dequantize_q8_0_to_f32_colmajor_async(
                self.inner.raw.as_ptr(),
                source,
                output,
                capacity_bytes,
                shape.input_dim() as i32,
                shape.output_dim() as i32,
            )
        };
        ensure!(
            rc == 0,
            "m40llm_dequantize_q8_0_to_f32_colmajor_async failed: {rc}"
        );
        record_async_kernel("dequantize_q8_0_to_f32_colmajor");
        Ok(())
    }
}
