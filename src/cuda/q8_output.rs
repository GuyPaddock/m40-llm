use super::{ffi, record_async_kernel, CudaContext, CudaStream};
use anyhow::{ensure, Result};
use std::ffi::c_void;

impl CudaContext {
    /// One fixed raw-Q8 four-column/256-thread original-tree warp finish.
    /// Enqueues only; it does not select a generation policy.
    ///
    /// # Safety
    /// Ordered producers and live, disjoint spans on this context are required
    /// through stream completion. Input holds K f32 at four-byte alignment,
    /// weights hold N*(K/32)*34 raw Q8_0 bytes at two-byte alignment, and output
    /// holds N f32 at four-byte alignment. Output may not overlap either input.
    pub unsafe fn gemm_f32xq8_0_gguf_coltile4_warp_async(
        &self,
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        n: i32,
        k: i32,
        stream: CudaStream,
    ) -> Result<()> {
        ensure!(
            n > 0 && k > 0 && k % 32 == 0,
            "output warp requires positive N and complete Q8 blocks"
        );
        ensure!(
            !input.is_null()
                && !weight.is_null()
                && !output.is_null()
                && input != output.cast_const()
                && weight != output.cast_const(),
            "output warp requires disjoint non-null spans"
        );
        ensure!(
            (input as usize).is_multiple_of(4)
                && (weight as usize).is_multiple_of(2)
                && (output as usize).is_multiple_of(4),
            "output warp span alignment mismatch"
        );
        let _guard = self.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_gemm_f32xq8_0_gguf_coltile4_warp_async(
            self.inner.raw.as_ptr(),
            input,
            weight,
            output,
            n,
            k,
            stream.ffi_kind(),
        );
        ensure!(
            rc == 0,
            "m40llm_gemm_f32xq8_0_gguf_coltile4_warp_async failed: {rc}"
        );
        record_async_kernel("gemm_f32xq8_0_gguf_coltile4_warp");
        Ok(())
    }
}
