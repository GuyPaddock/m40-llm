use super::{ffi, record_async_kernel, CudaContext};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

impl CudaContext {
    /// # Safety
    /// Input holds rows*K f32 values, scales each hold N*(K/32) u16 values,
    /// and packet holds 2*N*K bytes in the task-pair format. Output/captures
    /// each hold rows*N f32 values. Retain all context owners and order their
    /// producers through decode completion; no output may overlap any span.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn ffn_q8_task_pair_packet_async(
        &self,
        input: *const c_void,
        gate_scales: *const c_void,
        up_scales: *const c_void,
        packets: *const c_void,
        output: *mut c_void,
        rows: i32,
        n: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
    ) -> Result<()> {
        ensure!(
            matches!(rows, 1 | 4) && n > 0 && k > 0 && k % 1024 == 0 && k <= i32::MAX - 31,
            "FFN task pair requires rows 1/4, positive N and K divisible by 1024"
        );
        let weights = (n as usize)
            .checked_mul(k as usize)
            .context("task-pair weight size overflow")?;
        let packet_bytes = weights
            .checked_mul(2)
            .context("task-pair payload overflow")?;
        let input_bytes = (rows as usize)
            .checked_mul(k as usize)
            .and_then(|v| v.checked_mul(4))
            .context("task-pair input overflow")?;
        let output_bytes = (rows as usize)
            .checked_mul(n as usize)
            .and_then(|v| v.checked_mul(4))
            .context("task-pair output overflow")?;
        let span = |pointer: *const c_void, bytes: usize, align: usize| -> Result<_> {
            let start = pointer as usize;
            ensure!(
                start != 0 && start.is_multiple_of(align),
                "unaligned/null task-pair span"
            );
            Ok((
                start,
                start
                    .checked_add(bytes)
                    .context("task-pair address overflow")?,
            ))
        };
        let reads = [
            span(input, input_bytes, 16)?,
            span(gate_scales, weights / 16, 2)?,
            span(up_scales, weights / 16, 2)?,
            span(packets, packet_bytes, 16)?,
        ];
        let (g, u) = intermediates.unwrap_or((std::ptr::null_mut(), std::ptr::null_mut()));
        let pointers = [output, g, u];
        let mut writes = [(0, 0); 3];
        for index in 0..if intermediates.is_some() { 3 } else { 1 } {
            let (start, end) = span(pointers[index].cast_const(), output_bytes, 4)?;
            for &(read_start, read_end) in &reads {
                ensure!(
                    end <= read_start || read_end <= start,
                    "task-pair output overlaps read span"
                );
            }
            for &(other_start, other_end) in &writes[..index] {
                ensure!(
                    end <= other_start || other_end <= start,
                    "task-pair outputs overlap"
                );
            }
            writes[index] = (start, end);
        }
        let _guard = self.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_ffn_q8_task_pair_packet_async(
            self.inner.raw.as_ptr(),
            input,
            gate_scales,
            up_scales,
            packets,
            output,
            g,
            u,
            rows,
            n,
            k,
        );
        ensure!(rc == 0, "m40llm_ffn_q8_task_pair_packet_async failed: {rc}");
        record_async_kernel("ffn_q8_task_pair_packet_128");
        Ok(())
    }
}
