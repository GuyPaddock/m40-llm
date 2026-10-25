use super::{ffi, record_async_kernel, CudaContext};
use anyhow::{ensure, Result};
use std::ffi::c_void;

/// M03's entire bounded geometry set. These do not select a runtime policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
#[repr(i32)]
pub enum Q8FfnFp32Variant {
    ScalarShared256 = 0,
    PackedShared256 = 1,
    ScalarWarp256 = 2,
    PackedWarp256 = 3,
    PackedWarp128 = 4,
}

impl Q8FfnFp32Variant {
    pub const ALL: [Self; 5] = [
        Self::ScalarShared256,
        Self::PackedShared256,
        Self::ScalarWarp256,
        Self::PackedWarp256,
        Self::PackedWarp128,
    ];

    pub fn threads(self) -> u32 {
        if self == Self::PackedWarp128 {
            128
        } else {
            256
        }
    }

    pub fn has_packed_loads(self) -> bool {
        matches!(
            self,
            Self::PackedShared256 | Self::PackedWarp256 | Self::PackedWarp128
        )
    }

    pub fn has_warp_reduction(self) -> bool {
        matches!(
            self,
            Self::ScalarWarp256 | Self::PackedWarp256 | Self::PackedWarp128
        )
    }

    fn operation_name(self) -> &'static str {
        match self {
            Self::ScalarShared256 => "ffn_q8_f32_scalar_shared_256",
            Self::PackedShared256 => "ffn_q8_f32_packed_shared_256",
            Self::ScalarWarp256 => "ffn_q8_f32_scalar_warp_256",
            Self::PackedWarp256 => "ffn_q8_f32_packed_warp_256",
            Self::PackedWarp128 => "ffn_q8_f32_packed_warp_128",
        }
    }
}

impl CudaContext {
    /// Enqueue one genuine 1/4-row FP32-activation FFN ablation on decode.
    /// No allocation, transfer, environment selection or synchronization occurs.
    ///
    /// # Safety
    /// Buffers belong to this context and outlive decode completion. Input/output
    /// hold rows*k / rows*hidden f32 values and may not alias. Each weight matrix
    /// holds hidden*(k/32)*34 bytes of native Q8_0 at two-byte alignment.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_f32_ablation_async(
        &self,
        input: *const c_void,
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        variant: Q8FfnFp32Variant,
    ) -> Result<()> {
        ensure!(
            !input.is_null()
                && !gate.is_null()
                && !up.is_null()
                && !output.is_null()
                && input != output.cast_const(),
            "invalid/aliased FP32 FFN buffers"
        );
        ensure!(matches!(rows,1|4)&&hidden>0&&k>0&&k%32==0,"FP32 FFN ablations require rows 1/4, positive hidden and complete 32-value Q8_0 blocks");
        ensure!(
            (gate as usize).is_multiple_of(2) && (up as usize).is_multiple_of(2),
            "FP32 FFN Q8_0 weights require two-byte alignment"
        );
        let _guard = self.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_ffn_q8_f32_ablation_async(
                self.inner.raw.as_ptr(),
                input,
                gate,
                up,
                output,
                rows,
                hidden,
                k,
                variant as i32,
            )
        };
        ensure!(rc==0,"m40llm_ffn_q8_f32_ablation_async failed: {rc}, variant={variant:?}, rows={rows}, hidden={hidden}, k={k}");
        record_async_kernel(variant.operation_name());
        Ok(())
    }
}
