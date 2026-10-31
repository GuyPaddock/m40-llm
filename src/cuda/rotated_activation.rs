use super::{ffi, record_async_kernel, CudaContext, CudaStream, DeviceBuffer};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

const GUARD_BYTES: usize = 32;
const GUARD: u8 = 0xa5;

/// Research producer schedules with identical canonical and encoded outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RotatedPreparation {
    Staged,
    Fused,
}

/// Invocation-owned sign/Walsh128/Q8_1 producer. Every enqueue recomputes its
/// outputs; no input identity cache or runtime dispatch uses this owner.
pub struct DeviceRotatedActivation {
    ctx: CudaContext,
    signs: DeviceBuffer,
    signed_input: DeviceBuffer,
    rotated: DeviceBuffer,
    quantized: DeviceBuffer,
    rows: usize,
    columns: usize,
}

fn upload_bytes(ctx: &CudaContext, bytes: &[u8]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), "rotated_activation:owned")?;
    unsafe {
        ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
    }
    Ok(buffer)
}

fn get_payload_pointer(buffer: &DeviceBuffer) -> *mut c_void {
    unsafe { buffer.as_mut_ptr().cast::<u8>().add(GUARD_BYTES).cast() }
}

impl DeviceRotatedActivation {
    /// Upload immutable signs and allocate equal guarded storage for both
    /// schedules. Shape/sign validation precedes every allocation.
    pub fn upload(ctx: &CudaContext, rows: usize, signs: &[i8]) -> Result<Self> {
        let columns = signs.len();
        let count = rows
            .checked_mul(columns)
            .context("rotation shape overflow")?;
        ensure!(
            matches!(rows, 1 | 2 | 4 | 8)
                && columns > 0
                && columns.is_multiple_of(128)
                && count <= (i32::MAX - 255) as usize,
            "rotation requires rows 1/2/4/8 and bounded complete Walsh128 blocks"
        );
        ensure!(
            signs.iter().all(|s| matches!(s, -1 | 1)),
            "invalid rotation signs"
        );
        let bytes = count
            .checked_mul(4)
            .context("rotation allocation overflow")?;
        let encoded_bytes = count / 32 * 36;
        let sign_bytes: Vec<_> = signs.iter().map(|&s| s as u8).collect();
        Ok(Self {
            ctx: ctx.clone(),
            signs: upload_bytes(ctx, &sign_bytes)?,
            signed_input: upload_bytes(ctx, &vec![GUARD; bytes + 2 * GUARD_BYTES])?,
            rotated: upload_bytes(ctx, &vec![GUARD; bytes + 2 * GUARD_BYTES])?,
            quantized: upload_bytes(ctx, &vec![GUARD; encoded_bytes + 2 * GUARD_BYTES])?,
            rows,
            columns,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn workspace_bytes(&self) -> usize {
        self.signed_input.bytes() + self.rotated.bytes() + self.quantized.bytes()
    }

    pub fn get_sign_buffer(&self) -> &DeviceBuffer {
        &self.signs
    }

    pub fn get_rotated_pointer(&self) -> *const c_void {
        get_payload_pointer(&self.rotated).cast_const()
    }

    pub fn get_quantized_pointer(&self) -> *const c_void {
        get_payload_pointer(&self.quantized).cast_const()
    }

    /// Synchronize both streams and read output plus all scratch canaries.
    /// This diagnostic is outside the asynchronous producer's timing boundary.
    pub fn read_preparation(&self) -> Result<(Vec<f32>, Vec<u8>)> {
        self.ctx.synchronize_stream(CudaStream::Prefill)?;
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        let mut payloads = Vec::new();
        for buffer in [&self.signed_input, &self.rotated, &self.quantized] {
            let mut bytes = vec![0; buffer.bytes()];
            unsafe {
                self.ctx
                    .memcpy_d2h(bytes.as_mut_ptr().cast(), buffer.as_ptr(), bytes.len())?;
            }
            ensure!(
                bytes[..GUARD_BYTES]
                    .iter()
                    .chain(&bytes[bytes.len() - GUARD_BYTES..])
                    .all(|&b| b == GUARD),
                "rotation scratch guard changed"
            );
            payloads.push(bytes[GUARD_BYTES..bytes.len() - GUARD_BYTES].to_vec());
        }
        let quantized = payloads.pop().expect("three guarded buffers");
        let rotated = payloads[1]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
            .collect();
        Ok((rotated, quantized))
    }

    /// Recompute canonical FP32 rotation and Q8_1 encoding on the chosen stream.
    ///
    /// # Safety
    /// Input holds rows*columns aligned FP32 values on this context and survives
    /// stream completion. Values must produce finite Walsh sums and nonzero
    /// non-underflowing Q8_1 scales except for all-zero blocks. Order producers
    /// and consumers explicitly, serialize this owner's use across streams,
    /// and retain the owner through completion even when enqueue returns an error.
    pub unsafe fn enqueue_async(
        &self,
        input: *const c_void,
        mode: RotatedPreparation,
        stream: CudaStream,
    ) -> Result<()> {
        let begin = input as usize;
        ensure!(
            begin != 0 && begin.is_multiple_of(4),
            "invalid rotation input pointer"
        );
        let end = begin
            .checked_add(self.rows * self.columns * 4)
            .context("rotation input span overflow")?;
        for buffer in [
            &self.signs,
            &self.signed_input,
            &self.rotated,
            &self.quantized,
        ] {
            let owned_begin = buffer.as_ptr() as usize;
            let owned_end = owned_begin + buffer.bytes();
            ensure!(
                end <= owned_begin || owned_end <= begin,
                "rotation input aliases owned storage"
            );
        }
        let _guard = self.ctx.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_prepare_rotated_q8_1_async(
            self.ctx.inner.raw.as_ptr(),
            input,
            self.signs.as_ptr(),
            get_payload_pointer(&self.signed_input),
            get_payload_pointer(&self.rotated),
            get_payload_pointer(&self.quantized),
            self.rows as i32,
            self.columns as i32,
            i32::from(mode == RotatedPreparation::Fused),
            stream.ffi_kind(),
        );
        ensure!(rc == 0, "m40llm_prepare_rotated_q8_1_async failed: {rc}");
        match mode {
            RotatedPreparation::Staged => {
                record_async_kernel("rotated_activation_signs");
                record_async_kernel("rotated_activation_walsh128");
                record_async_kernel("quantize_q8_1");
            }
            RotatedPreparation::Fused => record_async_kernel("rotated_activation_fused"),
        }
        Ok(())
    }
}
