use super::{ffi, record_async_kernel, CudaContext, CudaStream, DeviceBuffer};
use crate::evaluation::ternary_screen::TernarySketch;
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

const GUARD_BYTES: usize = 32;
const GUARD: u8 = 0xa5;

/// Fixed batch-one screening probe. Owns immutable encoded weights and reusable
/// guarded rotation/selection scratch. No runtime dispatch uses this owner.
pub struct DeviceTernaryScreen {
    ctx: CudaContext,
    immutable: [DeviceBuffer; 6],
    rotated: DeviceBuffer,
    coordinates: DeviceBuffer,
    k: usize,
    h: usize,
}

fn upload_bytes(ctx: &CudaContext, bytes: &[u8]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), "ternary_screen:owned")?;
    unsafe {
        ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
    }
    Ok(buffer)
}

fn validate_span(pointer: *const c_void, bytes: usize) -> Result<(usize, usize)> {
    let begin = pointer as usize;
    ensure!(
        begin != 0 && begin.is_multiple_of(4),
        "invalid screening pointer"
    );
    Ok((
        begin,
        begin
            .checked_add(bytes)
            .context("screening pointer overflow")?,
    ))
}

fn validate_disjoint(a: (usize, usize), b: (usize, usize)) -> Result<()> {
    ensure!(a.1 <= b.0 || b.1 <= a.0, "overlapping screening spans");
    Ok(())
}

fn payload_pointer(buffer: &DeviceBuffer) -> *mut c_void {
    // Both guarded workspace allocations include this prefix and a suffix.
    unsafe { buffer.as_mut_ptr().cast::<u8>().add(GUARD_BYTES).cast() }
}

impl DeviceTernaryScreen {
    /// Validate shapes, matching rotations and finite nonnegative down-column
    /// norms before allocating. Uploads complete before returning ownership.
    pub fn upload(
        ctx: &CudaContext,
        gate: &TernarySketch,
        up: &TernarySketch,
        norms: &[f32],
    ) -> Result<Self> {
        let k = gate.input_dim();
        let h = gate.channels();
        ensure!(
            k <= i32::MAX as usize
                && h <= (i32::MAX - 31) as usize
                && up.input_dim() == k
                && up.channels() == h
                && up.signs() == gate.signs(),
            "incompatible screening shapes or signs"
        );
        ensure!(
            norms.len() == h && norms.iter().all(|&n| n.is_finite() && n >= 0.0),
            "invalid screening down-column norms"
        );
        let rotated_bytes = k
            .checked_mul(4)
            .and_then(|v| v.checked_add(2 * GUARD_BYTES))
            .context("rotation size overflow")?;
        let coordinate_bytes = (k / 128)
            .checked_mul(80)
            .and_then(|v| v.checked_add(2 * GUARD_BYTES))
            .context("selection size overflow")?;
        h.checked_mul(16)
            .context("screening output size overflow")?;
        let signs: Vec<_> = gate.signs().iter().map(|&s| s as u8).collect();
        let norms: Vec<_> = norms.iter().flat_map(|n| n.to_le_bytes()).collect();
        let [gate_scales, gate_codes] = gate.planes();
        let [up_scales, up_codes] = up.planes();
        let immutable = [
            upload_bytes(ctx, &signs)?,
            upload_bytes(ctx, gate_scales)?,
            upload_bytes(ctx, gate_codes)?,
            upload_bytes(ctx, up_scales)?,
            upload_bytes(ctx, up_codes)?,
            upload_bytes(ctx, &norms)?,
        ];
        let rotated = upload_bytes(ctx, &vec![GUARD; rotated_bytes])?;
        let coordinates = upload_bytes(ctx, &vec![GUARD; coordinate_bytes])?;
        Ok(Self {
            ctx: ctx.clone(),
            immutable,
            rotated,
            coordinates,
            k,
            h,
        })
    }

    pub fn immutable_buffers(&self) -> [&DeviceBuffer; 6] {
        self.immutable.each_ref()
    }

    pub fn workspace_bytes(&self) -> usize {
        self.rotated.bytes() + self.coordinates.bytes()
    }

    /// Read scratch and validate its canaries after the caller has synchronized
    /// the producer stream. This diagnostic is outside every timing boundary.
    pub fn read_preparation(&self) -> Result<(Vec<f32>, Vec<u8>)> {
        let mut payloads = Vec::new();
        for buffer in [&self.rotated, &self.coordinates] {
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
                "screening scratch guard changed"
            );
            payloads.push(bytes[GUARD_BYTES..bytes.len() - GUARD_BYTES].to_vec());
        }
        let coordinates = payloads.pop().expect("two workspace buffers");
        let rotated = payloads[0]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
            .collect();
        Ok((rotated, coordinates))
    }

    /// Write four H-element FP32 planes: gate, up, estimated SwiGLU and score.
    /// The boundary includes rotation and per-block top-80 coordinate selection.
    ///
    /// # Safety
    /// Input must contain K finite FP32 values; output must have 4*H writable
    /// FP32 elements. Both must be live on this context and disjoint. Order
    /// producers and consumers on stream, retain owners until completion, and
    /// serialize workspace use, including diagnostics, across all streams.
    pub unsafe fn enqueue_async(
        &self,
        input: *const c_void,
        output: *mut c_void,
        stream: CudaStream,
    ) -> Result<()> {
        let input_span = validate_span(input, self.k * 4)?;
        let output_span = validate_span(output.cast_const(), self.h * 16)?;
        validate_disjoint(input_span, output_span)?;
        for buffer in self
            .immutable
            .iter()
            .chain([&self.rotated, &self.coordinates])
        {
            let begin = buffer.as_ptr() as usize;
            let span = (begin, begin + buffer.bytes());
            validate_disjoint(input_span, span)?;
            validate_disjoint(output_span, span)?;
        }
        let _guard = self.ctx.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_ternary_screen_async(
            self.ctx.inner.raw.as_ptr(),
            input,
            self.immutable[0].as_ptr(),
            self.immutable[1].as_ptr(),
            self.immutable[2].as_ptr(),
            self.immutable[3].as_ptr(),
            self.immutable[4].as_ptr(),
            self.immutable[5].as_ptr(),
            payload_pointer(&self.rotated),
            payload_pointer(&self.coordinates),
            output,
            self.k as i32,
            self.h as i32,
            stream.ffi_kind(),
        );
        ensure!(rc == 0, "m40llm_ternary_screen_async failed: {rc}");
        record_async_kernel("ternary_rotate_select");
        record_async_kernel("ternary_score_channels");
        Ok(())
    }
}
