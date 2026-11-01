use super::{ffi, record_async_kernel, CudaContext, CudaStream, DeviceBuffer};
use crate::infer::q4_channels::Q4ChannelWeights;
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

/// Change only down-weight addressing; both paths use the same reduction tree.
#[derive(Debug, Clone, Copy)]
pub enum Q4DownLayout {
    Native,
    ChannelMajor,
}

/// Isolated research owner containing both weight layouts and reusable scratch.
/// Selection is immutable and may differ across rows. No runtime dispatch uses it.
pub struct DeviceQ4SelectedFfn {
    ctx: CudaContext,
    weights: [DeviceBuffer; 5],
    indices: DeviceBuffer,
    hidden: DeviceBuffer,
    partial: DeviceBuffer,
    rows: usize,
    k: usize,
    h: usize,
    count: usize,
}

fn upload_bytes(ctx: &CudaContext, bytes: &[u8]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), "q4_selected:immutable")?;
    unsafe {
        ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
    }
    Ok(buffer)
}

fn validate_span(pointer: *const c_void, bytes: usize) -> Result<(usize, usize)> {
    let begin = pointer as usize;
    ensure!(
        begin != 0 && begin.is_multiple_of(4),
        "Q4 null/unaligned activation"
    );
    Ok((
        begin,
        begin.checked_add(bytes).context("Q4 pointer overflow")?,
    ))
}

fn validate_disjoint_spans(a: (usize, usize), b: (usize, usize)) -> Result<()> {
    ensure!(
        a.1 <= b.0 || b.1 <= a.0,
        "Q4 activation overlaps another span"
    );
    Ok(())
}

impl DeviceQ4SelectedFfn {
    /// Validate all dimensions, byte counts and per-row sorted unique indices
    /// before allocating. Each successful allocation immediately has RAII ownership.
    pub fn upload(
        ctx: &CudaContext,
        gate: &[u8],
        up: &[u8],
        down: &Q4ChannelWeights,
        indices: &[u32],
        rows: usize,
    ) -> Result<Self> {
        let k = down.output_dim();
        let h = down.input_dim();
        ensure!(matches!(rows, 1 | 2 | 4 | 8), "Q4 rows must be 1/2/4/8");
        ensure!(
            k.is_multiple_of(32) && k <= (i32::MAX - 255) as usize && h <= i32::MAX as usize,
            "Q4 unsupported dimensions"
        );
        ensure!(
            indices.len().is_multiple_of(rows),
            "Q4 unequal row selection sizes"
        );
        let count = indices.len() / rows;
        ensure!(count > 0 && count <= h, "Q4 selected count out of range");
        for row in indices.chunks_exact(count) {
            ensure!(
                row.iter().all(|&i| (i as usize) < h)
                    && row.windows(2).all(|pair| pair[0] < pair[1]),
                "Q4 indices must be sorted, unique and in bounds"
            );
        }
        let bytes = (k / 32)
            .checked_mul(h)
            .and_then(|v| v.checked_mul(18))
            .context("Q4 gate/up size overflow")?;
        ensure!(
            gate.len() == bytes && up.len() == bytes,
            "Q4 gate/up byte mismatch"
        );
        let hidden_bytes = rows
            .checked_mul(count)
            .and_then(|v| v.checked_mul(4))
            .context("Q4 hidden size overflow")?;
        let partial_bytes = rows
            .checked_mul(k)
            .and_then(|v| v.checked_mul(32 * 4))
            .context("Q4 partial size overflow")?;
        let native = down.unpack();
        let [scales, codes] = down.planes();
        let index_bytes: Vec<_> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        let weights = [
            upload_bytes(ctx, gate)?,
            upload_bytes(ctx, up)?,
            upload_bytes(ctx, &native)?,
            upload_bytes(ctx, scales)?,
            upload_bytes(ctx, codes)?,
        ];
        let indices = upload_bytes(ctx, &index_bytes)?;
        let hidden = DeviceBuffer::new_tagged(ctx, hidden_bytes, "q4_selected:hidden")?;
        let partial = DeviceBuffer::new_tagged(ctx, partial_bytes, "q4_selected:partial")?;
        Ok(Self {
            ctx: ctx.clone(),
            weights,
            indices,
            hidden,
            partial,
            rows,
            k,
            h,
            count,
        })
    }

    /// Weight and selection owners for immutable readback diagnostics.
    pub fn immutable_buffers(&self) -> [&DeviceBuffer; 6] {
        [
            &self.weights[0],
            &self.weights[1],
            &self.weights[2],
            &self.weights[3],
            &self.weights[4],
            &self.indices,
        ]
    }

    pub fn workspace_bytes(&self) -> usize {
        self.hidden.bytes() + self.partial.bytes()
    }

    /// Execute selected gate/up, SwiGLU, down partials and final reduction.
    ///
    /// # Safety
    /// Input and output must be live disjoint rows*K FP32 device spans on this
    /// context. Order producers on stream, retain owners through completion,
    /// and serialize uses of this owner's workspace even across different streams.
    pub unsafe fn enqueue_async(
        &self,
        input: *const c_void,
        output: *mut c_void,
        layout: Q4DownLayout,
        stream: CudaStream,
    ) -> Result<()> {
        let activation_bytes = self.rows * self.k * 4;
        let input_span = validate_span(input, activation_bytes)?;
        let output_span = validate_span(output.cast_const(), activation_bytes)?;
        validate_disjoint_spans(input_span, output_span)?;
        for buffer in self
            .immutable_buffers()
            .into_iter()
            .chain([&self.hidden, &self.partial])
        {
            let begin = buffer.as_ptr() as usize;
            let internal = (begin, begin + buffer.bytes());
            validate_disjoint_spans(input_span, internal)?;
            validate_disjoint_spans(output_span, internal)?;
        }
        let _guard = self.ctx.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_q4_selected_ffn_async(
            self.ctx.inner.raw.as_ptr(),
            input,
            self.weights[0].as_ptr(),
            self.weights[1].as_ptr(),
            self.weights[2].as_ptr(),
            self.weights[3].as_ptr(),
            self.weights[4].as_ptr(),
            self.indices.as_ptr(),
            self.hidden.as_mut_ptr(),
            self.partial.as_mut_ptr(),
            output,
            self.rows as i32,
            self.k as i32,
            self.h as i32,
            self.count as i32,
            i32::from(matches!(layout, Q4DownLayout::ChannelMajor)),
            stream.ffi_kind(),
        );
        ensure!(rc == 0, "m40llm_q4_selected_ffn_async failed: {rc}");
        record_async_kernel("q4_selected_gate_up");
        record_async_kernel("q4_selected_down_partial");
        record_async_kernel("q4_selected_down_reduce");
        Ok(())
    }
}
