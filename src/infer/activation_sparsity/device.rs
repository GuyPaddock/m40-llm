use anyhow::{ensure, Context, Result};

use super::{ActivationThreshold, SparseInputSelection};
use crate::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    infer::execution_weights::{DeviceQ8ExecutionView, DeviceQ8ExecutionWeight, Q8TiledLaunch},
};

/// Session/probe-owned single-row scratch. Selection and consumption enqueue
/// on decode; the owner must survive stream completion, including on errors.
#[derive(Debug)]
pub struct DeviceSparseInputWorkspace {
    ctx: CudaContext,
    input_dim: usize,
    indices: DeviceBuffer,
    values: DeviceBuffer,
    count: DeviceBuffer,
    has_selection: bool,
}

impl DeviceSparseInputWorkspace {
    pub fn new(ctx: &CudaContext, input_dim: usize) -> Result<Self> {
        ensure!(
            input_dim > 0 && input_dim <= i32::MAX as usize - 255,
            "sparse input dimension is outside the supported indexing range"
        );
        let bytes = input_dim
            .checked_mul(4)
            .context("sparse input byte overflow")?;
        let indices = DeviceBuffer::new_tagged(ctx, bytes, "sparse_input:indices")?;
        let values = DeviceBuffer::new_tagged(ctx, bytes, "sparse_input:values")?;
        let count = DeviceBuffer::new_tagged(ctx, 4, "sparse_input:count")?;
        Ok(Self {
            ctx: ctx.clone(),
            input_dim,
            indices,
            values,
            count,
            has_selection: false,
        })
    }

    pub fn bytes(&self) -> usize {
        self.indices.bytes() + self.values.bytes() + self.count.bytes()
    }

    /// # Safety
    /// Input must belong to this context and hold input_dim FP32 values. Order
    /// other-stream producers before decode; retain all buffers through drain.
    pub unsafe fn select_async(
        &mut self,
        input: &DeviceBuffer,
        threshold: ActivationThreshold,
    ) -> Result<()> {
        ensure!(
            input.bytes() >= self.input_dim * 4,
            "sparse input buffer too small"
        );
        self.ctx.select_sparse_inputs_f32_async(
            input.as_ptr(),
            self.indices.as_mut_ptr(),
            self.values.as_mut_ptr(),
            self.count.as_mut_ptr(),
            self.input_dim as i32,
            threshold.magnitude(),
        )?;
        self.has_selection = true;
        Ok(())
    }

    /// # Safety
    /// Weight/output must belong to this context. Keep them and this workspace
    /// alive through decode completion. Enqueued selection must not be modified
    /// by another stream; calls on the same decode stream are ordered normally.
    pub unsafe fn project_tiled_async(
        &self,
        weight: &DeviceQ8ExecutionWeight,
        output: &DeviceBuffer,
        config: Q8TiledLaunch,
    ) -> Result<()> {
        ensure!(self.has_selection, "select sparse inputs before projection");
        ensure!(
            matches!(config.columns(), 8 | 16),
            "sparse input projection supports tile widths 8 or 16"
        );
        let DeviceQ8ExecutionView::InputTiled { key, tiles } = weight.view() else {
            anyhow::bail!("sparse input projection requires input-tiled Q8 weights")
        };
        ensure!(
            key.layout
                == crate::infer::execution_weights::Q8ExecutionLayout::InputTiled {
                    columns: config.columns() as usize
                },
            "sparse input launch/layout mismatch"
        );
        let shape = key.source.shape();
        ensure!(
            shape.input_dim() == self.input_dim,
            "sparse input dimension mismatch"
        );
        let n = i32::try_from(shape.output_dim())?;
        ensure!(
            n <= i32::MAX - 15,
            "sparse output dimension exceeds indexing"
        );
        ensure!(
            output.bytes()
                >= shape
                    .output_dim()
                    .checked_mul(4)
                    .context("sparse output byte overflow")?,
            "sparse projection output buffer too small"
        );
        self.ctx.project_sparse_inputs_q8_async(
            self.indices.as_ptr(),
            self.values.as_ptr(),
            self.count.as_ptr(),
            tiles.as_ptr(),
            output.as_mut_ptr(),
            n,
            self.input_dim as i32,
            config.columns() as i32,
            config.threads() as i32,
        )
    }

    /// Diagnostic readback only, outside warm-path timing. Completes decode
    /// before reading count/payload; rejects malformed indices instead of
    /// reconstructing a misleading dense reference.
    pub fn read_selection_sync(&self) -> Result<SparseInputSelection> {
        ensure!(self.has_selection, "select sparse inputs before readback");
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        let mut count = 0u32;
        unsafe {
            self.ctx
                .memcpy_d2h((&mut count as *mut u32).cast(), self.count.as_ptr(), 4)?;
        }
        ensure!(
            count as usize <= self.input_dim,
            "invalid sparse retained count"
        );
        let mut indices = vec![0u32; count as usize];
        let mut values = vec![0.0f32; count as usize];
        if count != 0 {
            unsafe {
                self.ctx.memcpy_d2h(
                    indices.as_mut_ptr().cast(),
                    self.indices.as_ptr(),
                    indices.len() * 4,
                )?;
                self.ctx.memcpy_d2h(
                    values.as_mut_ptr().cast(),
                    self.values.as_ptr(),
                    values.len() * 4,
                )?;
            }
        }
        ensure!(
            indices
                .iter()
                .all(|&index| (index as usize) < self.input_dim)
                && indices.windows(2).all(|pair| pair[0] < pair[1]),
            "sparse indices must be in range and strictly increasing"
        );
        Ok(SparseInputSelection {
            input_dim: self.input_dim,
            indices,
            values,
        })
    }
}
