use anyhow::{ensure, Context, Result};
use serde::Serialize;

use super::{HostSparseQ8Weight, SparseQ8Storage};
use crate::{
    cuda::{CudaContext, DeviceBuffer},
    infer::execution_weights::Q8SourceIdentity,
};

/// Bounded experimental geometry, not a model/runtime dispatch policy.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SparseQ8Launch {
    columns: u32,
    threads: u32,
}

impl SparseQ8Launch {
    pub fn new(columns: u32, threads: u32) -> Result<Self> {
        ensure!(matches!(columns, 1 | 4), "sparse Q8 columns must be 1 or 4");
        ensure!(
            matches!(threads, 128 | 256),
            "sparse Q8 threads must be 128 or 256"
        );
        Ok(Self { columns, threads })
    }
    pub fn columns(self) -> u32 {
        self.columns
    }
    pub fn threads(self) -> u32 {
        self.threads
    }
}

#[derive(Debug)]
pub struct DeviceSparseQ8Weight {
    ctx: CudaContext,
    source: Q8SourceIdentity,
    storage: SparseQ8Storage,
    blocks: DeviceBuffer,
}

impl DeviceSparseQ8Weight {
    pub fn upload(ctx: &CudaContext, host: &HostSparseQ8Weight) -> Result<Self> {
        let shape = host.source().shape();
        ensure!(
            shape.input_dim() <= (i32::MAX - 31) as usize
                && shape.output_dim() <= i32::MAX as usize,
            "sparse Q8 shape exceeds kernel indexing"
        );
        let blocks = DeviceBuffer::new_tagged(ctx, host.blocks().len(), "sparse_q8:blocks")?;
        unsafe {
            ctx.memcpy_h2d(
                blocks.as_mut_ptr(),
                host.blocks().as_ptr().cast(),
                blocks.bytes(),
            )?;
        }
        Ok(Self {
            ctx: ctx.clone(),
            source: host.source().clone(),
            storage: host.storage(),
            blocks,
        })
    }

    pub fn storage(&self) -> &SparseQ8Storage {
        &self.storage
    }

    /// Enqueue without allocation or synchronization on the decode stream.
    ///
    /// # Safety
    /// All buffers must belong to the owning context and remain alive until
    /// decode-stream completion. Input producers must finish before this call.
    pub unsafe fn project_async(
        &self,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        launch: SparseQ8Launch,
    ) -> Result<()> {
        ensure!(
            (1..=65535).contains(&rows),
            "sparse Q8 rows must be in 1..=65535"
        );
        ensure!(
            input.as_ptr() != output.as_ptr(),
            "sparse Q8 input/output must not alias"
        );
        let shape = self.source.shape();
        for (buffer, dimension) in [(input, shape.input_dim()), (output, shape.output_dim())] {
            let bytes = rows
                .checked_mul(dimension)
                .and_then(|v| v.checked_mul(4))
                .context("sparse Q8 projection span overflow")?;
            ensure!(
                buffer.bytes() >= bytes,
                "sparse Q8 projection buffer too small: need {bytes}"
            );
        }
        self.ctx.gemm_f32xq8_sparse_f32_async(
            input.as_ptr(),
            self.blocks.as_ptr(),
            output.as_mut_ptr(),
            rows as i32,
            shape.output_dim() as i32,
            shape.input_dim() as i32,
            launch.columns() as i32,
            launch.threads() as i32,
        )
    }
}
