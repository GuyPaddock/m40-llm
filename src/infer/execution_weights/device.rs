use anyhow::{ensure, Context, Result};

use super::{
    prepare_buffers, HostQ8ExecutionWeight, Q8AllocationPlan, Q8ExecutionKey, Q8ExecutionLayout,
    Q8SeparateLaunch, Q8TiledAccess, Q8TiledLaunch,
};
use crate::cuda::{CudaContext, DeviceBuffer};

#[derive(Debug)]
pub struct DeviceQ8ExecutionWeight {
    ctx: CudaContext,
    key: Q8ExecutionKey,
    plan: Q8AllocationPlan,
    buffers: Vec<DeviceBuffer>,
}

/// A borrowed view cannot silently reinterpret separate/tiled data as GGUF.
#[derive(Debug)]
pub enum DeviceQ8ExecutionView<'a> {
    ColumnBlocks {
        key: &'a Q8ExecutionKey,
        blocks: &'a DeviceBuffer,
    },
    SeparateScalesValues {
        key: &'a Q8ExecutionKey,
        scales: &'a DeviceBuffer,
        values: &'a DeviceBuffer,
    },
    InputTiled {
        key: &'a Q8ExecutionKey,
        tiles: &'a DeviceBuffer,
    },
}

impl DeviceQ8ExecutionWeight {
    /// Synchronous setup only. Each buffer is owned immediately, so a later
    /// allocation or H2D failure releases all previously created planes.
    pub fn upload(ctx: &CudaContext, host: &HostQ8ExecutionWeight) -> Result<Self> {
        let buffers = prepare_buffers(
            host,
            |spec| {
                DeviceBuffer::new_tagged(
                    ctx,
                    spec.bytes,
                    &format!(
                        "q8_execution:{}:{:?}",
                        host.key().source.tensor_name,
                        spec.kind
                    ),
                )
            },
            |buffer, bytes| unsafe {
                ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())
            },
        )?;
        Ok(Self {
            ctx: ctx.clone(),
            key: host.key().clone(),
            plan: host.plan().clone(),
            buffers,
        })
    }

    pub fn key(&self) -> &Q8ExecutionKey {
        &self.key
    }
    pub fn plan(&self) -> &Q8AllocationPlan {
        &self.plan
    }
    pub fn view(&self) -> DeviceQ8ExecutionView<'_> {
        match self.key.layout {
            Q8ExecutionLayout::ColumnBlocks => DeviceQ8ExecutionView::ColumnBlocks {
                key: &self.key,
                blocks: &self.buffers[0],
            },
            Q8ExecutionLayout::SeparateScalesValues => {
                DeviceQ8ExecutionView::SeparateScalesValues {
                    key: &self.key,
                    scales: &self.buffers[0],
                    values: &self.buffers[1],
                }
            }
            Q8ExecutionLayout::InputTiled { .. } => DeviceQ8ExecutionView::InputTiled {
                key: &self.key,
                tiles: &self.buffers[0],
            },
        }
    }

    /// Enqueue this experiment on the owning context's decode stream.
    ///
    /// # Safety
    /// Input/output must belong to the owning context. Complete their producers
    /// before calling, and retain all buffers/this weight until stream drain.
    pub unsafe fn project_separate_async(
        &self,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        config: Q8SeparateLaunch,
    ) -> Result<()> {
        let DeviceQ8ExecutionView::SeparateScalesValues { scales, values, .. } = self.view() else {
            anyhow::bail!("separate Q8 projection requires SeparateScalesValues weights");
        };
        let (m, n, k) = self.validate_projection_buffers(input, output, rows)?;
        self.ctx.gemm_f32xq8_separate_f32_async(
            input.as_ptr(),
            scales.as_ptr(),
            values.as_ptr(),
            output.as_mut_ptr(),
            m,
            n,
            k,
            config.columns() as i32,
            config.threads() as i32,
        )
    }

    /// # Safety
    /// Same context, producer completion, and lifetime rules as separate planes.
    /// Enqueues on decode; the supplied tile width must match the stored layout.
    pub unsafe fn project_tiled_async(
        &self,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        config: Q8TiledLaunch,
    ) -> Result<()> {
        let DeviceQ8ExecutionView::InputTiled { key, tiles } = self.view() else {
            anyhow::bail!("tiled Q8 projection requires InputTiled weights");
        };
        ensure!(
            key.layout
                == Q8ExecutionLayout::InputTiled {
                    columns: config.columns() as usize
                },
            "Q8 launch tile width must match execution layout"
        );
        let (m, n, k) = self.validate_projection_buffers(input, output, rows)?;
        self.ctx.gemm_f32xq8_tiled_f32_async(
            input.as_ptr(),
            tiles.as_ptr(),
            output.as_mut_ptr(),
            m,
            n,
            k,
            config.columns() as i32,
            config.threads() as i32,
            match config.access() {
                Q8TiledAccess::Vector => 0,
                Q8TiledAccess::CoalescedLanes => 1,
            },
        )
    }

    fn validate_projection_buffers(
        &self,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
    ) -> Result<(i32, i32, i32)> {
        ensure!(
            (1..=65535).contains(&rows),
            "Q8 projection rows must be in 1..=65535"
        );
        let shape = self.key.source.shape();
        let k = i32::try_from(shape.input_dim())?;
        let n = i32::try_from(shape.output_dim())?;
        ensure!(
            k <= i32::MAX - 31,
            "Q8 input dimension exceeds kernel indexing"
        );
        for (buffer, dimension) in [(input, shape.input_dim()), (output, shape.output_dim())] {
            let bytes = rows
                .checked_mul(dimension)
                .and_then(|v| v.checked_mul(4))
                .context("Q8 projection buffer size overflow")?;
            ensure!(
                buffer.bytes() >= bytes,
                "Q8 projection buffer too small: need {bytes}"
            );
        }
        ensure!(
            input.as_ptr() != output.as_ptr(),
            "Q8 projection input/output must not alias"
        );
        Ok((rows as i32, n, k))
    }

    /// Diagnostic/setup readback using the owning context, not a hot projection
    /// operation.
    ///
    /// # Safety
    /// Any asynchronous consumer must complete before readback.
    pub unsafe fn read_planes(&self) -> Result<Vec<Vec<u8>>> {
        self.buffers
            .iter()
            .map(|buffer| {
                let mut bytes = vec![0; buffer.bytes()];
                self.ctx
                    .memcpy_d2h(bytes.as_mut_ptr().cast(), buffer.as_ptr(), bytes.len())
                    .context("read Q8 execution plane")?;
                Ok(bytes)
            })
            .collect()
    }
}

#[cfg(all(test, nvcc))]
mod tests {
    use super::*;
    use crate::{
        infer::execution_weights::{Q8MatrixShape, Q8SourceIdentity},
        profile,
    };

    /// Real second-plane allocation failure must release the first plane after
    /// its successful upload; no host-sized OOM fixture or global fault flag.
    #[test]
    fn test_partial_device_allocation_failure_frees_uploaded_plane() -> Result<()> {
        let ctx = CudaContext::new(-1)?;
        let props = ctx.current_device_props()?;
        anyhow::ensure!(
            props.major == 5 && props.minor == 2,
            "Q8 ownership test requires sm_52"
        );
        let source = Q8SourceIdentity::from_column_blocks(
            [1; 32],
            "failure_fixture".into(),
            0,
            Q8MatrixShape::new(32, 2)?,
        )?;
        let host = HostQ8ExecutionWeight::new(
            Q8ExecutionKey {
                source,
                layout: Q8ExecutionLayout::SeparateScalesValues,
            },
            vec![vec![3; 4], vec![4; 64]],
        )?;
        let initial_bytes = CudaContext::total_device_bytes();
        let before = profile::snapshot();
        let mut allocations = 0;
        let result = prepare_buffers(
            &host,
            |spec| {
                allocations += 1;
                DeviceBuffer::new_tagged(
                    &ctx,
                    if allocations == 2 {
                        usize::MAX
                    } else {
                        spec.bytes
                    },
                    "q8_failure_test",
                )
            },
            |buffer, data| unsafe {
                ctx.memcpy_h2d(buffer.as_mut_ptr(), data.as_ptr().cast(), data.len())
            },
        );
        assert!(result.is_err());
        assert_eq!(allocations, 2);
        assert_eq!(CudaContext::total_device_bytes(), initial_bytes);
        let delta = profile::counter_delta(&before, &profile::snapshot());
        assert_eq!(delta.device_allocs, 1);
        assert_eq!(delta.device_frees, 1);
        assert_eq!(delta.h2d_copies, 1);
        Ok(())
    }
}
