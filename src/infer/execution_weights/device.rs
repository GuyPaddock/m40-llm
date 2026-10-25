use anyhow::{Context, Result};

use super::{
    prepare_buffers, HostQ8ExecutionWeight, Q8AllocationPlan, Q8ExecutionKey, Q8ExecutionLayout,
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
