//! Lossless Q8 execution representations with typed ownership, separate from
//! GGUF tensor views and FP32 materialization. Runtime preparation is opt-in.

use anyhow::{ensure, Context, Result};
use serde::Serialize;

use crate::gguf::{GgmlDType, GgufTensor};

mod runtime;
pub use runtime::{
    Q8ExecutionConfig, Q8ExecutionFamilies, Q8ExecutionMode, Q8ExecutionStats,
    Q8PreparedTensorStats, Q8ProjectionGroup,
};
#[cfg(feature = "cuda")]
mod cache;
#[cfg(feature = "cuda")]
pub use cache::Q8ExecutionCache;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub(crate) enum Q8ProjectionLaunch {
    Separate(Q8SeparateLaunch),
    Tiled(Q8TiledLaunch),
}

#[cfg(feature = "cuda")]
mod device;
#[cfg(feature = "cuda")]
pub use device::{DeviceQ8ExecutionView, DeviceQ8ExecutionWeight};
#[cfg(test)]
mod tests;

pub const Q8_BLOCK_ELEMENTS: usize = 32;
pub const Q8_BLOCK_BYTES: usize = 34;

/// Logical GGUF `[K,N]` shape with separately padded column-block tails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct Q8MatrixShape {
    input_dim: usize,
    output_dim: usize,
}

impl Q8MatrixShape {
    pub fn new(input_dim: usize, output_dim: usize) -> Result<Self> {
        ensure!(
            input_dim > 0 && output_dim > 0,
            "Q8 dimensions must be nonzero"
        );
        input_dim
            .checked_mul(output_dim)
            .context("Q8 element count overflow")?;
        let shape = Self {
            input_dim,
            output_dim,
        };
        shape.source_bytes()?;
        Ok(shape)
    }

    pub fn input_dim(self) -> usize {
        self.input_dim
    }
    pub fn output_dim(self) -> usize {
        self.output_dim
    }
    pub fn blocks_per_column(self) -> usize {
        self.input_dim.div_ceil(Q8_BLOCK_ELEMENTS)
    }

    pub fn block_count(self) -> Result<usize> {
        self.output_dim
            .checked_mul(self.blocks_per_column())
            .context("Q8 block count overflow")
    }

    pub fn source_bytes(self) -> Result<usize> {
        self.block_count()?
            .checked_mul(Q8_BLOCK_BYTES)
            .context("Q8 source bytes overflow")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Q8SourceIdentity {
    model_sha256: [u8; 32],
    tensor_name: String,
    byte_offset: u64,
    dtype: GgmlDType,
    shape: Q8MatrixShape,
}

impl Q8SourceIdentity {
    pub fn model_sha256(&self) -> &[u8; 32] {
        &self.model_sha256
    }
    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
    }
    pub fn byte_offset(&self) -> u64 {
        self.byte_offset
    }
    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }
    pub fn shape(&self) -> Q8MatrixShape {
        self.shape
    }
    pub fn from_gguf_tensor(model_sha256: [u8; 32], tensor: &GgufTensor) -> Result<Self> {
        ensure!(
            tensor.dtype == GgmlDType::Q8_0,
            "Q8 execution requires a Q8_0 source tensor"
        );
        ensure!(
            tensor.shape.len() == 2,
            "Q8 projection requires a rank-two [K,N] tensor"
        );
        let shape = Q8MatrixShape::new(
            usize::try_from(tensor.shape[0])?,
            usize::try_from(tensor.shape[1])?,
        )?;
        ensure!(shape.input_dim().is_multiple_of(Q8_BLOCK_ELEMENTS),
            "GGUF Q8_0 rows require K divisible by 32; use explicit column blocks for synthetic tails");
        Self::from_column_blocks(model_sha256, tensor.name.clone(), tensor.offset, shape)
    }

    /// Synthetic ragged-K fixtures store all 32 bytes in each final column block.
    pub fn from_column_blocks(
        model_sha256: [u8; 32],
        tensor_name: String,
        byte_offset: u64,
        shape: Q8MatrixShape,
    ) -> Result<Self> {
        ensure!(
            !tensor_name.trim().is_empty(),
            "Q8 source tensor name must not be empty"
        );
        Ok(Self {
            model_sha256,
            tensor_name,
            byte_offset,
            dtype: GgmlDType::Q8_0,
            shape,
        })
    }
}

/// Borrows exactly one tensor from a tensor-data-relative weights blob.
#[derive(Debug, Clone, Copy)]
pub struct Q8SourceView<'a> {
    identity: &'a Q8SourceIdentity,
    blocks: &'a [u8],
}

impl<'a> Q8SourceView<'a> {
    /// An already sliced tensor keeps its original offset only as identity.
    pub fn from_tensor_bytes(identity: &'a Q8SourceIdentity, blocks: &'a [u8]) -> Result<Self> {
        ensure!(
            blocks.len() == identity.shape.source_bytes()?,
            "Q8 tensor byte length mismatch"
        );
        Ok(Self { identity, blocks })
    }

    pub fn new(identity: &'a Q8SourceIdentity, weights_blob: &'a [u8]) -> Result<Self> {
        ensure!(
            identity.dtype == GgmlDType::Q8_0,
            "Q8 source identity dtype mismatch"
        );
        let start = usize::try_from(identity.byte_offset).context("Q8 offset exceeds usize")?;
        let end = start
            .checked_add(identity.shape.source_bytes()?)
            .context("Q8 source end overflow")?;
        let blocks = weights_blob
            .get(start..end)
            .context("Q8 tensor exceeds weights blob")?;
        Ok(Self { identity, blocks })
    }
    pub fn identity(self) -> &'a Q8SourceIdentity {
        self.identity
    }
    pub fn blocks(self) -> &'a [u8] {
        self.blocks
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum Q8ExecutionLayout {
    ColumnBlocks,
    SeparateScalesValues,
    /// Input-block-major records; values are `[quant_index][output_column]`.
    InputTiled {
        columns: usize,
    },
}

/// Four fixed configurations for the independent separate-plane experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Q8SeparateLaunch {
    columns: u32,
    threads: u32,
}

impl Q8SeparateLaunch {
    pub fn new(columns: u32, threads: u32) -> Result<Self> {
        ensure!(
            matches!(columns, 1 | 4),
            "Q8 separate columns must be 1 or 4"
        );
        ensure!(
            matches!(threads, 128 | 256),
            "Q8 separate threads must be 128 or 256"
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

/// Six retained vector configurations; storage width is part of identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Q8TiledLaunch {
    columns: u32,
    threads: u32,
}

impl Q8TiledLaunch {
    pub fn new(columns: u32, threads: u32) -> Result<Self> {
        ensure!(
            matches!(columns, 4 | 8 | 16),
            "Q8 tiled columns must be 4, 8, or 16"
        );
        ensure!(
            matches!(threads, 128 | 256),
            "Q8 tiled threads must be 128 or 256"
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Q8ExecutionKey {
    pub source: Q8SourceIdentity,
    pub layout: Q8ExecutionLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Q8PlaneKind {
    ColumnBlocks,
    Scales,
    Values,
    InputTiles,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Q8PlaneSpec {
    pub kind: Q8PlaneKind,
    pub bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Q8InputTileGeometry {
    pub columns: usize,
    pub output_tiles: usize,
    pub values_offset: usize,
    pub record_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Q8AllocationPlan {
    planes: Vec<Q8PlaneSpec>,
    source_bytes: usize,
    resident_bytes: usize,
    tile: Option<Q8InputTileGeometry>,
}

impl Q8AllocationPlan {
    pub fn planes(&self) -> &[Q8PlaneSpec] {
        &self.planes
    }
    pub fn source_bytes(&self) -> usize {
        self.source_bytes
    }
    /// Requested device payload bytes, not CUDA allocator/page granularity.
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
    pub fn padding_bytes(&self) -> usize {
        self.resident_bytes - self.source_bytes
    }
    pub fn tile(&self) -> Option<Q8InputTileGeometry> {
        self.tile
    }
}

impl Q8ExecutionLayout {
    pub fn allocation_plan(self, shape: Q8MatrixShape) -> Result<Q8AllocationPlan> {
        let source_bytes = shape.source_bytes()?;
        let mut tile = None;
        let planes = match self {
            Self::ColumnBlocks => vec![Q8PlaneSpec {
                kind: Q8PlaneKind::ColumnBlocks,
                bytes: source_bytes,
            }],
            Self::SeparateScalesValues => vec![
                Q8PlaneSpec {
                    kind: Q8PlaneKind::Scales,
                    bytes: shape
                        .block_count()?
                        .checked_mul(2)
                        .context("Q8 scale bytes overflow")?,
                },
                Q8PlaneSpec {
                    kind: Q8PlaneKind::Values,
                    bytes: shape
                        .block_count()?
                        .checked_mul(32)
                        .context("Q8 value bytes overflow")?,
                },
            ],
            Self::InputTiled { columns } => {
                ensure!(
                    columns.is_power_of_two() && columns <= 32,
                    "Q8 tile columns must be a power of two in 1..=32"
                );
                let values_offset = (columns * 2).div_ceil(16) * 16;
                let record_bytes = values_offset + columns * Q8_BLOCK_ELEMENTS;
                let output_tiles = shape.output_dim().div_ceil(columns);
                let bytes = output_tiles
                    .checked_mul(shape.blocks_per_column())
                    .and_then(|records| records.checked_mul(record_bytes))
                    .context("Q8 tile bytes overflow")?;
                tile = Some(Q8InputTileGeometry {
                    columns,
                    output_tiles,
                    values_offset,
                    record_bytes,
                });
                vec![Q8PlaneSpec {
                    kind: Q8PlaneKind::InputTiles,
                    bytes,
                }]
            }
        };
        let resident_bytes = planes.iter().try_fold(0usize, |sum, plane| {
            sum.checked_add(plane.bytes)
                .context("Q8 execution bytes overflow")
        })?;
        Ok(Q8AllocationPlan {
            planes,
            source_bytes,
            resident_bytes,
            tile,
        })
    }
}

/// Owns explicitly prepared host planes. This constructor validates sizes, not
/// their encoding; C09/C10 packers are responsible for lossless value mappings.
#[derive(Debug)]
pub struct HostQ8ExecutionWeight {
    key: Q8ExecutionKey,
    plan: Q8AllocationPlan,
    planes: Vec<Vec<u8>>,
}

impl HostQ8ExecutionWeight {
    /// Losslessly transpose byte payloads into input-block/output-tile records.
    /// Padded columns and alignment bytes are zero; stored K tails are copied.
    pub fn tile_inputs(source: Q8SourceView<'_>, columns: usize) -> Result<Self> {
        let shape = source.identity().shape();
        let layout = Q8ExecutionLayout::InputTiled { columns };
        let plan = layout.allocation_plan(shape)?;
        let geometry = plan.tile().expect("input-tiled plan has geometry");
        let mut tiles = vec![0; plan.resident_bytes()];
        for qblock in 0..shape.blocks_per_column() {
            for tile in 0..geometry.output_tiles {
                let base = (qblock * geometry.output_tiles + tile) * geometry.record_bytes;
                for local_col in 0..columns.min(shape.output_dim() - tile * columns) {
                    let col = tile * columns + local_col;
                    let block = (col * shape.blocks_per_column() + qblock) * Q8_BLOCK_BYTES;
                    tiles[base + local_col * 2..base + local_col * 2 + 2]
                        .copy_from_slice(&source.blocks()[block..block + 2]);
                    for qindex in 0..Q8_BLOCK_ELEMENTS {
                        tiles[base + geometry.values_offset + qindex * columns + local_col] =
                            source.blocks()[block + 2 + qindex];
                    }
                }
            }
        }
        Self::new(
            Q8ExecutionKey {
                source: source.identity().clone(),
                layout,
            },
            vec![tiles],
        )
    }

    /// Copy scale bit patterns and all signed bytes, including inactive tails.
    /// No floating-point conversion or re-quantization occurs during packing.
    pub fn separate_scales_values(source: Q8SourceView<'_>) -> Result<Self> {
        let blocks = source.identity().shape().block_count()?;
        let mut scales = Vec::with_capacity(blocks * 2);
        let mut values = Vec::with_capacity(blocks * Q8_BLOCK_ELEMENTS);
        for block in source.blocks().chunks_exact(Q8_BLOCK_BYTES) {
            scales.extend_from_slice(&block[..2]);
            values.extend_from_slice(&block[2..]);
        }
        Self::new(
            Q8ExecutionKey {
                source: source.identity().clone(),
                layout: Q8ExecutionLayout::SeparateScalesValues,
            },
            vec![scales, values],
        )
    }

    pub fn new(key: Q8ExecutionKey, planes: Vec<Vec<u8>>) -> Result<Self> {
        ensure!(
            key.source.dtype == GgmlDType::Q8_0,
            "Q8 execution source dtype mismatch"
        );
        let plan = key.layout.allocation_plan(key.source.shape)?;
        ensure!(
            planes.len() == plan.planes.len(),
            "Q8 execution plane count mismatch"
        );
        for (data, spec) in planes.iter().zip(&plan.planes) {
            ensure!(
                data.len() == spec.bytes,
                "Q8 {:?} plane bytes: expected {}, got {}",
                spec.kind,
                spec.bytes,
                data.len()
            );
        }
        Ok(Self { key, plan, planes })
    }

    pub fn copy_column_blocks(source: Q8SourceView<'_>) -> Result<Self> {
        Self::new(
            Q8ExecutionKey {
                source: source.identity().clone(),
                layout: Q8ExecutionLayout::ColumnBlocks,
            },
            vec![source.blocks().to_vec()],
        )
    }
    pub fn key(&self) -> &Q8ExecutionKey {
        &self.key
    }
    pub fn plan(&self) -> &Q8AllocationPlan {
        &self.plan
    }
    pub fn planes(&self) -> &[Vec<u8>] {
        &self.planes
    }
}

#[cfg(any(test, feature = "cuda"))]
fn prepare_buffers<B>(
    host: &HostQ8ExecutionWeight,
    mut allocate: impl FnMut(Q8PlaneSpec) -> Result<B>,
    mut upload: impl FnMut(&B, &[u8]) -> Result<()>,
) -> Result<Vec<B>> {
    let mut buffers = Vec::with_capacity(host.planes.len());
    for (spec, data) in host.plan.planes.iter().copied().zip(&host.planes) {
        let buffer = allocate(spec)?;
        upload(&buffer, data)?;
        buffers.push(buffer);
    }
    Ok(buffers)
}
