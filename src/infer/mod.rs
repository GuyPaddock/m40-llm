#![allow(dead_code)]

mod backend;
mod batch;
mod embeddings;
mod forward;
mod gemm;
mod kv;
mod load;
mod logits;
mod mapping;
mod meta;
mod tensor_views;
mod types;
#[cfg(feature = "cuda")]
mod workspace;

pub use backend::{
    ProjectionBackend, ProjectionBackendDecision, ProjectionBackendEstimate, ProjectionBackendMode,
};
#[cfg(feature = "cuda")]
pub use batch::VarlenPrefillPlan;
pub use batch::{
    BatchMetadata, BatchSequence, BucketStats, BucketedBatch, LengthBucket, PackedSequenceOffsets,
    VarlenPrefillTile, VarlenPrefillTileSelection,
};
#[cfg(feature = "cuda")]
pub use forward::{ForwardBatchItem, ForwardPrefillSequence};
#[cfg(feature = "cuda")]
pub use kv::with_exact_block_staging;
pub use types::{DeviceTensorView, LoadedModel, ModelConfig, StandardLayerWeights};
