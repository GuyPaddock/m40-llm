#![allow(dead_code)]

pub mod activation_sparsity;
mod backend;
mod batch;
mod embeddings;
pub mod execution_weights;
#[cfg(feature = "cuda")]
mod ffn_capture;
mod forward;
mod gemm;
mod kv;
#[cfg(feature = "cuda")]
mod layer_influence;
mod layer_mask;
mod load;
mod logits;
mod mapping;
mod meta;
pub mod q8_activation;
mod q8_ffn;
pub mod q8_prefill;
pub mod sparse_weights;
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
pub use ffn_capture::FfnCapture;
#[cfg(feature = "cuda")]
pub use forward::{ForwardBatchItem, ForwardPrefillSequence};
#[cfg(feature = "cuda")]
pub use kv::with_exact_block_staging;
#[cfg(feature = "cuda")]
pub use layer_influence::LayerInfluenceCapture;
pub use layer_mask::LayerMask;
pub use q8_ffn::{Q8FfnConfig, Q8FfnKernel, Q8FfnScratch};
pub use types::{DeviceTensorView, LoadedModel, ModelConfig, StandardLayerWeights};
