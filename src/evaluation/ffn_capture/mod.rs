//! Read-only paired FFN observations for projection benchmarks, not sparsity
//! calibration. Normalized gate/up input and completed SwiGLU input to down
//! belong to the same original layer, task and absolute generated position.

mod collector;
mod records;

pub use collector::FfnCaptureCollector;
pub use records::{
    FfnCaptureArtifact, FfnCapturePlan, FfnLayerSamples, FfnLayerSource, FfnRowPosition,
    FfnTaskObservation, FfnTensorSource,
};

#[cfg(test)]
mod tests;
