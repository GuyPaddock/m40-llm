//! Experimental magnitude selection for the completed SwiGLU vector. This
//! does not change model dispatch or use the rejected sparse-weight format.

use anyhow::{ensure, Result};
use serde::Serialize;

mod calibration;
pub use calibration::{
    ActivationCalibration, ActivationCalibrationCollector, ActivationCalibrationPlan,
    CalibrationLayer, CalibrationTaskObservation, LayerActivationCalibration,
    LayerActivationThreshold,
};

#[cfg(feature = "cuda")]
mod device;
#[cfg(feature = "cuda")]
pub use device::DeviceSparseInputWorkspace;
#[cfg(feature = "cuda")]
mod runtime;
#[cfg(feature = "cuda")]
pub use runtime::{ActivationLayerStats, ActivationResearch, ActivationResearchStats};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ActivationThreshold(f32);

impl ActivationThreshold {
    pub fn new(magnitude: f32) -> Result<Self> {
        ensure!(
            magnitude.is_finite() && magnitude >= 0.0,
            "activation threshold must be finite and nonnegative"
        );
        Ok(Self(magnitude.abs()))
    }

    /// Fit a magnitude quantile to caller-supplied calibration observations.
    /// Ties may discard more than the requested fraction; measure actual sparsity.
    /// The caller owns calibration/held-out separation and layer identity.
    pub fn calibrate(observations: &[f32], target_sparsity: f64) -> Result<Self> {
        ensure!(
            target_sparsity.is_finite() && (0.0..1.0).contains(&target_sparsity),
            "target activation sparsity must be in [0, 1)"
        );
        ensure!(
            !observations.is_empty() && observations.iter().all(|value| value.is_finite()),
            "activation calibration requires nonempty finite observations"
        );
        let discarded = (observations.len() as f64 * target_sparsity).floor() as usize;
        if discarded == 0 {
            return Self::new(0.0);
        }
        let mut magnitudes: Vec<_> = observations.iter().map(|value| value.abs()).collect();
        magnitudes.sort_unstable_by(f32::total_cmp);
        Self::new(magnitudes[discarded - 1])
    }

    pub fn magnitude(self) -> f32 {
        self.0
    }

    /// Never hide a nonfinite activation by thresholding it to zero.
    pub fn retains(self, value: f32) -> bool {
        !value.is_finite() || value.abs() > self.0
    }
}

/// Compact values retain their original increasing input indices, not packed
/// weight offsets. Omitted inputs are exactly zero in the matched reference.
#[derive(Debug, Clone)]
pub struct SparseInputSelection {
    input_dim: usize,
    indices: Vec<u32>,
    values: Vec<f32>,
}

impl SparseInputSelection {
    pub fn select(input: &[f32], threshold: ActivationThreshold) -> Result<Self> {
        ensure!(
            !input.is_empty() && input.len() <= i32::MAX as usize - 255,
            "sparse input dimension is outside the supported indexing range"
        );
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (index, &value) in input.iter().enumerate() {
            if threshold.retains(value) {
                indices.push(index as u32);
                values.push(value);
            }
        }
        Ok(Self {
            input_dim: input.len(),
            indices,
            values,
        })
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn sparsity(&self) -> f64 {
        1.0 - self.indices.len() as f64 / self.input_dim as f64
    }

    pub fn reconstruct(&self) -> Vec<f32> {
        let mut dense = vec![0.0; self.input_dim];
        for (&index, &value) in self.indices.iter().zip(&self.values) {
            dense[index as usize] = value;
        }
        dense
    }
}
