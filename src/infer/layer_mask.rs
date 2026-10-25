use anyhow::{ensure, Result};
use serde::Serialize;

/// Diagnostic compute skipping only. Original layer IDs, weights, and KV
/// capacity remain unchanged. Default means every layer executes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct LayerMask {
    layer_count: Option<u32>,
    skipped_layers: Vec<usize>,
}

impl LayerMask {
    pub fn new(layer_count: u32, skipped_layers: &[usize]) -> Result<Self> {
        ensure!(layer_count > 0, "layer mask requires at least one layer");
        let mut skipped_layers = skipped_layers.to_vec();
        skipped_layers.sort_unstable();
        for &layer in &skipped_layers {
            ensure!(
                layer < layer_count as usize,
                "skipped layer {layer} is outside {layer_count} layers"
            );
            ensure!(
                layer != 0 && layer + 1 != layer_count as usize,
                "diagnostic masks preserve the first and last layers; cannot skip layer {layer}"
            );
        }
        ensure!(
            !skipped_layers.windows(2).any(|pair| pair[0] == pair[1]),
            "duplicate skipped layer ID"
        );
        Ok(Self {
            layer_count: Some(layer_count),
            skipped_layers,
        })
    }

    pub fn validate(&self, layer_count: u32) -> Result<()> {
        ensure!(layer_count > 0, "layer mask requires at least one layer");
        ensure!(
            self.layer_count
                .is_none_or(|expected| expected == layer_count),
            "mask layer count {:?} differs from model layer count {layer_count}",
            self.layer_count
        );
        Ok(())
    }

    pub fn skipped_layers(&self) -> &[usize] {
        &self.skipped_layers
    }

    pub fn has_skipped_layers(&self) -> bool {
        !self.skipped_layers.is_empty()
    }

    pub fn active_layer_count(&self, layer_count: u32) -> Result<usize> {
        self.validate(layer_count)?;
        Ok(layer_count as usize - self.skipped_layers.len())
    }

    pub fn active_layers(
        &self,
        layer_count: u32,
    ) -> Result<impl DoubleEndedIterator<Item = usize> + '_> {
        self.validate(layer_count)?;
        Ok((0..layer_count as usize)
            .filter(|layer| self.skipped_layers.binary_search(layer).is_err()))
    }
}

impl super::LoadedModel {
    /// Freeze the diagnostic before KV/session state exists. This skips compute,
    /// not source tensors or physical KV layer slots; no layer IDs are renumbered.
    pub fn set_diagnostic_layer_mask(&mut self, skipped_layers: &[usize]) -> Result<()> {
        let mask = LayerMask::new(self.model_config.block_count, skipped_layers)?;
        ensure!(
            self.kv_cache.is_none(),
            "configure the layer mask before allocating KV or creating decode sessions"
        );
        #[cfg(feature = "cuda")]
        ensure!(
            self.ffn_capture.is_none(),
            "finish FFN capture before configuring a diagnostic mask"
        );
        #[cfg(feature = "cuda")]
        ensure!(
            self.layer_influence.is_none(),
            "finish influence capture before configuring a diagnostic mask"
        );
        #[cfg(feature = "cuda")]
        ensure!(
            self.activation_research.is_none(),
            "configure the layer mask before activation capture or application"
        );
        #[cfg(feature = "cuda")]
        ensure!(
            self.forward_workspace
                .lock()
                .map_err(|_| anyhow::anyhow!("poisoned forward workspace"))?
                .is_none(),
            "configure the layer mask before forward execution"
        );
        self.layer_mask = mask;
        Ok(())
    }

    pub fn active_layer_count(&self) -> Result<usize> {
        self.layer_mask
            .active_layer_count(self.model_config.block_count)
    }
}
