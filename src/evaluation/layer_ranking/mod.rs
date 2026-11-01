//! Calibration-only layer selection; held-out quality never chooses the mask.

mod influence;
mod removal;

pub use influence::{
    measure_block_influence, LayerInfluence, LayerInfluenceCollector, LayerInfluenceValues,
    LayerRankingPassage, LayerRankingPlan,
};
pub use removal::{LayerRemovalRanker, LayerRemovalRanking, LayerRemovalStage, LayerRemovalTrial};

#[cfg(test)]
mod tests;
