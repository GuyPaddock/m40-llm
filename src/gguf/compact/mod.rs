//! Research-only compact layer export; no source model or runtime default changes.

mod evidence;
mod plan;

pub use evidence::{CompactExportEvidence, CompactQualitySummary};
pub use plan::{CompactGgufPlan, CompactTensorCopy};
