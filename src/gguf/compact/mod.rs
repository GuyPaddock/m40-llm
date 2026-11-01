//! Research-only compact layer export; no source model or runtime default changes.

mod evidence;
mod metadata;
mod plan;
mod write;

pub use evidence::{CompactExportEvidence, CompactQualitySummary};
pub use plan::{CompactGgufPlan, CompactTensorCopy};
pub use write::{export_compact_gguf, CompactExportResult};
