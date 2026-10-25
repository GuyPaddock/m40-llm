//! Explicit research-quality evaluation, not generation policy.

pub mod corpus;
pub mod identity;
pub mod layer_ranking;
pub mod loss;
pub mod scoring;
#[cfg(feature = "cuda")]
pub mod teacher_forced;
