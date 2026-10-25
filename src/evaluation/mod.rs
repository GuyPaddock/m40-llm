//! Explicit research-quality evaluation, not generation policy.

pub mod corpus;
pub mod loss;
pub mod scoring;
#[cfg(feature = "cuda")]
pub mod teacher_forced;
