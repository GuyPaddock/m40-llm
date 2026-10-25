#![allow(clippy::items_after_test_module)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

pub mod cli;
pub mod cuda;
pub mod decode_batch;
pub mod gguf;
pub mod infer;

pub mod decode;
pub mod decode_session;
pub mod evaluation;
pub mod generate;
pub mod generation_metrics;
#[cfg(feature = "gguf_ext")]
pub mod gguf_ext;
#[cfg(feature = "cuda")]
pub mod gpu_profile;
pub mod kv_compression;
pub mod kv_selection;
pub mod model;
pub mod profile;
pub mod sampling;
#[cfg(feature = "server")]
pub mod server;
pub(crate) mod timing;
pub mod tokenizer;
