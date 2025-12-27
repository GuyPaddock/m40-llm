pub mod cli;
pub mod cuda;
pub mod gguf;
pub mod infer;

pub mod decode;
#[cfg(feature = "gguf_ext")]
pub mod gguf_ext;
pub mod model;
pub mod sampling;
#[cfg(feature = "server")]
pub mod server;
pub mod tokenizer;

// Automatically enable device weights when NVCC is available
#[cfg(nvcc)]
#[ctor::ctor]
fn auto_enable_nvcc() {
    // Only set if not already set
    if std::env::var("M40LLM_ENABLE_NVCC").is_err() {
        std::env::set_var("M40LLM_ENABLE_NVCC", "1");
    }
}
