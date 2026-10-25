#[path = "q8_projection.rs"]
pub mod projection;
#[path = "tiny_gguf.rs"]
mod tiny;

use m40_llm::gguf::{GgmlDType, GgufModel};

pub fn build_f16() -> (GgufModel, Vec<u8>) {
    tiny::make_tied_identity_tiny_gguf(tiny::TinyGgufConfig {
        vocab: 96,
        d_model: 64,
        hidden: 128,
        head_count: 1,
        block_count: 1,
        context_length: 128,
    })
}

/// Synthetic model descriptor plus tensor-data bytes, matching the existing
/// tiny-model fixtures. The seed changes signed payloads without changing layout.
pub fn build(seed: u8) -> (GgufModel, Vec<u8>) {
    let (mut model, _) = build_f16();
    let mut weights = Vec::new();
    for tensor in &mut model.tensors {
        let k = tensor.shape[0] as usize;
        let n = tensor.shape[1] as usize;
        weights.resize(weights.len().div_ceil(32) * 32, 0);
        tensor.offset = weights.len() as u64;
        tensor.dtype = GgmlDType::Q8_0;
        let mut data = projection::make_column_blocks(k, n);
        for block in data.chunks_exact_mut(34) {
            for value in &mut block[2..] {
                *value = value.wrapping_add(seed);
            }
        }
        weights.extend_from_slice(&data);
    }
    (model, weights)
}
