#[path = "q8_projection.rs"]
pub mod projection;
#[path = "tiny_gguf.rs"]
mod tiny;

use m40_llm::gguf::{GgmlDType, GgufModel};

pub const QKV_NAMES: [&str; 3] = [
    "layers.0.attention.wq.weight",
    "layers.0.attention.wk.weight",
    "layers.0.attention.wv.weight",
];

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
    build_with_layers(seed, 1)
}

/// Full-forward fixture with distinct original-layer payloads. Tensor offsets
/// remain relative to the synthetic data blob, as in the one-layer fixture.
pub fn build_with_layers(seed: u8, layer_count: u32) -> (GgufModel, Vec<u8>) {
    assert!(layer_count > 0);
    let (mut model, mut weights) = build_projection_fixture(seed, [64; 3], 128);
    let original = weights.clone();
    let tensors = model.tensors.clone();
    for layer in 1..layer_count {
        for tensor in &tensors {
            if !tensor.name.starts_with("layers.0.") {
                continue;
            }
            let mut copied = tensor.clone();
            copied.name = copied
                .name
                .replacen("layers.0.", &format!("layers.{layer}."), 1);
            weights.resize(weights.len().div_ceil(32) * 32, 0);
            copied.offset = weights.len() as u64;
            let size = tensor.shape.iter().product::<u64>() as usize / 32 * 34;
            let mut payload =
                original[tensor.offset as usize..tensor.offset as usize + size].to_vec();
            for block in payload.chunks_exact_mut(34) {
                for value in &mut block[2..] {
                    *value = value.wrapping_add((layer as u8).wrapping_mul(7));
                }
            }
            weights.extend_from_slice(&payload);
            model.tensors.push(copied);
        }
    }
    model.metadata.insert(
        "llama.block_count".to_owned(),
        m40_llm::gguf::GgufValue::Scalar(m40_llm::gguf::GgufScalar::U32(layer_count)),
    );
    (model, weights)
}

/// Two-layer prefill fixtures with nonzero quantized FFNs. The Qwen case has
/// head128 GQA, split-half RoPE and real Q/K/V bias tensors. Small weights keep
/// storage-rounding drift within the existing packed/sequential tolerance.
#[allow(dead_code)]
pub fn build_prefill_fixture(is_qwen: bool) -> (GgufModel, Vec<u8>) {
    let (mut model, original) = if is_qwen {
        let (mut model, bytes) = tiny::make_qwen2_attention_bias_tiny_gguf(tiny::TinyGgufConfig {
            vocab: 320,
            d_model: 256,
            hidden: 512,
            head_count: 2,
            block_count: 2,
            context_length: 128,
        });
        for prefix in ["llama", "qwen2"] {
            model.metadata.insert(
                format!("{prefix}.attention.head_count_kv"),
                m40_llm::gguf::GgufValue::Scalar(m40_llm::gguf::GgufScalar::U32(1)),
            );
        }
        (model, bytes)
    } else {
        build_with_layers(0, 2)
    };
    let mut weights = Vec::new();
    for tensor in &mut model.tensors {
        let old = tensor.clone();
        if is_qwen
            && (tensor.name.ends_with("attention.wk.weight")
                || tensor.name.ends_with("attention.wv.weight"))
        {
            tensor.shape[1] = 128;
        }
        if is_qwen
            && (tensor.name.ends_with("attention.wk.bias")
                || tensor.name.ends_with("attention.wv.bias"))
        {
            tensor.shape[0] = 128;
        }
        weights.resize(weights.len().div_ceil(32) * 32, 0);
        tensor.offset = weights.len() as u64;
        if tensor.shape.len() == 2 {
            let mut payload = if old.dtype == GgmlDType::Q8_0 {
                let size = tensor.shape.iter().product::<u64>() as usize / 32 * 34;
                original[old.offset as usize..old.offset as usize + size].to_vec()
            } else {
                projection::make_column_blocks(tensor.shape[0] as usize, tensor.shape[1] as usize)
            };
            for block in payload.chunks_exact_mut(34) {
                let scale =
                    half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32() / 32.0;
                block[..2].copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            }
            tensor.dtype = GgmlDType::Q8_0;
            weights.extend_from_slice(&payload);
        } else {
            assert_eq!(tensor.dtype, GgmlDType::F32);
            let size = tensor.shape.iter().product::<u64>() as usize * 4;
            for value in original[old.offset as usize..old.offset as usize + size].chunks_exact(4) {
                weights.extend_from_slice(
                    &(f32::from_le_bytes(value.try_into().unwrap()) / 16.0).to_le_bytes(),
                );
            }
        }
    }
    (model, weights)
}

/// Override projection widths for projection-only tail/GQA tests; these
/// synthetic descriptors are not intended to run a full transformer forward.
pub fn build_projection_fixture(
    seed: u8,
    widths: [usize; 3],
    mlp_width: usize,
) -> (GgufModel, Vec<u8>) {
    let (mut model, _) = build_f16();
    let mut weights = Vec::new();
    for (tensor_index, tensor) in model.tensors.iter_mut().enumerate() {
        if let Some(index) = QKV_NAMES.iter().position(|&name| name == tensor.name) {
            tensor.shape[1] = widths[index] as u64;
        }
        if tensor.name.ends_with("feed_forward.w1.weight")
            || tensor.name.ends_with("feed_forward.w3.weight")
        {
            tensor.shape[1] = mlp_width as u64;
        }
        let k = tensor.shape[0] as usize;
        let n = tensor.shape[1] as usize;
        weights.resize(weights.len().div_ceil(32) * 32, 0);
        tensor.offset = weights.len() as u64;
        tensor.dtype = GgmlDType::Q8_0;
        let mut data = projection::make_column_blocks(k, n);
        for block in data.chunks_exact_mut(34) {
            for value in &mut block[2..] {
                *value = value
                    .wrapping_add(seed)
                    .wrapping_add((tensor_index as u8).wrapping_mul(19));
            }
        }
        weights.extend_from_slice(&data);
    }
    (model, weights)
}
