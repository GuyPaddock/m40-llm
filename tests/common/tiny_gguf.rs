use half::f16;
use m40_llm::gguf::{GgmlDType, GgufModel, GgufScalar, GgufTensor, GgufValue};

#[derive(Debug, Clone)]
pub struct TinyGgufConfig {
    pub vocab: usize,
    pub d_model: usize,
    pub hidden: usize,
    pub head_count: u32,
    pub block_count: u32,
    pub context_length: u32,
}

impl Default for TinyGgufConfig {
    fn default() -> Self {
        Self {
            vocab: 256,
            d_model: 256,
            hidden: 16,
            head_count: 1,
            block_count: 1,
            context_length: 512,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum TinyOutput {
    Identity,
    ForceToken(usize),
}

fn f16_bytes(v: f32) -> [u8; 2] {
    f16::from_f32(v).to_bits().to_le_bytes()
}

#[allow(dead_code)]
pub fn make_identity_tiny_gguf(cfg: TinyGgufConfig) -> (GgufModel, Vec<u8>) {
    make_tiny_gguf(cfg, TinyOutput::Identity, false)
}

#[allow(dead_code)]
pub fn make_ascii_tiny_gguf(cfg: TinyGgufConfig) -> (GgufModel, Vec<u8>) {
    make_tiny_gguf(cfg, TinyOutput::ForceToken(b'A' as usize), false)
}

#[allow(dead_code)]
pub fn make_tied_identity_tiny_gguf(cfg: TinyGgufConfig) -> (GgufModel, Vec<u8>) {
    make_tiny_gguf(cfg, TinyOutput::Identity, true)
}

fn make_tiny_gguf(
    cfg: TinyGgufConfig,
    output: TinyOutput,
    tied_output: bool,
) -> (GgufModel, Vec<u8>) {
    assert!(cfg.vocab >= cfg.d_model);
    assert!(cfg.d_model > 0);
    assert!(cfg.hidden > 0);
    assert!(cfg.head_count > 0);
    assert!(cfg.block_count > 0);
    assert!(cfg.d_model.is_multiple_of(cfg.head_count as usize));
    if let TinyOutput::ForceToken(token) = output {
        assert!(token < cfg.vocab);
    }

    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::Scalar(GgufScalar::Str("llama".to_string())),
    );
    metadata.insert(
        "general.file_type".to_string(),
        GgufValue::Scalar(GgufScalar::U32(1)),
    );
    metadata.insert(
        "llama.embedding_length".to_string(),
        GgufValue::Scalar(GgufScalar::U32(cfg.d_model as u32)),
    );
    metadata.insert(
        "llama.vocab_size".to_string(),
        GgufValue::Scalar(GgufScalar::U32(cfg.vocab as u32)),
    );
    metadata.insert(
        "llama.block_count".to_string(),
        GgufValue::Scalar(GgufScalar::U32(cfg.block_count)),
    );
    metadata.insert(
        "llama.attention.head_count".to_string(),
        GgufValue::Scalar(GgufScalar::U32(cfg.head_count)),
    );
    metadata.insert(
        "llama.feed_forward_length".to_string(),
        GgufValue::Scalar(GgufScalar::U32(cfg.hidden as u32)),
    );
    metadata.insert(
        "llama.context_length".to_string(),
        GgufValue::Scalar(GgufScalar::U32(cfg.context_length)),
    );
    metadata.insert(
        "tokenizer.ggml.model".to_string(),
        GgufValue::Scalar(GgufScalar::Str("llama".to_string())),
    );

    let mut tensors: Vec<GgufTensor> = Vec::new();
    let mut weights: Vec<u8> = Vec::new();
    let mut add_tensor =
        |name: &str, dtype: GgmlDType, shape: &[u64], fill: &mut dyn FnMut(&mut Vec<u8>)| {
            let offset = weights.len() as u64;
            fill(&mut weights);
            tensors.push(GgufTensor {
                name: name.to_string(),
                dtype,
                shape: shape.to_vec(),
                offset,
            });
        };

    if tied_output {
        let mut fill = |buf: &mut Vec<u8>| {
            for col in 0..cfg.vocab {
                for row in 0..cfg.d_model {
                    let v = if row == col { 1.0 } else { 0.0 };
                    buf.extend_from_slice(&f16_bytes(v));
                }
            }
        };
        add_tensor(
            "tok_embeddings.weight",
            GgmlDType::F16,
            &[cfg.d_model as u64, cfg.vocab as u64],
            &mut fill,
        );
    } else {
        let mut fill = |buf: &mut Vec<u8>| {
            for row in 0..cfg.vocab {
                for col in 0..cfg.d_model {
                    let v = if row == col { 1.0 } else { 0.0 };
                    buf.extend_from_slice(&f16_bytes(v));
                }
            }
        };
        add_tensor(
            "tok_embeddings.weight",
            GgmlDType::F16,
            &[cfg.vocab as u64, cfg.d_model as u64],
            &mut fill,
        );
    }

    if !tied_output {
        let mut fill = |buf: &mut Vec<u8>| {
            for col in 0..cfg.vocab {
                for row in 0..cfg.d_model {
                    let v = match output {
                        TinyOutput::Identity => {
                            if row == col {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        TinyOutput::ForceToken(token) => {
                            if col == token {
                                2.0
                            } else if row == col {
                                1.0
                            } else {
                                0.0
                            }
                        }
                    };
                    buf.extend_from_slice(&f16_bytes(v));
                }
            }
        };
        add_tensor(
            "output.weight",
            GgmlDType::F16,
            &[cfg.d_model as u64, cfg.vocab as u64],
            &mut fill,
        );
    }

    let zeros_f16 = |n: usize| -> Vec<u8> { (0..n).flat_map(|_| f16_bytes(0.0)).collect() };
    let mut add_zeros = |name: &str, shape: &[u64]| {
        let elems: usize = shape.iter().copied().product::<u64>() as usize;
        let mut fill = |buf: &mut Vec<u8>| buf.extend_from_slice(&zeros_f16(elems));
        add_tensor(name, GgmlDType::F16, shape, &mut fill);
    };

    for layer in 0..cfg.block_count {
        for suffix in [
            "attention.wq.weight",
            "attention.wk.weight",
            "attention.wv.weight",
            "attention.wo.weight",
        ] {
            add_zeros(
                &format!("layers.{layer}.{suffix}"),
                &[cfg.d_model as u64, cfg.d_model as u64],
            );
        }
        add_zeros(
            &format!("layers.{layer}.feed_forward.w3.weight"),
            &[cfg.d_model as u64, cfg.hidden as u64],
        );
        add_zeros(
            &format!("layers.{layer}.feed_forward.w1.weight"),
            &[cfg.d_model as u64, cfg.hidden as u64],
        );
        add_zeros(
            &format!("layers.{layer}.feed_forward.w2.weight"),
            &[cfg.hidden as u64, cfg.d_model as u64],
        );
    }

    let gguf = GgufModel {
        version: 1,
        metadata,
        tensors,
        data_offset: 0,
    };
    (gguf, weights)
}
