#![cfg(not(feature = "cuda"))]

use anyhow::Result;
use half::f16;
use m40_llm::decode::{decode_loop_with, greedy_sampler, StoppingCriteria};
use m40_llm::gguf::{GgmlDType, GgufModel, GgufScalar, GgufTensor, GgufValue};
use m40_llm::infer::LoadedModel;
use m40_llm::tokenizer::Tokenizer;
use std::ffi::c_void;

fn f16_bytes(v: f32) -> [u8; 2] {
    let h = f16::from_f32(v);
    h.to_bits().to_le_bytes()
}

fn make_min_gguf(vocab: usize, d_model: usize, hidden: usize) -> (GgufModel, Vec<u8>) {
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::Scalar(GgufScalar::Str("llama".to_string())),
    );
    metadata.insert(
        "llama.embedding_length".to_string(),
        GgufValue::Scalar(GgufScalar::U32(d_model as u32)),
    );
    metadata.insert(
        "llama.attention.head_count".to_string(),
        GgufValue::Scalar(GgufScalar::U32(1)),
    );
    metadata.insert(
        "llama.vocab_size".to_string(),
        GgufValue::Scalar(GgufScalar::U32(vocab as u32)),
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

    // tok_embeddings.weight: one-hot rows
    {
        let mut fill = |buf: &mut Vec<u8>| {
            for row in 0..vocab {
                for col in 0..d_model {
                    let v = if row == col { 1.0 } else { 0.0 };
                    buf.extend_from_slice(&f16_bytes(v));
                }
            }
        };
        add_tensor(
            "tok_embeddings.weight",
            GgmlDType::F16,
            &[vocab as u64, d_model as u64],
            &mut fill,
        );
    }

    // output.weight (lm_head): identity D x V
    {
        let mut fill = |buf: &mut Vec<u8>| {
            for r in 0..d_model {
                for c in 0..vocab {
                    let v = if r == c { 1.0 } else { 0.0 };
                    buf.extend_from_slice(&f16_bytes(v));
                }
            }
        };
        add_tensor(
            "output.weight",
            GgmlDType::F16,
            &[d_model as u64, vocab as u64],
            &mut fill,
        );
    }

    // Minimal layer 0 weights
    let zeros_f16 = |n: usize| -> Vec<u8> { (0..n).flat_map(|_| f16_bytes(0.0)).collect() };
    let mut add_zeros = |name: &str, shape: &[u64]| {
        let elems: usize = shape.iter().copied().product::<u64>() as usize;
        let mut fill = |buf: &mut Vec<u8>| buf.extend_from_slice(&zeros_f16(elems));
        add_tensor(name, GgmlDType::F16, shape, &mut fill);
    };
    add_zeros(
        "layers.0.attention.wq.weight",
        &[d_model as u64, d_model as u64],
    );
    add_zeros(
        "layers.0.attention.wk.weight",
        &[d_model as u64, d_model as u64],
    );
    add_zeros(
        "layers.0.attention.wv.weight",
        &[d_model as u64, d_model as u64],
    );
    add_zeros(
        "layers.0.attention.wo.weight",
        &[d_model as u64, d_model as u64],
    );
    add_zeros(
        "layers.0.feed_forward.w3.weight",
        &[d_model as u64, hidden as u64],
    );
    add_zeros(
        "layers.0.feed_forward.w1.weight",
        &[d_model as u64, hidden as u64],
    );
    add_zeros(
        "layers.0.feed_forward.w2.weight",
        &[hidden as u64, d_model as u64],
    );

    let gguf = GgufModel {
        version: 1,
        metadata,
        tensors,
        data_offset: 0,
    };
    (gguf, weights)
}

#[test]
fn e2e_decode_tokens_cpu_identity() -> Result<()> {
    let vocab = 256usize;
    let d_model = 256usize;
    let hidden = 16usize;
    let (gguf, bytes) = make_min_gguf(vocab, d_model, hidden);

    let model = LoadedModel::from_gguf(gguf, bytes, 0)?;
    let tokenizer = Tokenizer::byte_level();

    let eos_id = tokenizer.eos_id();
    let stopping = StoppingCriteria::new(Some(4), eos_id);
    let sampler = greedy_sampler(1234);

    // logits_fn: host path
    let logits_fn = {
        move |ids: &[u32]| -> anyhow::Result<Vec<f32>> {
            let tok_id = *ids.last().ok_or_else(|| anyhow::anyhow!("empty ids"))? as usize;
            let logits = unsafe {
                // Build hidden from embedding row (F16->F32) and compute logits via lm_head
                let tok = model
                    .device_tensors
                    .get("tok_embeddings.weight")
                    .ok_or_else(|| anyhow::anyhow!("missing tok_embeddings.weight"))?;
                let d_model = tok.shape[1] as usize;
                let row_bytes = d_model * 2;
                let off = tok.byte_offset as usize + tok_id * row_bytes;
                let row = &model.host_weights[off..off + row_bytes];
                let mut hidden = vec![0f32; d_model];
                for i in 0..d_model {
                    let lo = row[2 * i] as u16;
                    let hi = row[2 * i + 1] as u16;
                    hidden[i] = f16::from_bits(lo | (hi << 8)).to_f32();
                }
                model.logits_from_hidden(hidden.as_ptr() as *const c_void)?
            };
            Ok(logits)
        }
    };

    let ids = decode_loop_with(&tokenizer, "A", true, sampler, &stopping, logits_fn)?;
    // ids includes BOS (optional) and prompt, then generated tokens.
    // For identity setup, greedy repeats the last prompt token (byte for 'A' = 65)
    let last_prompt = *ids.get(ids.len() - 1 - 4).unwrap(); // token before 4 generated
    assert_eq!(last_prompt, 'A' as u32);
    for t in ids[ids.len() - 4..].iter() {
        assert_eq!(*t, last_prompt);
    }
    Ok(())
}
