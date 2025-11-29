#![cfg(all(feature = "cuda", nvcc))]

use anyhow::Result;
use half::f16;
use m40_llm::decode::{decode_loop_with, greedy_sampler, StoppingCriteria};
use m40_llm::gguf::{GgmlDType, GgufModel, GgufScalar, GgufTensor, GgufValue};
use m40_llm::infer::LoadedModel;

mod cuda_env;

fn f16_bytes(v: f32) -> [u8; 2] {
    let h = f16::from_f32(v);
    h.to_bits().to_le_bytes()
}

fn make_min_gguf(vocab: usize, d_model: usize, hidden: usize) -> (GgufModel, Vec<u8>) {
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        "llama.embedding_length".to_string(),
        GgufValue::Scalar(GgufScalar::U32(d_model as u32)),
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
fn e2e_decode_tokens_cuda_vs_cpu_identity() -> Result<()> {
    let ctx = cuda_env::ctx_m40()?;
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("skipping: {}", e);
        return Ok(());
    }

    let vocab = 256usize;
    let d_model = 256usize;
    let hidden = 16usize;
    let (gguf, bytes) = make_min_gguf(vocab, d_model, hidden);

    let model = LoadedModel::from_gguf(gguf, bytes, -1)?;
    let tokenizer = m40_llm::tokenizer::Tokenizer::byte_level();

    let eos_id = tokenizer.eos_id();
    let stopping = StoppingCriteria::new(Some(4), eos_id);
    let sampler = greedy_sampler(1234);

    // GPU logits_fn: embed last token into device f32 buffer, then GEMM with lm_head
    let d_x = model.cuda.device_malloc(d_model * 4)?;
    let logits_fn_gpu = {
        let model = &model;
        move |ids: &[u32]| -> anyhow::Result<Vec<f32>> {
            let tok_id = *ids.last().ok_or_else(|| anyhow::anyhow!("empty ids"))? as u64;
            unsafe {
                model.load_token_embedding_f16_to_f32(tok_id, d_x)?;
                model.logits_from_hidden(d_x)
            }
        }
    };

    let ids_gpu = decode_loop_with(
        &tokenizer,
        "A",
        true,
        sampler.clone(),
        &stopping,
        logits_fn_gpu,
    )?;

    // CPU logits_fn: construct hidden from embedding row on host and do host dot with lm_head
    let logits_fn_cpu = {
        let model = &model;
        move |ids: &[u32]| -> anyhow::Result<Vec<f32>> {
            let tok_id = *ids.last().ok_or_else(|| anyhow::anyhow!("empty ids"))? as usize;
            // hidden from embedding row (copy F16 row from device → host; convert to f32)
            let tok = model
                .device_tensors
                .get("tok_embeddings.weight")
                .ok_or_else(|| anyhow::anyhow!("missing tok_embeddings.weight"))?;
            let d_model = tok.shape[1] as usize;
            let row_bytes = d_model * 2;
            let off = tok.byte_offset as usize + tok_id * row_bytes;
            let mut row_f16 = vec![0u8; row_bytes];
            unsafe {
                model.cuda.memcpy_d2h(
                    row_f16.as_mut_ptr() as *mut _,
                    (model.d_weights_base as usize + off) as *const _,
                    row_bytes,
                )?;
            }
            let mut hidden = vec![0f32; d_model];
            for i in 0..d_model {
                let lo = row_f16[2 * i] as u16;
                let hi = row_f16[2 * i + 1] as u16;
                hidden[i] = f16::from_bits(lo | (hi << 8)).to_f32();
            }
            // logits = hidden x output.weight (D x V) using host matmul over copied weights
            let (lm, _d_model, vocab, _tied) = model.map_lm_head()?;
            let w_off = lm.byte_offset as usize;
            let bytes = d_model * vocab * 2;
            let mut w_bytes = vec![0u8; bytes];
            unsafe {
                model.cuda.memcpy_d2h(
                    w_bytes.as_mut_ptr() as *mut _,
                    (model.d_weights_base as usize + w_off) as *const _,
                    bytes,
                )?;
            }
            let mut logits = vec![0f32; vocab];
            for (col, logit_ref) in logits.iter_mut().enumerate().take(vocab) {
                let mut acc = 0f32;
                let mut idx = col * 2;
                for (_rowi, h) in hidden.iter().copied().enumerate().take(d_model) {
                    let lo = w_bytes[idx] as u16;
                    let hi = w_bytes[idx + 1] as u16;
                    let w = f16::from_bits(lo | (hi << 8)).to_f32();
                    acc += h * w;
                    idx += vocab * 2;
                }
                *logit_ref = acc;
            }
            Ok(logits)
        }
    };

    let ids_cpu = decode_loop_with(&tokenizer, "A", true, sampler, &stopping, logits_fn_cpu)?;

    // Compare last 4 generated tokens; they should be identical and equal to last prompt token
    let last_prompt = *ids_cpu.get(ids_cpu.len() - 1 - 4).unwrap();
    assert_eq!(last_prompt, 'A' as u32);
    assert_eq!(&ids_cpu[ids_cpu.len() - 4..], &ids_gpu[ids_gpu.len() - 4..]);

    unsafe {
        model.cuda.device_free(d_x)?;
    }
    Ok(())
}
