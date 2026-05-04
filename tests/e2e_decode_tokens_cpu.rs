#![cfg(not(feature = "cuda"))]

use anyhow::Result;
use half::f16;
use m40_llm::decode::{decode_loop_with, greedy_sampler, StoppingCriteria};
use m40_llm::infer::LoadedModel;
use m40_llm::tokenizer::Tokenizer;
use std::ffi::c_void;

#[path = "common/tiny_gguf.rs"]
mod tiny_gguf;

#[test]
fn e2e_decode_tokens_cpu_identity() -> Result<()> {
    let (gguf, bytes) = tiny_gguf::make_identity_tiny_gguf(tiny_gguf::TinyGgufConfig::default());

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
