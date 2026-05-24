#![cfg(all(feature = "cuda", nvcc))]

use anyhow::Result;
use half::f16;
use m40_llm::decode::{decode_loop_with, greedy_sampler, StoppingCriteria};
use m40_llm::infer::LoadedModel;

mod cuda_env;
#[path = "common/tiny_gguf.rs"]
mod tiny_gguf;

#[test]
fn e2e_decode_tokens_cuda_vs_cpu_identity() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("skipping: {}", e);
        return Ok(());
    }

    let cfg = tiny_gguf::TinyGgufConfig {
        head_count: 16,
        ..Default::default()
    };
    let vocab = cfg.vocab;
    let d_model = cfg.d_model;
    let (gguf, bytes) = tiny_gguf::make_identity_tiny_gguf(cfg);

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
                let vocab_size = model.model_config.vocab_size as usize;
                assert_eq!(vocab_size, vocab);

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
            let (_name, lm, _d_model, vocab, _tied) = model.map_lm_head()?;
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

#[test]
fn e2e_logits_cuda_uses_tied_output_embeddings() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("skipping: {}", e);
        return Ok(());
    }

    let cfg = tiny_gguf::TinyGgufConfig {
        vocab: 128,
        d_model: 64,
        head_count: 8,
        ..Default::default()
    };
    let (gguf, bytes) = tiny_gguf::make_tied_identity_tiny_gguf(cfg.clone());
    let model = LoadedModel::from_gguf(gguf, bytes, -1)?;
    let (_name, _view, d_model, vocab, tied) = model.map_lm_head()?;
    assert_eq!(d_model, cfg.d_model);
    assert_eq!(vocab, cfg.vocab);
    assert!(tied);

    let mut hidden = vec![0f32; cfg.d_model];
    hidden[42] = 1.0;
    let hidden_bytes = hidden.len() * std::mem::size_of::<f32>();
    let d_hidden = model.cuda.device_malloc(hidden_bytes)?;
    unsafe {
        model.cuda.memcpy_h2d(
            d_hidden,
            hidden.as_ptr() as *const std::ffi::c_void,
            hidden_bytes,
        )?;
        let logits = model.logits_from_hidden(d_hidden)?;
        model.cuda.device_free(d_hidden)?;

        let (best_token, best_logit) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("logits non-empty");
        assert_eq!(best_token, 42);
        assert!((best_logit - 1.0).abs() < 1e-5);
    }
    Ok(())
}
