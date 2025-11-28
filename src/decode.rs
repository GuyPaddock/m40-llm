// src/decode.rs
//! Decode loop helpers: stopping criteria for EOS and max_tokens.

use anyhow::{anyhow, Result};

use crate::sampling::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Default)]
pub struct StoppingCriteria {
    pub max_tokens: Option<usize>,
    pub eos_id: Option<u32>,
}

impl StoppingCriteria {
    pub fn new(max_tokens: Option<usize>, eos_id: Option<u32>) -> Self {
        Self { max_tokens, eos_id }
    }

    /// Returns true if generation should stop based on:
    /// - max_tokens: if present and generated_len >= max_tokens
    /// - eos_id: if present and the last generated id equals eos_id
    pub fn should_stop(&self, generated: &[u32]) -> bool {
        if let Some(mt) = self.max_tokens {
            if generated.len() >= mt {
                return true;
            }
        }
        if let (Some(eos), Some(&last)) = (self.eos_id, generated.last()) {
            if last == eos {
                return true;
            }
        }
        false
    }
}

/// Minimal host-side decode loop that consumes logits from a user-provided callback.
/// - Tokenizer BOS is added when present if `add_bos` is true
/// - Stopping on EOS (if configured) or max_tokens (if configured)
pub fn decode_loop_with<F>(
    tokenizer: &Tokenizer,
    prompt: &str,
    add_bos: bool,
    mut sampler: Sampler,
    stopping: &StoppingCriteria,
    mut logits_fn: F,
) -> Result<Vec<u32>>
where
    F: FnMut(&[u32]) -> Result<Vec<f32>>,
{
    let mut ids = tokenizer.encode_with_specials(prompt, add_bos, false)?;
    let start_len = ids.len();
    let mut generated: Vec<u32> = Vec::new();

    loop {
        if stopping.should_stop(&generated) {
            break;
        }
        let logits = logits_fn(&ids)?;
        if logits.is_empty() {
            return Err(anyhow!("logits_fn returned empty logits"));
        }
        let next = sampler.sample(&logits)? as u32;
        ids.push(next);
        generated.push(next);
        if let Some(eos) = stopping.eos_id {
            if next == eos {
                break;
            }
        }
        if let Some(mt) = stopping.max_tokens {
            if generated.len() >= mt {
                break;
            }
        }
        // Safety guard: avoid runaway if neither criterion present (tests should set at least one)
        if stopping.max_tokens.is_none() && stopping.eos_id.is_none() {
            if generated.len() > 4 * (start_len.max(1)) && generated.len() > 4096 {
                return Err(anyhow!(
                    "decode_loop_with: refusing to continue without stopping criteria"
                ));
            }
        }
    }
    Ok(ids)
}

/// Convenience to build a greedy sampler (top_k=1) with a seed
pub fn greedy_sampler(seed: u64) -> Sampler {
    Sampler::new(SamplerConfig {
        top_k: Some(1),
        seed,
        ..Default::default()
    })
}
