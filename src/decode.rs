// src/decode.rs
//! Decode loop helpers: stopping criteria for EOS and max_tokens.

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
