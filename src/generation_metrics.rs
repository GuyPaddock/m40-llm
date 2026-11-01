//! Request-owned wall-clock measurements, excluding HTTP transport and admission queues.

use std::time::Instant;

use anyhow::{Context, Result};

use crate::decode::{StopReason, StoppingCriteria};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenerationMetrics {
    pub prompt_tokens: usize,
    /// Includes any sampled EOS/EOT token, even when it is omitted from output text.
    pub generated_tokens: usize,
    pub stop_reason: StopReason,
    /// Generation admission through the first sampled token, including request setup.
    pub prefill_elapsed_ms: f64,
    /// First through last sampled token; zero when fewer than two tokens are generated.
    pub decode_elapsed_ms: f64,
    pub total_elapsed_ms: f64,
}

pub(crate) struct GenerationTimer {
    started_at: Instant,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
}

impl GenerationTimer {
    pub(crate) fn new(started_at: Instant) -> Self {
        Self {
            started_at,
            first_token_at: None,
            last_token_at: None,
        }
    }

    pub(crate) fn record_token(&mut self) {
        let now = Instant::now();
        self.first_token_at.get_or_insert(now);
        self.last_token_at = Some(now);
    }

    pub(crate) fn finish(
        &self,
        prompt_tokens: usize,
        generated: &[u32],
        stopping: &StoppingCriteria,
    ) -> Result<GenerationMetrics> {
        let stop_reason = stopping
            .stop_reason(generated)
            .context("cannot report completed generation metrics before stopping")?;
        let prefill_elapsed_ms = self.first_token_at.map_or(0.0, |first| {
            (first - self.started_at).as_secs_f64() * 1000.0
        });
        let decode_elapsed_ms = self
            .first_token_at
            .zip(self.last_token_at)
            .map_or(0.0, |(first, last)| (last - first).as_secs_f64() * 1000.0);
        Ok(GenerationMetrics {
            prompt_tokens,
            generated_tokens: generated.len(),
            stop_reason,
            prefill_elapsed_ms,
            decode_elapsed_ms,
            total_elapsed_ms: self.started_at.elapsed().as_secs_f64() * 1000.0,
        })
    }
}
