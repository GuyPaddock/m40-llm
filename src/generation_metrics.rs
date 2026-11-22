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
    /// Diagnostic request-relative sample times, including any sampled stop token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_elapsed_ms: Option<Vec<f64>>,
}

pub(crate) struct GenerationTimer {
    started_at: Instant,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    token_elapsed_ms: Option<Vec<f64>>,
}

impl GenerationTimer {
    pub(crate) fn new(started_at: Instant) -> Self {
        Self {
            started_at,
            first_token_at: None,
            last_token_at: None,
            token_elapsed_ms: (std::env::var("M40LLM_RESEARCH_TOKEN_TIMINGS").as_deref()
                == Ok("1"))
            .then(Vec::new),
        }
    }

    pub(crate) fn record_token(&mut self) {
        let now = Instant::now();
        self.first_token_at.get_or_insert(now);
        self.last_token_at = Some(now);
        if let Some(samples) = &mut self.token_elapsed_ms {
            samples.push((now - self.started_at).as_secs_f64() * 1000.0);
        }
    }

    pub(crate) fn finish(
        &mut self,
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
            token_elapsed_ms: self.token_elapsed_ms.take(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Captured offsets agree with prefill/decode boundaries and include EOS.
    #[test]
    fn sampled_offsets_include_the_stop_token() -> Result<()> {
        let mut timer = GenerationTimer::new(Instant::now());
        timer.token_elapsed_ms = Some(Vec::new());
        timer.record_token();
        timer.record_token();
        let metrics = timer.finish(3, &[7, 9], &StoppingCriteria::new(None, Some(9)))?;
        let samples = metrics.token_elapsed_ms.as_ref().unwrap();
        assert_eq!(samples.len(), metrics.generated_tokens);
        assert_eq!(samples[0], metrics.prefill_elapsed_ms);
        assert!(samples[1] >= samples[0]);
        assert!((samples[1] - samples[0] - metrics.decode_elapsed_ms).abs() < 1e-6);
        Ok(())
    }

    /// Older reports and timing-disabled runs omit the optional diagnostic.
    #[test]
    fn metrics_round_trip_without_sample_offsets() -> Result<()> {
        let mut timer = GenerationTimer::new(Instant::now());
        timer.token_elapsed_ms = None;
        timer.record_token();
        let metrics = timer.finish(1, &[7], &StoppingCriteria::new(Some(1), None))?;
        let wire = serde_json::to_value(metrics)?;
        assert!(wire.get("token_elapsed_ms").is_none());
        let restored: GenerationMetrics = serde_json::from_value(wire)?;
        assert_eq!(restored.generated_tokens, 1);
        assert!(restored.token_elapsed_ms.is_none());
        Ok(())
    }
}
