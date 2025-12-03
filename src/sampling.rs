//! Host-side sampling: softmax with temperature, greedy, top-k, top-p (nucleus) sampling.
//! Deterministic RNG via xorshift64* with seed.

use anyhow::{anyhow, Result};

#[derive(Debug, Clone, Copy)]
pub struct SamplerConfig {
    pub temperature: f32, // 1.0 => pure softmax
    pub top_k: Option<usize>,
    pub top_p: Option<f32>, // 0<p<=1
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sampler {
    cfg: SamplerConfig,
    rng: XorShift64Star,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        let seed = if cfg.seed == 0 { 1 } else { cfg.seed };
        Self {
            cfg,
            rng: XorShift64Star::new(seed),
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<usize> {
        if logits.is_empty() {
            return Err(anyhow!("empty logits"));
        }
        // 1) Temperature adjust + stable softmax
        let probs = softmax_temp(logits, self.cfg.temperature);
        // 2) Apply top-k / top-p filtering into a working buffer of (idx, prob)
        let mut items: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        // Sanitize NaN/Inf probabilities to avoid panics
        for (_, pr) in &mut items {
            if !pr.is_finite() || *pr < 0.0 {
                *pr = 0.0;
            }
        }
        let mut nonzero = items.iter().any(|&(_, p)| p > 0.0);
        if let Some(k) = self.cfg.top_k {
            if k > 0 && k < items.len() {
                items.select_nth_unstable_by(k, |a, b| f32::total_cmp(&b.1, &a.1));
                items.truncate(k);
                nonzero = items.iter().any(|&(_, p)| p > 0.0);
            }
        }
        if let Some(p) = self.cfg.top_p {
            items.sort_by(|a, b| f32::total_cmp(&b.1, &a.1));
            let mut cum = 0.0f32;
            let mut keep = 0usize;
            for &(_, pr) in &items {
                cum += pr;
                keep += 1;
                if cum >= p {
                    break;
                }
            }
            items.truncate(keep.max(1));
            nonzero = items.iter().any(|&(_, p)| p > 0.0);
        }
        // If everything is zero (underflow or NaNs), fall back to greedy on logits
        if !nonzero {
            return self.sample_greedy(logits);
        }
        // 3) Normalize remaining and draw
        let sum: f32 = items.iter().map(|&(_, p)| p).sum();
        let mut r = self.rng.next_f32() * sum.max(f32::EPSILON);
        for (idx, p) in items {
            if r <= p {
                return Ok(idx);
            }
            r -= p;
        }
        Ok(logits.len() - 1)
    }

    pub fn sample_greedy(&self, logits: &[f32]) -> Result<usize> {
        if logits.is_empty() {
            return Err(anyhow!("empty logits"));
        }
        let mut max_i = 0usize;
        let mut max_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > max_v {
                max_v = v;
                max_i = i;
            }
        }
        Ok(max_i)
    }
}

fn softmax_temp(logits: &[f32], temperature: f32) -> Vec<f32> {
    let t = if temperature <= 0.0 {
        1e-6
    } else {
        temperature
    };
    let inv_t = 1.0f32 / t;
    let mut max_logit = f32::NEG_INFINITY;
    for &v in logits {
        max_logit = max_logit.max(v * inv_t);
    }
    let mut exps = Vec::with_capacity(logits.len());
    for &v in logits {
        let x = ((v * inv_t) - max_logit).exp();
        if x.is_finite() && x > 0.0 {
            exps.push(x);
        } else {
            exps.push(0.0);
        }
    }
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|x| x / sum).collect()
}

#[derive(Debug, Clone)]
struct XorShift64Star {
    state: u64,
}

impl XorShift64Star {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f32(&mut self) -> f32 {
        let v = self.next_u64();
        let mant = (v >> 41) as u32; // 23 bits
        (mant as f32) / ((1u32 << 23) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        let s = Sampler::new(SamplerConfig::default());
        let logits = [0.1, -0.2, 5.0, 1.0];
        assert_eq!(s.sample_greedy(&logits).unwrap(), 2);
    }

    #[test]
    fn softmax_temperature_effect() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: 0.5,
            ..Default::default()
        });
        let logits = [0.0, 1.0];
        // With low temperature, index 1 should be selected more often than 0
        let mut c1 = 0;
        for _ in 0..1000 {
            let idx = s.sample(&logits).unwrap();
            if idx == 1 {
                c1 += 1;
            }
        }
        assert!(c1 > 600, "idx1 should be sampled majority, got {}", c1);
    }

    #[test]
    fn top_k_limits_candidates() {
        let mut s = Sampler::new(SamplerConfig {
            top_k: Some(1),
            ..Default::default()
        });
        let logits = [0.0, 5.0, 1.0];
        // Only the max logit survives top-k=1, so always picks idx 1
        for _ in 0..100 {
            let idx = s.sample(&logits).unwrap();
            assert_eq!(idx, 1);
        }
    }

    #[test]
    fn top_p_truncates_cumulative() {
        let mut s = Sampler::new(SamplerConfig {
            top_p: Some(0.60),
            seed: 123,
            ..Default::default()
        });
        let logits = [0.0, 1.0, 2.0];
        // Expect only the top 2 to remain after top_p ~ 0.60 (softmax on [0,1,2])
        for _ in 0..100 {
            let idx = s.sample(&logits).unwrap();
            assert!(idx == 1 || idx == 2);
        }
    }
}
