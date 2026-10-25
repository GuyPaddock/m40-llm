#![cfg_attr(not(all(feature = "cuda", feature = "server", nvcc)), allow(dead_code))]

use anyhow::{bail, ensure, Context, Result};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;

pub const USER_PROMPT: &str = "Explain everything that happens between the time a user types a web address into their browser and the time the page is fully loaded and displayed on their screen in as much detail as possible";
pub const REPLACEMENT_PROMPT: &str = "Orchard inventory: seven copper baskets.";
pub const MAX_TOKENS: usize = 512;
pub const CONTEXT_TOKENS: u32 = 1024;

pub fn render_prompt() -> String {
    format!(
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{USER_PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    )
}

pub fn validate_prompt_ids(expected: &[u32], actual: &[u32]) -> Result<()> {
    ensure!(!expected.is_empty(), "empty benchmark prompt");
    ensure!(expected == actual, "tokenizer oracle prompt IDs differ");
    Ok(())
}

pub fn validate_replacement_ids(prompt: &[u32], replacement: &[u32]) -> Result<()> {
    ensure!(
        !prompt.is_empty() && !replacement.is_empty() && prompt[0] != replacement[0],
        "replacement prompt must have zero common token prefix, including special tokens"
    );
    Ok(())
}

pub fn discover_runner_url(log: &str) -> Result<String> {
    let port = log
        .lines()
        .rev()
        .find_map(|line| {
            line.contains("starting llama-server")
                .then(|| parse_count_after(line, "--port "))
                .flatten()
        })
        .context("no live llama-server port in Ollama log")?;
    ensure!((1..=65535).contains(&port), "invalid native runner port");
    Ok(format!("http://127.0.0.1:{port}"))
}

#[derive(Debug, Serialize)]
pub struct CacheEvidence {
    pub reused_prompt_tokens: usize,
    pub evaluated_prompt_tokens: usize,
    pub runner_lines: Vec<String>,
}

fn parse_count_after(line: &str, marker: &str) -> Option<usize> {
    line.split_once(marker)?
        .1
        .trim_start()
        .split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

pub fn verify_cache_reset(log: &str, prompt_tokens: usize) -> Result<CacheEvidence> {
    let cached_lines: Vec<&str> = log
        .lines()
        .filter(|line| line.contains("cached n_tokens ="))
        .collect();
    ensure!(
        cached_lines.len() == 1,
        "expected exactly one runner cache record, got {}",
        cached_lines.len()
    );
    let reused = parse_count_after(cached_lines[0], "cached n_tokens =")
        .context("invalid cached-token count")?;
    ensure!(reused == 0, "Ollama reused {reused} prompt KV tokens");
    let prompt_lines: Vec<&str> = log
        .lines()
        .filter(|line| line.contains("prompt eval time ="))
        .collect();
    ensure!(
        prompt_lines.len() == 1,
        "missing or ambiguous actual prompt evaluation record"
    );
    let evaluated = prompt_lines[0]
        .split_once(" / ")
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .and_then(|count| count.parse::<usize>().ok())
        .context("invalid evaluated-token count")?;
    ensure!(
        evaluated == prompt_tokens,
        "runner evaluated {evaluated} tokens; expected {prompt_tokens}"
    );
    Ok(CacheEvidence {
        reused_prompt_tokens: reused,
        evaluated_prompt_tokens: evaluated,
        runner_lines: vec![cached_lines[0].to_owned(), prompt_lines[0].to_owned()],
    })
}

pub fn validate_fixed_workload(
    prompt_tokens: usize,
    generated_tokens: usize,
    stop: &str,
    expected_prompt: usize,
) -> Result<()> {
    ensure!(
        prompt_tokens == expected_prompt,
        "prompt token count differs"
    );
    ensure!(
        generated_tokens == MAX_TOKENS,
        "generated {generated_tokens}, expected {MAX_TOKENS}"
    );
    ensure!(
        stop == "length",
        "natural stop {stop:?} does not qualify for fixed-workload timing"
    );
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct Pair {
    pub control_tps: f64,
    pub candidate_tps: f64,
}

#[derive(Debug, Serialize)]
pub struct Distribution {
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub mad_over_median: f64,
}

pub fn distribution(values: &[f64]) -> Result<Distribution> {
    ensure!(!values.is_empty(), "no observations");
    ensure!(
        values.iter().all(|v| v.is_finite() && *v > 0.0),
        "invalid throughput observation"
    );
    let median = median(values);
    let deviations: Vec<f64> = values.iter().map(|value| (value - median).abs()).collect();
    Ok(Distribution {
        median,
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mad_over_median: self::median(&deviations) / median,
    })
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    }
}

#[derive(Debug, Serialize)]
pub struct Comparison {
    pub pairs: usize,
    pub control: Distribution,
    pub candidate: Distribution,
    pub gain: f64,
    pub promotion_threshold: f64,
    pub bootstrap_gain_ci95: [f64; 2],
    pub bootstrap_resamples: usize,
    pub bootstrap_seed: u64,
    pub needs_variation_extension: bool,
    pub needs_fresh_confirmation: bool,
    pub passes_promotion_gate: bool,
    pub reaches_original_target: bool,
}

pub fn compare_pairs(pairs: &[Pair]) -> Result<Comparison> {
    ensure!(pairs.len() >= 5, "need at least five contemporaneous pairs");
    let control_values: Vec<f64> = pairs.iter().map(|p| p.control_tps).collect();
    let candidate_values: Vec<f64> = pairs.iter().map(|p| p.candidate_tps).collect();
    let control = distribution(&control_values)?;
    let candidate = distribution(&candidate_values)?;
    let gain = candidate.median / control.median - 1.0;
    let variation = control.mad_over_median.max(candidate.mad_over_median);
    let promotion_threshold = 0.03_f64.max(2.0 * variation);
    let mut rng = StdRng::seed_from_u64(0);
    let mut gains = Vec::with_capacity(10_000);
    let mut sampled_control = Vec::with_capacity(pairs.len());
    let mut sampled_candidate = Vec::with_capacity(pairs.len());
    for _ in 0..10_000 {
        sampled_control.clear();
        sampled_candidate.clear();
        for _ in pairs {
            let pair = pairs[rng.gen_range(0..pairs.len())];
            sampled_control.push(pair.control_tps);
            sampled_candidate.push(pair.candidate_tps);
        }
        gains.push(median(&sampled_candidate) / median(&sampled_control) - 1.0);
    }
    gains.sort_by(f64::total_cmp);
    let ci = [gains[249], gains[9749]];
    let needs_fresh_confirmation =
        (gain - promotion_threshold).abs() <= 0.01 || (gain >= promotion_threshold && ci[0] <= 0.0);
    Ok(Comparison {
        pairs: pairs.len(),
        reaches_original_target: candidate.median >= 53.06
            && candidate.median >= 1.30 * control.median,
        control,
        candidate,
        gain,
        promotion_threshold,
        bootstrap_gain_ci95: ci,
        bootstrap_resamples: 10_000,
        bootstrap_seed: 0,
        needs_variation_extension: variation > 0.05,
        needs_fresh_confirmation,
        passes_promotion_gate: gain >= promotion_threshold && ci[0] > 0.0 && variation <= 0.05,
    })
}

pub fn next_trial_count(comparison: &Comparison, is_confirmation: bool) -> Result<Option<usize>> {
    if is_confirmation {
        ensure!(
            comparison.pairs == 20,
            "confirmation must contain exactly twenty fresh pairs"
        );
        return Ok(None);
    }
    if comparison.needs_variation_extension {
        if comparison.pairs == 5 {
            return Ok(Some(10));
        }
        bail!("timing remains unstable after ten pairs; investigate before making claims");
    }
    Ok(comparison.needs_fresh_confirmation.then_some(20))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Match actual token IDs rather than accepting equal prompt lengths.
    #[test]
    fn test_prompt_identity_rejects_equal_length_mismatch() {
        assert!(validate_prompt_ids(&[1, 2], &[1, 3]).is_err());
        assert!(validate_prompt_ids(&[1, 2], &[1, 2]).is_ok());
        assert!(validate_replacement_ids(&[1, 2], &[1, 99]).is_err());
        assert!(validate_replacement_ids(&[1, 2], &[3]).is_ok());
    }

    /// Resolve the current native runner rather than using a stale prior port.
    #[test]
    fn test_runner_discovery_uses_latest_start() -> Result<()> {
        let log = "starting llama-server --port 1234 --host 127.0.0.1\nstarting llama-server --port 5678 --host 127.0.0.1";
        assert_eq!(discover_runner_url(log)?, "http://127.0.0.1:5678");
        assert!(discover_runner_url("unrelated --port 5678").is_err());
        assert!(discover_runner_url("starting llama-server --port 99999").is_err());
        Ok(())
    }

    /// Require both zero cached KV and complete actual runner prompt evaluation.
    #[test]
    fn test_cache_reset_requires_runner_evidence() -> Result<()> {
        let log = "slot cached n_tokens = 0, memory_seq_rm [0, end)\nslot prompt eval time = 120.00 ms / 65 tokens";
        let evidence = verify_cache_reset(log, 65)?;
        assert_eq!(evidence.evaluated_prompt_tokens, 65);
        assert_eq!(evidence.reused_prompt_tokens, 0);
        assert!(verify_cache_reset(&log.replace("= 0,", "= 4,"), 65).is_err());
        assert!(verify_cache_reset(log, 44).is_err());
        assert!(verify_cache_reset("prompt_eval_count=65", 65).is_err());
        assert!(verify_cache_reset(&format!("{log}\n{log}"), 65).is_err());
        Ok(())
    }

    /// Keep early EOS and unequal actual work out of the 512-token speed claim.
    #[test]
    fn test_fixed_workload_counts_and_stops() {
        assert!(validate_fixed_workload(65, 512, "length", 65).is_ok());
        assert!(validate_fixed_workload(65, 511, "stop", 65).is_err());
        assert!(validate_fixed_workload(65, 512, "stop", 65).is_err());
        assert!(validate_fixed_workload(44, 512, "length", 65).is_err());
    }

    /// Report median absolute deviation without excluding slow observations.
    #[test]
    fn test_distribution_preserves_outliers() -> Result<()> {
        let result = distribution(&[10.0, 20.0, 30.0, 40.0, 100.0])?;
        assert_eq!(result.median, 30.0);
        assert_eq!(result.min, 10.0);
        assert_eq!(result.max, 100.0);
        assert_eq!(result.mad_over_median, 10.0 / 30.0);
        assert!(distribution(&[f64::NAN]).is_err());
        Ok(())
    }

    /// Preserve contemporaneous pairs and make bootstrap decisions reproducible.
    #[test]
    fn test_paired_bootstrap_and_borderline_confirmation() -> Result<()> {
        let pairs: Vec<Pair> = [10.0, 20.0, 30.0, 40.0, 50.0]
            .into_iter()
            .map(|control| Pair {
                control_tps: control,
                candidate_tps: control * 1.04,
            })
            .collect();
        let first = compare_pairs(&pairs)?;
        let repeated = compare_pairs(&pairs)?;
        assert_eq!(first.bootstrap_gain_ci95, repeated.bootstrap_gain_ci95);
        assert!((first.bootstrap_gain_ci95[0] - 0.04).abs() < 1e-12);
        assert!((first.bootstrap_gain_ci95[1] - 0.04).abs() < 1e-12);
        let borderline = compare_pairs(
            &[Pair {
                control_tps: 40.0,
                candidate_tps: 41.4,
            }; 5],
        )?;
        assert_eq!(next_trial_count(&borderline, false)?, Some(20));
        assert!(next_trial_count(&borderline, true).is_err());
        let confirmation = compare_pairs(
            &[Pair {
                control_tps: 40.0,
                candidate_tps: 41.4,
            }; 20],
        )?;
        assert_eq!(next_trial_count(&confirmation, true)?, None);
        Ok(())
    }

    /// Extend noisy screening once, then reject unresolved instability.
    #[test]
    fn test_variation_extension_is_bounded() -> Result<()> {
        let pairs: Vec<Pair> = [10.0, 20.0, 30.0, 40.0, 50.0]
            .into_iter()
            .map(|value| Pair {
                control_tps: value,
                candidate_tps: value * 2.0,
            })
            .collect();
        assert_eq!(next_trial_count(&compare_pairs(&pairs)?, false)?, Some(10));
        let doubled: Vec<Pair> = pairs.iter().chain(pairs.iter()).copied().collect();
        assert!(next_trial_count(&compare_pairs(&doubled)?, false).is_err());
        Ok(())
    }
}
