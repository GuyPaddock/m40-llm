use m40_llm::sampling::{Sampler, SamplerConfig};

#[test]
fn greedy_picks_max_integration() {
    let s = Sampler::new(SamplerConfig::default());
    let logits = [0.1, -0.2, 5.0, 1.0];
    assert_eq!(s.sample_greedy(&logits).unwrap(), 2);
}

#[test]
fn top_k_limits_candidates_integration() {
    let mut s = Sampler::new(SamplerConfig {
        top_k: Some(1),
        ..Default::default()
    });
    let logits = [0.0, 5.0, 1.0];
    for _ in 0..50 {
        let idx = s.sample(&logits).unwrap();
        assert_eq!(idx, 1);
    }
}

#[test]
fn top_p_truncates_integration() {
    let mut s = Sampler::new(SamplerConfig {
        top_p: Some(0.60),
        seed: 123,
        ..Default::default()
    });
    let logits = [0.0, 1.0, 2.0];
    for _ in 0..50 {
        let idx = s.sample(&logits).unwrap();
        assert!(idx == 1 || idx == 2);
    }
}

#[test]
fn deterministic_given_seed() {
    let logits = [0.5f32, 0.1, 0.4];
    let mut s1 = Sampler::new(SamplerConfig {
        temperature: 0.8,
        top_k: Some(2),
        top_p: Some(0.9),
        seed: 999,
    });
    let mut s2 = Sampler::new(SamplerConfig {
        temperature: 0.8,
        top_k: Some(2),
        top_p: Some(0.9),
        seed: 999,
    });
    let mut seq1 = Vec::new();
    let mut seq2 = Vec::new();
    for _ in 0..10 {
        seq1.push(s1.sample(&logits).unwrap());
        seq2.push(s2.sample(&logits).unwrap());
    }
    assert_eq!(seq1, seq2);
}
