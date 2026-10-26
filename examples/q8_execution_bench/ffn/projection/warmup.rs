//! Bounded untimed projection warmup, independent of primary pair settings.
use anyhow::{ensure, Result};
use serde_json::{json, Value};

/// Reach the requested accumulated GPU duration using finite positive samples.
/// A sample is average per-call GPU milliseconds for the supplied batch size.
pub(super) fn measure(
    minimum_ms: f64,
    mut sample: impl FnMut(usize) -> Result<f64>,
) -> Result<Value> {
    ensure!(
        minimum_ms.is_finite() && minimum_ms > 0.0,
        "invalid projection warmup minimum"
    );
    let mut elapsed = 0.0;
    for batch in 1..=256 {
        let gpu_ms = sample(256)?;
        ensure!(
            gpu_ms.is_finite() && gpu_ms > 0.0,
            "invalid projection warmup sample"
        );
        elapsed += gpu_ms * 256.0;
        ensure!(elapsed.is_finite(), "projection warmup duration overflow");
        if elapsed >= minimum_ms {
            return Ok(json!({"gpu_ms":elapsed,"minimum_gpu_ms":minimum_ms,
                "batches":batch,"iterations_per_batch":256,"maximum_batches":256,
                "scope":"untimed warmup, never pooled with primary pairs"}));
        }
    }
    anyhow::bail!("projection bounded warmup exhausted")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 27-us packed control must reach one second before the bounded budget
    /// expires; its actual accumulated duration and iteration count agree.
    #[test]
    fn test_fast_projection_reaches_minimum_without_exhausting_budget() -> Result<()> {
        let mut iterations = 0;
        let mut batches = 0;
        let result = measure(1000.0, |count| {
            iterations += count;
            batches += 1;
            Ok(0.027)
        })?;
        let elapsed = result["gpu_ms"].as_f64().unwrap();
        assert!(elapsed >= 1000.0 && elapsed < 1000.0 + 256.0 * 0.027);
        assert!((elapsed - iterations as f64 * 0.027).abs() < 1e-8);
        assert_eq!(result["batches"], batches);
        assert_eq!(
            iterations as u64,
            batches as u64 * result["iterations_per_batch"].as_u64().unwrap()
        );
        assert!(batches as u64 <= result["maximum_batches"].as_u64().unwrap());
        Ok(())
    }

    /// Invalid targets do not call the sampler; invalid measurements and an
    /// unattainable duration fail rather than returning a fake warm result.
    #[test]
    fn test_warmup_rejects_invalid_samples_and_remains_bounded() {
        for minimum in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut calls = 0;
            assert!(measure(minimum, |_| {
                calls += 1;
                Ok(0.027)
            })
            .is_err());
            assert_eq!(calls, 0);
        }
        for gpu_ms in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX] {
            let mut calls = 0;
            assert!(measure(1000.0, |_| {
                calls += 1;
                Ok(gpu_ms)
            })
            .is_err());
            assert_eq!(calls, 1);
        }
        let mut calls = 0;
        assert!(measure(1000.0, |_| {
            calls += 1;
            Ok(1e-12)
        })
        .is_err());
        assert_eq!(calls, 256);
    }
}
