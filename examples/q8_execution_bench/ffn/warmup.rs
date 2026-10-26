//! Bounded untimed operation warmup, independent of primary pair settings.
use anyhow::{ensure, Result};
use serde_json::{json, Value};

/// Reach the requested accumulated GPU duration using finite positive samples.
/// A sample is average per-call GPU milliseconds for the supplied batch size.
pub(super) fn measure(minimum_ms: f64, sample: impl FnMut(usize) -> Result<f64>) -> Result<Value> {
    measure_with_budget(minimum_ms, 256, 256, sample)
}

/// Choose a finite batch budget for short operations without changing the
/// required GPU warmup duration or the primary timing interval.
pub(super) fn measure_with_budget(
    minimum_ms: f64,
    iterations_per_batch: usize,
    maximum_batches: usize,
    mut sample: impl FnMut(usize) -> Result<f64>,
) -> Result<Value> {
    ensure!(
        minimum_ms.is_finite()
            && minimum_ms > 0.0
            && iterations_per_batch > 0
            && maximum_batches > 0,
        "invalid operation warmup target or budget"
    );
    let mut elapsed = 0.0;
    for batch in 1..=maximum_batches {
        let gpu_ms = sample(iterations_per_batch)?;
        ensure!(
            gpu_ms.is_finite() && gpu_ms > 0.0,
            "invalid operation warmup sample"
        );
        elapsed += gpu_ms * iterations_per_batch as f64;
        ensure!(elapsed.is_finite(), "operation warmup duration overflow");
        if elapsed >= minimum_ms {
            return Ok(json!({"gpu_ms":elapsed,"minimum_gpu_ms":minimum_ms,
                "batches":batch,"iterations_per_batch":iterations_per_batch,"maximum_batches":maximum_batches,
                "scope":"untimed warmup, never pooled with primary pairs"}));
        }
    }
    anyhow::bail!("operation bounded warmup exhausted")
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

    /// A two-microsecond reduction reaches the same one-second target with
    /// a larger untimed batch; insufficient and empty budgets fail explicitly.
    #[test]
    fn test_short_reduction_uses_explicit_finite_warmup_budget() -> Result<()> {
        let mut iterations = 0;
        let result = measure_with_budget(1000.0, 4096, 256, |count| {
            iterations += count;
            Ok(0.002)
        })?;
        let elapsed = result["gpu_ms"].as_f64().unwrap();
        assert!(elapsed >= 1000.0 && elapsed < 1000.0 + 4096.0 * 0.002);
        assert!((elapsed - iterations as f64 * 0.002).abs() < 1e-8);
        assert_eq!(result["iterations_per_batch"], 4096);
        assert_eq!(result["maximum_batches"], 256);
        for (batch_size, limit) in [(0, 256), (4096, 0)] {
            let mut calls = 0;
            assert!(measure_with_budget(1000.0, batch_size, limit, |_| {
                calls += 1;
                Ok(0.002)
            })
            .is_err());
            assert_eq!(calls, 0);
        }
        let mut calls = 0;
        assert!(measure_with_budget(1000.0, 4096, 2, |_| {
            calls += 1;
            Ok(0.002)
        })
        .is_err());
        assert_eq!(calls, 2);
        Ok(())
    }
}
