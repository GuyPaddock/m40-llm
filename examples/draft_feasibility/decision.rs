use anyhow::{ensure, Result};
use serde::Serialize;

/// Median hardware costs indexed by accepted guesses, excluding the known seed.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct DecisionCosts {
    pub serial_ms: [f64; 4],
    pub candidate_ms: [f64; 4],
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Decision {
    pub should_attempt: bool,
    pub reason: &'static str,
    pub histogram: [usize; 4],
    pub last_attempt_position: Option<usize>,
    pub full_acceptance_gain: Option<f64>,
    pub expected_gain: Option<f64>,
}

/// Owns only outcomes of executed attempts in this request, never skipped labels.
#[derive(Default)]
pub struct Controller {
    histogram: [usize; 4],
    last_attempt_position: Option<usize>,
}

impl Controller {
    pub fn decide(
        &self,
        position: usize,
        remaining: usize,
        has_proposal: bool,
        costs: Option<&DecisionCosts>,
    ) -> Decision {
        let mut result = Decision {
            should_attempt: false,
            reason: "budget",
            histogram: self.histogram,
            last_attempt_position: self.last_attempt_position,
            full_acceptance_gain: None,
            expected_gain: None,
        };
        if remaining < 4 {
            return result;
        }
        if !has_proposal {
            result.reason = "source_miss";
            return result;
        }
        let Some(costs) = costs.filter(|_| (65..=569).contains(&position)) else {
            result.reason = "unmeasured_position";
            return result;
        };
        let full_gain = costs.serial_ms[3] / (costs.candidate_ms[3] + 0.1) - 1.;
        result.full_acceptance_gain = Some(full_gain);
        if full_gain < 0.03 {
            result.reason = "hardware_ceiling";
            return result;
        }
        let count: usize = self.histogram.iter().sum();
        if count == 0 {
            result.should_attempt = true;
            result.reason = "first_probe";
            return result;
        }
        let weighted = |values: &[f64; 4]| {
            values
                .iter()
                .zip(self.histogram)
                .map(|(value, count)| value * count as f64)
                .sum::<f64>()
                / count as f64
        };
        let gain = weighted(&costs.serial_ms) / (weighted(&costs.candidate_ms) + 0.1) - 1.;
        result.expected_gain = Some(gain);
        if gain >= 0.03 {
            result.should_attempt = true;
            result.reason = "expected_gain";
        } else if self
            .last_attempt_position
            .is_some_and(|last| position.saturating_sub(last) >= 32)
        {
            result.should_attempt = true;
            result.reason = "recovery_probe";
        } else {
            result.reason = "expected_loss";
        }
        result
    }

    pub fn observe(&mut self, position: usize, accepted_guesses: usize) -> Result<()> {
        ensure!(accepted_guesses < 4, "known seed is not a guess");
        ensure!(
            self.last_attempt_position
                .is_none_or(|last| position > last),
            "attempt positions must increase"
        );
        self.histogram[accepted_guesses] += 1;
        self.last_attempt_position = Some(position);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn costs() -> DecisionCosts {
        DecisionCosts {
            serial_ms: [40., 60., 80., 80.],
            candidate_ms: [80., 80., 80., 60.],
        }
    }

    /// A rejected probe disables normal attempts; only 32 committed rows allow
    /// another probe. Misses cannot advance the attempt clock or teach outcomes.
    #[test]
    fn test_rejection_and_recovery_follow_committed_positions() -> Result<()> {
        let mut controller = Controller::default();
        assert!(
            controller
                .decide(65, 100, true, Some(&costs()))
                .should_attempt
        );
        controller.observe(65, 0)?;
        assert!(
            !controller
                .decide(96, 100, true, Some(&costs()))
                .should_attempt
        );
        assert!(
            !controller
                .decide(97, 100, false, Some(&costs()))
                .should_attempt
        );
        let recovery = controller.decide(97, 100, true, Some(&costs()));
        assert!(recovery.should_attempt);
        assert_eq!(recovery.reason, "recovery_probe");
        assert_eq!(recovery.histogram, [1, 0, 0, 0]);
        assert!(controller.observe(65, 3).is_err());
        assert!(controller.observe(97, 4).is_err());
        Ok(())
    }

    /// Perfect acceptance is useful only when measured verification is cheap;
    /// no amount of favorable history admits unmeasured positions or short tails.
    #[test]
    fn test_hardware_costs_and_bounds_override_acceptance() -> Result<()> {
        let mut controller = Controller::default();
        controller.observe(65, 3)?;
        assert_eq!(
            controller.decide(69, 100, true, Some(&costs())).reason,
            "expected_gain"
        );
        let expensive = DecisionCosts {
            candidate_ms: [100.; 4],
            ..costs()
        };
        assert!(
            !controller
                .decide(97, 100, true, Some(&expensive))
                .should_attempt
        );
        for (position, remaining) in [(64, 10), (570, 10), (69, 3)] {
            assert!(
                !controller
                    .decide(position, remaining, true, Some(&costs()))
                    .should_attempt
            );
        }
        Ok(())
    }
}
