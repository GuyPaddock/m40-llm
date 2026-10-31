//! Fixed research comparison arms. No environment or generation default.
use anyhow::{bail, Result};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Q8PortfolioArm {
    Control,
    Down,
    Vocabulary,
    Argmax,
    All,
}

impl Q8PortfolioArm {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "control" => Ok(Self::Control),
            "down" => Ok(Self::Down),
            "vocabulary" => Ok(Self::Vocabulary),
            "argmax" => Ok(Self::Argmax),
            "all" => Ok(Self::All),
            _ => bail!("unknown research portfolio arm {value:?}; use control, down, vocabulary, argmax or all"),
        }
    }
    pub fn has_packet_down(self) -> bool {
        matches!(self, Self::Down | Self::All)
    }
    pub fn has_packet_vocabulary(self) -> bool {
        matches!(self, Self::Vocabulary | Self::All)
    }
    pub fn has_partitioned_argmax(self) -> bool {
        matches!(self, Self::Argmax | Self::All)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Q8PortfolioTensorStats {
    pub key: super::Q8WarpPacketKey,
    pub is_vocabulary: bool,
    pub bytes: usize,
    pub plane_sha256: [String; 2],
}

#[derive(Debug, Clone, Serialize)]
pub struct Q8PortfolioStats {
    pub arm: Q8PortfolioArm,
    pub tensors: Vec<Q8PortfolioTensorStats>,
    pub resident_bytes: usize,
    pub original_prepared_bytes: usize,
    pub combined_prepared_bytes: usize,
    pub budget_bytes: usize,
    pub peak_temporary_host_bytes: usize,
    pub packing_and_validation_ms: f64,
    pub upload_ms: f64,
    pub preparation_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixed arms change only their named consumer, with no implicit all-on
    /// alias or unknown-value fallback.
    #[test]
    fn test_portfolio_arms_are_explicit_and_disjoint() -> Result<()> {
        for (name, mask) in [
            ("control", [false, false, false]),
            ("down", [true, false, false]),
            ("vocabulary", [false, true, false]),
            ("argmax", [false, false, true]),
            ("all", [true, true, true]),
        ] {
            let arm = Q8PortfolioArm::parse(name)?;
            assert_eq!(
                [
                    arm.has_packet_down(),
                    arm.has_packet_vocabulary(),
                    arm.has_partitioned_argmax()
                ],
                mask
            );
        }
        assert!(Q8PortfolioArm::parse("off").is_err());
        assert!(Q8PortfolioArm::parse("").is_err());
        Ok(())
    }
}
