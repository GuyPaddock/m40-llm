use anyhow::{ensure, Context, Result};
use serde::{Serialize, Serializer};

use super::{Q8ExecutionKey, Q8ExecutionLayout};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Q8ExecutionMode {
    Off,
    Split,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Q8ProjectionGroup {
    Qkv,
    Output,
    GateUp,
    Down,
    Vocabulary,
}

impl Q8ProjectionGroup {
    const ALL: [Self; 5] = [
        Self::Qkv,
        Self::Output,
        Self::GateUp,
        Self::Down,
        Self::Vocabulary,
    ];
    fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// Immutable selection of projection groups to prepare, not a per-token switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q8ExecutionFamilies(u8);

impl Q8ExecutionFamilies {
    pub const fn all() -> Self {
        Self(0b11111)
    }
    pub const fn none() -> Self {
        Self(0)
    }

    pub fn contains(self, group: Q8ProjectionGroup) -> bool {
        self.0 & group.bit() != 0
    }

    pub fn groups(self) -> Vec<Q8ProjectionGroup> {
        Q8ProjectionGroup::ALL
            .into_iter()
            .filter(|&group| self.contains(group))
            .collect()
    }

    pub fn parse(value: &str) -> Result<Self> {
        if value == "all" {
            return Ok(Self::all());
        }
        let mut families = Self::none();
        for item in value.split(',').map(str::trim) {
            let group = match item {
                "qkv" => Q8ProjectionGroup::Qkv,
                "output" => Q8ProjectionGroup::Output,
                "gate-up" => Q8ProjectionGroup::GateUp,
                "down" => Q8ProjectionGroup::Down,
                "vocabulary" => Q8ProjectionGroup::Vocabulary,
                _ => anyhow::bail!("invalid Q8 execution family {item:?}; use all or a comma-separated subset of qkv,output,gate-up,down,vocabulary"),
            };
            ensure!(
                !families.contains(group),
                "duplicate Q8 execution family: {item}"
            );
            families.0 |= group.bit();
        }
        Ok(families)
    }
}

impl Serialize for Q8ExecutionFamilies {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.groups().serialize(serializer)
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Q8ExecutionConfig {
    pub mode: Q8ExecutionMode,
    pub budget_bytes: usize,
    pub families: Q8ExecutionFamilies,
}

impl Q8ExecutionConfig {
    pub fn disabled() -> Self {
        Self {
            mode: Q8ExecutionMode::Off,
            budget_bytes: 0,
            families: Q8ExecutionFamilies::none(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let mode = match std::env::var("M40LLM_Q8_EXECUTION_WEIGHTS") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => "off".into(),
            Err(e) => return Err(e.into()),
        };
        if mode == "off" || mode == "0" {
            return Ok(Self::disabled());
        }
        let budget = match std::env::var("M40LLM_Q8_EXECUTION_BUDGET_MB") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(e) => return Err(e.into()),
        };
        let families = match std::env::var("M40LLM_Q8_EXECUTION_FAMILIES") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(e) => return Err(e.into()),
        };
        Self::parse(&mode, budget.as_deref(), families.as_deref())
    }

    pub fn parse(mode: &str, budget_mb: Option<&str>, families: Option<&str>) -> Result<Self> {
        let mode = match mode {
            "off" | "0" => return Ok(Self::disabled()),
            "split" => Q8ExecutionMode::Split,
            "mixed" => Q8ExecutionMode::Mixed,
            _ => anyhow::bail!("M40LLM_Q8_EXECUTION_WEIGHTS must be off, split, or mixed"),
        };
        let mb = budget_mb
            .unwrap_or("6144")
            .parse::<usize>()
            .context("invalid M40LLM_Q8_EXECUTION_BUDGET_MB")?;
        Ok(Self {
            mode,
            families: Q8ExecutionFamilies::parse(families.unwrap_or("all"))?,
            budget_bytes: mb
                .checked_mul(1024 * 1024)
                .context("Q8 execution budget overflow")?,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Q8PreparedTensorStats {
    pub key: Q8ExecutionKey,
    pub source_bytes: usize,
    pub resident_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Q8ExecutionStats {
    pub mode: Q8ExecutionMode,
    pub families: Q8ExecutionFamilies,
    pub has_prepared_vocabulary: bool,
    pub budget_bytes: usize,
    pub estimated_bytes: usize,
    pub resident_bytes: usize,
    pub source_weights_bytes: usize,
    pub total_weight_payload_bytes: usize,
    pub peak_setup_host_payload_bytes: usize,
    pub preparation_elapsed_ms: f64,
    pub model_hash_elapsed_ms: f64,
    pub tensors: Vec<Q8PreparedTensorStats>,
}

impl Q8ExecutionStats {
    pub(super) fn empty() -> Self {
        Self {
            mode: Q8ExecutionMode::Off,
            families: Q8ExecutionFamilies::none(),
            has_prepared_vocabulary: false,
            budget_bytes: 0,
            estimated_bytes: 0,
            resident_bytes: 0,
            source_weights_bytes: 0,
            total_weight_payload_bytes: 0,
            peak_setup_host_payload_bytes: 0,
            preparation_elapsed_ms: 0.0,
            model_hash_elapsed_ms: 0.0,
            tensors: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProjectionFamily {
    Query,
    Key,
    Value,
    Output,
    Gate,
    Up,
    Down,
    Vocabulary,
}

impl ProjectionFamily {
    pub(super) fn group(self) -> Q8ProjectionGroup {
        match self {
            Self::Query | Self::Key | Self::Value => Q8ProjectionGroup::Qkv,
            Self::Output => Q8ProjectionGroup::Output,
            Self::Gate | Self::Up => Q8ProjectionGroup::GateUp,
            Self::Down => Q8ProjectionGroup::Down,
            Self::Vocabulary => Q8ProjectionGroup::Vocabulary,
        }
    }
    pub(super) fn from_name(name: &str, lm_head_name: &str) -> Option<Self> {
        if name == lm_head_name {
            return Some(Self::Vocabulary);
        }
        let rest = name
            .strip_prefix("blk.")
            .or_else(|| name.strip_prefix("layers."))?;
        let (index, suffix) = rest.split_once('.')?;
        index.parse::<u32>().ok()?;
        match suffix {
            "attn_q.weight" | "attention.wq.weight" => Some(Self::Query),
            "attn_k.weight" | "attention.wk.weight" => Some(Self::Key),
            "attn_v.weight" | "attention.wv.weight" => Some(Self::Value),
            "attn_output.weight" | "attention.wo.weight" => Some(Self::Output),
            "ffn_gate.weight" | "feed_forward.w3.weight" => Some(Self::Gate),
            "ffn_up.weight" | "feed_forward.w1.weight" => Some(Self::Up),
            "ffn_down.weight" | "feed_forward.w2.weight" => Some(Self::Down),
            _ => None,
        }
    }

    pub(super) fn layouts(self, mode: Q8ExecutionMode) -> Vec<Q8ExecutionLayout> {
        if mode == Q8ExecutionMode::Off {
            return Vec::new();
        }
        let mut layouts = vec![Q8ExecutionLayout::SeparateScalesValues];
        if mode == Q8ExecutionMode::Mixed && matches!(self, Self::Query | Self::Output | Self::Down)
        {
            layouts.push(Q8ExecutionLayout::InputTiled { columns: 16 });
        }
        layouts
    }

    pub(super) fn split_columns(self) -> u32 {
        if matches!(self, Self::Key | Self::Value) {
            1
        } else {
            4
        }
    }
}

pub(super) fn check_budget(estimated: usize, budget: usize) -> Result<()> {
    ensure!(
        estimated <= budget,
        "Q8 execution preparation requires {estimated} bytes, budget={budget}; increase M40LLM_Q8_EXECUTION_BUDGET_MB or use M40LLM_Q8_EXECUTION_WEIGHTS=off"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-CUDA caller requesting preparation must get the configuration
    /// error before metadata parsing or any attempt to create device storage.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_non_cuda_loading_rejects_prepared_execution() {
        for mode in [Q8ExecutionMode::Split, Q8ExecutionMode::Mixed] {
            let error = crate::infer::LoadedModel::from_gguf_with_q8_execution(
                crate::gguf::GgufModel::new(0),
                Vec::new(),
                -1,
                Q8ExecutionConfig {
                    mode,
                    budget_bytes: 1024,
                    families: Q8ExecutionFamilies::all(),
                },
            )
            .err()
            .expect("non-CUDA preparation must be rejected");
            assert!(error.to_string().contains("require CUDA"));
        }
    }

    /// Invalid requests must fail rather than silently bypassing the prepared
    /// backend; off ignores the unused budget and stays allocation-free.
    #[test]
    fn test_execution_config_rejects_invalid_modes_and_budgets() {
        assert_eq!(
            Q8ExecutionConfig::parse("off", Some("bad"), Some("bad"))
                .unwrap()
                .mode,
            Q8ExecutionMode::Off
        );
        assert_eq!(
            Q8ExecutionConfig::parse("split", Some("1"), None)
                .unwrap()
                .budget_bytes,
            1024 * 1024
        );
        assert_eq!(
            Q8ExecutionConfig::parse("mixed", Some("0"), None)
                .unwrap()
                .budget_bytes,
            0
        );
        for (mode, budget) in [
            ("unknown", None),
            ("split", Some("bad")),
            ("mixed", Some("-1")),
        ] {
            assert!(Q8ExecutionConfig::parse(mode, budget, None).is_err());
        }
        assert!(Q8ExecutionConfig::parse("split", Some(&usize::MAX.to_string()), None).is_err());
        let config = Q8ExecutionConfig::parse("split", None, Some("qkv, down,vocabulary")).unwrap();
        assert!(config.families.contains(Q8ProjectionGroup::Down));
        assert!(!config.families.contains(Q8ProjectionGroup::GateUp));
        assert_eq!(
            serde_json::to_value(config.families).unwrap(),
            serde_json::json!(["qkv", "down", "vocabulary"])
        );
        for invalid in ["", "none", "all,down", "gate", "qkv,qkv", "down,"] {
            assert!(Q8ExecutionConfig::parse("split", None, Some(invalid)).is_err());
        }
    }

    /// Selection is per projection family: mixed only adds tiled Q/O/down and
    /// off must not allocate any layout. K/V retain the measured narrow launch.
    #[test]
    fn test_layout_policy_preserves_projection_families_and_bounds() {
        for (name, family) in [
            ("blk.0.attn_q.weight", ProjectionFamily::Query),
            ("layers.1.attention.wk.weight", ProjectionFamily::Key),
            ("blk.2.attn_v.weight", ProjectionFamily::Value),
            ("blk.0.attn_output.weight", ProjectionFamily::Output),
            ("layers.0.feed_forward.w3.weight", ProjectionFamily::Gate),
            ("blk.0.ffn_up.weight", ProjectionFamily::Up),
            ("layers.0.feed_forward.w2.weight", ProjectionFamily::Down),
            ("token_embd.weight", ProjectionFamily::Vocabulary),
        ] {
            assert_eq!(
                ProjectionFamily::from_name(name, "token_embd.weight"),
                Some(family)
            );
            assert!(family.layouts(Q8ExecutionMode::Off).is_empty());
            assert_eq!(
                family.layouts(Q8ExecutionMode::Split),
                [Q8ExecutionLayout::SeparateScalesValues]
            );
            let mixed = family.layouts(Q8ExecutionMode::Mixed);
            assert_eq!(
                mixed.len(),
                if matches!(
                    family,
                    ProjectionFamily::Query | ProjectionFamily::Output | ProjectionFamily::Down
                ) {
                    2
                } else {
                    1
                }
            );
            assert_eq!(
                family.split_columns(),
                if matches!(family, ProjectionFamily::Key | ProjectionFamily::Value) {
                    1
                } else {
                    4
                }
            );
        }
        assert!(ProjectionFamily::from_name("blk.bad.ffn_down.weight", "output.weight").is_none());
        assert!(ProjectionFamily::from_name("blk.0.attn_q.bias", "output.weight").is_none());
        assert!(check_budget(1024, 1024).is_ok());
        assert!(check_budget(1024, 1023).is_err());
    }
}
