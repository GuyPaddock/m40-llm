use std::cell::RefCell;
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCompressMode {
    #[default]
    Off,
    DenseRecentOnly,
    BlockSelectExact,
    RecentOnly,
    BlockSummary,
    BlockSelectLossy,
}

impl KvCompressMode {
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvRepresentativePolicy {
    #[default]
    Last,
    Stride,
}

impl KvRepresentativePolicy {
    pub fn as_ffi(self) -> u32 {
        match self {
            Self::Last => 0,
            Self::Stride => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvExactOldBacking {
    #[default]
    Dense,
    Q8,
    Fp16KQ4V,
}

impl KvExactOldBacking {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Q8 => "q8",
            Self::Fp16KQ4V => "fp16-k-q4-v",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvExactOldAttention {
    #[default]
    Staged,
    Q8Direct,
    Fp16KQ4VDirect,
}

impl KvExactOldAttention {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Staged => "staged",
            Self::Q8Direct => "q8-direct",
            Self::Fp16KQ4VDirect => "fp16-k-q4-v-direct",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCompressionConfig {
    pub mode: KvCompressMode,
    pub recent_window: u32,
    pub block_size: u32,
    pub top_blocks: u32,
    pub representatives: u32,
    pub representative_policy: KvRepresentativePolicy,
    pub exact_old_backing: KvExactOldBacking,
    pub exact_old_attention: KvExactOldAttention,
}

impl Default for KvCompressionConfig {
    fn default() -> Self {
        Self {
            mode: KvCompressMode::BlockSelectExact,
            recent_window: 1024,
            block_size: 32,
            top_blocks: 8,
            representatives: 0,
            representative_policy: KvRepresentativePolicy::Last,
            exact_old_backing: KvExactOldBacking::Fp16KQ4V,
            exact_old_attention: KvExactOldAttention::Fp16KQ4VDirect,
        }
    }
}

impl KvCompressionConfig {
    pub fn dense_reference() -> Self {
        Self {
            mode: KvCompressMode::Off,
            recent_window: 1024,
            block_size: 32,
            top_blocks: 8,
            representatives: 0,
            representative_policy: KvRepresentativePolicy::Last,
            exact_old_backing: KvExactOldBacking::Dense,
            exact_old_attention: KvExactOldAttention::Staged,
        }
    }

    pub fn for_mode(mode: KvCompressMode) -> Self {
        let mut config = if mode == KvCompressMode::BlockSelectExact {
            Self::default()
        } else {
            Self::dense_reference()
        };
        config.mode = mode;
        config
    }

    pub fn is_preferred_batched_runtime(&self) -> bool {
        self.mode == KvCompressMode::BlockSelectExact
            && self.effective_exact_old_backing() == KvExactOldBacking::Fp16KQ4V
            && self.effective_exact_old_attention() == KvExactOldAttention::Fp16KQ4VDirect
            && self.representatives == 0
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.recent_window == 0 {
            anyhow::bail!("kv_recent_window must be > 0");
        }
        if self.block_size == 0 {
            anyhow::bail!("kv_compress_block must be > 0");
        }
        if matches!(
            self.mode,
            KvCompressMode::BlockSelectExact | KvCompressMode::BlockSelectLossy
        ) && self.top_blocks == 0
        {
            anyhow::bail!("kv_compress_top_blocks must be > 0 for block-select modes");
        }
        let backing = if self.mode == KvCompressMode::BlockSelectExact {
            self.effective_exact_old_backing()
        } else {
            self.exact_old_backing
        };
        let attention = if self.mode == KvCompressMode::BlockSelectExact {
            self.effective_exact_old_attention()
        } else {
            self.exact_old_attention
        };
        if backing != KvExactOldBacking::Dense && self.mode != KvCompressMode::BlockSelectExact {
            anyhow::bail!(
                "kv_exact_old_backing={} requires kv_compress_mode=block-select-exact",
                backing.as_str()
            );
        }
        match attention {
            KvExactOldAttention::Staged => {}
            KvExactOldAttention::Q8Direct => {
                if backing != KvExactOldBacking::Q8 {
                    anyhow::bail!(
                        "kv_exact_old_attention=q8-direct requires kv_exact_old_backing=q8"
                    );
                }
            }
            KvExactOldAttention::Fp16KQ4VDirect => {
                if backing != KvExactOldBacking::Fp16KQ4V {
                    anyhow::bail!(
                        "kv_exact_old_attention=fp16-k-q4-v-direct requires kv_exact_old_backing=fp16-k-q4-v"
                    );
                }
            }
        }
        Ok(())
    }

    pub fn validate_runtime_support(&self, max_seq_len: u32, head_dim: u32) -> anyhow::Result<()> {
        self.validate()?;
        if self.mode == KvCompressMode::Off {
            return Ok(());
        }
        if max_seq_len == 0 {
            anyhow::bail!("compressed KV requires model context_length > 0");
        }
        if self.recent_window > max_seq_len {
            anyhow::bail!(
                "kv_recent_window {} exceeds model context_length {}; use a smaller --kv-recent-window or --kv-compress-mode off",
                self.recent_window,
                max_seq_len
            );
        }
        if self.block_size > max_seq_len {
            anyhow::bail!(
                "kv_compress_block {} exceeds model context_length {}",
                self.block_size,
                max_seq_len
            );
        }
        if matches!(
            self.mode,
            KvCompressMode::BlockSelectExact | KvCompressMode::BlockSelectLossy
        ) && self.top_blocks > 64
        {
            anyhow::bail!(
                "kv_compress_top_blocks {} exceeds the CUDA block-select limit of 64",
                self.top_blocks
            );
        }
        match self.mode {
            KvCompressMode::Off => Ok(()),
            KvCompressMode::DenseRecentOnly
            | KvCompressMode::RecentOnly
            | KvCompressMode::BlockSummary
            | KvCompressMode::BlockSelectLossy => {
                if head_dim != 64 {
                    anyhow::bail!(
                        "{:?} currently requires head_dim=64, got head_dim={}",
                        self.mode,
                        head_dim
                    );
                }
                Ok(())
            }
            KvCompressMode::BlockSelectExact => {
                let backing = self.effective_exact_old_backing();
                let attention = self.effective_exact_old_attention();
                if head_dim == 64 {
                    return Ok(());
                }
                if head_dim == 128
                    && backing == KvExactOldBacking::Fp16KQ4V
                    && attention == KvExactOldAttention::Fp16KQ4VDirect
                {
                    return Ok(());
                }
                anyhow::bail!(
                    "block-select-exact currently requires head_dim=64, or head_dim=128 with --kv-exact-old-backing fp16-k-q4-v and --kv-exact-old-attention fp16-k-q4-v-direct; got head_dim={} backing={} attention={}",
                    head_dim,
                    backing.as_str(),
                    attention.as_str()
                )
            }
        }
    }

    pub fn effective_exact_old_backing(&self) -> KvExactOldBacking {
        if self.exact_old_backing != KvExactOldBacking::Dense {
            return self.exact_old_backing;
        }
        match std::env::var("M40LLM_KV_EXACT_OLD_BACKING").ok().as_deref() {
            Some("q8") | Some("Q8") => KvExactOldBacking::Q8,
            Some("fp16-k-q4-v") | Some("FP16-K-Q4-V") => KvExactOldBacking::Fp16KQ4V,
            _ => KvExactOldBacking::Dense,
        }
    }

    pub fn effective_exact_old_attention(&self) -> KvExactOldAttention {
        if self.exact_old_attention != KvExactOldAttention::Staged {
            return self.exact_old_attention;
        }
        match std::env::var("M40LLM_KV_EXACT_OLD_ATTENTION")
            .ok()
            .as_deref()
        {
            Some("q8-direct") | Some("Q8-DIRECT") | Some("direct-q8") => {
                KvExactOldAttention::Q8Direct
            }
            Some("fp16-k-q4-v-direct")
            | Some("FP16-K-Q4-V-DIRECT")
            | Some("direct-fp16-k-q4-v") => KvExactOldAttention::Fp16KQ4VDirect,
            _ => KvExactOldAttention::Staged,
        }
    }

    pub fn uses_compressed_exact_old_backing(&self) -> bool {
        self.mode == KvCompressMode::BlockSelectExact
            && self.effective_exact_old_backing() != KvExactOldBacking::Dense
    }
}

static RUNTIME_CONFIG: OnceLock<Mutex<KvCompressionConfig>> = OnceLock::new();

thread_local! {
    static SCOPED_RUNTIME_CONFIG: RefCell<Option<KvCompressionConfig>> = const { RefCell::new(None) };
}

pub struct ScopedRuntimeConfig {
    previous: Option<KvCompressionConfig>,
}

impl ScopedRuntimeConfig {
    pub fn new(config: KvCompressionConfig) -> Self {
        let previous = SCOPED_RUNTIME_CONFIG.with(|slot| slot.replace(Some(config)));
        Self { previous }
    }
}

impl Drop for ScopedRuntimeConfig {
    fn drop(&mut self) {
        SCOPED_RUNTIME_CONFIG.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

pub fn set_runtime_config(config: KvCompressionConfig) {
    let cell = RUNTIME_CONFIG.get_or_init(|| Mutex::new(KvCompressionConfig::default()));
    *cell.lock().expect("kv compression runtime config lock") = config;
}

pub fn runtime_config() -> KvCompressionConfig {
    if let Some(config) = SCOPED_RUNTIME_CONFIG.with(|slot| slot.borrow().clone()) {
        return config;
    }
    RUNTIME_CONFIG
        .get_or_init(|| Mutex::new(KvCompressionConfig::default()))
        .lock()
        .expect("kv compression runtime config lock")
        .clone()
}

#[cfg(test)]
mod tests {
    use super::{
        runtime_config, set_runtime_config, KvCompressMode, KvCompressionConfig,
        KvExactOldAttention, KvExactOldBacking, ScopedRuntimeConfig,
    };

    #[test]
    fn validates_direct_fp16_k_q4_v_runtime_config() {
        let cfg = KvCompressionConfig::default();
        cfg.validate().expect("preferred runtime config validates");
        cfg.validate_runtime_support(2048, 64)
            .expect("preferred runtime config supports head64");
        cfg.validate_runtime_support(2048, 128)
            .expect("preferred runtime config supports head128");
        assert_eq!(cfg.mode, KvCompressMode::BlockSelectExact);
        assert_eq!(cfg.top_blocks, 8);
        assert_eq!(cfg.exact_old_backing, KvExactOldBacking::Fp16KQ4V);
        assert_eq!(cfg.exact_old_attention, KvExactOldAttention::Fp16KQ4VDirect);
        assert!(cfg.is_preferred_batched_runtime());
    }

    #[test]
    fn dense_reference_config_is_explicit() {
        let cfg = KvCompressionConfig::dense_reference();
        cfg.validate().expect("dense reference config validates");
        cfg.validate_runtime_support(16, 128)
            .expect("dense reference has no compressed-KV head_dim restriction");
        assert_eq!(cfg.mode, KvCompressMode::Off);
        assert_eq!(cfg.top_blocks, 8);
        assert_eq!(cfg.exact_old_backing, KvExactOldBacking::Dense);
        assert_eq!(cfg.exact_old_attention, KvExactOldAttention::Staged);
        assert!(!cfg.is_preferred_batched_runtime());
    }

    #[test]
    fn preferred_batched_runtime_rejects_diagnostic_variants() {
        let q8 = KvCompressionConfig {
            exact_old_backing: KvExactOldBacking::Q8,
            exact_old_attention: KvExactOldAttention::Q8Direct,
            ..KvCompressionConfig::default()
        };
        assert!(!q8.is_preferred_batched_runtime());

        let reps = KvCompressionConfig {
            representatives: 1,
            ..KvCompressionConfig::default()
        };
        assert!(!reps.is_preferred_batched_runtime());
    }

    #[test]
    fn runtime_support_rejects_oversized_recent_window() {
        let err = KvCompressionConfig::default()
            .validate_runtime_support(512, 64)
            .expect_err("recent window larger than context should fail");
        assert!(err.to_string().contains("kv_recent_window"));
    }

    #[test]
    fn runtime_support_rejects_unsupported_head128_combo() {
        let cfg = KvCompressionConfig {
            exact_old_backing: KvExactOldBacking::Q8,
            exact_old_attention: KvExactOldAttention::Q8Direct,
            ..KvCompressionConfig::default()
        };
        let err = cfg
            .validate_runtime_support(2048, 128)
            .expect_err("q8 direct head128 should fail");
        assert!(err.to_string().contains("head_dim=128"));
    }

    #[test]
    fn runtime_support_rejects_top_blocks_above_cuda_limit() {
        let cfg = KvCompressionConfig {
            top_blocks: 65,
            ..KvCompressionConfig::default()
        };
        let err = cfg
            .validate_runtime_support(2048, 64)
            .expect_err("top_blocks above CUDA limit should fail");
        assert!(err.to_string().contains("limit of 64"));
    }

    #[test]
    fn rejects_direct_attention_without_matching_backing() {
        let cfg = KvCompressionConfig {
            mode: KvCompressMode::BlockSelectExact,
            exact_old_attention: KvExactOldAttention::Fp16KQ4VDirect,
            ..KvCompressionConfig::dense_reference()
        };
        let err = cfg.validate().expect_err("mismatched backend must fail");
        assert!(
            err.to_string().contains("requires kv_exact_old_backing"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_exact_old_backing_outside_block_select_exact() {
        let cfg = KvCompressionConfig {
            mode: KvCompressMode::Off,
            exact_old_backing: KvExactOldBacking::Fp16KQ4V,
            ..KvCompressionConfig::dense_reference()
        };
        let err = cfg.validate().expect_err("mismatched mode must fail");
        assert!(
            err.to_string().contains("requires kv_compress_mode"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn scoped_runtime_config_shadows_global_config() {
        set_runtime_config(KvCompressionConfig::dense_reference());

        let scoped = KvCompressionConfig {
            top_blocks: 4,
            ..KvCompressionConfig::default()
        };
        {
            let _guard = ScopedRuntimeConfig::new(scoped.clone());
            assert_eq!(runtime_config(), scoped);
        }

        assert_eq!(runtime_config(), KvCompressionConfig::dense_reference());
    }
}
