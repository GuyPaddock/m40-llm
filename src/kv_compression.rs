#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCompressMode {
    #[default]
    Off,
    BlockSelectExact,
    BlockSummary,
    BlockSelectLossy,
}

impl KvCompressMode {
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCompressionConfig {
    pub mode: KvCompressMode,
    pub recent_window: u32,
    pub block_size: u32,
    pub top_blocks: u32,
    pub representatives: u32,
}

impl Default for KvCompressionConfig {
    fn default() -> Self {
        Self {
            mode: KvCompressMode::Off,
            recent_window: 1024,
            block_size: 32,
            top_blocks: 16,
            representatives: 2,
        }
    }
}

impl KvCompressionConfig {
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
        Ok(())
    }
}

static RUNTIME_CONFIG: OnceLock<Mutex<KvCompressionConfig>> = OnceLock::new();

pub fn set_runtime_config(config: KvCompressionConfig) {
    let cell = RUNTIME_CONFIG.get_or_init(|| Mutex::new(KvCompressionConfig::default()));
    *cell.lock().expect("kv compression runtime config lock") = config;
}

pub fn runtime_config() -> KvCompressionConfig {
    RUNTIME_CONFIG
        .get_or_init(|| Mutex::new(KvCompressionConfig::default()))
        .lock()
        .expect("kv compression runtime config lock")
        .clone()
}
use std::sync::{Mutex, OnceLock};
