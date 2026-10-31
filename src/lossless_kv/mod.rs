//! Experiment-only bit-exact FP16 exponent packing, independent of runtime KV.
mod codec;
#[cfg(test)]
mod tests;

use anyhow::{ensure, Result};
use serde::Serialize;

pub const BLOCK_VALUES: usize = 512;
const HEADER_BYTES: usize = 8;
const ENTRY_BYTES: usize = 8;
const RAW_WIDTH: u8 = 255;
const MAGIC: &[u8; 4] = b"M4KV";

#[derive(Debug, Clone, Serialize)]
pub struct LosslessKvStats {
    pub values: usize,
    pub original_bytes: usize,
    pub encoded_bytes: usize,
    pub metadata_bytes: usize,
    pub payload_bytes: usize,
    pub raw_blocks: usize,
    pub packed_blocks_by_width: [usize; 5],
}

/// Owned research representation. Every directory byte counts toward size;
/// all-raw input can expand. It is never selected by inference dispatch.
#[derive(Debug, Clone)]
pub struct LosslessKvBlocks {
    bytes: Vec<u8>,
    stats: LosslessKvStats,
}

impl LosslessKvBlocks {
    pub fn encode(words: &[u16]) -> Result<Self> {
        Self::from_bytes(codec::encode(words)?)
    }

    /// Validate the complete directory and every packed exponent before owning
    /// a serialized stream. Untrusted counts never control unchecked allocation.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let stats = codec::validate(&bytes)?;
        Ok(Self { bytes, stats })
    }

    pub fn get_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn get_stats(&self) -> &LosslessKvStats {
        &self.stats
    }

    pub fn decode(&self) -> Result<Vec<u16>> {
        let mut output = Vec::with_capacity(self.stats.values);
        codec::visit(&self.bytes, |word| output.push(word))?;
        ensure!(
            output.len() == self.stats.values,
            "lossless KV decode count mismatch"
        );
        Ok(output)
    }
}
