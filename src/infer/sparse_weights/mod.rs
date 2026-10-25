//! Experimental fixed-popcount Q8 storage. Masks are supplied by the caller;
//! this module neither calibrates pruning nor participates in model dispatch.

use anyhow::{ensure, Result};
use half::f16;
use serde::Serialize;

use super::execution_weights::{Q8SourceIdentity, Q8SourceView, Q8_BLOCK_BYTES};

#[cfg(test)]
mod tests;

pub const SPARSE_Q8_RETAINED: usize = 16;
pub const SPARSE_Q8_IDEAL_BYTES: usize = 22;
pub const SPARSE_Q8_BLOCK_BYTES: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SparseQ8Storage {
    pub blocks: usize,
    pub dense_bytes: usize,
    pub ideal_bytes: usize,
    pub allocated_bytes: usize,
    pub value_bytes: usize,
    pub mask_bytes: usize,
    pub scale_bytes: usize,
    pub alignment_bytes: usize,
    pub retained_logical_values: usize,
    pub logical_values: usize,
}

/// Each 24-byte block is LE mask:u32, scale:f16, zero padding:u16,
/// then sixteen i8 values in increasing mask-bit order.
#[derive(Debug)]
pub struct HostSparseQ8Weight {
    source: Q8SourceIdentity,
    blocks: Vec<u8>,
}

impl HostSparseQ8Weight {
    pub fn pack(source: Q8SourceView<'_>, masks: &[u32]) -> Result<Self> {
        let shape = source.identity().shape();
        ensure!(
            masks.len() == shape.block_count()?,
            "sparse Q8 mask count mismatch"
        );
        let bytes = shape
            .block_count()?
            .checked_mul(SPARSE_Q8_BLOCK_BYTES)
            .ok_or_else(|| anyhow::anyhow!("sparse Q8 byte count overflow"))?;
        let mut packed = vec![0; bytes];
        for (index, ((dense, block), &mask)) in source
            .blocks()
            .chunks_exact(Q8_BLOCK_BYTES)
            .zip(packed.chunks_exact_mut(SPARSE_Q8_BLOCK_BYTES))
            .zip(masks)
            .enumerate()
        {
            ensure!(
                mask.count_ones() == 16,
                "sparse Q8 mask must retain exactly sixteen entries"
            );
            block[..4].copy_from_slice(&mask.to_le_bytes());
            block[4..6].copy_from_slice(&dense[..2]);
            let block_in_column = index % shape.blocks_per_column();
            let valid = (shape.input_dim() - block_in_column * 32).min(32);
            let mut rank = 0;
            for position in 0..32 {
                if mask & (1u32 << position) != 0 {
                    block[8 + rank] = if position < valid {
                        dense[2 + position]
                    } else {
                        0
                    };
                    rank += 1;
                }
            }
        }
        Self::from_packed(source.identity().clone(), packed)
    }

    /// Validate external bytes before they can reach a kernel. Partial final
    /// blocks retain sixteen physical slots, with zero payload outside K.
    pub fn from_packed(source: Q8SourceIdentity, blocks: Vec<u8>) -> Result<Self> {
        let shape = source.shape();
        ensure!(
            Some(blocks.len()) == shape.block_count()?.checked_mul(SPARSE_Q8_BLOCK_BYTES),
            "sparse Q8 byte length mismatch"
        );
        for (index, block) in blocks.chunks_exact(SPARSE_Q8_BLOCK_BYTES).enumerate() {
            let mask = read_mask(block);
            ensure!(
                mask.count_ones() == 16,
                "sparse Q8 mask must retain exactly sixteen entries"
            );
            ensure!(
                block[6..8] == [0, 0],
                "sparse Q8 alignment bytes must be zero"
            );
            ensure!(
                read_scale(block).is_finite(),
                "sparse Q8 scale must be finite"
            );
            let start = index % shape.blocks_per_column() * 32;
            let valid = (shape.input_dim() - start).min(32);
            let mut rank = 0;
            for position in 0..32 {
                if mask & (1u32 << position) != 0 {
                    ensure!(
                        position < valid || block[8 + rank] == 0,
                        "sparse Q8 tail payload must be zero"
                    );
                    rank += 1;
                }
            }
        }
        Ok(Self { source, blocks })
    }

    pub fn source(&self) -> &Q8SourceIdentity {
        &self.source
    }
    pub fn blocks(&self) -> &[u8] {
        &self.blocks
    }

    pub fn storage(&self) -> SparseQ8Storage {
        let shape = self.source.shape();
        let blocks = self.blocks.len() / SPARSE_Q8_BLOCK_BYTES;
        let retained_logical_values = self
            .blocks
            .chunks_exact(SPARSE_Q8_BLOCK_BYTES)
            .enumerate()
            .map(|(index, block)| {
                let valid = (shape.input_dim() - index % shape.blocks_per_column() * 32).min(32);
                let valid_mask = u32::MAX >> (32 - valid);
                (read_mask(block) & valid_mask).count_ones() as usize
            })
            .sum();
        SparseQ8Storage {
            blocks,
            dense_bytes: blocks * Q8_BLOCK_BYTES,
            ideal_bytes: blocks * SPARSE_Q8_IDEAL_BYTES,
            allocated_bytes: self.blocks.len(),
            value_bytes: blocks * 16,
            mask_bytes: blocks * 4,
            scale_bytes: blocks * 2,
            alignment_bytes: blocks * 2,
            retained_logical_values,
            logical_values: shape.input_dim() * shape.output_dim(),
        }
    }

    /// Native GGUF column blocks for the identical pruned matrix, not the
    /// original nonzero matrix. Scales retain their original FP16 bits.
    pub fn reconstruct_column_blocks(&self) -> Vec<u8> {
        let mut dense = vec![0; self.blocks.len() / SPARSE_Q8_BLOCK_BYTES * Q8_BLOCK_BYTES];
        for (packed, block) in self
            .blocks
            .chunks_exact(SPARSE_Q8_BLOCK_BYTES)
            .zip(dense.chunks_exact_mut(Q8_BLOCK_BYTES))
        {
            block[..2].copy_from_slice(&packed[4..6]);
            let mask = read_mask(packed);
            let mut rank = 0;
            for position in 0..32 {
                if mask & (1u32 << position) != 0 {
                    block[2 + position] = packed[8 + rank];
                    rank += 1;
                }
            }
        }
        dense
    }

    /// Scalar FP32 reconstruction, column-major `[N,K]` without padded tails.
    pub fn decode_column_major(&self) -> Vec<f32> {
        let shape = self.source.shape();
        let mut decoded = vec![0.0; shape.input_dim() * shape.output_dim()];
        for (index, block) in self.blocks.chunks_exact(SPARSE_Q8_BLOCK_BYTES).enumerate() {
            let column = index / shape.blocks_per_column();
            let start = index % shape.blocks_per_column() * 32;
            let mask = read_mask(block);
            let scale = read_scale(block);
            let mut rank = 0;
            for position in 0..32 {
                if mask & (1u32 << position) != 0 {
                    if start + position < shape.input_dim() {
                        decoded[column * shape.input_dim() + start + position] =
                            scale * f32::from(block[8 + rank] as i8);
                    }
                    rank += 1;
                }
            }
        }
        decoded
    }
}

fn read_mask(block: &[u8]) -> u32 {
    u32::from_le_bytes(block[..4].try_into().expect("validated sparse Q8 block"))
}

fn read_scale(block: &[u8]) -> f32 {
    f16::from_le_bytes([block[4], block[5]]).to_f32()
}
