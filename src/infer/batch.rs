use anyhow::{Context, Result};
#[cfg(feature = "cuda")]
use std::ffi::c_void;

#[cfg(feature = "cuda")]
use crate::cuda::CudaContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchSequence {
    pub seq_len: u32,
    pub kv_len: u32,
    pub query_len: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedSequenceOffsets {
    pub token_offset: u32,
    pub q_offset: u32,
    pub kv_offset: u32,
    pub attention_offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchMetadata {
    sequences: Vec<BatchSequence>,
    offsets: Vec<PackedSequenceOffsets>,
    token_offsets: Vec<u32>,
    q_offsets: Vec<u32>,
    kv_offsets: Vec<u32>,
    attention_offsets: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LengthBucket {
    Tokens1To64,
    Tokens65To128,
    Tokens129To256,
    Tokens257To512,
    Tokens513To1024,
    Tokens1025To2048,
    Tokens2049To4096,
    Tokens4097To8192,
    Tokens8193Plus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BucketedBatch {
    pub bucket: LengthBucket,
    pub sequence_indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BucketStats {
    pub max_query_len: u32,
    pub max_kv_len: u32,
    pub total_query_tokens: u32,
    pub total_kv_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarlenPrefillTile {
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
}

impl VarlenPrefillTile {
    pub const CONSERVATIVE_HEAD64: Self = Self {
        tile_m: 1,
        tile_n: 128,
        tile_k: 64,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarlenPrefillTileSelection {
    pub head_dim: u32,
    pub max_query_len: u32,
    pub max_kv_len: u32,
    pub tile: VarlenPrefillTile,
}

impl VarlenPrefillTileSelection {
    pub fn select(meta: &BatchMetadata, head_dim: u32) -> Result<Self> {
        if head_dim != 64 {
            anyhow::bail!(
                "variable-length prefill currently supports head_dim=64 only, got {head_dim}"
            );
        }
        let max_query_len = meta
            .sequences()
            .iter()
            .map(|seq| seq.query_len)
            .max()
            .unwrap_or(0);
        let max_kv_len = meta
            .sequences()
            .iter()
            .map(|seq| seq.kv_len)
            .max()
            .unwrap_or(0);
        Ok(Self {
            head_dim,
            max_query_len,
            max_kv_len,
            tile: VarlenPrefillTile::CONSERVATIVE_HEAD64,
        })
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct VarlenPrefillPlan {
    meta: BatchMetadata,
    selection: VarlenPrefillTileSelection,
    ctx: CudaContext,
    d_q_offsets: *mut c_void,
    d_kv_offsets: *mut c_void,
    d_q_lens: *mut c_void,
    d_kv_lens: *mut c_void,
    bytes: usize,
}

#[cfg(feature = "cuda")]
impl VarlenPrefillPlan {
    pub fn new(ctx: &CudaContext, meta: BatchMetadata, head_dim: u32) -> Result<Self> {
        let selection = VarlenPrefillTileSelection::select(&meta, head_dim)?;
        let batch_size = meta.sequences().len();
        let bytes = batch_size
            .checked_mul(std::mem::size_of::<u32>())
            .context("variable-length prefill metadata size overflow")?;
        let q_lens: Vec<u32> = meta.sequences().iter().map(|seq| seq.query_len).collect();
        let kv_lens: Vec<u32> = meta.sequences().iter().map(|seq| seq.kv_len).collect();

        let mut allocations = Vec::new();
        let result = (|| -> Result<Self> {
            let d_q_offsets = ctx.device_malloc_tagged(bytes, "varlen_prefill.q_offsets")?;
            allocations.push(d_q_offsets);
            let d_kv_offsets = ctx.device_malloc_tagged(bytes, "varlen_prefill.kv_offsets")?;
            allocations.push(d_kv_offsets);
            let d_q_lens = ctx.device_malloc_tagged(bytes, "varlen_prefill.q_lens")?;
            allocations.push(d_q_lens);
            let d_kv_lens = ctx.device_malloc_tagged(bytes, "varlen_prefill.kv_lens")?;
            allocations.push(d_kv_lens);

            unsafe {
                ctx.memcpy_h2d(
                    d_q_offsets,
                    meta.q_offsets().as_ptr() as *const c_void,
                    bytes,
                )?;
                ctx.memcpy_h2d(
                    d_kv_offsets,
                    meta.kv_offsets().as_ptr() as *const c_void,
                    bytes,
                )?;
                ctx.memcpy_h2d(d_q_lens, q_lens.as_ptr() as *const c_void, bytes)?;
                ctx.memcpy_h2d(d_kv_lens, kv_lens.as_ptr() as *const c_void, bytes)?;
            }

            allocations.clear();
            Ok(Self {
                meta,
                selection,
                ctx: ctx.clone(),
                d_q_offsets,
                d_kv_offsets,
                d_q_lens,
                d_kv_lens,
                bytes,
            })
        })();

        if result.is_err() {
            for ptr in allocations {
                unsafe {
                    let _ = ctx.device_free(ptr);
                }
            }
        }
        result
    }

    pub fn metadata(&self) -> &BatchMetadata {
        &self.meta
    }

    pub fn tile_selection(&self) -> VarlenPrefillTileSelection {
        self.selection
    }

    pub fn batch_size(&self) -> u32 {
        self.meta.sequences().len() as u32
    }

    /// # Safety
    /// Packed buffers must use [total_q_tokens, q_heads, 64] and
    /// [total_kv_tokens, kv_heads, 64] row-major f32 layouts.
    pub unsafe fn dispatch_head64(
        &self,
        d_q_f32: *const c_void,
        d_k_f32: *const c_void,
        d_v_f32: *const c_void,
        q_heads: u32,
        kv_heads: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        self.ctx.attention_prefill_f32_gqa_varlen_head64(
            d_q_f32,
            d_k_f32,
            d_v_f32,
            self.d_q_offsets as *const u32,
            self.d_kv_offsets as *const u32,
            self.d_q_lens as *const u32,
            self.d_kv_lens as *const u32,
            self.batch_size(),
            q_heads,
            kv_heads,
            d_out_f32,
        )
    }

    /// # Safety
    /// Same layout requirements as `dispatch_head64`. The call only enqueues
    /// work on the prefill stream; synchronize before reading outputs.
    pub unsafe fn dispatch_head64_async(
        &self,
        d_q_f32: *const c_void,
        d_k_f32: *const c_void,
        d_v_f32: *const c_void,
        q_heads: u32,
        kv_heads: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        self.ctx.attention_prefill_f32_gqa_varlen_head64_async(
            d_q_f32,
            d_k_f32,
            d_v_f32,
            self.d_q_offsets as *const u32,
            self.d_kv_offsets as *const u32,
            self.d_q_lens as *const u32,
            self.d_kv_lens as *const u32,
            self.batch_size(),
            q_heads,
            kv_heads,
            d_out_f32,
        )
    }
}

#[cfg(feature = "cuda")]
impl Drop for VarlenPrefillPlan {
    fn drop(&mut self) {
        let _ = self.bytes;
        unsafe {
            let _ = self.ctx.device_free(self.d_q_offsets);
            let _ = self.ctx.device_free(self.d_kv_offsets);
            let _ = self.ctx.device_free(self.d_q_lens);
            let _ = self.ctx.device_free(self.d_kv_lens);
        }
    }
}

impl BatchMetadata {
    pub fn new(sequences: Vec<BatchSequence>) -> Result<Self> {
        if sequences.is_empty() {
            anyhow::bail!("batch metadata requires at least one sequence");
        }

        let mut offsets = Vec::with_capacity(sequences.len());
        let mut token_offsets = Vec::with_capacity(sequences.len() + 1);
        let mut q_offsets = Vec::with_capacity(sequences.len() + 1);
        let mut kv_offsets = Vec::with_capacity(sequences.len() + 1);
        let mut attention_offsets = Vec::with_capacity(sequences.len() + 1);

        let mut token_sum = 0u32;
        let mut q_sum = 0u32;
        let mut kv_sum = 0u32;
        let mut attention_sum = 0u64;
        token_offsets.push(token_sum);
        q_offsets.push(q_sum);
        kv_offsets.push(kv_sum);
        attention_offsets.push(attention_sum);

        for (idx, seq) in sequences.iter().copied().enumerate() {
            if seq.seq_len == 0 {
                anyhow::bail!("sequence {idx} has zero seq_len");
            }
            if seq.query_len == 0 {
                anyhow::bail!("sequence {idx} has zero query_len");
            }
            if seq.kv_len == 0 {
                anyhow::bail!("sequence {idx} has zero kv_len");
            }
            if seq.query_len > seq.seq_len {
                anyhow::bail!(
                    "sequence {idx} query_len {} exceeds seq_len {}",
                    seq.query_len,
                    seq.seq_len
                );
            }
            if seq.kv_len < seq.query_len {
                anyhow::bail!(
                    "sequence {idx} kv_len {} is smaller than query_len {}",
                    seq.kv_len,
                    seq.query_len
                );
            }

            offsets.push(PackedSequenceOffsets {
                token_offset: token_sum,
                q_offset: q_sum,
                kv_offset: kv_sum,
                attention_offset: attention_sum,
            });

            token_sum = token_sum
                .checked_add(seq.seq_len)
                .with_context(|| format!("token_offsets overflow after sequence {idx}"))?;
            q_sum = q_sum
                .checked_add(seq.query_len)
                .with_context(|| format!("q_offsets overflow after sequence {idx}"))?;
            kv_sum = kv_sum
                .checked_add(seq.kv_len)
                .with_context(|| format!("kv_offsets overflow after sequence {idx}"))?;
            attention_sum = attention_sum
                .checked_add((seq.query_len as u64) * (seq.kv_len as u64))
                .with_context(|| format!("attention_offsets overflow after sequence {idx}"))?;

            token_offsets.push(token_sum);
            q_offsets.push(q_sum);
            kv_offsets.push(kv_sum);
            attention_offsets.push(attention_sum);
        }

        Ok(Self {
            sequences,
            offsets,
            token_offsets,
            q_offsets,
            kv_offsets,
            attention_offsets,
        })
    }

    pub fn sequences(&self) -> &[BatchSequence] {
        &self.sequences
    }

    pub fn offsets(&self) -> &[PackedSequenceOffsets] {
        &self.offsets
    }

    pub fn token_offsets(&self) -> &[u32] {
        &self.token_offsets
    }

    pub fn q_offsets(&self) -> &[u32] {
        &self.q_offsets
    }

    pub fn kv_offsets(&self) -> &[u32] {
        &self.kv_offsets
    }

    pub fn attention_offsets(&self) -> &[u64] {
        &self.attention_offsets
    }

    pub fn total_tokens(&self) -> u32 {
        *self.token_offsets.last().unwrap_or(&0)
    }

    pub fn total_q_tokens(&self) -> u32 {
        *self.q_offsets.last().unwrap_or(&0)
    }

    pub fn total_kv_tokens(&self) -> u32 {
        *self.kv_offsets.last().unwrap_or(&0)
    }

    pub fn total_attention_cells(&self) -> u64 {
        *self.attention_offsets.last().unwrap_or(&0)
    }

    pub fn buckets(&self) -> Vec<BucketedBatch> {
        let mut grouped: Vec<BucketedBatch> = Vec::new();
        for (idx, seq) in self.sequences.iter().enumerate() {
            let bucket = LengthBucket::for_len(seq.kv_len.max(seq.query_len));
            match grouped.iter_mut().find(|g| g.bucket == bucket) {
                Some(group) => group.sequence_indices.push(idx),
                None => grouped.push(BucketedBatch {
                    bucket,
                    sequence_indices: vec![idx],
                }),
            }
        }
        grouped.sort_by_key(|g| g.bucket);
        grouped
    }

    pub fn bucket_stats(&self, bucket: &BucketedBatch) -> BucketStats {
        let mut stats = BucketStats {
            max_query_len: 0,
            max_kv_len: 0,
            total_query_tokens: 0,
            total_kv_tokens: 0,
        };
        for &idx in &bucket.sequence_indices {
            let seq = self.sequences[idx];
            stats.max_query_len = stats.max_query_len.max(seq.query_len);
            stats.max_kv_len = stats.max_kv_len.max(seq.kv_len);
            stats.total_query_tokens = stats.total_query_tokens.saturating_add(seq.query_len);
            stats.total_kv_tokens = stats.total_kv_tokens.saturating_add(seq.kv_len);
        }
        stats
    }

    pub fn bucket_sequence_indices(&self, bucket: LengthBucket) -> Vec<usize> {
        self.sequences
            .iter()
            .enumerate()
            .filter_map(|(idx, seq)| {
                (LengthBucket::for_len(seq.kv_len.max(seq.query_len)) == bucket).then_some(idx)
            })
            .collect()
    }
}

impl LengthBucket {
    pub fn for_len(len: u32) -> Self {
        match len {
            0..=64 => Self::Tokens1To64,
            65..=128 => Self::Tokens65To128,
            129..=256 => Self::Tokens129To256,
            257..=512 => Self::Tokens257To512,
            513..=1024 => Self::Tokens513To1024,
            1025..=2048 => Self::Tokens1025To2048,
            2049..=4096 => Self::Tokens2049To4096,
            4097..=8192 => Self::Tokens4097To8192,
            _ => Self::Tokens8193Plus,
        }
    }

    pub fn max_len(self) -> Option<u32> {
        match self {
            Self::Tokens1To64 => Some(64),
            Self::Tokens65To128 => Some(128),
            Self::Tokens129To256 => Some(256),
            Self::Tokens257To512 => Some(512),
            Self::Tokens513To1024 => Some(1024),
            Self::Tokens1025To2048 => Some(2048),
            Self::Tokens2049To4096 => Some(4096),
            Self::Tokens4097To8192 => Some(8192),
            Self::Tokens8193Plus => None,
        }
    }
}
