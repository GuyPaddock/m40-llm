use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::infer::{BatchMetadata, BatchSequence};

#[cfg(feature = "cuda")]
use crate::cuda::{CudaContext, DeviceBuffer, KVCache};
#[cfg(feature = "cuda")]
use std::ffi::c_void;

pub type DecodeRequestId = u64;

#[derive(Debug, Clone)]
pub struct DecodeSequencePool {
    inner: Arc<Mutex<DecodeSequencePoolInner>>,
}

#[derive(Debug)]
struct DecodeSequencePoolInner {
    free: VecDeque<u32>,
    capacity: u32,
}

#[derive(Debug)]
pub struct DecodeSequenceLease {
    sequence_id: u32,
    inner: Option<Arc<Mutex<DecodeSequencePoolInner>>>,
}

impl DecodeSequencePool {
    pub fn new(capacity: u32) -> Self {
        let free = (0..capacity).collect();
        Self {
            inner: Arc::new(Mutex::new(DecodeSequencePoolInner { free, capacity })),
        }
    }

    pub fn capacity(&self) -> u32 {
        self.inner
            .lock()
            .expect("decode sequence pool poisoned")
            .capacity
    }

    pub fn available(&self) -> usize {
        self.inner
            .lock()
            .expect("decode sequence pool poisoned")
            .free
            .len()
    }

    pub fn try_lease(&self) -> Option<DecodeSequenceLease> {
        let sequence_id = self
            .inner
            .lock()
            .expect("decode sequence pool poisoned")
            .free
            .pop_front()?;
        Some(DecodeSequenceLease {
            sequence_id,
            inner: Some(self.inner.clone()),
        })
    }
}

impl DecodeSequenceLease {
    pub fn sequence_id(&self) -> u32 {
        self.sequence_id
    }
}

impl Drop for DecodeSequenceLease {
    fn drop(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        inner
            .lock()
            .expect("decode sequence pool poisoned")
            .free
            .push_back(self.sequence_id);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeRequestStatus {
    Active,
    Completed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeRequestState {
    pub request_id: DecodeRequestId,
    pub sequence_id: u32,
    pub kv_len: u32,
    pub status: DecodeRequestStatus,
}

impl DecodeRequestState {
    pub const fn active(request_id: DecodeRequestId, sequence_id: u32, kv_len: u32) -> Self {
        Self {
            request_id,
            sequence_id,
            kv_len,
            status: DecodeRequestStatus::Active,
        }
    }

    pub const fn completed(request_id: DecodeRequestId, sequence_id: u32, kv_len: u32) -> Self {
        Self {
            request_id,
            sequence_id,
            kv_len,
            status: DecodeRequestStatus::Completed,
        }
    }

    pub const fn cancelled(request_id: DecodeRequestId, sequence_id: u32, kv_len: u32) -> Self {
        Self {
            request_id,
            sequence_id,
            kv_len,
            status: DecodeRequestStatus::Cancelled,
        }
    }

    pub const fn is_active(self) -> bool {
        matches!(self.status, DecodeRequestStatus::Active)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeBatchEntry {
    pub request_id: DecodeRequestId,
    pub sequence_id: u32,
    pub kv_len: u32,
    pub batch_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeBatchPlan {
    entries: Vec<DecodeBatchEntry>,
    metadata: BatchMetadata,
}

impl DecodeBatchPlan {
    pub fn from_requests(requests: &[DecodeRequestState]) -> Result<Self> {
        let active: Vec<_> = requests
            .iter()
            .copied()
            .filter(|request| request.is_active())
            .collect();
        if active.is_empty() {
            anyhow::bail!("decode batch requires at least one active request");
        }

        let mut entries = Vec::with_capacity(active.len());
        let mut sequences = Vec::with_capacity(active.len());
        for (batch_index, request) in active.into_iter().enumerate() {
            if request.kv_len == 0 {
                anyhow::bail!("active request {} has zero kv_len", request.request_id);
            }
            entries.push(DecodeBatchEntry {
                request_id: request.request_id,
                sequence_id: request.sequence_id,
                kv_len: request.kv_len,
                batch_index,
            });
            sequences.push(BatchSequence {
                seq_len: request.kv_len,
                kv_len: request.kv_len,
                query_len: 1,
            });
        }

        let metadata = BatchMetadata::new(sequences).context("decode batch metadata")?;
        Ok(Self { entries, metadata })
    }

    pub fn entries(&self) -> &[DecodeBatchEntry] {
        &self.entries
    }

    pub fn metadata(&self) -> &BatchMetadata {
        &self.metadata
    }

    pub fn batch_size(&self) -> u32 {
        self.entries.len() as u32
    }

    pub fn sequence_ids(&self) -> Vec<u32> {
        self.entries.iter().map(|entry| entry.sequence_id).collect()
    }

    pub fn sequence_lens(&self) -> Vec<u32> {
        self.entries.iter().map(|entry| entry.kv_len).collect()
    }

    pub fn batch_index_for_request(&self, request_id: DecodeRequestId) -> Option<usize> {
        self.entries
            .iter()
            .find(|entry| entry.request_id == request_id)
            .map(|entry| entry.batch_index)
    }

    pub fn output_offset_f32(
        &self,
        request_id: DecodeRequestId,
        q_heads: u32,
        head_dim: u32,
    ) -> Option<usize> {
        let elems_per_item = (q_heads as usize).checked_mul(head_dim as usize)?;
        self.batch_index_for_request(request_id)
            .and_then(|idx| idx.checked_mul(elems_per_item))
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct CudaDecodeBatchPlan {
    plan: DecodeBatchPlan,
    d_sequence_ids: DeviceBuffer,
    d_sequence_lens: DeviceBuffer,
}

#[cfg(feature = "cuda")]
impl CudaDecodeBatchPlan {
    pub fn new(ctx: &CudaContext, plan: DecodeBatchPlan) -> Result<Self> {
        let bytes = (plan.batch_size() as usize)
            .checked_mul(std::mem::size_of::<u32>())
            .context("decode batch metadata byte size overflow")?;
        let d_sequence_ids = DeviceBuffer::new_tagged(ctx, bytes, "decode_batch:sequence_ids")?;
        let d_sequence_lens = DeviceBuffer::new_tagged(ctx, bytes, "decode_batch:sequence_lens")?;
        let sequence_ids = plan.sequence_ids();
        let sequence_lens = plan.sequence_lens();
        unsafe {
            ctx.memcpy_h2d(
                d_sequence_ids.as_mut_ptr(),
                sequence_ids.as_ptr() as *const c_void,
                bytes,
            )?;
            ctx.memcpy_h2d(
                d_sequence_lens.as_mut_ptr(),
                sequence_lens.as_ptr() as *const c_void,
                bytes,
            )?;
        }
        Ok(Self {
            plan,
            d_sequence_ids,
            d_sequence_lens,
        })
    }

    pub fn plan(&self) -> &DecodeBatchPlan {
        &self.plan
    }

    /// # Safety
    /// `d_q_f32` and `d_out_f32` must be packed by `plan.entries()` order with
    /// one query token per active request: [batch, q_heads, head_dim].
    pub unsafe fn dispatch_attention(
        &self,
        ctx: &CudaContext,
        kv: &KVCache,
        d_q_f32: *const c_void,
        q_heads: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        unsafe {
            kv.attention_last_token_f32_gqa_batched(
                ctx,
                self.d_sequence_ids.as_ptr() as *const u32,
                self.d_sequence_lens.as_ptr() as *const u32,
                self.plan.batch_size(),
                d_q_f32,
                q_heads,
                d_out_f32,
            )
        }
    }
}

#[cfg(feature = "server")]
pub fn server_batch_decode_requested() -> bool {
    std::env::var("M40LLM_SERVER_BATCH_DECODE").ok().as_deref() == Some("1")
}

#[cfg(feature = "server")]
pub fn server_batch_decode_status() -> &'static str {
    "M40LLM_SERVER_BATCH_DECODE=1 requested; buffered /generate uses a queued decode scheduler with leased KV sequence slots, stepping all active requests each tick"
}

#[cfg(feature = "server")]
pub fn maybe_log_server_batch_decode_status() {
    if !server_batch_decode_requested() {
        return;
    }
    static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    LOGGED.get_or_init(|| {
        eprintln!("[server] {}", server_batch_decode_status());
    });
}

#[cfg(feature = "server")]
pub fn server_batch_prefill_requested() -> bool {
    std::env::var("M40LLM_SERVER_BATCH_PREFILL").ok().as_deref() == Some("1")
}

#[cfg(feature = "server")]
pub fn server_batch_prefill_status() -> &'static str {
    "M40LLM_SERVER_BATCH_PREFILL=1 requested; compatible dense buffered scheduler ticks and same-length preferred compressed head128 ticks use packed prompt-prefix prefill"
}

#[cfg(feature = "server")]
pub fn maybe_log_server_batch_prefill_status() {
    if !server_batch_prefill_requested() {
        return;
    }
    static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    LOGGED.get_or_init(|| {
        eprintln!("[server] {}", server_batch_prefill_status());
    });
}

#[cfg(feature = "cuda")]
impl CudaDecodeBatchPlan {
    /// # Safety
    /// `d_q_f32` and `d_out_f32` must be packed by `plan.entries()` order with
    /// one query token per active request: [batch, q_heads, head_dim].
    pub unsafe fn dispatch_attention_async(
        &self,
        ctx: &CudaContext,
        kv: &KVCache,
        d_q_f32: *const c_void,
        q_heads: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        unsafe {
            kv.attention_last_token_f32_gqa_batched_async(
                ctx,
                self.d_sequence_ids.as_ptr() as *const u32,
                self.d_sequence_lens.as_ptr() as *const u32,
                self.plan.batch_size(),
                d_q_f32,
                q_heads,
                d_out_f32,
            )
        }
    }

    /// # Safety
    /// `d_q_f32` and `d_out_f32` must be packed by `plan.entries()` order with
    /// one query token per active request: [batch, q_heads, head_dim].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dispatch_fp16_k_q4_v_direct_attention_async(
        &self,
        ctx: &CudaContext,
        kv: &KVCache,
        d_q_f32: *const c_void,
        q_heads: u32,
        recent_window: u32,
        block_size: u32,
        top_blocks: u32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        unsafe {
            kv.attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct_batched_async(
                ctx,
                self.d_sequence_ids.as_ptr() as *const u32,
                self.d_sequence_lens.as_ptr() as *const u32,
                self.plan.batch_size(),
                d_q_f32,
                q_heads,
                recent_window,
                block_size,
                top_blocks,
                d_out_f32,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_sequence_pool_leases_and_returns_slots() {
        let pool = DecodeSequencePool::new(2);
        assert_eq!(pool.capacity(), 2);
        assert_eq!(pool.available(), 2);

        let first = pool.try_lease().expect("first sequence");
        let second = pool.try_lease().expect("second sequence");
        assert_eq!(first.sequence_id(), 0);
        assert_eq!(second.sequence_id(), 1);
        assert_eq!(pool.available(), 0);
        assert!(pool.try_lease().is_none());

        drop(first);
        assert_eq!(pool.available(), 1);
        let third = pool.try_lease().expect("reused sequence");
        assert_eq!(third.sequence_id(), 0);
    }

    #[test]
    fn decode_batch_keeps_active_mixed_lengths() {
        let requests = [
            DecodeRequestState::active(10, 2, 5),
            DecodeRequestState::completed(11, 3, 7),
            DecodeRequestState::active(12, 4, 1),
            DecodeRequestState::cancelled(13, 5, 9),
            DecodeRequestState::active(14, 6, 3),
        ];
        let plan = DecodeBatchPlan::from_requests(&requests).expect("decode batch");
        assert_eq!(plan.sequence_ids(), vec![2, 4, 6]);
        assert_eq!(plan.sequence_lens(), vec![5, 1, 3]);
        assert_eq!(plan.metadata().total_q_tokens(), 3);
        assert_eq!(plan.metadata().total_kv_tokens(), 9);
        assert_eq!(plan.metadata().sequences()[0].query_len, 1);
        assert_eq!(plan.batch_index_for_request(12), Some(1));
        assert_eq!(plan.output_offset_f32(14, 4, 64), Some(2 * 4 * 64));
    }

    #[test]
    fn decode_batch_rejects_no_active_requests() {
        let err =
            DecodeBatchPlan::from_requests(&[DecodeRequestState::completed(1, 0, 1)]).unwrap_err();
        assert!(err.to_string().contains("at least one active request"));
    }

    #[test]
    fn decode_batch_rejects_zero_active_kv_len() {
        let err =
            DecodeBatchPlan::from_requests(&[DecodeRequestState::active(1, 0, 0)]).unwrap_err();
        assert!(err.to_string().contains("zero kv_len"));
    }
}
