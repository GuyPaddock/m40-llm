use std::ffi::c_void;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};

use super::{
    ffi, record_async_kernel, CudaContext, CudaStream, DeviceBuffer, KVCache, Q8DecodeEventDiag,
};

/// Captured once per compressed cache, never polled in the attention loop.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NoOldAttention {
    #[default]
    Off,
    DenseRecent,
}

impl NoOldAttention {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "off" => Ok(Self::Off),
            "dense-recent" => Ok(Self::DenseRecent),
            _ => anyhow::bail!("M40LLM_KV_NO_OLD_ATTENTION must be off or dense-recent"),
        }
    }

    pub(super) fn from_env() -> Result<Self> {
        let mode = Self::parse(
            &std::env::var("M40LLM_KV_NO_OLD_ATTENTION").unwrap_or_else(|_| "off".into()),
        )?;
        if mode == Self::DenseRecent
            && (std::env::var("M40LLM_KV_BLOCK_SELECT_POLICY").unwrap_or_else(|_| "topk".into())
                != "topk"
                || std::env::vars().any(|(key, _)| key.starts_with("M40LLM_KV_FORCE_")))
        {
            eprintln!(
                "[kv-no-old] ineligible diagnostic selection policy; retaining original attention"
            );
            return Ok(Self::Off);
        }
        Ok(mode)
    }
}

/// Bounded launch choices for the experimental full-context head128 kernel.
#[derive(Debug, Clone, Copy)]
pub struct Head128AttentionConfig {
    pub partitions: u32,
    pub threads: u32,
}

impl Head128AttentionConfig {
    pub fn validate(self) -> Result<()> {
        ensure!(
            matches!(self.partitions, 1 | 2 | 4 | 8),
            "head128 attention partitions must be 1, 2, 4, or 8"
        );
        ensure!(
            matches!(self.threads, 128 | 256),
            "head128 attention threads must be 128 or 256"
        );
        Ok(())
    }

    pub(super) fn from_env() -> Result<Option<Self>> {
        if std::env::var("M40LLM_ATTN_HEAD128_KERNEL").as_deref() != Ok("parallel") {
            return Ok(None);
        }
        let config = Self {
            partitions: std::env::var("M40LLM_ATTN_HEAD128_PARTITIONS")
                .unwrap_or_else(|_| "1".into())
                .parse()
                .context("invalid M40LLM_ATTN_HEAD128_PARTITIONS")?,
            threads: std::env::var("M40LLM_ATTN_HEAD128_THREADS")
                .unwrap_or_else(|_| "128".into())
                .parse()
                .context("invalid M40LLM_ATTN_HEAD128_THREADS")?,
        };
        config.validate()?;
        Ok(Some(config))
    }
}

/// FP32 maximum, exponential sum, and 128 unnormalized V values per partition.
/// One buffer serves every layer/sequence on its owning context's decode stream.
#[derive(Debug)]
pub struct Head128AttentionWorkspace {
    q_head_capacity: u32,
    partials: DeviceBuffer,
}

impl Head128AttentionWorkspace {
    pub fn new(ctx: &CudaContext, q_heads: u32) -> Result<Self> {
        ensure!(
            q_heads > 0,
            "head128 workspace requires nonzero query heads"
        );
        let bytes = (q_heads as usize)
            .checked_mul(8 * 130 * std::mem::size_of::<f32>())
            .context("head128 partial-result bytes overflow")?;
        Ok(Self {
            q_head_capacity: q_heads,
            partials: DeviceBuffer::new_tagged(ctx, bytes, "attention:head128_partials")?,
        })
    }

    pub fn bytes(&self) -> usize {
        self.partials.bytes()
    }
}

impl KVCache {
    /// Allocate scratch during cache/session preparation, never in attention.
    pub fn prepare_no_old_attention(&self, ctx: &CudaContext, q_heads: u32) -> Result<()> {
        if self.inner.no_old_attention == NoOldAttention::Off
            || self.inner.head_dim != 128
            || self.inner.exact_old_backing != "fp16-k-q4-v"
        {
            return Ok(());
        }
        ensure!(
            q_heads > 0 && q_heads.is_multiple_of(self.num_heads()),
            "invalid no-old GQA heads"
        );
        let mut workspace = self.inner.head128_attention_workspace.lock().unwrap();
        ensure!(workspace.is_none(), "no-old scratch already prepared");
        *workspace = Some(Head128AttentionWorkspace::new(ctx, q_heads)?);
        eprintln!("[kv-no-old] opt-in dense-recent head128 partitions=8 threads=256 window={} scratch_bytes={}",
            self.inner.recent_window, workspace.as_ref().unwrap().bytes());
        Ok(())
    }

    /// # Safety
    /// Q/output and this cache belong to ctx. seq_len is the authoritative
    /// valid prefix length for this exclusively owned physical slot. All
    /// producers must precede this enqueue on the decode stream.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn try_enqueue_no_old_attention(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q: *const c_void,
        q_heads: u32,
        seq_len: u32,
        recent_window: u32,
        d_out: *mut c_void,
    ) -> Result<bool> {
        if self.inner.no_old_attention == NoOldAttention::Off
            || !self.inner.compressed
            || self.inner.head_dim != 128
            || self.inner.exact_old_backing != "fp16-k-q4-v"
            || recent_window != self.inner.recent_window
            || seq_len == 0
            || seq_len > recent_window
            || seq_len > 8192
        {
            return Ok(false);
        }
        let workspace = self.inner.head128_attention_workspace.lock().unwrap();
        let workspace = workspace
            .as_ref()
            .context("prepare no-old attention scratch before enqueue")?;
        ensure!(
            q_heads <= workspace.q_head_capacity
                && Arc::ptr_eq(&ctx.inner, &workspace.partials.ctx.inner),
            "no-old scratch capacity/context mismatch"
        );
        let _guard = ctx.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_attention_no_old_head128_async(
            ctx.inner.raw.as_ptr(),
            self.inner.raw.as_ptr(),
            seq_id,
            d_q,
            q_heads,
            seq_len,
            recent_window,
            workspace.partials.as_mut_ptr(),
            workspace.bytes(),
            d_out,
        );
        ensure!(
            rc == 0,
            "m40llm_attention_no_old_head128_async failed: {rc}"
        );
        record_async_kernel("attention_no_old_head128_parallel");
        record_async_kernel("attention_no_old_head128_merge");
        Ok(true)
    }

    /// # Safety
    /// Q/output must hold `q_heads * 128` floats on this cache's CUDA context.
    /// They and the workspace must stay live until decode-stream completion.
    /// This API retains all absolute KV positions in `[0, seq_len)`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_head128_parallel_async(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q: *const c_void,
        q_heads: u32,
        seq_len: u32,
        d_out: *mut c_void,
        config: Head128AttentionConfig,
        workspace: Option<&Head128AttentionWorkspace>,
    ) -> Result<()> {
        config.validate()?;
        ensure!(
            !self.inner.compressed && self.inner.head_dim == 128,
            "parallel head128 attention requires dense FP16 KV with head_dim=128"
        );
        let (partials, bytes) = if config.partitions == 1 {
            (std::ptr::null_mut(), 0)
        } else {
            let workspace = workspace.context("split attention requires a reusable workspace")?;
            ensure!(
                q_heads <= workspace.q_head_capacity
                    && Arc::ptr_eq(&ctx.inner, &workspace.partials.ctx.inner),
                "head128 attention workspace capacity/context mismatch"
            );
            (workspace.partials.as_mut_ptr(), workspace.bytes())
        };
        let diag = Q8DecodeEventDiag::start_dynamic(
            ctx,
            format!(
                "attention_head128_parallel p={} seq_len={seq_len}",
                config.partitions
            ),
            CudaStream::Decode,
        )?;
        let guard = ctx.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_attention_head128_parallel_async(
            ctx.inner.raw.as_ptr(),
            self.inner.raw.as_ptr(),
            seq_id,
            d_q,
            q_heads,
            seq_len,
            config.partitions,
            config.threads,
            partials,
            bytes,
            d_out,
        );
        ensure!(
            rc == 0,
            "m40llm_attention_head128_parallel_async failed: {rc}"
        );
        record_async_kernel("attention_head128_parallel");
        if config.partitions > 1 {
            record_async_kernel("attention_head128_merge");
        }
        drop(guard);
        if let Some(diag) = diag {
            diag.finish(ctx)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn enqueue_head128_cached(
        &self,
        ctx: &CudaContext,
        seq_id: u32,
        d_q: *const c_void,
        q_heads: u32,
        seq_len: u32,
        d_out: *mut c_void,
        config: Head128AttentionConfig,
    ) -> Result<()> {
        let mut workspace = self.inner.head128_attention_workspace.lock().unwrap();
        if config.partitions > 1
            && workspace
                .as_ref()
                .is_none_or(|value| value.q_head_capacity < q_heads)
        {
            *workspace = Some(Head128AttentionWorkspace::new(ctx, q_heads)?);
        }
        self.attention_head128_parallel_async(
            ctx,
            seq_id,
            d_q,
            q_heads,
            seq_len,
            d_out,
            config,
            workspace.as_ref(),
        )
    }

    /// Scratch is separate from resident KV allocation, and zero when disabled.
    pub fn head128_attention_workspace_bytes(&self) -> usize {
        self.inner
            .head128_attention_workspace
            .lock()
            .unwrap()
            .as_ref()
            .map_or(0, Head128AttentionWorkspace::bytes)
    }
}
