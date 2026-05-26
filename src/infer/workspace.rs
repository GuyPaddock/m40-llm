use crate::cuda::{CudaContext, DeviceBuffer};
use anyhow::{Context, Result};
use std::ffi::c_void;

#[derive(Debug)]
pub struct ForwardWorkspace {
    d_model: usize,
    kv_dim: usize,
    hidden_dim: usize,
    rows: usize,
    dq: DeviceBuffer,
    dk: DeviceBuffer,
    dv: DeviceBuffer,
    datt: DeviceBuffer,
    dy_attn: DeviceBuffer,
    dgate: DeviceBuffer,
    dup: DeviceBuffer,
    dhid: DeviceBuffer,
    dy_mlp: DeviceBuffer,
    d_xn: DeviceBuffer,
    d_x1: DeviceBuffer,
    d_x1n: DeviceBuffer,
    scratch_a: DeviceBuffer,
    scratch_b: DeviceBuffer,
    d_attention_seq_ids: DeviceBuffer,
    d_attention_seq_lens: DeviceBuffer,
}

#[derive(Debug, Clone, Copy)]
pub struct ForwardWorkspacePtrs {
    pub rows: usize,
    pub dq: *mut c_void,
    pub dk: *mut c_void,
    pub dv: *mut c_void,
    pub datt: *mut c_void,
    pub dy_attn: *mut c_void,
    pub dgate: *mut c_void,
    pub dup: *mut c_void,
    pub dhid: *mut c_void,
    pub dy_mlp: *mut c_void,
    pub d_xn: *mut c_void,
    pub d_x1: *mut c_void,
    pub d_x1n: *mut c_void,
    pub scratch_a: *mut c_void,
    pub scratch_b: *mut c_void,
    pub d_attention_seq_ids: *mut c_void,
    pub d_attention_seq_lens: *mut c_void,
}

impl ForwardWorkspace {
    pub fn estimated_bytes(
        d_model: usize,
        kv_dim: usize,
        hidden_dim: usize,
        rows: usize,
    ) -> Result<usize> {
        if d_model == 0 || kv_dim == 0 || hidden_dim == 0 {
            anyhow::bail!(
                "ForwardWorkspace: invalid dims d_model={d_model} kv_dim={kv_dim} hidden_dim={hidden_dim}"
            );
        }
        if rows == 0 {
            anyhow::bail!("ForwardWorkspace: rows must be greater than zero");
        }

        let f32_bytes = std::mem::size_of::<f32>();
        let u32_bytes = std::mem::size_of::<u32>();
        let bytes_d = d_model
            .checked_mul(rows)
            .context("workspace d_model bytes overflow")?
            .checked_mul(f32_bytes)
            .context("workspace d_model bytes overflow")?;
        let bytes_kv = kv_dim
            .checked_mul(rows)
            .context("workspace kv bytes overflow")?
            .checked_mul(f32_bytes)
            .context("workspace kv bytes overflow")?;
        let bytes_h = hidden_dim
            .checked_mul(rows)
            .context("workspace hidden bytes overflow")?
            .checked_mul(f32_bytes)
            .context("workspace hidden bytes overflow")?;
        let bytes_attention_meta = rows
            .checked_mul(u32_bytes)
            .context("workspace attention metadata bytes overflow")?;

        bytes_d
            .checked_mul(9)
            .context("workspace d_model aggregate bytes overflow")?
            .checked_add(
                bytes_kv
                    .checked_mul(2)
                    .context("workspace kv aggregate bytes overflow")?,
            )
            .context("workspace aggregate bytes overflow")?
            .checked_add(
                bytes_h
                    .checked_mul(3)
                    .context("workspace hidden aggregate bytes overflow")?,
            )
            .context("workspace aggregate bytes overflow")?
            .checked_add(
                bytes_attention_meta
                    .checked_mul(2)
                    .context("workspace metadata aggregate bytes overflow")?,
            )
            .context("workspace aggregate bytes overflow")
    }

    pub fn new(
        ctx: &CudaContext,
        d_model: usize,
        kv_dim: usize,
        hidden_dim: usize,
    ) -> Result<Self> {
        Self::new_with_rows(ctx, d_model, kv_dim, hidden_dim, 1)
    }

    pub fn new_with_rows(
        ctx: &CudaContext,
        d_model: usize,
        kv_dim: usize,
        hidden_dim: usize,
        rows: usize,
    ) -> Result<Self> {
        if d_model == 0 || kv_dim == 0 || hidden_dim == 0 {
            anyhow::bail!(
                "ForwardWorkspace: invalid dims d_model={d_model} kv_dim={kv_dim} hidden_dim={hidden_dim}"
            );
        }
        if rows == 0 {
            anyhow::bail!("ForwardWorkspace: rows must be greater than zero");
        }

        let bytes_d = d_model
            .checked_mul(rows)
            .context("workspace d_model bytes overflow")?
            .checked_mul(std::mem::size_of::<f32>())
            .context("workspace d_model bytes overflow")?;
        let bytes_kv = kv_dim
            .checked_mul(rows)
            .context("workspace kv bytes overflow")?
            .checked_mul(std::mem::size_of::<f32>())
            .context("workspace kv bytes overflow")?;
        let bytes_h = hidden_dim
            .checked_mul(rows)
            .context("workspace hidden bytes overflow")?
            .checked_mul(std::mem::size_of::<f32>())
            .context("workspace hidden bytes overflow")?;
        let bytes_attention_meta = rows
            .checked_mul(std::mem::size_of::<u32>())
            .context("workspace attention metadata bytes overflow")?;

        Ok(Self {
            d_model,
            kv_dim,
            hidden_dim,
            rows,
            dq: DeviceBuffer::new_tagged(ctx, bytes_d, "fwd:dq_f32")?,
            dk: DeviceBuffer::new_tagged(ctx, bytes_kv, "fwd:dk_f32")?,
            dv: DeviceBuffer::new_tagged(ctx, bytes_kv, "fwd:dv_f32")?,
            datt: DeviceBuffer::new_tagged(ctx, bytes_d, "fwd:datt_f32")?,
            dy_attn: DeviceBuffer::new_tagged(ctx, bytes_d, "fwd:dy_attn_f32")?,
            dgate: DeviceBuffer::new_tagged(ctx, bytes_h, "fwd:dgate_f32")?,
            dup: DeviceBuffer::new_tagged(ctx, bytes_h, "fwd:dup_f32")?,
            dhid: DeviceBuffer::new_tagged(ctx, bytes_h, "fwd:dhid_f32")?,
            dy_mlp: DeviceBuffer::new_tagged(ctx, bytes_d, "fwd:dy_mlp_f32")?,
            d_xn: DeviceBuffer::new_tagged(ctx, bytes_d, "fwd:d_xn_f32")?,
            d_x1: DeviceBuffer::new_tagged(ctx, bytes_d, "fwd:d_x1_f32")?,
            d_x1n: DeviceBuffer::new_tagged(ctx, bytes_d, "fwd:d_x1n_f32")?,
            scratch_a: DeviceBuffer::new_tagged(ctx, bytes_d, "fwd_all:scratch_a_f32")?,
            scratch_b: DeviceBuffer::new_tagged(ctx, bytes_d, "fwd_all:scratch_b_f32")?,
            d_attention_seq_ids: DeviceBuffer::new_tagged(
                ctx,
                bytes_attention_meta,
                "fwd:batch_attention_seq_ids",
            )?,
            d_attention_seq_lens: DeviceBuffer::new_tagged(
                ctx,
                bytes_attention_meta,
                "fwd:batch_attention_seq_lens",
            )?,
        })
    }

    pub fn matches(&self, d_model: usize, kv_dim: usize, hidden_dim: usize, rows: usize) -> bool {
        self.d_model == d_model
            && self.kv_dim == kv_dim
            && self.hidden_dim == hidden_dim
            && self.rows == rows
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn ptrs(&self) -> ForwardWorkspacePtrs {
        ForwardWorkspacePtrs {
            rows: self.rows,
            dq: self.dq.as_mut_ptr(),
            dk: self.dk.as_mut_ptr(),
            dv: self.dv.as_mut_ptr(),
            datt: self.datt.as_mut_ptr(),
            dy_attn: self.dy_attn.as_mut_ptr(),
            dgate: self.dgate.as_mut_ptr(),
            dup: self.dup.as_mut_ptr(),
            dhid: self.dhid.as_mut_ptr(),
            dy_mlp: self.dy_mlp.as_mut_ptr(),
            d_xn: self.d_xn.as_mut_ptr(),
            d_x1: self.d_x1.as_mut_ptr(),
            d_x1n: self.d_x1n.as_mut_ptr(),
            scratch_a: self.scratch_a.as_mut_ptr(),
            scratch_b: self.scratch_b.as_mut_ptr(),
            d_attention_seq_ids: self.d_attention_seq_ids.as_mut_ptr(),
            d_attention_seq_lens: self.d_attention_seq_lens.as_mut_ptr(),
        }
    }

    pub fn free(self, _ctx: &CudaContext) {
        drop(self);
    }
}
