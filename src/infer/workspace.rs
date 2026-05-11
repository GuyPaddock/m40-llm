use crate::cuda::{CudaContext, DeviceBuffer};
use anyhow::{Context, Result};
use std::ffi::c_void;

#[derive(Debug)]
pub struct ForwardWorkspace {
    d_model: usize,
    kv_dim: usize,
    hidden_dim: usize,
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
}

#[derive(Debug, Clone, Copy)]
pub struct ForwardWorkspacePtrs {
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
}

impl ForwardWorkspace {
    pub fn new(
        ctx: &CudaContext,
        d_model: usize,
        kv_dim: usize,
        hidden_dim: usize,
    ) -> Result<Self> {
        if d_model == 0 || kv_dim == 0 || hidden_dim == 0 {
            anyhow::bail!(
                "ForwardWorkspace: invalid dims d_model={d_model} kv_dim={kv_dim} hidden_dim={hidden_dim}"
            );
        }

        let bytes_d = d_model
            .checked_mul(std::mem::size_of::<f32>())
            .context("workspace d_model bytes overflow")?;
        let bytes_kv = kv_dim
            .checked_mul(std::mem::size_of::<f32>())
            .context("workspace kv bytes overflow")?;
        let bytes_h = hidden_dim
            .checked_mul(std::mem::size_of::<f32>())
            .context("workspace hidden bytes overflow")?;

        Ok(Self {
            d_model,
            kv_dim,
            hidden_dim,
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
        })
    }

    pub fn matches(&self, d_model: usize, kv_dim: usize, hidden_dim: usize) -> bool {
        self.d_model == d_model && self.kv_dim == kv_dim && self.hidden_dim == hidden_dim
    }

    pub fn ptrs(&self) -> ForwardWorkspacePtrs {
        ForwardWorkspacePtrs {
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
        }
    }

    pub fn free(self, _ctx: &CudaContext) {
        drop(self);
    }
}
