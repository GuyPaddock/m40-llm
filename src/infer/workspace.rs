use crate::cuda::CudaContext;
use anyhow::{Context, Result};
use std::ffi::c_void;

#[derive(Debug)]
pub struct ForwardWorkspace {
    d_model: usize,
    kv_dim: usize,
    hidden_dim: usize,
    dq: *mut c_void,
    dk: *mut c_void,
    dv: *mut c_void,
    datt: *mut c_void,
    dy_attn: *mut c_void,
    dgate: *mut c_void,
    dup: *mut c_void,
    dhid: *mut c_void,
    dy_mlp: *mut c_void,
    d_xn: *mut c_void,
    d_x1: *mut c_void,
    d_x1n: *mut c_void,
    scratch_a: *mut c_void,
    scratch_b: *mut c_void,
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
            dq: ctx.device_malloc_tagged(bytes_d, "fwd:dq_f32")?,
            dk: ctx.device_malloc_tagged(bytes_kv, "fwd:dk_f32")?,
            dv: ctx.device_malloc_tagged(bytes_kv, "fwd:dv_f32")?,
            datt: ctx.device_malloc_tagged(bytes_d, "fwd:datt_f32")?,
            dy_attn: ctx.device_malloc_tagged(bytes_d, "fwd:dy_attn_f32")?,
            dgate: ctx.device_malloc_tagged(bytes_h, "fwd:dgate_f32")?,
            dup: ctx.device_malloc_tagged(bytes_h, "fwd:dup_f32")?,
            dhid: ctx.device_malloc_tagged(bytes_h, "fwd:dhid_f32")?,
            dy_mlp: ctx.device_malloc_tagged(bytes_d, "fwd:dy_mlp_f32")?,
            d_xn: ctx.device_malloc_tagged(bytes_d, "fwd:d_xn_f32")?,
            d_x1: ctx.device_malloc_tagged(bytes_d, "fwd:d_x1_f32")?,
            d_x1n: ctx.device_malloc_tagged(bytes_d, "fwd:d_x1n_f32")?,
            scratch_a: ctx.device_malloc_tagged(bytes_d, "fwd_all:scratch_a_f32")?,
            scratch_b: ctx.device_malloc_tagged(bytes_d, "fwd_all:scratch_b_f32")?,
        })
    }

    pub fn matches(&self, d_model: usize, kv_dim: usize, hidden_dim: usize) -> bool {
        self.d_model == d_model && self.kv_dim == kv_dim && self.hidden_dim == hidden_dim
    }

    pub fn ptrs(&self) -> ForwardWorkspacePtrs {
        ForwardWorkspacePtrs {
            dq: self.dq,
            dk: self.dk,
            dv: self.dv,
            datt: self.datt,
            dy_attn: self.dy_attn,
            dgate: self.dgate,
            dup: self.dup,
            dhid: self.dhid,
            dy_mlp: self.dy_mlp,
            d_xn: self.d_xn,
            d_x1: self.d_x1,
            d_x1n: self.d_x1n,
            scratch_a: self.scratch_a,
            scratch_b: self.scratch_b,
        }
    }

    pub fn free(self, ctx: &CudaContext) {
        unsafe {
            let _ = ctx.device_free(self.dq);
            let _ = ctx.device_free(self.dk);
            let _ = ctx.device_free(self.dv);
            let _ = ctx.device_free(self.datt);
            let _ = ctx.device_free(self.dy_attn);
            let _ = ctx.device_free(self.dgate);
            let _ = ctx.device_free(self.dup);
            let _ = ctx.device_free(self.dhid);
            let _ = ctx.device_free(self.dy_mlp);
            let _ = ctx.device_free(self.d_xn);
            let _ = ctx.device_free(self.d_x1);
            let _ = ctx.device_free(self.d_x1n);
            let _ = ctx.device_free(self.scratch_a);
            let _ = ctx.device_free(self.scratch_b);
        }
    }
}
