use super::LoadedModel;
use anyhow::{Context, Result};
use std::ffi::c_void;

impl LoadedModel {
    pub unsafe fn run_gemm(
        &self,
        d_a: *const c_void,
        d_b: *const c_void,
        d_c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        self.cuda.gemm_f16_f32(d_a, d_b, d_c, m, n, k)
    }

    pub unsafe fn mlp_gates_f32xf16_f32(
        &self,
        d_x_f32: *const c_void,
        m: i32,
        k: i32,
        d_w_gate_f16: *const c_void,
        d_w_up_f16: *const c_void,
        h: i32,
        d_gate_out_f32: *mut c_void,
        d_up_out_f32: *mut c_void,
    ) -> Result<()> {
        if m <= 0 || k <= 0 || h <= 0 {
            anyhow::bail!("mlp_gates: invalid dims");
        }
        self.matmul_f32xf16_f32(d_x_f32, d_w_gate_f16, d_gate_out_f32, m, h, k)
            .with_context(|| format!("mlp gate GEMM failed: m={m} n={h} k={k}"))?;
        self.matmul_f32xf16_f32(d_x_f32, d_w_up_f16, d_up_out_f32, m, h, k)
            .with_context(|| format!("mlp up GEMM failed: m={m} n={h} k={k}"))
    }

    /// # Safety
    /// MLP projections against GGUF F16 weights with logical shapes [K,H].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mlp_gates_f32xf16_gguf_f32(
        &self,
        d_x_f32: *const c_void,
        m: i32,
        k: i32,
        d_w_gate_f16: *const c_void,
        d_w_up_f16: *const c_void,
        h: i32,
        d_gate_out_f32: *mut c_void,
        d_up_out_f32: *mut c_void,
    ) -> Result<()> {
        if m <= 0 || k <= 0 || h <= 0 {
            anyhow::bail!("mlp_gates: invalid dims");
        }
        self.matmul_f32xf16_gguf_f32(d_x_f32, d_w_gate_f16, d_gate_out_f32, m, h, k)
            .with_context(|| format!("mlp gate GGUF GEMM failed: m={m} n={h} k={k}"))?;
        self.matmul_f32xf16_gguf_f32(d_x_f32, d_w_up_f16, d_up_out_f32, m, h, k)
            .with_context(|| format!("mlp up GGUF GEMM failed: m={m} n={h} k={k}"))
    }

    /// # Safety
    /// Down projection: Y = H · W_down, where H is hidden f32 (MxH), W_down is f16 (H x N), output f32 (MxN)
    pub unsafe fn mlp_down_proj_f32xf16_f32(
        &self,
        d_hidden_f32: *const c_void,
        m: i32,
        h: i32,
        d_w_down_f16: *const c_void,
        n: i32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        if m <= 0 || h <= 0 || n <= 0 {
            anyhow::bail!("mlp_down_proj: invalid dims");
        }
        self.matmul_f32xf16_f32(d_hidden_f32, d_w_down_f16, d_out_f32, m, n, h)
            .with_context(|| format!("mlp down GEMM failed: m={m} n={n} k={h}"))
    }

    /// # Safety
    /// Down projection against a GGUF F16 weight with logical shape [H,N].
    pub unsafe fn mlp_down_proj_f32xf16_gguf_f32(
        &self,
        d_hidden_f32: *const c_void,
        m: i32,
        h: i32,
        d_w_down_f16: *const c_void,
        n: i32,
        d_out_f32: *mut c_void,
    ) -> Result<()> {
        if m <= 0 || h <= 0 || n <= 0 {
            anyhow::bail!("mlp_down_proj: invalid dims");
        }
        self.matmul_f32xf16_gguf_f32(d_hidden_f32, d_w_down_f16, d_out_f32, m, n, h)
            .with_context(|| format!("mlp down GGUF GEMM failed: m={m} n={n} k={h}"))
    }
}

impl LoadedModel {
    /// # Safety
    /// f16 × f16 → f32 row-major GEMM. Device pointers must be valid on the same device as the context.
    pub unsafe fn matmul_f16xf16_f32(
        &self,
        d_a_f16: *const c_void,
        d_b_f16: *const c_void,
        d_c_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        self.cuda
            .gemm_f16xf16_f32(d_a_f16, d_b_f16, d_c_f32, m, n, k)
    }

    /// # Safety
    /// f32 × f16 → f32 row-major GEMM, useful when input activations are f32 and weights are f16.
    pub unsafe fn matmul_f32xf16_f32(
        &self,
        d_a_f32: *const c_void,
        d_b_f16: *const c_void,
        d_c_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        self.cuda
            .gemm_f32xf16_f32(d_a_f32, d_b_f16, d_c_f32, m, n, k)
    }

    /// # Safety
    /// f32 × GGUF F16 → f32 GEMM. The weight tensor has logical shape [K,N]
    /// with dimension 0 stored contiguously, as GGUF writes tensor data.
    pub unsafe fn matmul_f32xf16_gguf_f32(
        &self,
        d_a_f32: *const c_void,
        d_b_f16: *const c_void,
        d_c_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        self.cuda
            .gemm_f32xf16_gguf_f32(d_a_f32, d_b_f16, d_c_f32, m, n, k)
    }
}

impl LoadedModel {
    /// # Safety
    /// A f32 (MxK) × Wq/Wk/Wv f16 (KxNq/KxNk/KxNv) → Q/K/V f32 (MxN*)
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn qkv_project_f32xf16_f32(
        &self,
        d_x_f32: *const c_void,
        m: i32,
        k: i32,
        d_wq_f16: *const c_void,
        n_q: i32,
        d_wk_f16: *const c_void,
        n_k: i32,
        d_wv_f16: *const c_void,
        n_v: i32,
        d_q_out_f32: *mut c_void,
        d_k_out_f32: *mut c_void,
        d_v_out_f32: *mut c_void,
    ) -> Result<()> {
        if m <= 0 || k <= 0 || n_q <= 0 || n_k <= 0 || n_v <= 0 {
            anyhow::bail!("qkv_project: invalid dims");
        }
        self.matmul_f32xf16_f32(d_x_f32, d_wq_f16, d_q_out_f32, m, n_q, k)
            .with_context(|| format!("Q projection GEMM failed: m={m} n={n_q} k={k}"))?;
        self.matmul_f32xf16_f32(d_x_f32, d_wk_f16, d_k_out_f32, m, n_k, k)
            .with_context(|| format!("K projection GEMM failed: m={m} n={n_k} k={k}"))?;
        self.matmul_f32xf16_f32(d_x_f32, d_wv_f16, d_v_out_f32, m, n_v, k)
            .with_context(|| format!("V projection GEMM failed: m={m} n={n_v} k={k}"))
    }

    /// # Safety
    /// A f32 (MxK) × GGUF Wq/Wk/Wv F16 (logical KxN*, K-fastest) → Q/K/V f32.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn qkv_project_f32xf16_gguf_f32(
        &self,
        d_x_f32: *const c_void,
        m: i32,
        k: i32,
        d_wq_f16: *const c_void,
        n_q: i32,
        d_wk_f16: *const c_void,
        n_k: i32,
        d_wv_f16: *const c_void,
        n_v: i32,
        d_q_out_f32: *mut c_void,
        d_k_out_f32: *mut c_void,
        d_v_out_f32: *mut c_void,
    ) -> Result<()> {
        if m <= 0 || k <= 0 || n_q <= 0 || n_k <= 0 || n_v <= 0 {
            anyhow::bail!("qkv_project: invalid dims");
        }
        self.matmul_f32xf16_gguf_f32(d_x_f32, d_wq_f16, d_q_out_f32, m, n_q, k)
            .with_context(|| format!("Q projection GGUF GEMM failed: m={m} n={n_q} k={k}"))?;
        self.matmul_f32xf16_gguf_f32(d_x_f32, d_wk_f16, d_k_out_f32, m, n_k, k)
            .with_context(|| format!("K projection GGUF GEMM failed: m={m} n={n_k} k={k}"))?;
        self.matmul_f32xf16_gguf_f32(d_x_f32, d_wv_f16, d_v_out_f32, m, n_v, k)
            .with_context(|| format!("V projection GGUF GEMM failed: m={m} n={n_v} k={k}"))
    }

    /// # Safety
    /// A f32 (MxK) × W f16 (KxN) → Out f32 (MxN)
    pub unsafe fn out_proj_f32xf16_f32(
        &self,
        d_in_f32: *const c_void,
        d_w_out_f16: *const c_void,
        d_out_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        if m <= 0 || n <= 0 || k <= 0 {
            anyhow::bail!("out_proj: invalid dims");
        }
        self.matmul_f32xf16_f32(d_in_f32, d_w_out_f16, d_out_f32, m, n, k)
            .with_context(|| format!("attention output projection GEMM failed: m={m} n={n} k={k}"))
    }

    /// # Safety
    /// A f32 (MxK) × GGUF F16 W (logical KxN, K-fastest) → Out f32.
    pub unsafe fn out_proj_f32xf16_gguf_f32(
        &self,
        d_in_f32: *const c_void,
        d_w_out_f16: *const c_void,
        d_out_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        if m <= 0 || n <= 0 || k <= 0 {
            anyhow::bail!("out_proj: invalid dims");
        }
        self.matmul_f32xf16_gguf_f32(d_in_f32, d_w_out_f16, d_out_f32, m, n, k)
            .with_context(|| {
                format!("attention output projection GGUF GEMM failed: m={m} n={n} k={k}")
            })
    }

    /// # Safety
    /// Generic projection helper A f32 (MxK) × W f16 (KxN) → C f32 (MxN)
    pub unsafe fn project_f32xf16_f32(
        &self,
        d_a_f32: *const c_void,
        d_b_f16: *const c_void,
        d_c_f32: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        self.matmul_f32xf16_f32(d_a_f32, d_b_f16, d_c_f32, m, n, k)
            .with_context(|| format!("projection wrapper GEMM failed: m={m} n={n} k={k}"))
    }
}
