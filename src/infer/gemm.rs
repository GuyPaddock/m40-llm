#[cfg(feature = "cuda")]
use super::types::{MaterializedWeight, MaterializedWeightKey};
use super::LoadedModel;
#[cfg(feature = "cuda")]
use crate::gguf::GgmlDType;
use anyhow::{Context, Result};
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::sync::Once;

#[cfg(feature = "cuda")]
fn materialized_weights_enabled() -> bool {
    std::env::var("M40LLM_MATERIALIZE_F32_WEIGHTS")
        .map(|v| v != "0")
        .unwrap_or(true)
}

#[cfg(feature = "cuda")]
fn gemm_log_enabled() -> bool {
    std::env::var("M40LLM_GEMM_LOG").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
fn materialized_budget_bytes() -> Option<usize> {
    let mb = std::env::var("M40LLM_MATERIALIZE_F32_BUDGET_MB")
        .ok()?
        .parse::<usize>()
        .ok()?;
    mb.checked_mul(1024)?.checked_mul(1024)
}

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
        #[cfg(feature = "cuda")]
        {
            if materialized_weights_enabled() && cfg!(have_cublas) {
                match self.materialized_gguf_weight(d_b_f16, n, k) {
                    Ok(d_b_f32) => {
                        if let Err(err) = self
                            .cuda
                            .gemm_f32xf32_f32(d_a_f32, d_b_f32, d_c_f32, m, n, k)
                        {
                            if gemm_log_enabled() {
                                eprintln!(
                                    "[cuda] materialized f32 GEMM failed; falling back to GGUF F16 kernel: {err}"
                                );
                            }
                        } else {
                            return Ok(());
                        }
                    }
                    Err(err) => {
                        if gemm_log_enabled() {
                            eprintln!(
                                "[cuda] GGUF F16 weight materialization failed; falling back to GGUF F16 kernel: {err}"
                            );
                        }
                    }
                }
            }
        }
        self.cuda
            .gemm_f32xf16_gguf_f32(d_a_f32, d_b_f16, d_c_f32, m, n, k)
    }

    #[cfg(feature = "cuda")]
    fn materialized_gguf_weight(
        &self,
        d_b_f16: *const c_void,
        n: i32,
        k: i32,
    ) -> Result<*const c_void> {
        self.log_materialization_budget_once();
        let key = MaterializedWeightKey {
            src: d_b_f16 as usize,
            n,
            k,
        };

        if let Some(existing) = self.materialized_weights.lock().unwrap().get(&key) {
            return Ok(existing.dptr as *const c_void);
        }

        let elems = (n as usize)
            .checked_mul(k as usize)
            .context("materialized weight element count overflow")?;
        let bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .context("materialized weight byte size overflow")?;
        let tensor_name = self.materialized_tensor_name(d_b_f16);
        let current_bytes = self.current_materialized_f32_bytes();
        if let Some(budget) = materialized_budget_bytes() {
            let after_bytes = current_bytes
                .checked_add(bytes)
                .context("materialized weight budget byte overflow")?;
            if after_bytes > budget {
                anyhow::bail!(
                    "materialized f32 budget exceeded for tensor={} bytes={} current={} budget={}",
                    tensor_name,
                    bytes,
                    current_bytes,
                    budget
                );
            }
        }
        let dptr = self.cuda.device_malloc(bytes)?;
        let materialize_res = unsafe {
            self.cuda
                .materialize_gguf_f16_to_f32_colmajor_nt(d_b_f16, dptr, n, k)
        };
        if let Err(err) = materialize_res {
            unsafe {
                let _ = self.cuda.device_free(dptr);
            }
            return Err(err);
        }

        let mut weights = self.materialized_weights.lock().unwrap();
        if let Some(existing) = weights.get(&key) {
            unsafe {
                let _ = self.cuda.device_free(dptr);
            }
            return Ok(existing.dptr as *const c_void);
        }
        weights.insert(key, MaterializedWeight { dptr, bytes });
        if gemm_log_enabled() {
            let total = weights.values().map(|w| w.bytes).sum::<usize>();
            let budget = materialized_budget_bytes()
                .map(|b| b.to_string())
                .unwrap_or_else(|| "unbounded".to_string());
            eprintln!(
                "[cuda] materialized_f32_weight: tensor={} bytes={} total={} budget={}",
                tensor_name, bytes, total, budget
            );
        }
        Ok(dptr as *const c_void)
    }

    #[cfg(feature = "cuda")]
    fn current_materialized_f32_bytes(&self) -> usize {
        self.materialized_weights
            .lock()
            .unwrap()
            .values()
            .map(|w| w.bytes)
            .sum()
    }

    #[cfg(feature = "cuda")]
    fn estimated_materialized_f32_bytes(&self) -> usize {
        self.device_tensors
            .values()
            .filter(|tensor| tensor.dtype == GgmlDType::F16 && tensor.shape.len() == 2)
            .filter_map(|tensor| {
                tensor
                    .shape
                    .iter()
                    .try_fold(1usize, |acc, &dim| {
                        acc.checked_mul(usize::try_from(dim).ok()?)
                    })
                    .and_then(|elems| elems.checked_mul(std::mem::size_of::<f32>()))
            })
            .sum()
    }

    #[cfg(feature = "cuda")]
    fn materialized_tensor_name(&self, d_b_f16: *const c_void) -> String {
        self.device_tensors
            .iter()
            .find_map(|(name, tensor)| {
                (tensor.dptr as *const c_void == d_b_f16).then(|| name.clone())
            })
            .unwrap_or_else(|| format!("ptr={d_b_f16:?}"))
    }

    #[cfg(feature = "cuda")]
    fn log_materialization_budget_once(&self) {
        if !gemm_log_enabled() {
            return;
        }
        static LOG_ONCE: Once = Once::new();
        LOG_ONCE.call_once(|| {
            let estimated = self.estimated_materialized_f32_bytes();
            let budget = materialized_budget_bytes()
                .map(|b| b.to_string())
                .unwrap_or_else(|| "unbounded".to_string());
            eprintln!(
                "[cuda] materialized_f32_budget: estimated_f16_2d_bytes={} budget={}",
                estimated, budget
            );
        });
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
