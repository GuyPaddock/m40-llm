#[cfg(feature = "cuda")]
use super::types::{MaterializedWeight, MaterializedWeightKey};
#[cfg(feature = "cuda")]
use super::workspace::ForwardWorkspace;
use super::LoadedModel;
#[cfg(feature = "cuda")]
use super::{ProjectionBackendDecision, ProjectionBackendEstimate, ProjectionBackendMode};
#[cfg(feature = "cuda")]
use crate::cuda::CudaStream;
#[cfg(feature = "cuda")]
use crate::gguf::GgmlDType;
use anyhow::{Context, Result};
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::sync::Once;

#[cfg(feature = "cuda")]
fn materialized_weights_env_enabled() -> bool {
    std::env::var("M40LLM_MATERIALIZE_F32_WEIGHTS")
        .map(|v| v != "0")
        .unwrap_or(true)
}

#[cfg(feature = "cuda")]
fn gemm_log_enabled() -> bool {
    std::env::var("M40LLM_GEMM_LOG").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
pub(crate) fn decode_cublas_single_stream_enabled() -> bool {
    let requested = matches!(
        std::env::var("M40LLM_DECODE_CUBLAS_STREAM").ok().as_deref(),
        Some("decode") | Some("1") | Some("true") | Some("TRUE")
    );
    requested
        && materialized_weights_env_enabled()
        && std::env::var("M40LLM_PROJECTION_BACKEND")
            .map(|value| value != "large-model")
            .unwrap_or(true)
}

#[cfg(feature = "cuda")]
fn materialized_gemm_stream(m: i32) -> CudaStream {
    if m == 1 && decode_cublas_single_stream_enabled() {
        CudaStream::Decode
    } else {
        CudaStream::Prefill
    }
}

#[cfg(feature = "cuda")]
fn f16_decode_kernel_enabled() -> bool {
    matches!(
        std::env::var("M40LLM_F16_DECODE_KERNEL").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn f16_decode_kernel_decode_stream_enabled() -> bool {
    f16_decode_kernel_enabled()
        && matches!(
            std::env::var("M40LLM_F16_DECODE_STREAM").ok().as_deref(),
            Some("decode") | Some("DECODE")
        )
}

#[cfg(feature = "cuda")]
pub(crate) fn q8_decode_kernel_decode_stream_enabled() -> bool {
    matches!(
        std::env::var("M40LLM_Q8_DECODE_STREAM").ok().as_deref(),
        Some("decode") | Some("DECODE")
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn q4_decode_kernel_decode_stream_enabled() -> bool {
    matches!(
        std::env::var("M40LLM_Q4_DECODE_STREAM").ok().as_deref(),
        Some("decode") | Some("DECODE")
    ) || q8_decode_kernel_decode_stream_enabled()
}

#[cfg(feature = "cuda")]
fn fused_mlp_swiglu_decode_enabled() -> bool {
    f16_decode_kernel_enabled()
        && matches!(
            std::env::var("M40LLM_FUSED_MLP_SWIGLU").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE")
        )
}

#[cfg(feature = "cuda")]
fn fused_q8_mlp_swiglu_decode_enabled() -> bool {
    matches!(
        std::env::var("M40LLM_FUSED_Q8_MLP_SWIGLU").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    )
}

#[cfg(feature = "cuda")]
fn fused_q4_mlp_swiglu_decode_enabled() -> bool {
    matches!(
        std::env::var("M40LLM_FUSED_Q4_MLP_SWIGLU").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    ) || fused_q8_mlp_swiglu_decode_enabled()
}

#[cfg(feature = "cuda")]
fn fused_qkv_decode_enabled() -> bool {
    f16_decode_kernel_enabled()
        && matches!(
            std::env::var("M40LLM_FUSED_QKV").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE")
        )
}

#[cfg(feature = "cuda")]
fn fused_q8_qkv_decode_enabled() -> bool {
    matches!(
        std::env::var("M40LLM_FUSED_Q8_QKV").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    )
}

#[cfg(feature = "cuda")]
fn fused_q4_qkv_decode_enabled() -> bool {
    matches!(
        std::env::var("M40LLM_FUSED_Q4_QKV").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    ) || fused_q8_qkv_decode_enabled()
}

#[cfg(feature = "cuda")]
fn materialized_budget_bytes() -> Option<usize> {
    let mb = std::env::var("M40LLM_MATERIALIZE_F32_BUDGET_MB")
        .ok()?
        .parse::<usize>()
        .ok()?;
    mb.checked_mul(1024)?.checked_mul(1024)
}

#[cfg(feature = "cuda")]
fn fast_fits_budget_bytes() -> Option<usize> {
    let mb = std::env::var("M40LLM_FAST_FITS_BUDGET_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(22 * 1024);
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
    /// Async MLP projections against GGUF F16 weights with logical shapes [K,H].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mlp_gates_f32xf16_gguf_f32_async(
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
        self.matmul_f32xf16_gguf_f32_async(d_x_f32, d_w_gate_f16, d_gate_out_f32, m, h, k)
            .with_context(|| format!("mlp gate async GGUF GEMM failed: m={m} n={h} k={k}"))?;
        self.matmul_f32xf16_gguf_f32_async(d_x_f32, d_w_up_f16, d_up_out_f32, m, h, k)
            .with_context(|| format!("mlp up async GGUF GEMM failed: m={m} n={h} k={k}"))
    }

    /// # Safety
    /// Optional single-token fused GGUF F16 gate/up projection plus SwiGLU.
    /// Returns `Ok(false)` when the opt-in path is not applicable.
    pub unsafe fn try_mlp_gate_up_swiglu_f32xf16_gguf_decode_async(
        &self,
        d_x_f32: *const c_void,
        m: i32,
        k: i32,
        d_w_gate_f16: *const c_void,
        d_w_up_f16: *const c_void,
        h: i32,
        d_out_f32: *mut c_void,
    ) -> Result<bool> {
        #[cfg(feature = "cuda")]
        {
            if m == 1
                && k > 0
                && h > 0
                && fused_q4_mlp_swiglu_decode_enabled()
                && self.gguf_weight_dtype(d_w_gate_f16) == GgmlDType::Q4_0
                && self.gguf_weight_dtype(d_w_up_f16) == GgmlDType::Q4_0
            {
                self.cuda
                    .mlp_gate_up_swiglu_f32xq4_0_gguf_decode_async(
                        d_x_f32,
                        d_w_gate_f16,
                        d_w_up_f16,
                        d_out_f32,
                        h,
                        k,
                    )
                    .with_context(|| {
                        format!("fused Q4 MLP gate/up SwiGLU failed: m={m} h={h} k={k}")
                    })?;
                return Ok(true);
            }
            if m == 1
                && k > 0
                && h > 0
                && fused_q8_mlp_swiglu_decode_enabled()
                && self.gguf_weight_dtype(d_w_gate_f16) == GgmlDType::Q8_0
                && self.gguf_weight_dtype(d_w_up_f16) == GgmlDType::Q8_0
            {
                self.cuda
                    .mlp_gate_up_swiglu_f32xq8_0_gguf_decode_async(
                        d_x_f32,
                        d_w_gate_f16,
                        d_w_up_f16,
                        d_out_f32,
                        h,
                        k,
                    )
                    .with_context(|| {
                        format!("fused Q8 MLP gate/up SwiGLU failed: m={m} h={h} k={k}")
                    })?;
                return Ok(true);
            }
            if m == 1
                && k > 0
                && h > 0
                && fused_mlp_swiglu_decode_enabled()
                && self.gguf_weight_dtype(d_w_gate_f16) == GgmlDType::F16
                && self.gguf_weight_dtype(d_w_up_f16) == GgmlDType::F16
            {
                self.cuda
                    .mlp_gate_up_swiglu_f32xf16_gguf_decode_async(
                        d_x_f32,
                        d_w_gate_f16,
                        d_w_up_f16,
                        d_out_f32,
                        h,
                        k,
                    )
                    .with_context(|| {
                        format!("fused MLP gate/up SwiGLU failed: m={m} h={h} k={k}")
                    })?;
                return Ok(true);
            }
        }
        let _ = (d_x_f32, m, k, d_w_gate_f16, d_w_up_f16, h, d_out_f32);
        Ok(false)
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

    /// # Safety
    /// Async down projection against a GGUF F16 weight with logical shape [H,N].
    pub unsafe fn mlp_down_proj_f32xf16_gguf_f32_async(
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
        self.matmul_f32xf16_gguf_f32_async(d_hidden_f32, d_w_down_f16, d_out_f32, m, n, h)
            .with_context(|| format!("mlp down async GGUF GEMM failed: m={m} n={n} k={h}"))
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
            if self.gguf_weight_dtype(d_b_f16) == GgmlDType::Q8_0 {
                return self
                    .cuda
                    .gemm_f32xq8_0_gguf_f32(d_a_f32, d_b_f16, d_c_f32, m, n, k);
            }
            if self.gguf_weight_dtype(d_b_f16) == GgmlDType::Q4_0 {
                return self
                    .cuda
                    .gemm_f32xq4_0_gguf_f32(d_a_f32, d_b_f16, d_c_f32, m, n, k);
            }
            if self.gguf_weight_dtype(d_b_f16) == GgmlDType::Q6K && m == 1 {
                return self
                    .cuda
                    .gemm_f32xq6_k_gguf_f32_decode_async(d_a_f32, d_b_f16, d_c_f32, m, n, k)
                    .and_then(|_| {
                        self.cuda
                            .synchronize_stream(crate::cuda::CudaStream::Decode)
                    });
            }
            if self.projection_backend_allows_materialized_f32() {
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

    /// # Safety
    /// Async f32 × GGUF F16 → f32 GEMM when the materialized FP32 cuBLAS path is
    /// available. If materialization or async cuBLAS enqueue fails, falls back to
    /// the existing synchronized GGUF F16 kernel for correctness.
    pub unsafe fn matmul_f32xf16_gguf_f32_async(
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
            if self.gguf_weight_dtype(d_b_f16) == GgmlDType::Q8_0 {
                return self
                    .cuda
                    .gemm_f32xq8_0_gguf_f32_async(d_a_f32, d_b_f16, d_c_f32, m, n, k);
            }
            if self.gguf_weight_dtype(d_b_f16) == GgmlDType::Q4_0 {
                return self
                    .cuda
                    .gemm_f32xq4_0_gguf_f32_async(d_a_f32, d_b_f16, d_c_f32, m, n, k);
            }
            if self.gguf_weight_dtype(d_b_f16) == GgmlDType::Q6K && m == 1 {
                return self
                    .cuda
                    .gemm_f32xq6_k_gguf_f32_decode_async(d_a_f32, d_b_f16, d_c_f32, m, n, k);
            }
            if self.gguf_weight_dtype(d_b_f16) == GgmlDType::F16
                && m == 1
                && f16_decode_kernel_enabled()
            {
                return self
                    .cuda
                    .gemm_f32xf16_gguf_f32_decode_async(d_a_f32, d_b_f16, d_c_f32, m, n, k);
            }
            if self.projection_backend_allows_materialized_f32() {
                match self.materialized_gguf_weight(d_b_f16, n, k) {
                    Ok(d_b_f32) => {
                        if let Err(err) = self.cuda.gemm_f32xf32_f32_stream_async(
                            d_a_f32,
                            d_b_f32,
                            d_c_f32,
                            m,
                            n,
                            k,
                            materialized_gemm_stream(m),
                        ) {
                            if gemm_log_enabled() {
                                eprintln!(
                                    "[cuda] async materialized f32 GEMM failed; falling back to GGUF F16 kernel: {err}"
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
        let (tensor_name, byte_offset, dtype, shape) = self.materialized_tensor_identity(d_b_f16);
        let key = MaterializedWeightKey {
            src: d_b_f16 as usize,
            tensor_name: tensor_name.clone(),
            byte_offset,
            dtype,
            shape,
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
        let tensor_label = tensor_name.unwrap_or_else(|| format!("ptr={d_b_f16:?}"));
        let current_bytes = self.current_materialized_f32_bytes();
        if let Some(budget) = materialized_budget_bytes() {
            let after_bytes = current_bytes
                .checked_add(bytes)
                .context("materialized weight budget byte overflow")?;
            if after_bytes > budget {
                anyhow::bail!(
                    "materialized f32 budget exceeded for tensor={} bytes={} current={} budget={}",
                    tensor_label,
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
                tensor_label, bytes, total, budget
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
    pub fn materialized_f32_cache_stats(&self) -> (usize, usize) {
        let weights = self.materialized_weights.lock().unwrap();
        let bytes = weights.values().map(|w| w.bytes).sum();
        (weights.len(), bytes)
    }

    #[cfg(feature = "cuda")]
    pub fn estimated_materialized_f32_bytes(&self) -> usize {
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
    pub fn projection_backend_decision(&self) -> ProjectionBackendDecision {
        let d_model = self.model_config.embedding_length as usize;
        let kv_dim = (self.model_config.attention_head_count_kv
            * self.model_config.attention_key_length) as usize;
        let hidden_dim = self.model_config.feed_forward_length as usize;
        let workspace_bytes =
            ForwardWorkspace::estimated_bytes(d_model, kv_dim, hidden_dim, 1).unwrap_or(0);
        let kv_bytes = self
            .kv_cache
            .as_ref()
            .map(|kv| kv.actual_bytes())
            .unwrap_or(0);
        let estimate = ProjectionBackendEstimate {
            weights_bytes: self.weights_len,
            materialized_f32_bytes: self.estimated_materialized_f32_bytes(),
            workspace_bytes,
            kv_bytes,
        };
        super::backend::choose_projection_backend(
            ProjectionBackendMode::from_env(),
            estimate,
            fast_fits_budget_bytes(),
            cfg!(have_cublas),
            materialized_weights_env_enabled(),
        )
    }

    #[cfg(feature = "cuda")]
    fn projection_backend_allows_materialized_f32(&self) -> bool {
        let decision = self.projection_backend_decision();
        self.log_projection_backend_once(&decision);
        decision.allows_materialized_f32()
    }

    #[cfg(feature = "cuda")]
    fn log_projection_backend_once(&self, decision: &ProjectionBackendDecision) {
        if !gemm_log_enabled() {
            return;
        }
        static LOG_ONCE: Once = Once::new();
        LOG_ONCE.call_once(|| {
            let total = decision
                .estimate
                .total_with_materialization()
                .map(|value| value.to_string())
                .unwrap_or_else(|| "overflow".to_string());
            let budget = decision
                .budget_bytes
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unbounded".to_string());
            eprintln!(
                "[cuda] projection_backend: requested={:?} selected={:?} reason={} cublas={} materialization_enabled={} weights_bytes={} materialized_f32_bytes={} workspace_bytes={} kv_bytes={} fast_fits_total_bytes={} budget_bytes={}",
                decision.requested,
                decision.selected,
                decision.reason,
                decision.cublas_available,
                decision.materialization_enabled,
                decision.estimate.weights_bytes,
                decision.estimate.materialized_f32_bytes,
                decision.estimate.workspace_bytes,
                decision.estimate.kv_bytes,
                total,
                budget
            );
        });
    }

    #[cfg(feature = "cuda")]
    fn materialized_tensor_identity(
        &self,
        d_b_f16: *const c_void,
    ) -> (Option<String>, u64, GgmlDType, Vec<u64>) {
        self.device_tensors
            .iter()
            .find_map(|(name, tensor)| {
                std::ptr::eq(tensor.dptr.cast_const(), d_b_f16).then(|| {
                    (
                        Some(name.clone()),
                        tensor.byte_offset,
                        tensor.dtype,
                        tensor.shape.clone(),
                    )
                })
            })
            .unwrap_or_else(|| (None, 0, GgmlDType::Unknown(u32::MAX), Vec::new()))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn gguf_weight_dtype(&self, d_b: *const c_void) -> GgmlDType {
        self.device_tensors
            .values()
            .find_map(|tensor| std::ptr::eq(tensor.dptr.cast_const(), d_b).then_some(tensor.dtype))
            .unwrap_or(GgmlDType::Unknown(u32::MAX))
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
    /// Async f32 (MxK) × GGUF Wq/Wk/Wv F16 (logical KxN*, K-fastest) → Q/K/V f32.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn qkv_project_f32xf16_gguf_f32_async(
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
        self.matmul_f32xf16_gguf_f32_async(d_x_f32, d_wq_f16, d_q_out_f32, m, n_q, k)
            .with_context(|| format!("Q projection async GGUF GEMM failed: m={m} n={n_q} k={k}"))?;
        self.matmul_f32xf16_gguf_f32_async(d_x_f32, d_wk_f16, d_k_out_f32, m, n_k, k)
            .with_context(|| format!("K projection async GGUF GEMM failed: m={m} n={n_k} k={k}"))?;
        self.matmul_f32xf16_gguf_f32_async(d_x_f32, d_wv_f16, d_v_out_f32, m, n_v, k)
            .with_context(|| format!("V projection async GGUF GEMM failed: m={m} n={n_v} k={k}"))
    }

    /// # Safety
    /// Optional single-token fused Q/K/V projection for GGUF F16 weights.
    /// Bias pointers, when present, must point to f32 vectors.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn try_qkv_project_f32xf16_gguf_f32_decode_async(
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
        d_bq_f32: Option<*const c_void>,
        d_bk_f32: Option<*const c_void>,
        d_bv_f32: Option<*const c_void>,
        d_q_out_f32: *mut c_void,
        d_k_out_f32: *mut c_void,
        d_v_out_f32: *mut c_void,
    ) -> Result<bool> {
        #[cfg(feature = "cuda")]
        {
            if m == 1
                && k > 0
                && n_q > 0
                && n_k > 0
                && n_v > 0
                && fused_q4_qkv_decode_enabled()
                && self.gguf_weight_dtype(d_wq_f16) == GgmlDType::Q4_0
                && self.gguf_weight_dtype(d_wk_f16) == GgmlDType::Q4_0
                && self.gguf_weight_dtype(d_wv_f16) == GgmlDType::Q4_0
            {
                self.cuda
                    .qkv_f32xq4_0_gguf_decode_async(
                        d_x_f32,
                        d_wq_f16,
                        d_wk_f16,
                        d_wv_f16,
                        d_bq_f32,
                        d_bk_f32,
                        d_bv_f32,
                        d_q_out_f32,
                        d_k_out_f32,
                        d_v_out_f32,
                        n_q,
                        n_k,
                        n_v,
                        k,
                    )
                    .with_context(|| {
                        format!(
                            "fused Q4 QKV projection failed: m={m} n_q={n_q} n_k={n_k} n_v={n_v} k={k}"
                        )
                    })?;
                return Ok(true);
            }
            if m == 1
                && k > 0
                && n_q > 0
                && n_k > 0
                && n_v > 0
                && fused_q8_qkv_decode_enabled()
                && self.gguf_weight_dtype(d_wq_f16) == GgmlDType::Q8_0
                && self.gguf_weight_dtype(d_wk_f16) == GgmlDType::Q8_0
                && self.gguf_weight_dtype(d_wv_f16) == GgmlDType::Q8_0
            {
                self.cuda
                    .qkv_f32xq8_0_gguf_decode_async(
                        d_x_f32,
                        d_wq_f16,
                        d_wk_f16,
                        d_wv_f16,
                        d_bq_f32,
                        d_bk_f32,
                        d_bv_f32,
                        d_q_out_f32,
                        d_k_out_f32,
                        d_v_out_f32,
                        n_q,
                        n_k,
                        n_v,
                        k,
                    )
                    .with_context(|| {
                        format!(
                            "fused Q8 QKV projection failed: m={m} n_q={n_q} n_k={n_k} n_v={n_v} k={k}"
                        )
                    })?;
                return Ok(true);
            }
            if m == 1
                && k > 0
                && n_q > 0
                && n_k > 0
                && n_v > 0
                && fused_qkv_decode_enabled()
                && self.gguf_weight_dtype(d_wq_f16) == GgmlDType::F16
                && self.gguf_weight_dtype(d_wk_f16) == GgmlDType::F16
                && self.gguf_weight_dtype(d_wv_f16) == GgmlDType::F16
            {
                self.cuda
                    .qkv_f32xf16_gguf_decode_async(
                        d_x_f32,
                        d_wq_f16,
                        d_wk_f16,
                        d_wv_f16,
                        d_bq_f32,
                        d_bk_f32,
                        d_bv_f32,
                        d_q_out_f32,
                        d_k_out_f32,
                        d_v_out_f32,
                        n_q,
                        n_k,
                        n_v,
                        k,
                    )
                    .with_context(|| {
                        format!(
                            "fused QKV projection failed: m={m} n_q={n_q} n_k={n_k} n_v={n_v} k={k}"
                        )
                    })?;
                return Ok(true);
            }
        }
        let _ = (
            d_x_f32,
            m,
            k,
            d_wq_f16,
            n_q,
            d_wk_f16,
            n_k,
            d_wv_f16,
            n_v,
            d_bq_f32,
            d_bk_f32,
            d_bv_f32,
            d_q_out_f32,
            d_k_out_f32,
            d_v_out_f32,
        );
        Ok(false)
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
    /// Async f32 (MxK) × GGUF F16 W (logical KxN, K-fastest) → Out f32.
    pub unsafe fn out_proj_f32xf16_gguf_f32_async(
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
        self.matmul_f32xf16_gguf_f32_async(d_in_f32, d_w_out_f16, d_out_f32, m, n, k)
            .with_context(|| {
                format!("attention output projection async GGUF GEMM failed: m={m} n={n} k={k}")
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
