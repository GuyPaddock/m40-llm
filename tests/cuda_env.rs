// tests/helpers.rs
#![cfg(all(feature = "cuda", nvcc))]
#![allow(dead_code)]

use anyhow::{anyhow, Result};
use m40_llm::cuda::CudaContext;

/// Create a CUDA context preferring Tesla M40 (sm_52).
/// If env M40LLM_FORCE_M40=1 is set, kernels.cu will also auto-select M40 even if device_id >= 0.
pub fn ctx_m40() -> Result<CudaContext> {
    // Pass -1 to trigger auto-selection logic in the CUDA layer we added.
    CudaContext::new(-1)
}

/// Optional guard to ensure we are actually on device with sm_52.
/// Call at top of CUDA tests that must validate on M40 specifically.
pub fn require_sm52(ctx: &CudaContext) -> Result<()> {
    let props = ctx.current_device_props()?;
    if props.major == 5 && props.minor == 2 {
        Ok(())
    } else {
        Err(anyhow!(
            "skipping: active device is '{}', sm_{}{} (id {}), not sm_52",
            props.name,
            props.major,
            props.minor,
            props.device_id
        ))
    }
}
