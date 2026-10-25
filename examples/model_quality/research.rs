use super::records;
use anyhow::{ensure, Result};
use serde_json::json;
use std::collections::BTreeMap;

pub(super) const CUDA_FLAGS: &str = "build.rs -O3 -std=c++17 -cudart=shared -fPIC -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52 -allow-unsupported-compiler; no --use_fast_math; host header defines in build.rs";

pub(super) fn execution_hash(
    environment: &BTreeMap<String, String>,
    context: u32,
) -> Result<String> {
    records::sha256(&serde_json::to_vec(
        &json!({"environment":environment,"context_tokens":context,
        "reference":"native full-layer Q8, explicit dense sequential prefill/decode, greedy top_k=1 seed=0",
        "task_template":"canonical Qwen system+user ChatML; Llama3 auto chat"}),
    )?)
}

pub(super) fn validate_execution(environment: &BTreeMap<String, String>) -> Result<()> {
    ensure!(
        environment
            .get("M40LLM_Q8_DECODE_STREAM")
            .map(String::as_str)
            == Some("decode"),
        "model research requires M40LLM_Q8_DECODE_STREAM=decode"
    );
    for key in [
        "M40LLM_PREFILL_CHUNK_SIZE",
        "M40LLM_KV_COMPRESSED_PREFILL_CHUNK_SIZE",
    ] {
        ensure!(
            environment.get(key).is_none_or(|value| value == "0"),
            "model research requires sequential prefill; unset {key} or set 0"
        );
    }
    Ok(())
}
