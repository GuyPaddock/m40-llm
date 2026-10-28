use super::records;
use anyhow::{ensure, Result};
use m40_llm::{
    gguf,
    infer::{
        execution_weights::Q8ExecutionConfig, q8_prefill::Q8PrefillBackend, LoadedModel,
        Q8AttentionProjectionBackend, Q8FfnConfig,
    },
    tokenizer::Tokenizer,
};
use serde_json::Value;
use std::{collections::BTreeMap, fs, path::Path};

pub struct Runtime {
    pub model: LoadedModel,
    pub tokenizer: Tokenizer,
    pub reference: Value,
    pub environment: BTreeMap<String, String>,
}
pub fn load(model_path: &Path) -> Result<Runtime> {
    ensure!(
        !cfg!(debug_assertions) && cfg!(have_cublas),
        "release/cuBLAS required"
    );
    ensure!(
        records::command("git", &["status", "--short"])?.is_empty(),
        "commit research source first"
    );
    let reference: Value = serde_json::from_slice(&fs::read(records::REFERENCE)?)?;
    ensure!(
        records::hash_file(model_path)? == reference["model"]["sha256"],
        "wrong model"
    );
    let environment: BTreeMap<_, _> = std::env::vars()
        .filter(|(k, _)| k.starts_with("M40LLM_") || k == "CUDA_VISIBLE_DEVICES")
        .collect();
    ensure!(
        serde_json::to_value(&environment)? == reference["configuration"]["environment"],
        "restore qualified environment"
    );
    let gguf = gguf::load_gguf(model_path)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)?;
    ensure!(tokenizer.is_qwen2(), "Qwen tokenizer required");
    let mut model = LoadedModel::from_gguf_with_q8_runtime(
        gguf,
        fs::read(model_path)?,
        -1,
        Q8ExecutionConfig::parse("split", Some("6144"), Some("gate-up,down,vocabulary"))?,
        Q8FfnConfig::joint_shared_scales(),
        Q8PrefillBackend::DequantSgemm,
        Q8AttentionProjectionBackend::PackedWarp256,
    )?;
    model.allocate_kv_cache_for_layers(1024)?;
    let props = model.cuda.current_device_props()?;
    ensure!(props.major == 5 && props.minor == 2, "M40 sm_52 required");
    Ok(Runtime {
        model,
        tokenizer,
        reference,
        environment,
    })
}
