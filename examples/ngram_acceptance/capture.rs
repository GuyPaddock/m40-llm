use super::records::{self, REGRESSION_PROMPTS};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    generate::{generate_text, GenerateOptions, PromptFormat},
    gguf,
    infer::{execution_weights::Q8ExecutionConfig, LoadedModel},
    kv_compression::KvCompressionConfig,
    tokenizer::Tokenizer,
};
use serde_json::json;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
};

/// Capture actual IDs with the selected runtime, not tokens re-encoded from text.
/// These are untimed quality traces; no HTTP speed claim is made from them.
pub fn capture(model_path: PathBuf, output: PathBuf) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "capture with --release to match C11"
    );
    let environment =
        serde_json::to_value(std::env::vars().collect::<std::collections::BTreeMap<_, _>>())?;
    let configuration = records::configuration(&environment)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let model_sha256 = records::command_text(
        "sha256sum",
        &[model_path.to_str().context("model path must be UTF-8")?],
    )?
    .split_whitespace()
    .next()
    .context("model hash")?
    .to_owned();
    let gguf = gguf::load_gguf(&model_path)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)?;
    ensure!(
        tokenizer.is_qwen2(),
        "C11 regression capture expects Qwen2 tokenizer"
    );
    let mut model = LoadedModel::from_gguf_with_q8_execution(
        gguf,
        fs::read(&model_path)?,
        -1,
        Q8ExecutionConfig::disabled(),
    )?;
    let props = model.cuda.current_device_props()?;
    ensure!(
        props.major == 5 && props.minor == 2,
        "capture requires the selected M40"
    );
    model.allocate_kv_cache_for_layers(1024)?;
    writeln!(
        file,
        "{}",
        json!({"kind":"ngram_regression_manifest","model_path":model_path,"model_sha256":model_sha256,
        "configuration":configuration,"configuration_sha256":records::sha256(&serde_json::to_vec(&configuration)?)?,
        "revision":records::command_text("git",&["rev-parse","HEAD"])?,"worktree":records::command_text("git",&["status","--short"])?,
        "is_timed":false,"build_profile":"release","nvcc":records::command_text("nvcc",&["--version"])?,
        "timing_scope":"untimed in-process actual-ID capture, not HTTP performance","device":format!("{:?}",props)})
    )?;
    for &(label, user_prompt, max_tokens, expected) in REGRESSION_PROMPTS {
        let prompt = format!("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n");
        let options = GenerateOptions {
            prompt: prompt.clone(),
            prompt_format: PromptFormat::Raw,
            max_tokens: Some(max_tokens),
            top_k: Some(1),
            seed: Some(0),
            kv_compression: KvCompressionConfig::dense_reference(),
            ..Default::default()
        };
        let generated = generate_text(&model, options.clone())?;
        let repeated = generate_text(&model, options)?;
        ensure!(
            generated.output == repeated.output && generated.token_ids == repeated.token_ids,
            "unstable reference trace: {label}"
        );
        let expected_prompt = tokenizer.encode_with_specials(&prompt, false, false)?;
        ensure!(
            generated.token_ids[..generated.prompt_token_len] == expected_prompt,
            "capture prompt IDs changed"
        );
        let generated_ids = &generated.token_ids[generated.prompt_token_len..];
        ensure!(
            generated.metrics.generated_tokens == generated_ids.len(),
            "capture token accounting changed"
        );
        writeln!(
            file,
            "{}",
            json!({"kind":"ngram_regression_trace","case":label,"user_prompt":user_prompt,"rendered_prompt":prompt,
            "prompt_token_ids":expected_prompt,"generated_token_ids":generated_ids,"stop_ids":tokenizer.stop_ids(),
            "max_tokens":max_tokens,"output":generated.output,"expected_output":expected,"metrics":generated.metrics,
            "is_stable_two_runs":true,"is_timed":false})
        )?;
        file.flush()?;
        if let Some(expected) = expected {
            ensure!(
                generated.output.trim() == expected,
                "native quality smoke failed: {label}"
            );
        }
        eprintln!("captured {label}: {} generated IDs", generated_ids.len());
    }
    Ok(())
}
