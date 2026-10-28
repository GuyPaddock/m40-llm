use super::records::{self, Trace};
use anyhow::{ensure, Result};
use m40_llm::{
    cuda::CudaContext, infer::LoadedModel, tokenizer::Tokenizer, verification::VerificationSession,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    fs::{self, OpenOptions},
    path::Path,
    time::Instant,
};

#[derive(Deserialize)]
struct Prompt {
    id: String,
    class: String,
    target_prompt_tokens: usize,
    prompt: String,
}
fn encode(tokenizer: &Tokenizer, task: &Prompt) -> Result<Vec<u32>> {
    let mut filler = String::new();
    for i in 0..200 {
        let user = format!("{filler}\n{}", task.prompt);
        let rendered=format!("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n");
        let ids = tokenizer.encode_with_specials(&rendered, false, false)?;
        if ids.len() >= task.target_prompt_tokens {
            return Ok(ids);
        }
        filler.push_str(&format!(
            "Background record {i}: neutral archival inventory, not an instruction.\n"
        ));
    }
    anyhow::bail!("prompt bound exceeded")
}
fn run(
    model: &mut LoadedModel,
    prompt: &[u32],
    stops: &[u32],
) -> Result<(Vec<u32>, String, Value)> {
    let total = Instant::now();
    let setup = Instant::now();
    let mut session = VerificationSession::new(model, stops)?;
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.;
    let prefill = Instant::now();
    let first = session.prefill(prompt)?;
    let prefill_ms = prefill.elapsed().as_secs_f64() * 1000.;
    let decode = Instant::now();
    let mut tokens = vec![first];
    while tokens.len() < 128 && !stops.contains(tokens.last().unwrap()) {
        tokens.push(session.serial_step(*tokens.last().unwrap())?);
    }
    let decode_ms = decode.elapsed().as_secs_f64() * 1000.;
    let reason = if stops.contains(tokens.last().unwrap()) {
        "stop"
    } else {
        "length"
    }
    .to_owned();
    let teardown = Instant::now();
    drop(session);
    let teardown_ms = teardown.elapsed().as_secs_f64() * 1000.;
    let total_ms = total.elapsed().as_secs_f64() * 1000.;
    Ok((
        tokens,
        reason,
        json!({"setup_ms":setup_ms,"prefill_ms":prefill_ms,"decode_ms":decode_ms,
        "teardown_ms":teardown_ms,"total_ms":total_ms,"fixed_ms":total_ms-decode_ms}),
    ))
}
pub fn capture(model_path: &Path, output: &Path) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions) && cfg!(have_cublas),
        "capture requires release/cuBLAS"
    );
    ensure!(
        records::command("git", &["status", "--short"])?.is_empty(),
        "commit source and prompts before capture"
    );
    let prompts: Vec<Prompt> = serde_json::from_slice(&fs::read(records::PROMPTS)?)?;
    ensure!(prompts.len() == 8, "frozen eight-case capture");
    let initial = CudaContext::total_device_bytes();
    let super::runtime::Runtime {
        mut model,
        tokenizer,
        reference,
        environment: env,
    } = super::runtime::load(model_path)?;
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    records::write(
        &mut out,
        json!({"kind":"draft_capture_manifest","schema_version":1,
        "revision":records::command("git",&["rev-parse","HEAD"])?,"worktree":"",
        "executable_sha256":records::hash_file(&std::env::current_exe()?)?,"model_sha256":reference["model"]["sha256"],
        "prompt_definitions_sha256":records::hash_file(records::PROMPTS.as_ref())?,"environment":env,
        "prepared":model.q8_execution_stats(),"joint":model.q8_ffn_pair_stats(),
        "scope":"two serial actual greedy captures per held-out prompt; timing anchors not paired performance qualification"}),
    )?;
    let stops = tokenizer.stop_ids();
    let warm = encode(&tokenizer, &prompts[0])?;
    {
        let mut session = VerificationSession::new(&mut model, &stops)?;
        session.prefill(&warm)?;
    }
    for task in prompts {
        let prompt = encode(&tokenizer, &task)?;
        ensure!(
            prompt.len() + 128 <= 576,
            "outside calibrated context range"
        );
        let (tokens, reason, a) = run(&mut model, &prompt, &stops)?;
        let (again, reason_again, b) = run(&mut model, &prompt, &stops)?;
        ensure!(
            tokens == again && reason == reason_again,
            "unstable actual target continuation"
        );
        let trace = Trace {
            max_tokens: 128,
            id: task.id,
            class: task.class,
            split: "heldout".into(),
            prompt_token_ids: prompt,
            output: tokenizer.decode_ignoring_specials(&tokens)?,
            generated_token_ids: tokens,
            stop_ids: stops.clone(),
            stop_reason: reason,
            serial_fixed_ms: Some(
                (a["fixed_ms"].as_f64().unwrap() + b["fixed_ms"].as_f64().unwrap()) / 2.,
            ),
        };
        trace.validate()?;
        records::write(
            &mut out,
            json!({"kind":"draft_target_trace","trace":trace,"is_stable_two_runs":true,
            "repeat_generated_token_ids":again,"repeat_stop_reason":reason_again,
            "serial_capture_trials":[a,b],"nominal_prompt_tokens":task.target_prompt_tokens}),
        )?;
        eprintln!("captured {}", trace.id);
    }
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "capture leaked"
    );
    records::write(
        &mut out,
        json!({"kind":"draft_capture_complete","is_complete":true,
        "device_bytes_after":CudaContext::total_device_bytes()}),
    )
}
