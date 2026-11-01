//! Short resident requests for external-collector calibration, never HTTP qualification.
use super::{
    load_model,
    q8_kv_stack::{collect_launch_counts, read_gpu_state, run_command, workload},
};
use anyhow::{ensure, Result};
use m40_llm::{
    cuda::CudaContext,
    decode::StopReason,
    evaluation::identity::file_sha256,
    generate::{generate_text, GenerateOptions, PromptFormat, PromptPrefill},
    infer::{q8_prefill::Q8PrefillBackend, Q8AttentionProjectionBackend, Q8FfnConfig},
    kv_compression::KvCompressionConfig,
    profile,
};
use serde_json::{json, Value};
use std::{
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn parse_reference(contents: &str) -> Result<(Vec<u32>, Vec<u32>)> {
    let rows: Vec<Value> = contents
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let summaries: Vec<_> = rows
        .iter()
        .filter(|r| r["kind"] == "q8_runtime_summary")
        .collect();
    ensure!(
        summaries.len() == 1 && summaries[0]["passes_exact_quality_gate"] == true,
        "reference must have one passing quality summary"
    );
    let candidates: Vec<_> = rows
        .iter()
        .filter(|r| r["kind"] == "q8_runtime_browser_generation" && r["label"] == "candidate")
        .collect();
    ensure!(
        candidates.len() == 1,
        "reference must have one candidate browser generation"
    );
    let row = candidates[0];
    ensure!(
        row["config"]
            == json!({"attention_projection":"packed-warp-256","ffn":"split-f32x4-128","prefill":"dequant-sgemm"}),
        "reference runtime differs"
    );
    let prompt: Vec<u32> = serde_json::from_value(row["prompt_token_ids"].clone())?;
    let ids: Vec<u32> = serde_json::from_value(row["generation"]["generated_token_ids"].clone())?;
    ensure!(
        prompt.len() == 65 && ids.len() == workload::MAX_TOKENS,
        "reference must contain actual canonical 65/512 IDs"
    );
    Ok((prompt, ids[..16].to_vec()))
}

/// Calibrate external observation against the same resident dense runtime,
/// requiring actual frozen-prefix IDs and dispatch after four fresh-KV warmups.
#[test]
#[ignore = "explicit release M40 diagnostic; requires model, reference and new report paths"]
fn test_qualified_runtime_activity_probe() -> Result<()> {
    ensure!(!cfg!(debug_assertions), "use --release");
    let path = PathBuf::from(std::env::var("M40LLM_QWEN_THROUGHPUT_MODEL")?);
    let reference = PathBuf::from(std::env::var("M40LLM_QWEN_ACTIVITY_REFERENCE")?);
    let destination = PathBuf::from(std::env::var("M40LLM_QWEN_THROUGHPUT_REPORT")?);
    let model_sha256 = file_sha256(&path)?;
    ensure!(
        model_sha256 == "4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d",
        "calibration requires the pinned Qwen2.5-3B Q8_0 model"
    );
    let (prompt_ids, expected_ids) = parse_reference(&std::fs::read_to_string(&reference)?)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    let mut record = |row: Value| -> Result<()> {
        serde_json::to_writer(&mut file, &row)?;
        writeln!(file)?;
        file.flush()?;
        Ok(())
    };
    let initial = CudaContext::total_device_bytes();
    let dense = KvCompressionConfig::dense_reference();
    let model = load_model(&path, workload::CONTEXT_TOKENS, &dense)?;
    super::cuda_env::require_sm52(&model.cuda)?;
    ensure!(
        model.q8_ffn == Q8FfnConfig::split_f32x4()
            && model.q8_prefill == Q8PrefillBackend::DequantSgemm
            && model.q8_attention_projection.backend()
                == Q8AttentionProjectionBackend::PackedWarp256,
        "configure qualified split FFN, SGEMM prefix and packed projections explicitly"
    );
    let prepared = serde_json::to_value(model.q8_execution_stats())?;
    ensure!(
        prepared["resident_bytes"] == 2917580800_u64
            && prepared["tensors"]
                .as_array()
                .is_some_and(|t| t.len() == 109),
        "unexpected preparation"
    );
    record(
        json!({"kind":"activity_manifest","revision":run_command("git", &["rev-parse","HEAD"])?,
        "worktree":run_command("git", &["status","--short"])?,"build_profile":"release",
        "pid":std::process::id(),"model":path,"model_sha256":model_sha256,"reference":reference,
        "reference_sha256":file_sha256(&reference)?,
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,
        "ld_preload":std::env::var_os("LD_PRELOAD"),
        "prepared":prepared,"prompt":workload::render_user_prompt(workload::USER_PROMPT),
        "prompt_ids":prompt_ids,"max_tokens":16,"warmups":4,
        "environment":std::env::vars().filter(|(k,_)| k.starts_with("M40LLM_")||k=="CUDA_VISIBLE_DEVICES").collect::<std::collections::BTreeMap<_,_>>(),
        "scope":"resident short-request collector calibration; not HTTP or actual512 qualification"}),
    )?;
    for round in 0..5 {
        let before_gpu = read_gpu_state()?;
        let before = profile::snapshot();
        let started_unix_us = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let start = Instant::now();
        let generated = generate_text(
            &model,
            GenerateOptions {
                prompt: workload::render_user_prompt(workload::USER_PROMPT),
                prompt_format: PromptFormat::Raw,
                prompt_prefill: PromptPrefill::PackedPrefix {
                    max_prompt_tokens: 128,
                },
                max_tokens: Some(16),
                top_k: Some(1),
                seed: Some(0),
                kv_compression: dense.clone(),
                log_prefix: "activity-calibration",
                ..Default::default()
            },
        )?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let completed_unix_us = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let after = profile::snapshot();
        let after_gpu = read_gpu_state()?;
        ensure!(
            generated.prompt_token_len == prompt_ids.len()
                && generated.token_ids[..generated.prompt_token_len] == prompt_ids
                && generated.token_ids[generated.prompt_token_len..] == expected_ids
                && generated.metrics.stop_reason == StopReason::Length,
            "generation differs from frozen prefix"
        );
        ensure!(
            generated.prefill_mode == "packed-prefix-dequant-sgemm"
                && generated.q8_prefill_temporary_weight_bytes == Some(90177536),
            "prefix dispatch changed"
        );
        let operations = collect_launch_counts(&before, &after);
        for op in model
            .q8_attention_projection
            .backend()
            .get_operations()
            .unwrap()
        {
            ensure!(
                operations.get(op) == Some(&(36 * 16)),
                "wrong packed projection dispatch: {op}"
            );
        }
        ensure!(
            operations.get("ffn_q8_split_f32x4_128") == Some(&(36 * 16)),
            "wrong FFN dispatch"
        );
        record(
            json!({"kind":"activity_generation","phase":if round<4 {"warmup"} else {"measured"},
            "round":round,"elapsed_ms":elapsed_ms,"started_unix_us":started_unix_us,"completed_unix_us":completed_unix_us,
            "metrics":generated.metrics,"output":generated.output,
            "generated_ids":&generated.token_ids[generated.prompt_token_len..],
            "prompt_ids":&generated.token_ids[..generated.prompt_token_len],"prefill_mode":generated.prefill_mode,
            "operations":operations,"counters":profile::counter_delta(&before,&after),
            "gpu_before":before_gpu,"gpu_after":after_gpu}),
        )?;
    }
    ensure!(
        serde_json::to_value(model.q8_execution_stats())? == prepared,
        "prepared cache changed"
    );
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "tracked allocation leak"
    );
    record(
        json!({"kind":"activity_complete","tracked_initial":initial,"tracked_after":CudaContext::total_device_bytes()}),
    )?;
    Ok(())
}

/// A calibration must not silently use an absent or differently configured
/// generation as its expected token prefix.
#[test]
fn test_activity_reference_rejects_incomplete_and_wrong_runtime() {
    assert!(parse_reference("").is_err());
    let summary = json!({"kind":"q8_runtime_summary","passes_exact_quality_gate":true});
    let mut candidate = json!({"kind":"q8_runtime_browser_generation","label":"candidate",
        "config":{"attention_projection":"packed-warp-256","ffn":"split-f32x4-128","prefill":"dequant-sgemm"},
        "prompt_token_ids":vec![3;65],"generation":{"generated_token_ids":vec![7;512]}});
    let contents = format!("{summary}\n{candidate}\n");
    assert_eq!(
        parse_reference(&contents).unwrap(),
        (vec![3; 65], vec![7; 16])
    );
    candidate["config"]["ffn"] = json!("native");
    assert!(parse_reference(&format!("{summary}\n{candidate}\n")).is_err());
    assert!(parse_reference(&format!("{contents}{contents}")).is_err());
}
