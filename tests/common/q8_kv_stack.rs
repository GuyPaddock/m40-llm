//! Bounded resident-generation comparison; not buffered HTTP qualification.
use super::load_model;
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    generate::{generate_text, GenerateOptions, PromptFormat},
    infer::{q8_prefill::Q8PrefillBackend, Q8AttentionProjectionBackend, Q8FfnConfig},
    kv_compression::{KvCompressionConfig, KvExactOldAttention, KvExactOldBacking},
    profile,
};
use serde_json::{json, Value};
use std::{fs::OpenOptions, io::Write, path::PathBuf, process::Command, time::Instant};

#[path = "../../examples/common/qwen_workload.rs"]
mod workload;

fn run_command(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    ensure!(
        output.status.success(),
        "{program} failed: {:?}",
        output.status
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn read_gpu_state() -> Result<Value> {
    let device = std::env::var("CUDA_VISIBLE_DEVICES")?;
    let processes = run_command(
        "nvidia-smi",
        &[
            "-i",
            &device,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    )?;
    ensure!(
        processes.lines().all(|line| line
            .split(',')
            .nth(1)
            .and_then(|pid| pid.trim().parse::<u32>().ok())
            == Some(std::process::id())),
        "another compute process is using the selected GPU: {processes}"
    );
    Ok(json!({
        "clocks":run_command("nvidia-smi", &["-i", &device, "--query-gpu=uuid,pstate,clocks.sm,clocks.mem", "--format=csv,noheader,nounits"])?,
        "processes":processes
    }))
}

fn describe_kv(config: &KvCompressionConfig) -> Value {
    json!({"mode":format!("{:?}",config.mode),"recent_window":config.recent_window,
        "block_size":config.block_size,"top_blocks":config.top_blocks,
        "representatives":config.representatives,"policy":"topk",
        "exact_old_backing":config.effective_exact_old_backing().as_str(),
        "exact_old_attention":config.effective_exact_old_attention().as_str()})
}

/// Keep the qualified projection/FFN/prefill path fixed while comparing dense,
/// project-default KV and actual old-block retrieval. Fresh KV, warmups, actual
/// dispatch, natural stopping and output drift remain visible in every row.
#[test]
#[ignore = "explicit release M40 experiment; requires model and report paths"]
fn test_preferred_q8_runtime_with_compressed_kv() -> Result<()> {
    ensure!(!cfg!(debug_assertions), "use --release");
    ensure!(
        std::env::vars().all(|(k, v)| !k.starts_with("M40LLM_KV_")
            || (k == "M40LLM_KV_BLOCK_SELECT_POLICY" && v == "topk")),
        "disable unrelated KV diagnostics/policies for this comparison"
    );
    let path = PathBuf::from(std::env::var("M40LLM_QWEN_THROUGHPUT_MODEL")?);
    let destination = PathBuf::from(std::env::var("M40LLM_QWEN_THROUGHPUT_REPORT")?);
    let mut report = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    let mut record = |value: Value| -> Result<()> {
        serde_json::to_writer(&mut report, &value)?;
        writeln!(report)?;
        report.flush()?;
        Ok(())
    };
    let dense = KvCompressionConfig::dense_reference();
    let defaults = KvCompressionConfig::default();
    ensure!(
        defaults.top_blocks == 8
            && defaults.recent_window == 1024
            && defaults.effective_exact_old_backing() == KvExactOldBacking::Fp16KQ4V
            && defaults.effective_exact_old_attention() == KvExactOldAttention::Fp16KQ4VDirect,
        "experiment requires the documented preferred compressed defaults"
    );
    let cases = [
        ("dense", dense.clone()),
        ("compressed-default", defaults.clone()),
        (
            "compressed-window128-top8",
            KvCompressionConfig {
                recent_window: 128,
                ..defaults.clone()
            },
        ),
        (
            "compressed-window128-top4",
            KvCompressionConfig {
                recent_window: 128,
                top_blocks: 4,
                ..defaults
            },
        ),
    ];
    let initial = CudaContext::total_device_bytes();
    let mut model = load_model(&path, workload::CONTEXT_TOKENS, &dense)?;
    ensure!(
        model.q8_ffn == Q8FfnConfig::split_f32x4()
            && model.q8_prefill == Q8PrefillBackend::DequantSgemm
            && model.q8_attention_projection.backend()
                == Q8AttentionProjectionBackend::PackedWarp256,
        "configure the qualified split FFN, SGEMM prefix and packed projections explicitly"
    );
    ensure!(
        model.model_config.block_count == 36 && model.model_config.attention_key_length == 128,
        "this bounded comparison requires Qwen2.5-3B head128"
    );
    let prepared = serde_json::to_value(model.q8_execution_stats())?;
    ensure!(
        prepared["resident_bytes"] == 2917580800_u64
            && prepared["tensors"]
                .as_array()
                .is_some_and(|t| t.len() == 109),
        "unexpected prepared weights"
    );
    record(
        json!({"kind":"kv_stack_manifest", "revision":run_command("git", &["rev-parse","HEAD"])?,
            "worktree":run_command("git", &["status","--short"])?, "build_profile":"release",
            "executable_sha256":run_command("sha256sum", &[std::env::current_exe()?.to_str().context("executable path")?])?,
            "model":path,"model_sha256":run_command("sha256sum", &[path.to_str().context("model path")?])?,
            "prepared":prepared,"cases":cases.iter().map(|(label,config)| json!({"case":label,"kv":describe_kv(config)})).collect::<Vec<_>>(),"context":workload::CONTEXT_TOKENS,
            "prompt":workload::render_user_prompt(workload::USER_PROMPT),"max_tokens":workload::MAX_TOKENS,
            "sampling":"greedy, natural stopping", "pairs":5,
            "scope":"resident generation wall time; KV allocation and load excluded; NOT HTTP/Ollama qualification",
            "environment":std::env::vars().filter(|(k,_)| k.starts_with("M40LLM_")||k=="CUDA_VISIBLE_DEVICES").collect::<std::collections::BTreeMap<_,_>>()
        }),
    )?;
    let mut dense_ids = Vec::new();
    for round in 0..6 {
        let order: Vec<_> = if round % 2 == 0 {
            (0..cases.len()).collect()
        } else {
            (0..cases.len()).rev().collect()
        };
        for index in order {
            let (label, config) = &cases[index];
            let allocation = Instant::now();
            if config.mode.is_enabled() {
                model.allocate_compressed_kv_cache_for_layers(workload::CONTEXT_TOKENS, config)?;
            } else {
                model.allocate_kv_cache_for_layers(workload::CONTEXT_TOKENS)?;
            }
            let allocation_ms = allocation.elapsed().as_secs_f64() * 1000.0;
            let before_gpu = read_gpu_state()?;
            let before = profile::snapshot();
            let start = Instant::now();
            let generated = generate_text(
                &model,
                GenerateOptions {
                    prompt: workload::render_user_prompt(workload::USER_PROMPT),
                    prompt_format: PromptFormat::Raw,
                    prompt_prefill: m40_llm::generate::PromptPrefill::PackedPrefix {
                        max_prompt_tokens: 128,
                    },
                    max_tokens: Some(workload::MAX_TOKENS),
                    top_k: Some(1),
                    seed: Some(0),
                    kv_compression: config.clone(),
                    log_prefix: "kv-stack",
                    ..Default::default()
                },
            )?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            let after = profile::snapshot();
            let after_gpu = read_gpu_state()?;
            let ids = &generated.token_ids[generated.prompt_token_len..];
            ensure!(
                generated.prompt_token_len == 65,
                "canonical prompt IDs changed"
            );
            ensure!(
                generated.q8_prefill_temporary_weight_bytes == Some(90177536)
                    && generated.prefill_mode
                        == if config.mode.is_enabled() {
                            "packed-prefix-block-select-exact-dequant-sgemm"
                        } else {
                            "packed-prefix-dequant-sgemm"
                        },
                "requested SGEMM prefix did not execute"
            );
            if index == 0 {
                if round == 0 {
                    dense_ids = ids.to_vec();
                } else {
                    ensure!(ids == dense_ids, "dense generation changed between rounds");
                }
            }
            let operations: std::collections::BTreeMap<_, _> = after
                .by_op
                .iter()
                .filter_map(|(op, v)| {
                    let previous = before.by_op.get(op).copied().unwrap_or_default();
                    let n = v.launches.saturating_sub(previous.launches);
                    (n > 0).then_some((*op, n))
                })
                .collect();
            for op in model
                .q8_attention_projection
                .backend()
                .get_operations()
                .unwrap()
            {
                ensure!(
                    operations.get(op) == Some(&(36 * generated.metrics.generated_tokens as u64)),
                    "requested packed projection did not execute: {op}"
                );
            }
            if config.mode.is_enabled() {
                ensure!(
                    operations.get(
                        "attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct"
                    ) == Some(&(36 * generated.metrics.generated_tokens as u64)),
                    "requested direct mixed-KV attention did not execute"
                );
            }
            let sequence_tokens =
                generated.prompt_token_len + generated.metrics.generated_tokens - 1;
            let old_tokens = if config.mode.is_enabled() {
                sequence_tokens.saturating_sub(config.recent_window as usize)
            } else {
                0
            };
            let attended_upper = if config.mode.is_enabled() {
                sequence_tokens.min(config.recent_window as usize)
                    + old_tokens.min(config.top_blocks as usize * config.block_size as usize)
            } else {
                sequence_tokens
            };
            let divergence = dense_ids
                .iter()
                .zip(ids)
                .position(|(a, b)| a != b)
                .or_else(|| {
                    (dense_ids.len() != ids.len()).then_some(dense_ids.len().min(ids.len()))
                });
            record(
                json!({"kind":"kv_stack_generation","phase":if round==0 {"warmup"} else {"measured"},
                    "round":round,"case":label,"kv":describe_kv(config),"kv_allocation_ms":allocation_ms,"elapsed_ms":elapsed_ms,
                    "metrics":generated.metrics,"resident_tps":generated.metrics.generated_tokens as f64*1000.0/elapsed_ms,
                    "prefill_mode":generated.prefill_mode,"output":generated.output,"prompt_ids":&generated.token_ids[..generated.prompt_token_len],
                    "generated_ids":ids,"first_divergence_vs_dense":divergence,
                    "final_kv_bytes":generated.final_kv_allocated_bytes,"dense_equivalent_kv_bytes":generated.dense_equivalent_kv_bytes,
                    "recent_fp16_bytes":generated.recent_fp16_bytes,"old_k_fp16_bytes":generated.old_k_fp16_bytes,
                    "old_v_q4_payload_bytes":generated.q4_old_v_payload_bytes,"old_v_q4_scale_bytes":generated.q4_old_v_scale_bytes,
                    "summary_index_bytes":generated.summary_index_bytes,"temporary_prefill_weight_bytes":generated.q8_prefill_temporary_weight_bytes,
                    "old_tokens_final_forward":old_tokens,"active_tokens_upper_bound_per_head_final_forward":attended_upper,
                    "operations":operations,"counters":profile::counter_delta(&before,&after),"gpu_before":before_gpu,"gpu_after":after_gpu
                }),
            )?;
            eprintln!("[kv-stack] round={round} case={label} generated={} elapsed_ms={elapsed_ms:.3} decode_ms={:.3} final_kv={:?} divergence={divergence:?}",
                generated.metrics.generated_tokens,generated.metrics.decode_elapsed_ms,generated.final_kv_allocated_bytes);
        }
    }
    ensure!(
        serde_json::to_value(model.q8_execution_stats())? == prepared,
        "prepared weights changed across KV cases"
    );
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "stack experiment leaked tracked allocations"
    );
    record(
        json!({"kind":"kv_stack_complete","tracked_initial":initial,"tracked_after":CudaContext::total_device_bytes()}),
    )?;
    Ok(())
}

/// JSON records expose the physical backend and explicit dense reference,
/// rather than a label that could conceal different attention semantics.
#[test]
fn test_stack_report_describes_dense_and_compressed_semantics() {
    let dense = describe_kv(&KvCompressionConfig::dense_reference());
    let compressed = describe_kv(&KvCompressionConfig::default());
    assert_eq!(dense["mode"], "Off");
    assert_eq!(compressed["mode"], "BlockSelectExact");
    assert_eq!(compressed["exact_old_backing"], "fp16-k-q4-v");
    assert_eq!(compressed["exact_old_attention"], "fp16-k-q4-v-direct");
    assert_eq!(compressed["top_blocks"], 8);
    assert_eq!(compressed["recent_window"], 1024);
    assert_eq!(compressed["policy"], "topk");
}
