// src/main.rs
// Use the library crate instead of re-declaring modules to avoid duplicate code
use anyhow::Result;
use clap::Parser;
use m40_llm::cli::{
    Cli, Commands, KvCompressModeArg, KvExactOldAttentionArg, KvExactOldBackingArg,
    KvRepresentativePolicyArg,
};
use m40_llm::generate::{generate_text, GenerateOptions};
use m40_llm::kv_compression::{KvCompressMode, KvCompressionConfig};
#[cfg(not(feature = "server"))]
#[allow(unused_imports)]
use m40_llm::{gguf, infer, model};
use std::fs;
#[cfg(feature = "server")]
use {
    m40_llm::{gguf, infer, model, server},
    std::sync::Arc,
    tokio::net::TcpListener,
};

#[cfg(all(feature = "server", feature = "gguf_ext"))]
#[allow(unused_imports)]
use m40_llm::gguf_ext;

struct CliKvOptions {
    mode: KvCompressModeArg,
    recent_window: u32,
    block_size: u32,
    top_blocks: Option<u32>,
    representatives: u32,
    representative_policy: KvRepresentativePolicyArg,
    exact_old_backing: KvExactOldBackingArg,
    exact_old_attention: KvExactOldAttentionArg,
}

fn kv_config_from_cli(options: CliKvOptions) -> KvCompressionConfig {
    let mode: KvCompressMode = options.mode.into();
    let mut config = if mode == KvCompressMode::Off {
        KvCompressionConfig::dense_reference()
    } else {
        KvCompressionConfig::default()
    };
    config.mode = mode;
    config.recent_window = options.recent_window;
    config.block_size = options.block_size;
    config.top_blocks = options.top_blocks.unwrap_or(config.top_blocks);
    config.representatives = options.representatives;
    config.representative_policy = options.representative_policy.into();
    if mode != KvCompressMode::Off {
        config.exact_old_backing = options.exact_old_backing.into();
        config.exact_old_attention = options.exact_old_attention.into();
    }
    config
}

fn bounded_context_len(
    model_context_len: usize,
    max_context_tokens: Option<usize>,
) -> Result<usize> {
    match max_context_tokens {
        Some(0) => anyhow::bail!("--max-context-tokens must be greater than zero"),
        Some(limit) => Ok(limit.min(model_context_len)),
        None => Ok(model_context_len),
    }
}

fn warn_if_gpu_has_other_compute_processes(device_id: i32) {
    if std::env::var("M40LLM_GPU_BUSY_WARN").ok().as_deref() == Some("0") {
        return;
    }
    let Ok(output) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
            "-i",
            &device_id.to_string(),
        ])
        .output()
    else {
        return;
    };
    if !output.status.success() {
        return;
    }

    let current_pid = std::process::id();
    let mut other_processes = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let line = line.trim();
        if line.is_empty() || line.eq_ignore_ascii_case("[not supported]") {
            continue;
        }
        let mut parts = line.split(',').map(str::trim);
        let Some(pid_text) = parts.next() else {
            continue;
        };
        let Ok(pid) = pid_text.parse::<u32>() else {
            continue;
        };
        if pid == current_pid {
            continue;
        }
        let used_memory_mib = parts.next().unwrap_or("unknown");
        other_processes.push(format!("pid={pid} used_memory_mib={used_memory_mib}"));
    }

    if !other_processes.is_empty() {
        eprintln!(
            "[cuda] warning: device {device_id} has other compute process(es): {}; \
             concurrent large generation can cause CUDA allocation/GEMM failures. \
             Set M40LLM_GPU_BUSY_WARN=0 to suppress this warning.",
            other_processes.join("; ")
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Pull { model, source } => {
            let m = model::pull_model(&model, source).await?;
            println!(
                "Pulled model: {} ({:.2} MiB)",
                m.name,
                m.size_bytes as f64 / (1024.0 * 1024.0)
            );
        }
        Commands::List {} => {
            let models = model::list_models()?;
            if models.is_empty() {
                println!("No models found. Use `m40-llm pull ...`.");
            } else {
                for m in models {
                    println!(
                        "{} \t {:.2} MiB \t {}",
                        m.name,
                        m.size_bytes as f64 / (1024.0 * 1024.0),
                        m.path.display()
                    );
                }
            }
        }
        Commands::Generate {
            model,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            device_id,
            require_sm52,
            max_context_tokens,
            kv_compress_mode,
            kv_recent_window,
            kv_compress_block,
            kv_compress_top_blocks,
            kv_compress_representatives,
            kv_compress_representative_policy,
            kv_exact_old_backing,
            kv_exact_old_attention,
            prompt_format,
        } => {
            let local = model::resolve_model_arg(&model)?;
            let gguf_bytes = fs::read(&local.path)?;
            let gguf_model = gguf::load_gguf(&local.path)?;
            let mut loaded = infer::LoadedModel::from_gguf(gguf_model, gguf_bytes, device_id)?;

            let props = loaded.cuda.current_device_props()?;
            if require_sm52 && !(props.major == 5 && props.minor == 2) {
                anyhow::bail!(
                    "require_sm52 set but active device is '{}' sm_{}{} (id {})",
                    props.name,
                    props.major,
                    props.minor,
                    props.device_id
                );
            }
            eprintln!(
                "[cuda] device: '{}' (id {}), sm_{}{}",
                props.name, props.device_id, props.major, props.minor
            );
            warn_if_gpu_has_other_compute_processes(props.device_id);

            let max_len = bounded_context_len(
                loaded.model_config.context_length as usize,
                max_context_tokens,
            )?;
            let kv_compression = kv_config_from_cli(CliKvOptions {
                mode: kv_compress_mode,
                recent_window: kv_recent_window,
                block_size: kv_compress_block,
                top_blocks: kv_compress_top_blocks,
                representatives: kv_compress_representatives,
                representative_policy: kv_compress_representative_policy,
                exact_old_backing: kv_exact_old_backing,
                exact_old_attention: kv_exact_old_attention,
            });
            kv_compression.validate().map_err(|err| {
                anyhow::anyhow!(
                    "{err}; use --kv-compress-mode off for dense reference/compatibility mode"
                )
            })?;
            let use_compressed_kv = matches!(
                kv_compression.mode,
                KvCompressMode::RecentOnly
                    | KvCompressMode::BlockSummary
                    | KvCompressMode::BlockSelectLossy
            ) || kv_compression.uses_compressed_exact_old_backing();
            if use_compressed_kv {
                loaded.allocate_compressed_kv_cache_for_layers(
                    max_len.try_into().unwrap(),
                    &kv_compression,
                )?;
            } else {
                loaded.allocate_kv_cache_for_layers(max_len.try_into().unwrap())?;
            }
            let generated = generate_text(
                &loaded,
                GenerateOptions {
                    prompt,
                    max_tokens: Some(max_tokens),
                    temperature,
                    top_k,
                    top_p,
                    seed,
                    log_prefix: "cli",
                    sequence_id: 0,
                    reset_kv_cache: true,
                    kv_compression,
                    prompt_format: prompt_format.into(),
                },
            )?;
            print!("{}", generated.output);
        }
        Commands::Run {
            model,
            addr,
            device_id,
            require_sm52,
            max_context_tokens,
            kv_compress_mode,
            kv_recent_window,
            kv_compress_block,
            kv_compress_top_blocks,
            kv_compress_representatives,
            kv_compress_representative_policy,
            kv_exact_old_backing,
            kv_exact_old_attention,
        } => {
            // Silence unused variable warnings when server feature is off
            let _ = (
                &model,
                &addr,
                device_id,
                require_sm52,
                max_context_tokens,
                kv_compress_mode,
                kv_recent_window,
                kv_compress_block,
                kv_compress_top_blocks,
                kv_compress_representatives,
                kv_compress_representative_policy,
                kv_exact_old_backing,
                kv_exact_old_attention,
            );
            #[cfg(feature = "server")]
            {
                let kv_compression = kv_config_from_cli(CliKvOptions {
                    mode: kv_compress_mode,
                    recent_window: kv_recent_window,
                    block_size: kv_compress_block,
                    top_blocks: kv_compress_top_blocks,
                    representatives: kv_compress_representatives,
                    representative_policy: kv_compress_representative_policy,
                    exact_old_backing: kv_exact_old_backing,
                    exact_old_attention: kv_exact_old_attention,
                });
                kv_compression.validate().map_err(|err| {
                    anyhow::anyhow!(
                        "{err}; use --kv-compress-mode off for dense reference/compatibility mode"
                    )
                })?;
                let local = model::resolve_model_arg(&model)?;

                // If gguf_ext feature is enabled, inspect via gguf-llms for a quick overview
                #[cfg(feature = "gguf_ext")]
                {
                    if let Ok(info) = gguf_ext::overview(&local.path) {
                        println!(
                            "[gguf_ext] model overview: tensors={}, kv_len={}",
                            info.n_tensors, info.kv_len
                        );
                    } else {
                        println!("[gguf_ext] overview unavailable for this file");
                    }
                }

                let gguf_bytes = fs::read(&local.path)?;
                let gguf_model = gguf::load_gguf(&local.path)?;
                let mut loaded = infer::LoadedModel::from_gguf(gguf_model, gguf_bytes, device_id)?; // device selection

                #[cfg(feature = "cuda")]
                eprintln!(
                    "[mem] (load) pid={} device_id={} TOTAL_DEVICE_BYTES={}",
                    std::process::id(),
                    loaded.cuda.device_id(),
                    m40_llm::cuda::CudaContext::total_device_bytes()
                );

                // Optional: enforce sm_52 guard
                if require_sm52 {
                    let props = loaded.cuda.current_device_props()?;
                    if !(props.major == 5 && props.minor == 2) {
                        anyhow::bail!(
                            "require_sm52 set but active device is '{}' sm_{}{} (id {})",
                            props.name,
                            props.major,
                            props.minor,
                            props.device_id
                        );
                    }
                    warn_if_gpu_has_other_compute_processes(props.device_id);
                } else {
                    // Informative log of active device
                    let props = loaded.cuda.current_device_props()?;
                    println!(
                        "[cuda] device: '{}' (id {}), sm_{}{}",
                        props.name, props.device_id, props.major, props.minor
                    );
                    warn_if_gpu_has_other_compute_processes(props.device_id);
                }

                let max_len = bounded_context_len(
                    loaded.model_config.context_length as usize,
                    max_context_tokens,
                )?;
                if max_len < loaded.model_config.context_length as usize {
                    eprintln!(
                        "[server] limiting allocated KV context to {} tokens (model context {})",
                        max_len, loaded.model_config.context_length
                    );
                }
                let max_sequences =
                    if std::env::var("M40LLM_SERVER_BATCH_DECODE").ok().as_deref() == Some("1") {
                        std::env::var("M40LLM_SERVER_BATCH_DECODE_SLOTS")
                            .ok()
                            .and_then(|value| value.parse::<u32>().ok())
                            .filter(|value| *value > 0)
                            .unwrap_or(2)
                    } else {
                        1
                    };
                let use_compressed_kv = matches!(
                    kv_compression.mode,
                    KvCompressMode::RecentOnly
                        | KvCompressMode::BlockSummary
                        | KvCompressMode::BlockSelectLossy
                ) || kv_compression.uses_compressed_exact_old_backing();
                if use_compressed_kv {
                    if max_sequences != 1 {
                        anyhow::bail!(
                            "compressed KV server startup currently supports one logical sequence; use --kv-compress-mode off for dense batched decode"
                        );
                    }
                    loaded
                        .allocate_compressed_kv_cache_for_layers(max_len.try_into().unwrap(), &kv_compression)
                        .map_err(|err| {
                            anyhow::anyhow!(
                                "{err}; use --kv-compress-mode off for dense reference/compatibility mode"
                            )
                        })?;
                } else {
                    // Allocate one KV slot per layer and logical sequence.
                    loaded.allocate_kv_cache_for_layer_sequences(
                        max_len.try_into().unwrap(),
                        max_sequences,
                    )?;
                }

                let state = Arc::new(server::AppState::new_with_kv_config(loaded, kv_compression));
                let router = server::app_router(state);

                let listener = TcpListener::bind(&addr).await?;
                println!("Serving {} on http://{}/generate", model, addr);

                axum::serve(listener, router.into_make_service()).await?;
            }
            #[cfg(not(feature = "server"))]
            {
                anyhow::bail!("server feature is disabled; enable with `--features server`");
            }
        }
    }

    Ok(())
}
