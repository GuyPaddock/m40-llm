// src/main.rs
// Use the library crate instead of re-declaring modules to avoid duplicate code
use anyhow::Result;
use clap::Parser;
use m40_llm::cli::{Cli, Commands};
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
            kv_compress_mode,
            kv_recent_window,
            kv_compress_block,
            kv_compress_top_blocks,
            kv_compress_representatives,
            kv_compress_representative_policy,
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

            let max_len = loaded.model_config.context_length as usize;
            let kv_compression = KvCompressionConfig {
                mode: kv_compress_mode.into(),
                recent_window: kv_recent_window,
                block_size: kv_compress_block,
                top_blocks: kv_compress_top_blocks,
                representatives: kv_compress_representatives,
                representative_policy: kv_compress_representative_policy.into(),
            };
            let use_compressed_kv = matches!(
                kv_compression.mode,
                KvCompressMode::RecentOnly
                    | KvCompressMode::BlockSummary
                    | KvCompressMode::BlockSelectLossy
            ) || (kv_compression.mode == KvCompressMode::BlockSelectExact
                && std::env::var("M40LLM_KV_EXACT_OLD_BACKING")
                    .map(|value| {
                        matches!(value.as_str(), "q8" | "Q8" | "fp16-k-q4-v" | "FP16-K-Q4-V")
                    })
                    .unwrap_or(false));
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
        } => {
            // Silence unused variable warnings when server feature is off
            let _ = (&model, &addr, device_id, require_sm52);
            #[cfg(feature = "server")]
            {
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
                } else {
                    // Informative log of active device
                    let props = loaded.cuda.current_device_props()?;
                    println!(
                        "[cuda] device: '{}' (id {}), sm_{}{}",
                        props.name, props.device_id, props.major, props.minor
                    );
                }

                // Allocate one KV slot per layer and logical sequence.
                let max_len = loaded.model_config.context_length as usize;
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
                loaded.allocate_kv_cache_for_layer_sequences(
                    max_len.try_into().unwrap(),
                    max_sequences,
                )?;

                let state = Arc::new(server::AppState::new(loaded));
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
