// src/main.rs
// Use the library crate instead of re-declaring modules to avoid duplicate code
use anyhow::Result;
use clap::Parser;
use m40_llm::cli::{Cli, Commands};
#[cfg(not(feature = "server"))]
#[allow(unused_imports)]
use m40_llm::{gguf, infer, model};
#[cfg(feature = "server")]
use {
    m40_llm::{gguf, infer, model, server},
    std::{fs, sync::Arc},
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
        Commands::Run { model, addr } => {
            // Silence unused variable warnings when server feature is off
            let _ = (&model, &addr);
            #[cfg(feature = "server")]
            {
                let local = model::list_models()?
                    .into_iter()
                    .find(|m| m.name == model.replace(':', "_"))
                    .ok_or_else(|| {
                        anyhow::anyhow!(format!("Model not found locally: {}", model))
                    })?;

                let gguf_bytes = fs::read(&local.path)?;
                let gguf_model = gguf::load_gguf(&local.path)?;
                let loaded = infer::LoadedModel::from_gguf(gguf_model, gguf_bytes, 0)?; // GPU 0

                let state = Arc::new(server::AppState { model: loaded });
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
