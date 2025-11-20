// src/main.rs
mod cli;
mod model;
mod gguf;
mod cuda;
mod infer;
mod server;
// mod tokenizer; // stub for now

use crate::cli::{Cli, Commands};
use clap::Parser;
use anyhow::Result;
use std::{fs, sync::Arc};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Pull { model, source } => {
            let m = model::pull_model(&model, source).await?;
            println!("Pulled model: {} ({:.2} MiB)", m.name, m.size_bytes as f64 / (1024.0 * 1024.0));
        }
        Commands::List {} => {
            let models = model::list_models()?;
            if models.is_empty() {
                println!("No models found. Use `m40-llm pull ...`.");
            } else {
                for m in models {
                    println!("{} \t {:.2} MiB \t {}", m.name, m.size_bytes as f64 / (1024.0*1024.0), m.path.display());
                }
            }
        }
        Commands::Run { model, addr } => {
            let local = model::list_models()?
                .into_iter()
                .find(|m| m.name == model.replace(':', "_"))
                .ok_or_else(|| anyhow::anyhow!("Model not found locally: {model}"))?;

            let gguf_bytes = fs::read(&local.path)?;
            let gguf_model = gguf::load_gguf(&local.path)?;
            let loaded = infer::LoadedModel::from_gguf(gguf_model, gguf_bytes, 0)?; // GPU 0

            let state = Arc::new(server::AppState { model: loaded });
            let router = server::app_router(state);

            let listener = TcpListener::bind(&addr).await?;
            println!("Serving {model} on http://{addr}/generate");

            axum::Server::from_tcp(listener)?
                .serve(router.into_make_service())
                .await?;
        }
    }

    Ok(())
}
