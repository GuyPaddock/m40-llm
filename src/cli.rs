// src/cli.rs
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "m40-llm")]
#[command(about = "M40-optimized Rust GGUF LLM server", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Pull a model (GGUF) from a remote registry / HF
    Pull {
        /// Model name, e.g. "mistral:7b-instruct"
        model: String,
        /// Optional remote URL/alias (future)
        #[arg(long)]
        source: Option<String>,
    },

    /// List locally available models
    List {},

    /// Run the HTTP server for a given model
    Run {
        model: String,
        #[arg(long, default_value = "0.0.0.0:11434")]
        addr: String,
    },
}
