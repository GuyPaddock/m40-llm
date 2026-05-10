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

    /// Generate text once from a local model name or GGUF path
    Generate {
        model: String,
        prompt: String,
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
        #[arg(long)]
        temperature: Option<f32>,
        #[arg(long)]
        top_k: Option<usize>,
        #[arg(long)]
        top_p: Option<f32>,
        #[arg(long)]
        seed: Option<u64>,
        /// CUDA device id to use; -1 = auto-select Tesla M40 (sm_52) if available
        #[arg(long, default_value_t = -1)]
        device_id: i32,
        /// If set, abort if the active device is not sm_52 (Tesla M40)
        #[arg(long, default_value_t = false)]
        require_sm52: bool,
    },

    /// Run the HTTP server for a local model name or GGUF path
    Run {
        model: String,
        #[arg(long, default_value = "0.0.0.0:11434")]
        addr: String,
        /// CUDA device id to use; -1 = auto-select Tesla M40 (sm_52) if available
        #[arg(long, default_value_t = -1)]
        device_id: i32,
        /// If set, abort if the active device is not sm_52 (Tesla M40)
        #[arg(long, default_value_t = false)]
        require_sm52: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::{Cli, Commands};
    use clap::Parser;

    #[test]
    fn parses_generate_command() {
        let cli = Cli::parse_from([
            "m40-llm",
            "generate",
            "tiny.gguf",
            "Hello",
            "--max-tokens",
            "2",
            "--top-k",
            "1",
            "--seed",
            "7",
            "--require-sm52",
        ]);

        match cli.command {
            Commands::Generate {
                model,
                prompt,
                max_tokens,
                top_k,
                seed,
                require_sm52,
                ..
            } => {
                assert_eq!(model, "tiny.gguf");
                assert_eq!(prompt, "Hello");
                assert_eq!(max_tokens, 2);
                assert_eq!(top_k, Some(1));
                assert_eq!(seed, Some(7));
                assert!(require_sm52);
            }
            other => panic!("expected generate command, got {other:?}"),
        }
    }
}
