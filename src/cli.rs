// src/cli.rs
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum KvCompressModeArg {
    Off,
    DenseRecentOnly,
    BlockSelectExact,
    RecentOnly,
    BlockSummary,
    BlockSelectLossy,
}

impl From<KvCompressModeArg> for crate::kv_compression::KvCompressMode {
    fn from(value: KvCompressModeArg) -> Self {
        match value {
            KvCompressModeArg::Off => Self::Off,
            KvCompressModeArg::DenseRecentOnly => Self::DenseRecentOnly,
            KvCompressModeArg::BlockSelectExact => Self::BlockSelectExact,
            KvCompressModeArg::RecentOnly => Self::RecentOnly,
            KvCompressModeArg::BlockSummary => Self::BlockSummary,
            KvCompressModeArg::BlockSelectLossy => Self::BlockSelectLossy,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum KvRepresentativePolicyArg {
    Last,
    Stride,
}

impl From<KvRepresentativePolicyArg> for crate::kv_compression::KvRepresentativePolicy {
    fn from(value: KvRepresentativePolicyArg) -> Self {
        match value {
            KvRepresentativePolicyArg::Last => Self::Last,
            KvRepresentativePolicyArg::Stride => Self::Stride,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum KvExactOldBackingArg {
    Dense,
    Q8,
    #[value(name = "fp16-k-q4-v")]
    Fp16KQ4V,
}

impl From<KvExactOldBackingArg> for crate::kv_compression::KvExactOldBacking {
    fn from(value: KvExactOldBackingArg) -> Self {
        match value {
            KvExactOldBackingArg::Dense => Self::Dense,
            KvExactOldBackingArg::Q8 => Self::Q8,
            KvExactOldBackingArg::Fp16KQ4V => Self::Fp16KQ4V,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum KvExactOldAttentionArg {
    Staged,
    #[value(name = "q8-direct")]
    Q8Direct,
    #[value(name = "fp16-k-q4-v-direct")]
    Fp16KQ4VDirect,
}

impl From<KvExactOldAttentionArg> for crate::kv_compression::KvExactOldAttention {
    fn from(value: KvExactOldAttentionArg) -> Self {
        match value {
            KvExactOldAttentionArg::Staged => Self::Staged,
            KvExactOldAttentionArg::Q8Direct => Self::Q8Direct,
            KvExactOldAttentionArg::Fp16KQ4VDirect => Self::Fp16KQ4VDirect,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum PromptFormatArg {
    Auto,
    Raw,
    Llama3Chat,
    QwenChat,
}

impl From<PromptFormatArg> for crate::generate::PromptFormat {
    fn from(value: PromptFormatArg) -> Self {
        match value {
            PromptFormatArg::Auto => Self::Auto,
            PromptFormatArg::Raw => Self::Raw,
            PromptFormatArg::Llama3Chat => Self::Llama3Chat,
            PromptFormatArg::QwenChat => Self::QwenChat,
        }
    }
}

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
        /// Experimental compressed KV-cache mode
        #[arg(long, value_enum, default_value_t = KvCompressModeArg::Off)]
        kv_compress_mode: KvCompressModeArg,
        /// Exact recent KV tokens to preserve in compressed modes
        #[arg(long, default_value_t = 1024)]
        kv_recent_window: u32,
        /// Older-context block size for compressed KV modes
        #[arg(long, default_value_t = 32)]
        kv_compress_block: u32,
        /// Number of old blocks selected by block-select modes
        #[arg(long, default_value_t = 16)]
        kv_compress_top_blocks: u32,
        /// Representative tokens retained per old block in lossy modes
        #[arg(long, default_value_t = 0)]
        kv_compress_representatives: u32,
        /// Representative token selection policy for lossy compressed KV modes
        #[arg(long, value_enum, default_value_t = KvRepresentativePolicyArg::Last)]
        kv_compress_representative_policy: KvRepresentativePolicyArg,
        /// Exact-old KV backing for block-select-exact mode
        #[arg(long, value_enum, default_value_t = KvExactOldBackingArg::Dense)]
        kv_exact_old_backing: KvExactOldBackingArg,
        /// Exact-old attention backend for block-select-exact mode
        #[arg(long, value_enum, default_value_t = KvExactOldAttentionArg::Staged)]
        kv_exact_old_attention: KvExactOldAttentionArg,
        /// Prompt wrapper to use before tokenization
        #[arg(long, value_enum, default_value_t = PromptFormatArg::Auto)]
        prompt_format: PromptFormatArg,
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
    use super::{
        Cli, Commands, KvCompressModeArg, KvExactOldAttentionArg, KvExactOldBackingArg,
        KvRepresentativePolicyArg, PromptFormatArg,
    };
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
                kv_compress_mode,
                kv_recent_window,
                kv_compress_block,
                kv_compress_top_blocks,
                kv_compress_representatives,
                kv_compress_representative_policy,
                kv_exact_old_backing,
                kv_exact_old_attention,
                prompt_format,
                ..
            } => {
                assert_eq!(model, "tiny.gguf");
                assert_eq!(prompt, "Hello");
                assert_eq!(max_tokens, 2);
                assert_eq!(top_k, Some(1));
                assert_eq!(seed, Some(7));
                assert!(require_sm52);
                assert_eq!(kv_compress_mode, KvCompressModeArg::Off);
                assert_eq!(kv_recent_window, 1024);
                assert_eq!(kv_compress_block, 32);
                assert_eq!(kv_compress_top_blocks, 16);
                assert_eq!(kv_compress_representatives, 0);
                assert_eq!(
                    kv_compress_representative_policy,
                    KvRepresentativePolicyArg::Last
                );
                assert_eq!(kv_exact_old_backing, KvExactOldBackingArg::Dense);
                assert_eq!(kv_exact_old_attention, KvExactOldAttentionArg::Staged);
                assert_eq!(prompt_format, PromptFormatArg::Auto);
            }
            other => panic!("expected generate command, got {other:?}"),
        }
    }

    #[test]
    fn parses_qwen_chat_prompt_format() {
        let cli = Cli::parse_from([
            "m40-llm",
            "generate",
            "qwen.gguf",
            "Hello",
            "--prompt-format",
            "qwen-chat",
        ]);

        match cli.command {
            Commands::Generate { prompt_format, .. } => {
                assert_eq!(prompt_format, PromptFormatArg::QwenChat);
            }
            other => panic!("expected generate command, got {other:?}"),
        }
    }

    #[test]
    fn parses_exact_old_kv_runtime_flags() {
        let cli = Cli::parse_from([
            "m40-llm",
            "generate",
            "llama.gguf",
            "Find the config value",
            "--kv-compress-mode",
            "block-select-exact",
            "--kv-compress-top-blocks",
            "8",
            "--kv-exact-old-backing",
            "fp16-k-q4-v",
            "--kv-exact-old-attention",
            "fp16-k-q4-v-direct",
        ]);

        match cli.command {
            Commands::Generate {
                kv_compress_mode,
                kv_compress_top_blocks,
                kv_exact_old_backing,
                kv_exact_old_attention,
                ..
            } => {
                assert_eq!(kv_compress_mode, KvCompressModeArg::BlockSelectExact);
                assert_eq!(kv_compress_top_blocks, 8);
                assert_eq!(kv_exact_old_backing, KvExactOldBackingArg::Fp16KQ4V);
                assert_eq!(
                    kv_exact_old_attention,
                    KvExactOldAttentionArg::Fp16KQ4VDirect
                );
            }
            other => panic!("expected generate command, got {other:?}"),
        }
    }
}
