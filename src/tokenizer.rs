// src/tokenizer.rs
//! Minimal tokenizer scaffolding with a safe, deterministic byte-level fallback.
//! Later, integrate SentencePiece/BPE from GGUF metadata via gguf_ext.

use anyhow::{anyhow, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerKind {
    ByteLevel,
    SentencePiece,
    Bpe,
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    kind: TokenizerKind,
    // Special token IDs when provided by GGUF metadata
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    pad_id: Option<u32>,
    unk_id: Option<u32>,
}

impl Tokenizer {
    pub fn byte_level() -> Self {
        Self {
            kind: TokenizerKind::ByteLevel,
            bos_id: None,
            eos_id: None,
            pad_id: None,
            unk_id: None,
        }
    }

    pub fn kind(&self) -> &TokenizerKind {
        &self.kind
    }

    /// Construct from GGUF metadata when available. Fallback to byte-level.
    /// Detection heuristics (non-exhaustive):
    /// - If metadata contains a SentencePiece model (e.g., "tokenizer.ggml.model" == "spm" or
    ///   keys like "sentencepiece.model"), we would select SentencePiece (not yet implemented).
    /// - If BPE merges/vocab present, we would select BPE (not yet implemented).
    /// For now: always return byte-level; wire-up keeps interface stable for later integration.
    pub fn from_gguf_metadata(
        metadata: &std::collections::HashMap<String, crate::gguf::GgufValue>,
    ) -> Result<Self> {
        // Heuristic detection of tokenizer kind:
        let model_str = metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_ascii_lowercase());
        let has_sp = metadata
            .get("sentencepiece.model")
            .and_then(|v| v.as_str())
            .is_some()
            || matches!(model_str.as_deref(), Some("spm" | "sentencepiece"));
        let has_bpe = metadata
            .get("tokenizer.ggml.bpe_merges")
            .and_then(|v| v.as_str())
            .is_some()
            || matches!(model_str.as_deref(), Some("bpe"));

        let kind = if has_sp {
            TokenizerKind::SentencePiece
        } else if has_bpe {
            TokenizerKind::Bpe
        } else {
            TokenizerKind::ByteLevel
        };

        // Special token IDs are optional; try GGUF keys commonly used by GGUF/llama.cpp
        use crate::gguf::{GgufScalar, GgufValue};
        let get_u32 = |k: &str| -> Option<u32> {
            metadata.get(k).and_then(|v| match v {
                GgufValue::Scalar(GgufScalar::U32(x)) => Some(*x),
                GgufValue::Scalar(GgufScalar::I32(x)) => u32::try_from(*x).ok(),
                _ => None,
            })
        };
        // Common GGUF keys: tokenizer.ggml.* or special_tokens.*
        let bos_id =
            get_u32("tokenizer.ggml.bos_token_id").or_else(|| get_u32("special_tokens.bos_id"));
        let eos_id =
            get_u32("tokenizer.ggml.eos_token_id").or_else(|| get_u32("special_tokens.eos_id"));
        let pad_id =
            get_u32("tokenizer.ggml.pad_token_id").or_else(|| get_u32("special_tokens.pad_id"));
        let unk_id =
            get_u32("tokenizer.ggml.unknown_token_id").or_else(|| get_u32("special_tokens.unk_id"));

        Ok(Self {
            kind,
            bos_id,
            eos_id,
            pad_id,
            unk_id,
        })
    }

    /// Encode a UTF-8 string into token IDs.
    /// Byte-level fallback: one byte per token (u32 in 0..=255).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self.kind {
            TokenizerKind::ByteLevel | TokenizerKind::SentencePiece | TokenizerKind::Bpe => {
                Ok(text.as_bytes().iter().map(|&b| b as u32).collect())
            }
        }
    }

    /// Decode token IDs back into a String.
    /// Invalid byte values (>255) are replaced with 0x3F ('?').
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match self.kind {
            TokenizerKind::ByteLevel | TokenizerKind::SentencePiece | TokenizerKind::Bpe => {
                let bytes: Vec<u8> = ids
                    .iter()
                    .map(|&id| if id <= 255 { id as u8 } else { b'?' })
                    .collect();
                String::from_utf8(bytes).map_err(|e| anyhow!("decode produced invalid UTF-8: {e}"))
            }
        }
    }
}
