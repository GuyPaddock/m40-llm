// src/tokenizer.rs
//! Minimal tokenizer scaffolding with a safe, deterministic byte-level fallback.
//! Later, integrate SentencePiece/BPE from GGUF metadata via gguf_ext.

use anyhow::{anyhow, Result};

#[derive(Debug, Clone)]
pub enum TokenizerKind {
    ByteLevel,
    // Future: SentencePiece, BPE
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    kind: TokenizerKind,
    // For future use: vocab, merges, special tokens, etc.
}

impl Tokenizer {
    pub fn byte_level() -> Self {
        Self {
            kind: TokenizerKind::ByteLevel,
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
        // placeholder detection; currently returns byte-level until SP/BPE added
        let _maybe_kind = metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str());
        let _maybe_sp = metadata.get("sentencepiece.model").and_then(|v| v.as_str());
        let _maybe_bpe = metadata
            .get("tokenizer.ggml.bpe_merges")
            .and_then(|v| v.as_str());
        Ok(Self::byte_level())
    }

    /// Encode a UTF-8 string into token IDs.
    /// Byte-level fallback: one byte per token (u32 in 0..=255).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self.kind {
            TokenizerKind::ByteLevel => Ok(text.as_bytes().iter().map(|&b| b as u32).collect()),
        }
    }

    /// Decode token IDs back into a String.
    /// Invalid byte values (>255) are replaced with 0x3F ('?').
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match self.kind {
            TokenizerKind::ByteLevel => {
                let bytes: Vec<u8> = ids
                    .iter()
                    .map(|&id| if id <= 255 { id as u8 } else { b'?' })
                    .collect();
                String::from_utf8(bytes).map_err(|e| anyhow!("decode produced invalid UTF-8: {e}"))
            }
        }
    }
}
