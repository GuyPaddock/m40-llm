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

    /// Construct from GGUF metadata when available. For now, returns byte-level fallback.
    pub fn from_gguf_metadata(
        _metadata: &std::collections::HashMap<String, crate::gguf::GgufValue>,
    ) -> Result<Self> {
        // TODO: inspect keys such as tokenizer.ggml.* or sentencepiece.model and select appropriate backend.
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
