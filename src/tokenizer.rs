// src/tokenizer.rs
//! Minimal tokenizer with GGUF-derived SentencePiece/BPE vocab/merges when available.
//! Falls back to deterministic byte-level encoding when metadata is missing.

use anyhow::{anyhow, Result};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerKind {
    ByteLevel,
    SentencePiece,
    Bpe,
    Llama3,
    Qwen2,
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    kind: TokenizerKind,
    // Special token IDs when provided by GGUF metadata
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    pad_id: Option<u32>,
    unk_id: Option<u32>,
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    bpe_ranks: HashMap<(String, String), usize>,
    non_content_special_ids: Vec<u32>,
    has_chat_template: bool,
}

impl Tokenizer {
    pub fn byte_level() -> Self {
        Self {
            kind: TokenizerKind::ByteLevel,
            bos_id: None,
            eos_id: None,
            pad_id: None,
            unk_id: None,
            vocab: Vec::new(),
            token_to_id: HashMap::new(),
            bpe_ranks: HashMap::new(),
            non_content_special_ids: Vec::new(),
            has_chat_template: false,
        }
    }

    pub fn kind(&self) -> &TokenizerKind {
        &self.kind
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }
    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }
    pub fn pad_id(&self) -> Option<u32> {
        self.pad_id
    }
    pub fn unk_id(&self) -> Option<u32> {
        self.unk_id
    }
    pub fn is_llama3(&self) -> bool {
        self.kind == TokenizerKind::Llama3
    }
    pub fn is_qwen2(&self) -> bool {
        self.kind == TokenizerKind::Qwen2
    }
    pub fn has_chat_template(&self) -> bool {
        self.has_chat_template
    }
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }
    pub fn stop_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        if let Some(eos) = self.eos_id {
            ids.push(eos);
        }
        if self.is_llama3() {
            if let Some(eot) = self.token_id("<|eot_id|>") {
                ids.push(eot);
            }
            if let Some(end_text) = self.token_id("<|end_of_text|>") {
                ids.push(end_text);
            }
        }
        if self.is_qwen2() {
            if let Some(im_end) = self.token_id("<|im_end|>") {
                ids.push(im_end);
            }
            if let Some(end_text) = self.token_id("<|endoftext|>") {
                ids.push(end_text);
            }
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    // Mutable setters used in tests to simulate GGUF-provided IDs
    pub fn set_bos_id(&mut self, id: Option<u32>) {
        self.bos_id = id;
    }
    pub fn set_eos_id(&mut self, id: Option<u32>) {
        self.eos_id = id;
    }
    pub fn set_pad_id(&mut self, id: Option<u32>) {
        self.pad_id = id;
    }
    pub fn set_unk_id(&mut self, id: Option<u32>) {
        self.unk_id = id;
    }

    /// Construct from GGUF metadata when available. Falls back to byte-level if metadata is absent.
    /// Detection heuristics (non-exhaustive):
    /// - If metadata contains a SentencePiece model (e.g., "tokenizer.ggml.model" == "spm" or
    ///   keys like "sentencepiece.model"), we select SentencePiece.
    /// - If BPE merges/vocab present, we select BPE.
    pub fn from_gguf_metadata(
        metadata: &std::collections::HashMap<String, crate::gguf::GgufValue>,
    ) -> Result<Self> {
        use crate::gguf::{GgufScalar, GgufValue};
        // Heuristic detection of tokenizer kind:
        let model_str = metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_ascii_lowercase());
        let pre_str = metadata
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .map(|s| s.to_ascii_lowercase());
        let has_tokens = metadata.get("tokenizer.ggml.tokens").is_some();
        let has_chat_template = metadata.get("tokenizer.chat_template").is_some();
        let architecture = metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .map(|s| s.to_ascii_lowercase());
        let has_sp = metadata
            .get("sentencepiece.model")
            .and_then(|v| v.as_str())
            .is_some()
            || matches!(
                model_str.as_deref(),
                Some("spm" | "sentencepiece" | "llama")
            );
        let has_bpe = metadata
            .get("tokenizer.ggml.bpe_merges")
            .and_then(|v| v.as_str())
            .is_some()
            || matches!(model_str.as_deref(), Some("bpe"))
            || has_tokens;
        let is_llama_bpe = model_str.as_deref() == Some("gpt2")
            && pre_str.as_deref() == Some("llama-bpe")
            && metadata.get("llama.vocab_size").and_then(|v| match v {
                GgufValue::Scalar(GgufScalar::U32(x)) => Some(*x),
                GgufValue::Scalar(GgufScalar::I32(x)) => u32::try_from(*x).ok(),
                _ => None,
            }) == Some(128_256);
        let is_qwen_bpe = architecture
            .as_deref()
            .is_some_and(|arch| arch.starts_with("qwen"))
            || pre_str
                .as_deref()
                .is_some_and(|pre| pre.starts_with("qwen"));

        let kind = if is_llama_bpe {
            TokenizerKind::Llama3
        } else if is_qwen_bpe {
            TokenizerKind::Qwen2
        } else if has_sp {
            TokenizerKind::SentencePiece
        } else if has_bpe {
            TokenizerKind::Bpe
        } else {
            TokenizerKind::ByteLevel
        };

        // Special token IDs are optional; try GGUF keys commonly used by GGUF/llama.cpp
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

        // Extract vocab and merges when present
        let vocab = extract_string_array(metadata, "tokenizer.ggml.tokens").unwrap_or_default();
        let token_to_id: HashMap<_, _> = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        let non_content_special_ids = vocab
            .iter()
            .enumerate()
            .filter_map(|(i, token)| {
                let is_special = token.starts_with("<|") && token.ends_with("|>");
                is_special.then_some(i as u32)
            })
            .collect();
        let merges =
            extract_string_array(metadata, "tokenizer.ggml.bpe_merges").unwrap_or_else(|| {
                extract_string_array(metadata, "tokenizer.ggml.merges").unwrap_or_default()
            });
        let mut bpe_ranks = HashMap::new();
        for (rank, merge) in merges.iter().enumerate() {
            let parts: Vec<&str> = merge.split_whitespace().collect();
            if parts.len() == 2 {
                bpe_ranks.insert((parts[0].to_string(), parts[1].to_string()), rank);
            }
        }

        Ok(Self {
            kind,
            bos_id,
            eos_id,
            pad_id,
            unk_id,
            vocab,
            token_to_id,
            bpe_ranks,
            non_content_special_ids,
            has_chat_template,
        })
    }

    /// Encode a UTF-8 string into token IDs.
    /// Byte-level fallback: one byte per token (u32 in 0..=255).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self.kind {
            TokenizerKind::ByteLevel => Ok(text.as_bytes().iter().map(|&b| b as u32).collect()),
            TokenizerKind::SentencePiece => Ok(self.encode_sentencepiece(text)),
            TokenizerKind::Bpe => Ok(self.encode_bpe(text)),
            TokenizerKind::Qwen2 => {
                let enc = tiktoken::get_encoding("qwen2")
                    .ok_or_else(|| anyhow!("qwen2 tokenizer encoding unavailable"))?;
                Ok(enc.encode(text))
            }
            TokenizerKind::Llama3 => {
                let enc = tiktoken::get_encoding("llama3")
                    .ok_or_else(|| anyhow!("llama3 tokenizer encoding unavailable"))?;
                Ok(enc.encode(text))
            }
        }
    }

    pub fn encode_allowing_specials(&self, text: &str) -> Result<Vec<u32>> {
        match self.kind {
            TokenizerKind::Llama3 => {
                let enc = tiktoken::get_encoding("llama3")
                    .ok_or_else(|| anyhow!("llama3 tokenizer encoding unavailable"))?;
                Ok(enc.encode_with_special_tokens(text))
            }
            TokenizerKind::Qwen2 => {
                let enc = tiktoken::get_encoding("qwen2")
                    .ok_or_else(|| anyhow!("qwen2 tokenizer encoding unavailable"))?;
                Ok(enc.encode_with_special_tokens(text))
            }
            _ => self.encode(text),
        }
    }

    /// Encode and optionally add BOS/EOS if the IDs are known from metadata.
    pub fn encode_with_specials(
        &self,
        text: &str,
        add_bos: bool,
        add_eos: bool,
    ) -> Result<Vec<u32>> {
        let mut ids = self.encode_allowing_specials(text)?;
        if add_bos {
            if let Some(bos) = self.bos_id {
                ids.insert(0, bos);
            }
        }
        if add_eos {
            if let Some(eos) = self.eos_id {
                ids.push(eos);
            }
        }
        Ok(ids)
    }

    /// Determine if an id is a non-content special token (BOS/EOS/PAD).
    /// UNK is treated as content-bearing and returns false here.
    pub fn is_special(&self, id: u32) -> bool {
        self.bos_id == Some(id) || self.eos_id == Some(id) || self.pad_id == Some(id)
    }

    /// Filter out BOS/EOS/PAD tokens. Leaves UNK in place since it's a content-bearing token.
    pub fn strip_non_content_specials(&self, ids: &[u32]) -> Vec<u32> {
        ids.iter()
            .copied()
            .filter(|&id| {
                Some(id) != self.bos_id
                    && Some(id) != self.eos_id
                    && Some(id) != self.pad_id
                    && !self.non_content_special_ids.contains(&id)
            })
            .collect()
    }

    /// Decode token IDs back into a String.
    /// Invalid byte values (>255) are replaced with 0x3F ('?').
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        if self.vocab.is_empty() {
            let bytes: Vec<u8> = ids
                .iter()
                .filter(|&id| *id != 0)
                .map(|&id| if id <= 255 { id as u8 } else { b'?' })
                .collect();
            return String::from_utf8(bytes)
                .map_err(|e| anyhow!("decode produced invalid UTF-8: {e}"));
        }

        if self.kind == TokenizerKind::Llama3 || self.kind == TokenizerKind::Qwen2 {
            let encoding_name = if self.kind == TokenizerKind::Qwen2 {
                "qwen2"
            } else {
                "llama3"
            };
            let enc = tiktoken::get_encoding(encoding_name)
                .ok_or_else(|| anyhow!("{encoding_name} tokenizer encoding unavailable"))?;
            return enc
                .decode_to_string(ids)
                .map_err(|e| anyhow!("decode produced invalid UTF-8: {e}"));
        }

        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(tok) = self.vocab.get(id as usize) {
                if let Some(byte) = parse_byte_fallback_token(tok) {
                    bytes.push(byte);
                } else {
                    bytes.extend_from_slice(tok.as_bytes());
                }
            } else if id <= 255 {
                bytes.push(id as u8);
            } else {
                bytes.push(b'?');
            }
        }

        let joined = decode_utf8_with_qmark(&bytes);
        let text = match self.kind {
            TokenizerKind::SentencePiece => {
                let mut replaced = joined.replace('▁', " ");
                if replaced.starts_with(' ') {
                    replaced = replaced.trim_start().to_string();
                }
                replaced
            }
            TokenizerKind::Qwen2 => joined.replace('Ġ', " ").replace('Ċ', "\n"),
            _ => joined,
        };

        Ok(text)
    }

    /// Decode while ignoring BOS/EOS/PAD tokens if they are configured.
    pub fn decode_ignoring_specials(&self, ids: &[u32]) -> Result<String> {
        let filtered = self.strip_non_content_specials(ids);
        self.decode(&filtered)
    }

    fn encode_bpe(&self, text: &str) -> Vec<u32> {
        if self.vocab.is_empty() {
            return text.as_bytes().iter().map(|&b| b as u32).collect();
        }
        let mut ids = Vec::new();
        for (i, word) in text.split_whitespace().enumerate() {
            let piece = if i == 0 {
                word.to_string()
            } else {
                format!(" {}", word)
            };
            if piece.is_empty() {
                continue;
            }
            self.encode_piece(&piece, &mut ids);
        }
        ids
    }

    fn encode_sentencepiece(&self, text: &str) -> Vec<u32> {
        if self.vocab.is_empty() {
            return text.as_bytes().iter().map(|&b| b as u32).collect();
        }
        let mut ids = Vec::new();
        let mut current = String::new();
        let mut at_word_start = true;
        for ch in text.chars() {
            if ch.is_whitespace() {
                at_word_start = true;
                continue;
            }
            if at_word_start {
                if !current.is_empty() {
                    self.encode_piece(&current, &mut ids);
                    current.clear();
                }
                current.push('▁');
                at_word_start = false;
            }
            current.push(ch);
        }
        if !current.is_empty() {
            self.encode_piece(&current, &mut ids);
        }
        ids
    }

    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        if self.vocab.is_empty() {
            out.extend(piece.as_bytes().iter().map(|&b| b as u32));
            return;
        }

        if let Some(id) = self.token_to_id.get(piece) {
            out.push(*id);
            return;
        }

        if self.kind == TokenizerKind::SentencePiece && self.bpe_ranks.is_empty() {
            self.encode_piece_longest_match(piece, out);
            return;
        }

        let mut symbols: Vec<String> = piece.chars().map(|c| c.to_string()).collect();
        if !self.bpe_ranks.is_empty() {
            symbols = self.apply_bpe(symbols);
        }
        if symbols.len() == 1 {
            if let Some(id) = self.token_to_id.get(&symbols[0]) {
                out.push(*id);
                return;
            }
        }
        for sym in symbols {
            if let Some(id) = self.token_to_id.get(&sym) {
                out.push(*id);
            } else if let Some(unk) = self.unk_id {
                out.push(unk);
            } else {
                out.extend(sym.as_bytes().iter().map(|&b| b as u32));
            }
        }
    }

    fn encode_piece_longest_match(&self, piece: &str, out: &mut Vec<u32>) {
        let starts: Vec<usize> = piece
            .char_indices()
            .map(|(idx, _)| idx)
            .chain(std::iter::once(piece.len()))
            .collect();
        let mut pos_idx = 0usize;
        while pos_idx + 1 < starts.len() {
            let start = starts[pos_idx];
            let mut matched = None;
            for end_idx in ((pos_idx + 1)..starts.len()).rev() {
                let candidate = &piece[start..starts[end_idx]];
                if let Some(id) = self.token_to_id.get(candidate) {
                    matched = Some((*id, end_idx));
                    break;
                }
            }
            if let Some((id, next_idx)) = matched {
                out.push(id);
                pos_idx = next_idx;
                continue;
            }

            let ch = &piece[start..starts[pos_idx + 1]];
            if let Some(id) = self.token_to_id.get(ch) {
                out.push(*id);
            } else if let Some(unk) = self.unk_id {
                out.push(unk);
            } else {
                out.extend(ch.as_bytes().iter().map(|&b| b as u32));
            }
            pos_idx += 1;
        }
    }

    fn apply_bpe(&self, mut symbols: Vec<String>) -> Vec<String> {
        if symbols.len() < 2 || self.bpe_ranks.is_empty() {
            return symbols;
        }
        loop {
            let mut best_rank: Option<usize> = None;
            let mut best_idx: usize = 0;
            for i in 0..symbols.len() - 1 {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.bpe_ranks.get(&pair) {
                    if best_rank.is_none_or(|r| rank < r) {
                        best_rank = Some(rank);
                        best_idx = i;
                    }
                }
            }
            if best_rank.is_none() {
                break;
            }
            let merged = format!("{}{}", symbols[best_idx], symbols[best_idx + 1]);
            symbols.splice(best_idx..=best_idx + 1, [merged]);
            if symbols.len() < 2 {
                break;
            }
        }
        symbols
    }
}

fn parse_byte_fallback_token(token: &str) -> Option<u8> {
    let hex = token
        .strip_prefix("<0x")
        .and_then(|rest| rest.strip_suffix('>'))?;
    if hex.len() != 2 || !hex.as_bytes().iter().all(u8::is_ascii_hexdigit) {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

fn decode_utf8_with_qmark(bytes: &[u8]) -> String {
    let mut out = String::new();
    let mut remaining = bytes;
    while !remaining.is_empty() {
        match std::str::from_utf8(remaining) {
            Ok(valid) => {
                out.push_str(valid);
                break;
            }
            Err(err) => {
                let valid_up_to = err.valid_up_to();
                if valid_up_to > 0 {
                    out.push_str(std::str::from_utf8(&remaining[..valid_up_to]).expect("valid"));
                }
                out.push('?');
                let skip = err.error_len().unwrap_or(1);
                remaining = &remaining[valid_up_to + skip..];
            }
        }
    }
    out
}

fn extract_string_array(
    metadata: &std::collections::HashMap<String, crate::gguf::GgufValue>,
    key: &str,
) -> Option<Vec<String>> {
    use crate::gguf::{GgufScalar, GgufValue};
    match metadata.get(key)? {
        GgufValue::Array(arr) => {
            let mut out = Vec::new();
            for item in arr {
                if let GgufScalar::Str(s) = item {
                    out.push(s.clone());
                }
            }
            Some(out)
        }
        GgufValue::Scalar(GgufScalar::Str(s)) => {
            let lines: Vec<String> = s
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .map(|l| l.to_string())
                .collect();
            Some(lines)
        }
        _ => None,
    }
}
