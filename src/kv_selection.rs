use std::collections::BTreeSet;
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Default)]
pub struct KvAttentionGroupStats {
    pub prob_mass: f32,
    pub logit_max: f32,
    pub logit_mean: f32,
    pub count: u32,
}

#[derive(Debug, Clone)]
pub struct KvSelectedBlock {
    pub rank: u32,
    pub block_index: u32,
    pub score: f32,
    pub absolute_start: u32,
    pub absolute_end: u32,
}

#[derive(Debug, Clone)]
pub struct KvSelectionRecord {
    pub layer: Option<u32>,
    pub token: Option<u32>,
    pub selected_blocks: Vec<KvSelectedBlock>,
    pub total_old_blocks: u32,
    pub top_blocks: u32,
}

#[derive(Debug, Clone)]
pub struct KvAttentionBlockMass {
    pub block_index: u32,
    pub prob_mass: f32,
    pub logit_max: f32,
    pub logit_mean: f32,
    pub count: u32,
}

#[derive(Debug, Clone)]
pub struct KvAttentionTopEntry {
    pub group: String,
    pub block_index: Option<u32>,
    pub token_position: Option<u32>,
    pub score: f32,
    pub probability: f32,
}

#[derive(Debug, Clone)]
pub struct KvAttentionTelemetrySummary {
    pub recent: KvAttentionGroupStats,
    pub selected_old_exact: KvAttentionGroupStats,
    pub summary: KvAttentionGroupStats,
    pub representatives: KvAttentionGroupStats,
    pub other: KvAttentionGroupStats,
    pub needle_block_mass: Option<f32>,
    pub selected_block_masses: Vec<KvAttentionBlockMass>,
    pub top_entries: Vec<KvAttentionTopEntry>,
}

#[derive(Debug, Clone)]
pub struct KvAttentionTelemetryRecord {
    pub layer: Option<u32>,
    pub token: Option<u32>,
    pub attention: KvAttentionTelemetrySummary,
}

#[derive(Debug, Clone)]
pub struct KvSelectionSummary {
    pub selected_block_indices: Vec<u32>,
    pub total_old_blocks: u32,
    pub top_blocks: u32,
    pub selection_records: Vec<KvSelectionRecord>,
    pub attention: Option<KvAttentionTelemetrySummary>,
    pub attentions: Vec<KvAttentionTelemetryRecord>,
}

#[derive(Debug, Default)]
struct KvSelectionState {
    selected: BTreeSet<u32>,
    selection_records: Vec<KvSelectionRecord>,
    total_old_blocks: u32,
    top_blocks: u32,
    attentions: Vec<KvAttentionTelemetryRecord>,
    current_layer: Option<u32>,
    current_token: Option<u32>,
}

static STATE: OnceLock<Mutex<KvSelectionState>> = OnceLock::new();

fn state() -> &'static Mutex<KvSelectionState> {
    STATE.get_or_init(|| Mutex::new(KvSelectionState::default()))
}

pub fn enabled() -> bool {
    std::env::var("M40LLM_KV_SELECTION_TELEMETRY")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

pub fn needle_block() -> Option<u32> {
    std::env::var("M40LLM_KV_TELEMETRY_NEEDLE_BLOCK")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CapturePolicy {
    First,
    All,
    Layer(u32),
    Token(u32),
    LayerToken { layer: u32, token: u32 },
}

impl CapturePolicy {
    fn from_env() -> Self {
        let Ok(value) = std::env::var("M40LLM_KV_ATTENTION_CAPTURE") else {
            return Self::First;
        };
        let value = value.trim();
        if value.eq_ignore_ascii_case("all") {
            return Self::All;
        }
        if value.eq_ignore_ascii_case("first") || value.is_empty() {
            return Self::First;
        }
        let mut layer = None;
        let mut token = None;
        for part in value.split(',') {
            let Some((key, raw)) = part.trim().split_once(':') else {
                continue;
            };
            match key.trim() {
                "layer" => layer = raw.trim().parse::<u32>().ok(),
                "token" => token = raw.trim().parse::<u32>().ok(),
                _ => {}
            }
        }
        match (layer, token) {
            (Some(layer), Some(token)) => Self::LayerToken { layer, token },
            (Some(layer), None) => Self::Layer(layer),
            (None, Some(token)) => Self::Token(token),
            (None, None) => Self::First,
        }
    }

    fn matches(self, layer: Option<u32>, token: Option<u32>) -> bool {
        match self {
            Self::First | Self::All => true,
            Self::Layer(expected) => layer == Some(expected),
            Self::Token(expected) => token == Some(expected),
            Self::LayerToken { layer: l, token: t } => layer == Some(l) && token == Some(t),
        }
    }
}

pub fn reset() {
    if let Ok(mut guard) = state().lock() {
        *guard = KvSelectionState::default();
    }
}

pub fn set_attention_context(layer: Option<u32>, token: Option<u32>) {
    if let Ok(mut guard) = state().lock() {
        guard.current_layer = layer;
        guard.current_token = token;
    }
}

pub fn record(selected_blocks: &[u32], total_old_blocks: u32, top_blocks: u32) {
    let scored: Vec<KvSelectedBlock> = selected_blocks
        .iter()
        .copied()
        .enumerate()
        .map(|(rank, block_index)| KvSelectedBlock {
            rank: rank as u32,
            block_index,
            score: 0.0,
            absolute_start: 0,
            absolute_end: 0,
        })
        .collect();
    record_scored(scored, total_old_blocks, top_blocks);
}

pub fn record_scored(
    selected_blocks: Vec<KvSelectedBlock>,
    total_old_blocks: u32,
    top_blocks: u32,
) {
    if let Ok(mut guard) = state().lock() {
        guard.total_old_blocks = guard.total_old_blocks.max(total_old_blocks);
        guard.top_blocks = guard.top_blocks.max(top_blocks);
        for block in &selected_blocks {
            guard.selected.insert(block.block_index);
        }
        let layer = guard.current_layer;
        let token = guard.current_token;
        guard.selection_records.push(KvSelectionRecord {
            layer,
            token,
            selected_blocks,
            total_old_blocks,
            top_blocks,
        });
    }
}

pub fn has_attention() -> bool {
    state()
        .lock()
        .map(|guard| !guard.attentions.is_empty())
        .unwrap_or(false)
}

pub fn should_capture_attention() -> bool {
    if !enabled() {
        return false;
    }
    let policy = CapturePolicy::from_env();
    state()
        .lock()
        .map(|guard| {
            if policy == CapturePolicy::First && !guard.attentions.is_empty() {
                return false;
            }
            policy.matches(guard.current_layer, guard.current_token)
        })
        .unwrap_or(false)
}

pub fn record_attention(attention: KvAttentionTelemetrySummary) {
    if let Ok(mut guard) = state().lock() {
        let policy = CapturePolicy::from_env();
        if policy == CapturePolicy::First && !guard.attentions.is_empty() {
            return;
        }
        let layer = guard.current_layer;
        let token = guard.current_token;
        guard.attentions.push(KvAttentionTelemetryRecord {
            layer,
            token,
            attention,
        });
    }
}

pub fn snapshot() -> Option<KvSelectionSummary> {
    let guard = state().lock().ok()?;
    if guard.total_old_blocks == 0 && guard.selected.is_empty() && guard.attentions.is_empty() {
        return None;
    }
    let attention = guard
        .attentions
        .first()
        .map(|record| record.attention.clone());
    Some(KvSelectionSummary {
        selected_block_indices: guard.selected.iter().copied().collect(),
        total_old_blocks: guard.total_old_blocks,
        top_blocks: guard.top_blocks,
        selection_records: guard.selection_records.clone(),
        attention,
        attentions: guard.attentions.clone(),
    })
}
