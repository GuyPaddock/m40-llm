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
    pub top_entries: Vec<KvAttentionTopEntry>,
}

#[derive(Debug, Clone)]
pub struct KvSelectionSummary {
    pub selected_block_indices: Vec<u32>,
    pub total_old_blocks: u32,
    pub top_blocks: u32,
    pub attention: Option<KvAttentionTelemetrySummary>,
}

#[derive(Debug, Default)]
struct KvSelectionState {
    selected: BTreeSet<u32>,
    total_old_blocks: u32,
    top_blocks: u32,
    attention: Option<KvAttentionTelemetrySummary>,
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

pub fn reset() {
    if let Ok(mut guard) = state().lock() {
        *guard = KvSelectionState::default();
    }
}

pub fn record(selected_blocks: &[u32], total_old_blocks: u32, top_blocks: u32) {
    if let Ok(mut guard) = state().lock() {
        guard.total_old_blocks = guard.total_old_blocks.max(total_old_blocks);
        guard.top_blocks = guard.top_blocks.max(top_blocks);
        guard.selected.extend(selected_blocks.iter().copied());
    }
}

pub fn has_attention() -> bool {
    state()
        .lock()
        .map(|guard| guard.attention.is_some())
        .unwrap_or(false)
}

pub fn record_attention(attention: KvAttentionTelemetrySummary) {
    if let Ok(mut guard) = state().lock() {
        if guard.attention.is_none() {
            guard.attention = Some(attention);
        }
    }
}

pub fn snapshot() -> Option<KvSelectionSummary> {
    let guard = state().lock().ok()?;
    if guard.total_old_blocks == 0 && guard.selected.is_empty() && guard.attention.is_none() {
        return None;
    }
    Some(KvSelectionSummary {
        selected_block_indices: guard.selected.iter().copied().collect(),
        total_old_blocks: guard.total_old_blocks,
        top_blocks: guard.top_blocks,
        attention: guard.attention.clone(),
    })
}
