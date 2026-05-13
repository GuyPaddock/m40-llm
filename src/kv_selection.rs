use std::collections::BTreeSet;
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone)]
pub struct KvSelectionSummary {
    pub selected_block_indices: Vec<u32>,
    pub total_old_blocks: u32,
    pub top_blocks: u32,
}

#[derive(Debug, Default)]
struct KvSelectionState {
    selected: BTreeSet<u32>,
    total_old_blocks: u32,
    top_blocks: u32,
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

pub fn snapshot() -> Option<KvSelectionSummary> {
    let guard = state().lock().ok()?;
    if guard.total_old_blocks == 0 && guard.selected.is_empty() {
        return None;
    }
    Some(KvSelectionSummary {
        selected_block_indices: guard.selected.iter().copied().collect(),
        total_old_blocks: guard.total_old_blocks,
        top_blocks: guard.top_blocks,
    })
}
