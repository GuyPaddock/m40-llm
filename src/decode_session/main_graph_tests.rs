#[path = "../../tests/common/env_restore.rs"]
mod env_restore;

use super::DecodeSession;
use crate::{
    cuda::{ffi_debug_read_kv_token, CudaContext, CudaStream},
    gguf,
    infer::LoadedModel,
    kv_compression::{KvCompressionConfig, ScopedRuntimeConfig},
};
use anyhow::{ensure, Context, Result};
use env_restore::EnvRestore;
use std::path::PathBuf;

fn session(model: &LoadedModel, slot: u32, graph: bool) -> Result<DecodeSession> {
    let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", if graph { "1" } else { "0" });
    DecodeSession::new_for_sequence_with_kv_config(
        model,
        slot,
        KvCompressionConfig::dense_reference(),
        model.model_config.embedding_length as usize,
        true,
        "main-graph",
        "main-graph:x",
        "main-graph:out",
    )
}

fn slot_bits(model: &LoadedModel, sequence: u32, length: u32) -> Result<Vec<u16>> {
    let kv = model.kv_cache.as_ref().context("missing dense KV")?;
    let width = (kv.num_heads() * kv.head_dim()) as usize;
    let mut words = Vec::new();
    model.cuda.synchronize_stream(CudaStream::Decode)?;
    model.cuda.synchronize_stream(CudaStream::Prefill)?;
    for layer in 0..model.model_config.block_count as usize {
        let slot = model.kv_cache_physical_slot_for_layer_sequence(layer, sequence)?;
        ensure!(
            kv.debug_sequence_length(&model.cuda, slot)? == length,
            "invalid KV length at layer {layer} sequence {sequence}"
        );
        for position in 0..length {
            let mut keys = vec![0u16; width];
            let mut values = vec![0u16; width];
            let rc = unsafe {
                ffi_debug_read_kv_token(
                    &model.cuda,
                    kv,
                    slot,
                    position,
                    keys.as_mut_ptr().cast(),
                    values.as_mut_ptr().cast(),
                )
            };
            ensure!(
                rc == 0,
                "KV read failed at slot {slot} position {position}: {rc}"
            );
            words.extend(keys);
            words.extend(values);
        }
    }
    Ok(words)
}

/// Force the same eight real prompt IDs through ordinary and captured dense
/// TinyLlama forward. Check every vocabulary value and all valid FP16 KV words
/// after each position, then check session-owned tracked allocations after drop.
#[test]
#[ignore = "requires the pinned TinyLlama F16 checkpoint and exclusive M40"]
fn tinyllama_graph_preserves_logits_and_valid_kv() -> Result<()> {
    let path = PathBuf::from(
        std::env::var_os("M40LLM_RESEARCH_TINY_MODEL").context("TinyLlama model path")?,
    );
    let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let _sync = EnvRestore::set("M40LLM_DECODE_GRAPH_DIAG_SYNC", "0");
    let mut model = LoadedModel::from_gguf(gguf::load_gguf(&path)?, std::fs::read(&path)?, -1)?;
    let props = model.cuda.current_device_props()?;
    ensure!((props.major, props.minor) == (5, 2), "expected sm_52");
    model.validate_full_layer_decode()?;
    ensure!(
        model.model_config.block_count == 22 && model.model_config.embedding_length == 2048,
        "unexpected TinyLlama shape"
    );
    model.allocate_kv_cache_for_layer_sequences(16, 2)?;
    let prompt = [1, 529, 29989, 5205, 29989, 29958, 887, 526];
    // Populate any lazily materialized model weights before measuring session cleanup.
    {
        let _ordinary = EnvRestore::set("M40LLM_DECODE_GRAPH", "0");
        session(&model, 0, false)?.logits_for_ids(&prompt[..1], |_| {})?;
    }
    model.reset_kv_cache()?;
    let resident = CudaContext::total_device_bytes();
    let mut replayed = 0;
    let mut logit_differences = 0;
    let mut cache_differences = 0;
    {
        let mut ordinary = session(&model, 0, false)?;
        let mut captured = session(&model, 1, true)?;
        for length in 1..=prompt.len() {
            let prefix = &prompt[..length];
            let reference = {
                let _ordinary = EnvRestore::set("M40LLM_DECODE_GRAPH", "0");
                ordinary.logits_for_ids(prefix, |_| {})?
            };
            let candidate = {
                let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "1");
                captured.logits_for_ids(prefix, |_| {})?
            };
            replayed += usize::from(captured.last_forward_used_graph);
            ensure!(
                !ordinary.last_forward_used_graph
                    && reference.len() == model.model_config.vocab_size as usize
                    && candidate.len() == reference.len(),
                "invalid dispatch or vocabulary size"
            );
            ensure!(
                reference.iter().chain(&candidate).all(|v| v.is_finite()),
                "nonfinite logits"
            );
            let different_logits = reference
                .iter()
                .zip(&candidate)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let reference_kv = slot_bits(&model, 0, length as u32)?;
            let candidate_kv = slot_bits(&model, 1, length as u32)?;
            ensure!(reference_kv.len() == candidate_kv.len(), "KV shapes differ");
            let different_kv = reference_kv
                .iter()
                .zip(&candidate_kv)
                .filter(|(a, b)| a != b)
                .count();
            logit_differences += different_logits;
            cache_differences += different_kv;
            eprintln!("positions={length} replay={} logits={} differing_logits={different_logits} kv_words={} differing_kv={different_kv}",captured.last_forward_used_graph,reference.len(),reference_kv.len());
        }
    }
    let after = CudaContext::total_device_bytes();
    eprintln!("replayed={replayed} differing_logits={logit_differences} differing_kv={cache_differences} resident_before={resident} resident_after={after}");
    ensure!(
        replayed == prompt.len(),
        "ordinary fallback prevented graph qualification"
    );
    ensure!(after == resident, "session allocation ownership changed");
    ensure!(
        logit_differences == 0 && cache_differences == 0,
        "graph and ordinary state differ"
    );
    Ok(())
}
