//! M40 behavior checks for interleaved dense prompt processing.
use super::*;
use crate::cuda::{ffi_debug_read_kv_token, CudaContext, KVCache};

mod tiny_gguf {
    use crate as m40_llm;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/common/tiny_gguf.rs"
    ));
}

fn fixture(slots: u32) -> Result<AppState> {
    let (gguf, bytes) = tiny_gguf::make_attention_bias_tiny_gguf(tiny_gguf::TinyGgufConfig {
        vocab: 128,
        d_model: 128,
        head_count: 2,
        ..Default::default()
    });
    let mut model = LoadedModel::from_gguf(gguf, bytes, 0)?;
    model.allocate_kv_cache_for_layer_sequences(64, slots)?;
    let mut state = AppState::new_with_kv_config(model, KvCompressionConfig::dense_reference());
    state.decode_batching_requested = true;
    state.decode_sequence_pool = Some(crate::decode_batch::DecodeSequencePool::new(slots));
    Ok(state)
}

fn request(state: &AppState, id: u64, prompt: &str) -> Result<DecodeSchedulerRequest> {
    DecodeSchedulerRequest::new(
        state,
        id,
        GenerateRequest {
            prompt: prompt.to_string(),
            prompt_format: PromptFormat::Raw,
            max_tokens: Some(16),
            seed: Some(0),
            top_k: Some(1),
            ..Default::default()
        },
    )
}

fn cache_snapshot(state: &AppState, request: &DecodeSchedulerRequest) -> Result<Vec<[Vec<u8>; 2]>> {
    let model = &state.model;
    let cache: &KVCache = model.kv_cache.as_ref().expect("fixture cache");
    let slot =
        model.kv_cache_physical_slot_for_layer_sequence(0, request.sequence_lease.sequence_id())?;
    let length = cache.debug_sequence_length(&model.cuda, slot)?;
    assert_eq!(length as usize, request.session.processed_len());
    let width = (cache.num_heads() * cache.head_dim()) as usize * 2;
    let mut rows = Vec::new();
    for position in 0..length {
        let mut keys = vec![0; width];
        let mut values = vec![0; width];
        let rc = unsafe {
            ffi_debug_read_kv_token(
                &model.cuda,
                cache,
                slot,
                position,
                keys.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0);
        rows.push([keys, values]);
    }
    Ok(rows)
}

fn tick(state: &AppState, active: &mut VecDeque<DecodeSchedulerRequest>) {
    process_scheduler_batch_tick(state, active.drain(..).collect(), active, true);
}

fn active_request(active: &VecDeque<DecodeSchedulerRequest>, id: u64) -> &DecodeSchedulerRequest {
    active
        .iter()
        .find(|request| request.request_id == id)
        .unwrap()
}

/// An arriving prompt and a later prefill leave both existing decodes moving.
/// The final partial chunk reproduces the single-sequence sample and valid KV.
#[test]
#[ignore = "requires serial M40 hardware execution"]
fn dense_prefill_chunks_preserve_samples_and_cache() -> Result<()> {
    let baseline_bytes = CudaContext::total_device_bytes();
    {
        let state = fixture(4)?;
        let prompt = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefg";
        let mut reference = request(&state, 0, prompt)?;
        assert_eq!(reference.prompt_token_len, 33);
        reference.step()?;
        let expected_ids = reference.generated.clone();
        let expected_kv = cache_snapshot(&state, &reference)?;
        drop(reference);

        let mut first = request(&state, 1, "A")?;
        let mut second = request(&state, 2, "B")?;
        first.step()?;
        second.step()?;
        let initial_counts = [first.generated.len(), second.generated.len()];
        assert_eq!(initial_counts, [1, 1]);
        let mut active = VecDeque::from([first, second, request(&state, 3, prompt)?]);
        tick(&state, &mut active);
        let partial = active_request(&active, 3);
        assert!(partial.generated.is_empty());
        assert_eq!(cache_snapshot(&state, partial)?.len(), 16);

        active.push_back(request(&state, 4, "abcdefghijklmnop")?);
        tick(&state, &mut active);
        let partial = active_request(&active, 3);
        assert!(partial.generated.is_empty());
        assert_eq!(cache_snapshot(&state, partial)?.len(), 32);
        assert_eq!(active_request(&active, 4).session.processed_len(), 0);

        tick(&state, &mut active);
        let exact_width = active_request(&active, 4);
        assert_eq!(cache_snapshot(&state, exact_width)?.len(), 16);
        assert_eq!(exact_width.generated.len(), 1);
        assert!(active_request(&active, 3).generated.is_empty());

        tick(&state, &mut active);
        let completed_prompt = active_request(&active, 3);
        assert_eq!(completed_prompt.generated, expected_ids);
        assert_eq!(cache_snapshot(&state, completed_prompt)?, expected_kv);
        assert_eq!(active_request(&active, 1).generated.len(), 5);
        assert_eq!(active_request(&active, 2).generated.len(), 5);
    }
    assert_eq!(CudaContext::total_device_bytes(), baseline_bytes);
    Ok(())
}

/// Cancelling after a partial prompt returns its slot for a new request.
/// A shorter prompt in that slot matches a freshly processed sequence.
#[test]
#[ignore = "requires serial M40 hardware execution"]
fn cancelled_dense_prefill_slot_can_be_reused() -> Result<()> {
    let state = fixture(2)?;
    let mut first = request(&state, 1, "A")?;
    first.step()?;
    let mut abandoned = request(&state, 2, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefg")?;
    let abandoned_slot = abandoned.sequence_lease.sequence_id();
    let (sender, receiver) = tokio::sync::oneshot::channel();
    abandoned.respond_to = Some(sender);
    let mut active = VecDeque::from([first, abandoned]);
    tick(&state, &mut active);
    assert_eq!(active_request(&active, 2).session.processed_len(), 16);
    drop(receiver);
    assert!(active_request(&active, 2).is_cancelled());
    let abandoned = active.pop_front().expect("partial prefill at front");
    assert_eq!(abandoned.request_id, 2);
    drop(abandoned);

    let mut replacement = request(&state, 3, "Q")?;
    assert_eq!(replacement.sequence_lease.sequence_id(), abandoned_slot);
    replacement.step()?;
    let replacement_ids = replacement.generated.clone();
    let replacement_cache = cache_snapshot(&state, &replacement)?;
    assert_eq!(replacement_cache.len(), 1);
    drop(active);
    let mut reference = request(&state, 4, "Q")?;
    reference.step()?;
    assert_eq!(reference.generated, replacement_ids);
    assert_eq!(cache_snapshot(&state, &reference)?, replacement_cache);
    Ok(())
}
