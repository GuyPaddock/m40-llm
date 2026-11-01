//! Native Q8 graph qualification using real model, stream and cache state.

#[path = "../../tests/common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "../../tests/common/env_restore.rs"]
mod env_restore;
#[path = "../../tests/common/q8_model.rs"]
#[allow(dead_code)]
mod fixture;

use super::DecodeSession;
use crate::{
    cuda::{ffi_debug_read_kv_token, CudaContext, CudaStream},
    infer::{
        execution_weights::Q8ExecutionConfig, q8_prefill::Q8PrefillBackend, LoadedModel,
        Q8AttentionProjectionBackend, Q8FfnConfig,
    },
    kv_compression::{KvCompressionConfig, ScopedRuntimeConfig},
};
use anyhow::Result;
use env_restore::EnvRestore;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GraphTestFailure {
    Capture,
    Launch,
}

fn configure() -> Vec<EnvRestore> {
    [
        ("M40LLM_ENABLE_NVCC", "1"),
        ("M40LLM_PROJECTION_BACKEND", "large-model"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_ATTN_HEAD128_KERNEL", "native"),
        ("M40LLM_FUSED_Q8_QKV", "1"),
        ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
        ("M40LLM_DECODE_GRAPH", "0"),
        ("M40LLM_DECODE_GRAPH_DIAG_SYNC", "0"),
        ("M40LLM_Q8_FFN_KERNEL", "native"),
    ]
    .into_iter()
    .map(|(key, value)| EnvRestore::set(key, value))
    .collect()
}

fn build_model() -> Result<LoadedModel> {
    let (gguf, bytes) = fixture::build_prefill_fixture(true);
    let mut model = LoadedModel::from_gguf_with_q8_runtime(
        gguf,
        bytes,
        -1,
        Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
        Q8FfnConfig::native(),
        Q8PrefillBackend::Native,
        Q8AttentionProjectionBackend::Native,
    )?;
    model.allocate_kv_cache_with_layout(32, 4, 1, 128)?;
    Ok(model)
}

fn create_session(
    model: &LoadedModel,
    sequence: u32,
    is_graph_enabled: bool,
) -> Result<DecodeSession> {
    let _graph = EnvRestore::set(
        "M40LLM_DECODE_GRAPH",
        if is_graph_enabled { "1" } else { "0" },
    );
    DecodeSession::new_for_sequence_with_kv_config(
        model,
        sequence,
        KvCompressionConfig::dense_reference(),
        256,
        true,
        "graph-qualification",
        "graph-qualification:x",
        "graph-qualification:out",
    )
}

#[derive(Debug, PartialEq, Eq)]
struct CacheSlot {
    length: u32,
    keys: Vec<u16>,
    values: Vec<u16>,
}

fn read_cache(model: &LoadedModel) -> Result<Vec<CacheSlot>> {
    model.cuda.synchronize_stream(CudaStream::Decode)?;
    model.cuda.synchronize_stream(CudaStream::Prefill)?;
    let kv = model.kv_cache.as_ref().expect("allocated dense KV");
    let mut slots = Vec::new();
    for sequence in 0..2 {
        for layer in 0..2 {
            let physical = model.kv_cache_physical_slot_for_layer_sequence(layer, sequence)?;
            let length = kv.debug_sequence_length(&model.cuda, physical)?;
            let mut slot = CacheSlot {
                length,
                keys: vec![0; length as usize * 128],
                values: vec![0; length as usize * 128],
            };
            for token in 0..length {
                let offset = token as usize * 128;
                // Both producer streams are complete. Each destination contains
                // one full head128 FP16 token, and physical addresses this cache.
                let rc = unsafe {
                    ffi_debug_read_kv_token(
                        &model.cuda,
                        kv,
                        physical,
                        token,
                        slot.keys[offset..].as_mut_ptr().cast(),
                        slot.values[offset..].as_mut_ptr().cast(),
                    )
                };
                anyhow::ensure!(rc == 0, "KV read failed: {rc}");
            }
            slots.push(slot);
        }
    }
    Ok(slots)
}

fn assert_logits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() <= 2e-3, "logit {index}: {a} != {b}");
    }
    let greedy = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0
    };
    assert_eq!(greedy(actual), greedy(expected));
}

/// Grow distinct interleaved requests and compare every token's observable
/// logits, callbacks, complete FP16 cache and layer/sequence lengths. Faults
/// reject capture after recording operations or reject replay before submission;
/// they do not model asynchronous device faults.
fn compare_sessions(failure: Option<(GraphTestFailure, usize)>) -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let _scope = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    let mut cache_differences = 0_usize;
    let mut maximum_cache_error = 0.0_f32;
    let mut maximum_logit_error = 0.0_f32;
    {
        let control = build_model()?;
        let candidate = build_model()?;
        for cycle in 0..2 {
            control.reset_kv_cache()?;
            candidate.reset_kv_cache()?;
            let mut ordinary = [
                create_session(&control, 0, false)?,
                create_session(&control, 1, false)?,
            ];
            let mut graphed = [
                create_session(&candidate, 0, true)?,
                create_session(&candidate, 1, true)?,
            ];
            let mut ids: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
            for (step, sequence) in [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
                .into_iter()
                .enumerate()
            {
                let position = ids[sequence].len();
                ids[sequence].push(((step * 17 + sequence * 31 + cycle * 7) % 300 + 1) as u32);
                if sequence == 0 {
                    if let Some((fault, fault_position)) = failure {
                        if position == fault_position {
                            graphed[sequence].graph_test_failure = Some(fault);
                        }
                    }
                }
                let mut callbacks = Vec::new();
                let expected = {
                    let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "0");
                    ordinary[sequence].logits_for_ids(&ids[sequence], |_| {})?
                };
                let actual = {
                    let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "1");
                    graphed[sequence]
                        .logits_for_ids(&ids[sequence], |logits| callbacks.push(logits.to_vec()))?
                };
                assert_eq!(callbacks.len(), 1, "token processed more than once");
                assert_eq!(callbacks[0], actual);
                assert_logits(&actual, &expected);
                maximum_logit_error = actual
                    .iter()
                    .zip(&expected)
                    .fold(maximum_logit_error, |maximum, (&a, &b)| {
                        maximum.max((a - b).abs())
                    });
                assert_eq!(graphed[sequence].processed_len, ids[sequence].len());
                let is_failed = sequence == 0 && failure.is_some_and(|(_, at)| position >= at);
                assert_eq!(graphed[sequence].graph_disabled, is_failed);
                assert_eq!(graphed[sequence].last_forward_used_graph, !is_failed);
                let expected_cache = read_cache(&control)?;
                let actual_cache = read_cache(&candidate)?;
                for (request, request_ids) in ids.iter().enumerate() {
                    for layer in 0..2 {
                        let slot = request * 2 + layer;
                        assert_eq!(actual_cache[slot].length as usize, request_ids.len());
                    }
                }
                for (slot, (actual, expected)) in
                    actual_cache.iter().zip(&expected_cache).enumerate()
                {
                    for (kind, a, b) in [
                        ("keys", &actual.keys, &expected.keys),
                        ("values", &actual.values, &expected.values),
                    ] {
                        let differences = a.iter().zip(b).filter(|(a, b)| a != b).count();
                        let maximum = a
                            .iter()
                            .zip(b)
                            .map(|(&a, &b)| {
                                (half::f16::from_bits(a).to_f32()
                                    - half::f16::from_bits(b).to_f32())
                                .abs()
                            })
                            .fold(0.0_f32, f32::max);
                        if differences > 0 {
                            eprintln!("KV mismatch cycle={cycle} step={step} slot={slot} kind={kind} count={differences} max_abs={maximum}");
                        }
                        cache_differences += differences;
                        maximum_cache_error = maximum_cache_error.max(maximum);
                    }
                }
            }
        }
    }
    ctx.synchronize_stream(CudaStream::Decode)?;
    ctx.synchronize_stream(CudaStream::Prefill)?;
    assert_eq!(
        CudaContext::total_device_bytes(),
        initial,
        "tracked allocation leak"
    );
    eprintln!("graph qualification failure={failure:?} compared_tokens=32 maximum_logit_error={maximum_logit_error} cache_differences={cache_differences} maximum_cache_error={maximum_cache_error} tracked_cleanup=pass");
    assert_eq!(cache_differences, 0, "exact KV parity gate");
    Ok(())
}

/// Native Q8 replay preserves output and KV state across positions, independent
/// sequences, graph teardown and reconstruction on the same model.
#[test]
#[ignore = "explicit serial M40 native Q8 graph qualification"]
fn test_native_q8_graph_sequences() -> Result<()> {
    compare_sessions(None)
}

/// Cancelling a real stream capture leaves exactly the warmed token's state,
/// then permanently falls back while another sequence continues replaying.
#[test]
#[ignore = "explicit serial M40 native Q8 graph qualification"]
fn test_native_q8_graph_capture_cancellation() -> Result<()> {
    compare_sessions(Some((GraphTestFailure::Capture, 0)))
}

/// Rejection before the first replay preserves the ordinary token output and
/// cache even though native warmup already ran at the same position.
#[test]
#[ignore = "explicit serial M40 native Q8 graph qualification"]
fn test_native_q8_graph_first_launch_rejection() -> Result<()> {
    compare_sessions(Some((GraphTestFailure::Launch, 0)))
}

/// Rejection after successful replays preserves previous cache entries and
/// processes the new position through ordinary decode exactly once logically.
#[test]
#[ignore = "explicit serial M40 native Q8 graph qualification"]
fn test_native_q8_graph_later_launch_rejection() -> Result<()> {
    compare_sessions(Some((GraphTestFailure::Launch, 3)))
}
