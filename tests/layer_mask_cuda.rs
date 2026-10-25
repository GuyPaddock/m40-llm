#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/env_restore.rs"]
mod env_restore;
#[path = "common/tiny_gguf.rs"]
mod fixture;

use anyhow::Result;
use env_restore::EnvRestore;
use m40_llm::{
    cuda::{ffi_debug_read_kv_token, CudaContext, CudaStream, DeviceBuffer},
    decode_session::DecodeSession,
    evaluation::teacher_forced::compare_loaded_models,
    infer::{ForwardBatchItem, ForwardPrefillSequence, LoadedModel},
    kv_compression::{KvCompressionConfig, ScopedRuntimeConfig},
    profile::{self, OpCounters},
};
use std::ffi::c_void;

fn make_model(
    skipped: &[usize],
    head_dim: usize,
    sequences: u32,
    compression: &KvCompressionConfig,
    should_poison_skipped: bool,
) -> Result<LoadedModel> {
    let (gguf, mut weights) =
        fixture::make_qwen2_attention_bias_tiny_gguf(fixture::TinyGgufConfig {
            vocab: 256,
            d_model: head_dim,
            hidden: 128,
            head_count: 1,
            block_count: 6,
            context_length: 16,
        });
    for tensor in &gguf.tensors {
        if !tensor.name.starts_with("layers.") {
            continue;
        }
        let layer: usize = tensor.name.split('.').nth(1).unwrap().parse()?;
        let start = tensor.offset as usize;
        if tensor.name.ends_with("attention.wq.bias")
            || tensor.name.ends_with("attention.wk.bias")
            || tensor.name.ends_with("attention.wv.bias")
        {
            // Distinguish physical layer slots; identical layer KV would conceal
            // accidental renumbering after a skipped layer.
            for value in weights[start..start + head_dim * 4].chunks_exact_mut(4) {
                let layer_scale = if tensor.name.ends_with("attention.wk.bias") {
                    (layer + 1) as f32
                } else {
                    1.0
                };
                // Keep six-layer hidden magnitudes bounded so ordinary packed
                // FP32-vs-stored-FP16 rounding stays within the existing tolerance.
                let bias = f32::from_le_bytes(value.try_into().unwrap()) * layer_scale / 16.0;
                value.copy_from_slice(&bias.to_le_bytes());
            }
        }
        if should_poison_skipped
            && skipped.contains(&layer)
            && tensor.name.ends_with("attention.wq.weight")
        {
            for value in weights[start..start + head_dim * head_dim * 2].chunks_exact_mut(2) {
                value.copy_from_slice(&half::f16::NAN.to_bits().to_le_bytes());
            }
        }
    }
    let mut model = LoadedModel::from_gguf(gguf, weights, -1)?;
    model.set_diagnostic_layer_mask(skipped)?;
    if compression.mode.is_enabled() {
        model.allocate_compressed_kv_cache_for_layer_sequences(16, sequences, compression)?;
    } else {
        model.allocate_kv_cache_for_layer_sequences(16, sequences)?;
    }
    Ok(model)
}

fn buffer(model: &LoadedModel) -> Result<DeviceBuffer> {
    DeviceBuffer::new_tagged(
        &model.cuda,
        model.model_config.embedding_length as usize * 4,
        "mask_test:hidden",
    )
}

fn drain(model: &LoadedModel) -> Result<()> {
    model.cuda.synchronize_stream(CudaStream::Decode)?;
    model.cuda.synchronize_stream(CudaStream::Prefill)
}

fn read_hidden(model: &LoadedModel, output: &DeviceBuffer) -> Result<Vec<f32>> {
    drain(model)?;
    let mut values = vec![0.0; model.model_config.embedding_length as usize];
    unsafe {
        model
            .cuda
            .memcpy_d2h(values.as_mut_ptr().cast(), output.as_ptr(), output.bytes())?;
    }
    Ok(values)
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() < 2e-3,
            "hidden[{index}]: {a} vs {b}"
        );
    }
}

fn run_sequence(
    model: &LoadedModel,
    ids: &[u32],
    sequence: u32,
    should_run_manually: bool,
) -> Result<(Vec<f32>, OpCounters)> {
    let input = buffer(model)?;
    let output = buffer(model)?;
    let a = buffer(model)?;
    let b = buffer(model)?;
    let before = profile::snapshot();
    for (position, &id) in ids.iter().enumerate() {
        unsafe {
            model.load_token_embedding_to_f32(id as u64, input.as_mut_ptr())?;
            if should_run_manually {
                let layers: Vec<_> = model.layer_mask.active_layers(6)?.collect();
                let mut current = input.as_ptr();
                for (index, &layer) in layers.iter().enumerate() {
                    let next = if index + 1 == layers.len() {
                        output.as_mut_ptr()
                    } else if index % 2 == 0 {
                        a.as_mut_ptr()
                    } else {
                        b.as_mut_ptr()
                    };
                    model.forward_one_token_with_layer(
                        current,
                        layer,
                        sequence,
                        position as u32 + 1,
                        next,
                    )?;
                    current = next;
                }
            } else {
                assert_eq!(
                    model.forward_one_token_all_layers_for_sequence(
                        input.as_ptr(),
                        sequence,
                        position as u32 + 1,
                        output.as_mut_ptr()
                    )?,
                    model.active_layer_count()?
                );
            }
        }
    }
    let values = read_hidden(model, &output)?;
    Ok((
        values,
        profile::counter_delta(&before, &profile::snapshot()),
    ))
}

fn read_kv(
    model: &LoadedModel,
    layer: u32,
    sequence: u32,
    token: u32,
) -> Result<(Vec<u8>, Vec<u8>)> {
    drain(model)?;
    let cache = model.kv_cache.as_ref().unwrap();
    let slot = sequence * model.model_config.block_count + layer;
    let mut k = vec![0; model.model_config.embedding_length as usize * 2];
    let mut v = k.clone();
    let result = unsafe {
        ffi_debug_read_kv_token(
            &model.cuda,
            cache,
            slot,
            token,
            k.as_mut_ptr(),
            v.as_mut_ptr(),
        )
    };
    assert_eq!(result, 0);
    Ok((k, v))
}

/// Empty-mask full logits match the unchanged path exactly; invalid/late setup
/// leaves allocations and a previously configured model usable.
#[test]
fn test_empty_mask_logits_and_setup_validation() -> Result<()> {
    let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "0");
    let config = KvCompressionConfig::dense_reference();
    let _runtime = ScopedRuntimeConfig::new(config.clone());
    let initial = CudaContext::total_device_bytes();
    {
        let mut reference = make_model(&[], 64, 1, &config, false)?;
        let mut explicit = make_model(&[], 64, 1, &config, false)?;
        reference.layer_mask = Default::default();
        let bytes = CudaContext::total_device_bytes();
        assert!(explicit.set_diagnostic_layer_mask(&[1]).is_err());
        assert!(explicit.set_diagnostic_layer_mask(&[0]).is_err());
        assert!(!explicit.layer_mask.has_skipped_layers());
        assert_eq!(CudaContext::total_device_bytes(), bytes);
        let report = compare_loaded_models(&mut reference, &mut explicit, &[1, 2, 3, 4])?;
        assert_eq!(report.max_abs_logit_diff, 0.0);
        assert_eq!(report.reference, report.candidate);
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Poisoned skipped weights must never execute. Retained original-layer KV and
/// launch counts match explicit layer-by-layer execution across two sequences;
/// seeded skipped KV stays untouched and resident capacity is not reclaimed.
#[test]
fn test_masked_forward_matches_manual_layers_and_preserves_kv_ids() -> Result<()> {
    let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "0");
    let config = KvCompressionConfig::dense_reference();
    let _runtime = ScopedRuntimeConfig::new(config.clone());
    let initial = CudaContext::total_device_bytes();
    for skipped in [&[1][..], &[1, 2, 4][..]] {
        let actual = make_model(skipped, 64, 2, &config, true)?;
        let manual = make_model(skipped, 64, 2, &config, true)?;
        let dense = make_model(&[], 64, 2, &config, false)?;
        assert_eq!(actual.weights_len, dense.weights_len);
        assert_eq!(
            actual.kv_cache.as_ref().unwrap().actual_bytes(),
            dense.kv_cache.as_ref().unwrap().actual_bytes()
        );
        let sentinel = vec![123.0; 64];
        for sequence in 0..2 {
            for &layer in skipped {
                actual.append_kv_token_f32_from_host_for_layer(
                    layer as u32,
                    sequence,
                    0,
                    &sentinel,
                    &sentinel,
                )?;
            }
            let (a, actual_work) = run_sequence(&actual, &[3, 4, 5], sequence, false)?;
            let (b, manual_work) = run_sequence(&manual, &[3, 4, 5], sequence, true)?;
            assert_close(&a, &b);
            assert_eq!(actual_work.launches, manual_work.launches);
            assert_eq!(actual_work.cublas_calls, manual_work.cublas_calls);
            for layer in actual.layer_mask.active_layers(6)? {
                for token in 0..3 {
                    assert_eq!(
                        read_kv(&actual, layer as u32, sequence, token)?,
                        read_kv(&manual, layer as u32, sequence, token)?
                    );
                }
            }
            for &layer in skipped {
                let (k, v) = read_kv(&actual, layer as u32, sequence, 0)?;
                let expected: Vec<_> = sentinel
                    .iter()
                    .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
                    .collect();
                assert_eq!(k, expected);
                assert_eq!(v, expected);
            }
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

fn check_packed_and_batched(head_dim: usize, config: KvCompressionConfig) -> Result<()> {
    let _runtime = ScopedRuntimeConfig::new(config.clone());
    let sequential = make_model(&[1], head_dim, 2, &config, true)?;
    let packed = make_model(&[1], head_dim, 2, &config, true)?;
    let ids = [&[1, 2, 3][..], &[4, 5][..]];
    let outputs = [buffer(&packed)?, buffer(&packed)?];
    let expected = [
        run_sequence(&sequential, ids[0], 0, false)?.0,
        run_sequence(&sequential, ids[1], 1, false)?.0,
    ];
    unsafe {
        assert_eq!(
            packed.forward_prefill_all_layers_varlen_for_sequences(&[
                ForwardPrefillSequence {
                    token_ids: ids[0],
                    sequence_id: 0,
                    d_out_f32: outputs[0].as_mut_ptr()
                },
                ForwardPrefillSequence {
                    token_ids: ids[1],
                    sequence_id: 1,
                    d_out_f32: outputs[1].as_mut_ptr()
                },
            ])?,
            5
        );
    }
    for index in 0..2 {
        assert_close(&read_hidden(&packed, &outputs[index])?, &expected[index]);
    }
    let inputs = [buffer(&packed)?, buffer(&packed)?];
    let ref_input = buffer(&sequential)?;
    let ref_output = buffer(&sequential)?;
    // Initial prefixes fit the exact ring on both paths. Subsequent decode
    // crosses its boundary so old q4-V retrieval is compared under equal semantics.
    for step in 0..3 {
        let mut expected = Vec::new();
        for index in 0..2 {
            unsafe {
                sequential.load_token_embedding_to_f32(6 + step as u64, ref_input.as_mut_ptr())?;
                packed.load_token_embedding_to_f32(6 + step as u64, inputs[index].as_mut_ptr())?;
                sequential.forward_one_token_all_layers_for_sequence(
                    ref_input.as_ptr(),
                    index as u32,
                    ids[index].len() as u32 + step + 1,
                    ref_output.as_mut_ptr(),
                )?;
            }
            expected.push(read_hidden(&sequential, &ref_output)?);
        }
        unsafe {
            assert_eq!(
                packed.forward_one_token_all_layers_batched_for_sequences(&[
                    ForwardBatchItem {
                        d_x_f32: inputs[0].as_mut_ptr(),
                        sequence_id: 0,
                        seq_len: 4 + step,
                        d_out_f32: outputs[0].as_mut_ptr()
                    },
                    ForwardBatchItem {
                        d_x_f32: inputs[1].as_mut_ptr(),
                        sequence_id: 1,
                        seq_len: 3 + step,
                        d_out_f32: outputs[1].as_mut_ptr()
                    },
                ])?,
                5
            );
        }
        for index in 0..2 {
            assert_close(&read_hidden(&packed, &outputs[index])?, &expected[index]);
        }
    }
    if config.mode.is_enabled() {
        for sequence in 0..2 {
            for layer in 0..6 {
                let slot = sequence * packed.model_config.block_count + layer;
                let a = packed
                    .kv_cache
                    .as_ref()
                    .unwrap()
                    .debug_compressed_snapshot(&packed.cuda, slot)?;
                let b = sequential
                    .kv_cache
                    .as_ref()
                    .unwrap()
                    .debug_compressed_snapshot(&sequential.cuda, slot)?;
                assert_eq!(a, b, "head={head_dim} layer={layer} seq={sequence}");
                assert_eq!(
                    a.seq_len,
                    if layer == 1 {
                        0
                    } else {
                        ids[sequence as usize].len() as u32 + 3
                    }
                );
            }
        }
    } else {
        for sequence in 0..2 {
            for layer in packed.layer_mask.active_layers(6)? {
                for token in 0..ids[sequence as usize].len() as u32 + 3 {
                    assert_eq!(
                        read_kv(&packed, layer as u32, sequence, token)?,
                        read_kv(&sequential, layer as u32, sequence, token)?
                    );
                }
            }
        }
    }
    Ok(())
}

/// Mixed-length packed prefill and row-batched decode must honor the same mask
/// on head64/head128, including Qwen biases/RoPE and physical compressed state.
#[test]
fn test_masked_packed_prefill_and_batched_decode() -> Result<()> {
    let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "0");
    let initial = CudaContext::total_device_bytes();
    for head_dim in [64, 128] {
        check_packed_and_batched(head_dim, KvCompressionConfig::dense_reference())?;
        check_packed_and_batched(
            head_dim,
            KvCompressionConfig {
                recent_window: 4,
                block_size: 2,
                top_blocks: 2,
                ..KvCompressionConfig::default()
            },
        )?;
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// The device-parameter forward entrypoint must preserve mask routing, while
/// unvalidated session graph replay fails explicitly before scratch allocation.
#[test]
fn test_masked_device_params_and_graph_session_guard() -> Result<()> {
    let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "0");
    let config = KvCompressionConfig::dense_reference();
    let _runtime = ScopedRuntimeConfig::new(config.clone());
    let initial = CudaContext::total_device_bytes();
    {
        let ordinary = make_model(&[1, 2, 4], 64, 1, &config, true)?;
        let device_params = make_model(&[1, 2, 4], 64, 1, &config, true)?;
        let expected = run_sequence(&ordinary, &[3], 0, false)?.0;
        let input = buffer(&device_params)?;
        let output = buffer(&device_params)?;
        let position = DeviceBuffer::new_tagged(&device_params.cuda, 4, "mask_test:position")?;
        let length = DeviceBuffer::new_tagged(&device_params.cuda, 4, "mask_test:length")?;
        unsafe {
            device_params.cuda.memcpy_h2d(
                position.as_mut_ptr(),
                (&0u32 as *const u32).cast::<c_void>(),
                4,
            )?;
            device_params.cuda.memcpy_h2d(
                length.as_mut_ptr(),
                (&1u32 as *const u32).cast::<c_void>(),
                4,
            )?;
            device_params.load_token_embedding_to_f32(3, input.as_mut_ptr())?;
            assert_eq!(
                device_params.forward_one_token_all_layers_for_sequence_graph_params(
                    input.as_ptr(),
                    0,
                    position.as_ptr().cast(),
                    length.as_ptr().cast(),
                    output.as_mut_ptr()
                )?,
                3
            );
        }
        assert_close(&read_hidden(&device_params, &output)?, &expected);
        let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "1");
        let before = CudaContext::total_device_bytes();
        let session = DecodeSession::new_for_sequence_with_kv_config(
            &ordinary,
            0,
            config,
            64,
            true,
            "mask_test",
            "mask_test:x",
            "mask_test:out",
        );
        assert!(session.is_err());
        assert_eq!(CudaContext::total_device_bytes(), before);
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Reconstruction currently fills every physical slot. A masked session must
/// reject it before mutation/allocation and remain usable for sequential decode.
#[test]
fn test_masked_reconstruction_rejects_without_mutating_session() -> Result<()> {
    let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "0");
    let config = KvCompressionConfig {
        recent_window: 4,
        block_size: 2,
        top_blocks: 2,
        ..KvCompressionConfig::default()
    };
    let _runtime = ScopedRuntimeConfig::new(config.clone());
    let initial = CudaContext::total_device_bytes();
    {
        let model = make_model(&[1], 64, 1, &config, true)?;
        let mut session = DecodeSession::new_for_sequence_with_kv_config(
            &model,
            0,
            config,
            64,
            true,
            "mask_test",
            "mask_test:x",
            "mask_test:out",
        )?;
        let before = CudaContext::total_device_bytes();
        let error = session
            .logits_for_packed_then_compress_prefill_ids(&[1, 2, 3], |_| {})
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("masked packed-then-compress prefill is not supported"));
        assert_eq!(CudaContext::total_device_bytes(), before);
        assert_eq!(session.processed_len(), 0);
        assert!(session
            .logits_for_ids(&[1, 2, 3], |_| {})?
            .iter()
            .all(|v| v.is_finite()));
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}
