use anyhow::Result;
use m40_llm::infer::q8_prefill::Q8PrefillBackend;

#[path = "common/q8_model.rs"]
#[allow(dead_code)]
mod fixture;

/// Prefill is native by default; explicit experimental selection accepts no
/// ambiguous aliases or malformed values and has stable report names.
#[test]
fn test_q8_prefill_backend_requires_explicit_selection() -> Result<()> {
    assert_eq!(Q8PrefillBackend::default(), Q8PrefillBackend::Native);
    assert_eq!(Q8PrefillBackend::parse("native")?, Q8PrefillBackend::Native);
    assert_eq!(
        Q8PrefillBackend::parse("dequant-sgemm")?,
        Q8PrefillBackend::DequantSgemm
    );
    assert_eq!(
        serde_json::to_value(Q8PrefillBackend::DequantSgemm)?,
        "dequant-sgemm"
    );
    for value in ["", "off", "1", "sgemm", "dequant-sgemm "] {
        assert!(Q8PrefillBackend::parse(value).is_err());
    }
    Ok(())
}

/// A build without CUDA/cuBLAS must reject the requested backend before context
/// creation or device work, rather than silently using native prefill.
#[cfg(not(all(feature = "cuda", nvcc, have_cublas)))]
#[test]
fn test_q8_prefill_explicit_selection_requires_cuda_and_cublas() -> Result<()> {
    let (gguf, bytes) = fixture::build(0);
    let before = m40_llm::profile::snapshot();
    let error = m40_llm::infer::LoadedModel::from_gguf_with_q8_backends(
        gguf,
        bytes,
        -1,
        m40_llm::infer::execution_weights::Q8ExecutionConfig::disabled(),
        m40_llm::infer::Q8FfnConfig::native(),
        Q8PrefillBackend::DequantSgemm,
    )
    .err()
    .expect("unsupported backend must fail");
    assert!(error
        .to_string()
        .contains("M40LLM_Q8_PREFILL_BACKEND=native"));
    assert_eq!(
        m40_llm::profile::counter_delta(&before, &m40_llm::profile::snapshot()),
        Default::default()
    );
    Ok(())
}

#[cfg(all(feature = "cuda", nvcc, have_cublas))]
#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[cfg(all(feature = "cuda", nvcc, have_cublas))]
#[path = "common/env_restore.rs"]
mod env_restore;

#[cfg(all(feature = "cuda", nvcc, have_cublas))]
mod hardware {
    use super::*;
    use crate::env_restore::EnvRestore;
    use half::f16;
    use m40_llm::{
        cuda::{ffi_debug_read_kv_token, CudaContext, DeviceBuffer, ROPE_LAYOUT_NEOX},
        decode_session::DecodeSession,
        infer::{
            execution_weights::Q8ExecutionConfig, ForwardPrefillSequence, LoadedModel, Q8FfnConfig,
        },
        kv_compression::KvCompressionConfig,
        profile,
    };

    fn configure() -> Vec<EnvRestore> {
        [
            ("M40LLM_ENABLE_NVCC", "1"),
            ("M40LLM_Q8_EXECUTION_WEIGHTS", "off"),
            ("M40LLM_Q8_FFN_KERNEL", "native"),
            ("M40LLM_Q8_PREFILL_BACKEND", "native"),
            ("M40LLM_Q8_DECODE_STREAM", "decode"),
            ("M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK", "4"),
            ("M40LLM_Q8_DECODE_COL_TILE", "4"),
            ("M40LLM_PROJECTION_BACKEND", "large-model"),
            ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
            ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
            ("M40LLM_FUSED_Q8_QKV", "1"),
            ("M40LLM_PREFILL_CHUNK_SIZE", "0"),
            ("M40LLM_DECODE_GRAPH", "0"),
        ]
        .into_iter()
        .map(|(k, v)| EnvRestore::set(k, v))
        .collect()
    }

    fn model(is_qwen: bool, backend: Q8PrefillBackend) -> Result<LoadedModel> {
        let (gguf, bytes) = fixture::build_prefill_fixture(is_qwen);
        let mut model = LoadedModel::from_gguf_with_q8_backends(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
            Q8FfnConfig::native(),
            backend,
        )?;
        model.allocate_kv_cache_for_layers(32)?;
        if is_qwen {
            assert_eq!(model.model_config.rope_layout_code(), ROPE_LAYOUT_NEOX);
            assert_eq!(model.model_config.attention_head_count, 2);
            assert_eq!(model.model_config.attention_head_count_kv, 1);
        }
        Ok(model)
    }

    fn session(model: &LoadedModel) -> Result<DecodeSession> {
        DecodeSession::new_for_sequence_with_kv_config(
            model,
            0,
            KvCompressionConfig::dense_reference(),
            model.model_config.embedding_length as usize,
            true,
            "q8-prefill-runtime",
            "q8-prefill:session-input",
            "q8-prefill:session-output",
        )
    }

    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-3,
                "index={index} actual={a} expected={b}"
            );
        }
    }

    fn read_hidden(model: &LoadedModel, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
        let mut values = vec![0.0; buffer.bytes() / 4];
        unsafe {
            model
                .cuda
                .memcpy_d2h(values.as_mut_ptr().cast(), buffer.as_ptr(), buffer.bytes())?;
        }
        Ok(values)
    }

    fn compare_kv(a: &LoadedModel, b: &LoadedModel, length: u32) -> Result<()> {
        let kv = a.kv_cache.as_ref().unwrap();
        let elements = (kv.num_heads() * kv.head_dim()) as usize;
        for layer in 0..2 {
            for model in [a, b] {
                assert_eq!(
                    model
                        .kv_cache
                        .as_ref()
                        .unwrap()
                        .debug_sequence_length(&model.cuda, layer)?,
                    length
                );
            }
            for position in 0..length {
                let mut values = Vec::new();
                for model in [a, b] {
                    let mut key = vec![0u8; elements * 2];
                    let mut value = key.clone();
                    assert_eq!(
                        unsafe {
                            ffi_debug_read_kv_token(
                                &model.cuda,
                                model.kv_cache.as_ref().unwrap(),
                                layer,
                                position,
                                key.as_mut_ptr(),
                                value.as_mut_ptr(),
                            )
                        },
                        0
                    );
                    let decoded: Vec<_> = key
                        .chunks_exact(2)
                        .chain(value.chunks_exact(2))
                        .map(|pair| f16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32())
                        .collect();
                    values.push(decoded);
                }
                close(&values[0], &values[1]);
            }
        }
        Ok(())
    }

    fn packed(model: &LoadedModel, ids: &[u32], output: &DeviceBuffer) -> Result<()> {
        unsafe {
            model.forward_prefill_all_layers_varlen_for_sequences(&[ForwardPrefillSequence {
                token_ids: ids,
                sequence_id: 0,
                d_out_f32: output.as_mut_ptr(),
            }])?;
        }
        Ok(())
    }

    /// Real packed projections reuse exactly one temporary buffer across seven
    /// projections/layer. Llama head64 and Qwen head128 GQA with bias/split-half
    /// RoPE preserve sequential/packed logits and absolute KV positions/lengths.
    /// Subsequent one-row decode keeps the original dispatch and warm allocations.
    #[test]
    #[ignore = "requires M40/cuBLAS; run with --include-ignored --test-threads=1"]
    fn test_q8_prefill_sgemm_preserves_state_reuses_scratch_and_keeps_decode() -> Result<()> {
        let _lock = crate::cuda_test_scope::lock_cuda();
        let _env = configure();
        let _hardware = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        for is_qwen in [false, true] {
            let sequential = model(is_qwen, Q8PrefillBackend::Native)?;
            let reference = model(is_qwen, Q8PrefillBackend::Native)?;
            let candidate = model(is_qwen, Q8PrefillBackend::DequantSgemm)?;
            let before_cache = serde_json::to_value(candidate.q8_execution_stats())?;
            let width = candidate.model_config.embedding_length as usize;
            let capacity = width * candidate.model_config.feed_forward_length as usize * 4;
            let reference_out = DeviceBuffer::new_tagged(
                &reference.cuda,
                width * 4,
                "q8-prefill:reference-hidden",
            )?;
            let candidate_out = DeviceBuffer::new_tagged(
                &candidate.cuda,
                width * 4,
                "q8-prefill:candidate-hidden",
            )?;
            let ids = [2, 3, 4, 7, 8];
            let before = profile::snapshot();
            packed(&reference, &ids[..4], &reference_out)?;
            let baseline = profile::counter_delta(&before, &profile::snapshot());
            let before = profile::snapshot();
            packed(&candidate, &ids[..4], &candidate_out)?;
            let after = profile::snapshot();
            let delta = profile::counter_delta(&before, &after);
            assert_eq!(
                after
                    .by_op
                    .get("dequantize_q8_0_to_f32_colmajor")
                    .unwrap()
                    .launches
                    - before
                        .by_op
                        .get("dequantize_q8_0_to_f32_colmajor")
                        .map_or(0, |v| v.launches),
                14
            );
            assert_eq!(delta.cublas_calls, 14);
            assert_eq!(delta.device_allocs, baseline.device_allocs + 1);
            close(
                &read_hidden(&candidate, &candidate_out)?,
                &read_hidden(&reference, &reference_out)?,
            );
            compare_kv(&reference, &candidate, 4)?;
            let scratch = candidate
                .forward_workspace
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .ptrs()
                .q8_prefill_weight
                .unwrap();
            assert_eq!(scratch.bytes, capacity);
            let resident = CudaContext::total_device_bytes();
            let before = profile::snapshot();
            packed(&reference, &ids[..4], &reference_out)?;
            let baseline = profile::counter_delta(&before, &profile::snapshot());
            let before = profile::snapshot();
            packed(&candidate, &ids[..4], &candidate_out)?;
            let delta = profile::counter_delta(&before, &profile::snapshot());
            assert_eq!(delta.device_allocs, baseline.device_allocs);
            assert_eq!(delta.device_frees, baseline.device_frees);
            assert_eq!(CudaContext::total_device_bytes(), resident);
            let after = candidate
                .forward_workspace
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .ptrs()
                .q8_prefill_weight
                .unwrap();
            assert_eq!(
                (after.pointer, after.bytes),
                (scratch.pointer, scratch.bytes)
            );
            let mut seq = session(&sequential)?;
            let mut native = session(&reference)?;
            let mut selected = session(&candidate)?;
            native.mark_processed_through(4);
            selected.mark_processed_through(4);
            let expected = seq.logits_for_ids(&ids, |_| {})?;
            let packed_logits = native.logits_for_ids(&ids, |_| {})?;
            let before = profile::snapshot();
            let selected_logits = selected.logits_for_ids(&ids, |_| {})?;
            let after = profile::snapshot();
            assert_eq!(
                after
                    .by_op
                    .get("dequantize_q8_0_to_f32_colmajor")
                    .map_or(0, |v| v.launches),
                before
                    .by_op
                    .get("dequantize_q8_0_to_f32_colmajor")
                    .map_or(0, |v| v.launches)
            );
            close(&selected_logits, &expected);
            close(&selected_logits, &packed_logits);
            compare_kv(&sequential, &candidate, 5)?;
            compare_kv(&reference, &candidate, 5)?;
            assert!(candidate
                .forward_workspace
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .ptrs()
                .q8_prefill_weight
                .is_none());
            let before = profile::snapshot();
            let next = selected.logits_for_ids(&[2, 3, 4, 7, 8, 9], |_| {})?;
            let delta = profile::counter_delta(&before, &profile::snapshot());
            assert_eq!(delta.device_allocs + delta.device_frees, 0);
            close(&next, &seq.logits_for_ids(&[2, 3, 4, 7, 8, 9], |_| {})?);
            assert_eq!(
                serde_json::to_value(candidate.q8_execution_stats())?,
                before_cache
            );
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }

    /// Explicit packed prefill reaches SGEMM even with GPU-greedy sampling;
    /// prefix construction does not substitute a different vocabulary path.
    /// Generation reports actual temporary capacity and never silently ignores
    /// an experimental request whose prefix bound is disabled.
    #[test]
    #[ignore = "requires M40/cuBLAS; run with --include-ignored --test-threads=1"]
    fn test_q8_prefill_generation_honors_bound_for_cpu_and_gpu_greedy() -> Result<()> {
        let _lock = crate::cuda_test_scope::lock_cuda();
        let _env = configure();
        let _hardware = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        {
            let (mut gguf, bytes) = fixture::build_prefill_fixture(false);
            // This fixture has no vocabulary strings: explicitly use the byte
            // tokenizer for the ASCII prompt, not a fabricated Qwen tokenizer.
            gguf.metadata.remove("tokenizer.ggml.model");
            let mut candidate = LoadedModel::from_gguf_with_q8_backends(
                gguf,
                bytes,
                -1,
                Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
                Q8FfnConfig::native(),
                Q8PrefillBackend::DequantSgemm,
            )?;
            candidate.allocate_kv_cache_for_layers(32)?;
            let options = m40_llm::generate::GenerateOptions {
                prompt: "ABC".to_string(),
                max_tokens: Some(2),
                top_k: Some(1),
                seed: Some(0),
                kv_compression: KvCompressionConfig::dense_reference(),
                prompt_format: m40_llm::generate::PromptFormat::Raw,
                ..Default::default()
            };
            let _bound = EnvRestore::set("M40LLM_PREFILL_CHUNK_SIZE", "8");
            let _head = EnvRestore::set("M40LLM_Q8_LM_HEAD_ARGMAX", "1");
            let mut expected = None;
            for greedy in ["0", "1"] {
                let _greedy = EnvRestore::set("M40LLM_CUDA_GREEDY_ARGMAX", greedy);
                let before = profile::snapshot();
                let generated = m40_llm::generate::generate_text(&candidate, options.clone())?;
                let after = profile::snapshot();
                assert_eq!(generated.prefill_mode, "packed-prefix-dequant-sgemm");
                assert_eq!(generated.prompt_token_len, 3);
                assert_eq!(generated.q8_prefill_temporary_weight_bytes, Some(32768));
                assert_eq!(
                    after
                        .by_op
                        .get("dequantize_q8_0_to_f32_colmajor")
                        .unwrap()
                        .launches
                        - before
                            .by_op
                            .get("dequantize_q8_0_to_f32_colmajor")
                            .map_or(0, |v| v.launches),
                    14
                );
                if let Some(expected) = &expected {
                    assert_eq!(&generated.token_ids, expected);
                } else {
                    expected = Some(generated.token_ids);
                }
            }
            let _disabled = EnvRestore::set("M40LLM_PREFILL_CHUNK_SIZE", "0");
            let error = m40_llm::generate::generate_text(&candidate, options)
                .expect_err("explicit SGEMM request without packed prefix must fail");
            assert!(format!("{error:#}").contains("M40LLM_Q8_PREFILL_BACKEND=native"));
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }

    /// Load-time selection is model-local and unsupported model, runtime-disable,
    /// graph and multi-request prefill configurations fail without device work.
    #[test]
    #[ignore = "requires M40/cuBLAS; run with --include-ignored --test-threads=1"]
    fn test_q8_prefill_selection_and_unsupported_paths() -> Result<()> {
        let _lock = crate::cuda_test_scope::lock_cuda();
        let _env = configure();
        let _hardware = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        {
            let _selected = EnvRestore::set("M40LLM_Q8_PREFILL_BACKEND", "dequant-sgemm");
            let (gguf, bytes) = fixture::build_prefill_fixture(false);
            let selected = LoadedModel::from_gguf(gguf, bytes, -1)?;
            assert_eq!(selected.q8_prefill, Q8PrefillBackend::DequantSgemm);
            let (gguf, bytes) = fixture::build(0);
            let native = LoadedModel::from_gguf_with_q8_execution(
                gguf,
                bytes,
                -1,
                Q8ExecutionConfig::disabled(),
            )?;
            assert!(native.q8_prefill.is_native());
            let _changed = EnvRestore::set("M40LLM_Q8_PREFILL_BACKEND", "native");
            assert_eq!(selected.q8_prefill, Q8PrefillBackend::DequantSgemm);
        }
        let (gguf, bytes) = fixture::build_f16();
        let error = LoadedModel::from_gguf_with_q8_backends(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::disabled(),
            Q8FfnConfig::native(),
            Q8PrefillBackend::DequantSgemm,
        )
        .err()
        .expect("F16 projections must fail");
        assert!(error
            .to_string()
            .contains("M40LLM_Q8_PREFILL_BACKEND=native"));
        assert_eq!(CudaContext::total_device_bytes(), initial);
        {
            let candidate = model(false, Q8PrefillBackend::DequantSgemm)?;
            let before = profile::snapshot();
            assert!(
                unsafe { candidate.forward_prefill_all_layers_varlen_for_sequences(&[]) }.is_err()
            );
            {
                let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "1");
                assert!(session(&candidate).is_err());
            }
            {
                let _disabled = EnvRestore::set("M40LLM_ENABLE_NVCC", "0");
                let (gguf, bytes) = fixture::build(0);
                assert!(LoadedModel::from_gguf_with_q8_backends(
                    gguf,
                    bytes,
                    -1,
                    Q8ExecutionConfig::disabled(),
                    Q8FfnConfig::native(),
                    Q8PrefillBackend::DequantSgemm
                )
                .is_err());
            }
            assert_eq!(
                profile::counter_delta(&before, &profile::snapshot()),
                Default::default()
            );
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }
}
