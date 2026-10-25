use anyhow::Result;
use m40_llm::infer::{Q8FfnConfig, Q8FfnKernel};

#[path = "common/q8_model.rs"]
#[allow(dead_code)]
mod fixture;

/// Native is the default; the environment names choose only measured kernel
/// families. Typed constructors retain both qualified geometries for testing.
#[test]
fn test_q8_ffn_config_selects_only_screened_candidates() -> Result<()> {
    assert_eq!(Q8FfnConfig::default(), Q8FfnConfig::native());
    assert_eq!(Q8FfnConfig::parse("native")?.threads(), None);
    assert_eq!(
        Q8FfnConfig::parse("packed-f32")?,
        Q8FfnConfig::packed_f32(256)?
    );
    assert_eq!(Q8FfnConfig::parse("q8-1")?, Q8FfnConfig::q8_1(128)?);
    assert_eq!(
        Q8FfnConfig::parse("split-f32x4-128")?,
        Q8FfnConfig::split_f32x4()
    );
    assert_eq!(Q8FfnConfig::split_f32x4().threads(), Some(128));
    assert_eq!(
        serde_json::to_value(Q8FfnConfig::split_f32x4())?["kernel"],
        "split-f32x4-128"
    );
    for threads in [128, 256] {
        assert_eq!(
            Q8FfnConfig::packed_f32(threads)?.kernel(),
            Q8FfnKernel::PackedF32
        );
        assert_eq!(Q8FfnConfig::q8_1(threads)?.threads(), Some(threads));
    }
    for value in ["", "off", "1", "packed", "scalar-shared256", "q8-1 "] {
        assert!(Q8FfnConfig::parse(value).is_err());
    }
    for threads in [0, 32, 64, 512] {
        assert!(Q8FfnConfig::packed_f32(threads).is_err());
        assert!(Q8FfnConfig::q8_1(threads).is_err());
    }
    assert_eq!(
        serde_json::to_value(Q8FfnConfig::parse("q8-1")?)?["kernel"],
        "q8-1"
    );
    Ok(())
}

/// Only Q8_1 reserves activation storage, using its checked 36-byte block
/// contract. Bad dimensions cannot produce a misleading memory estimate.
#[test]
fn test_q8_ffn_config_accounts_for_quantized_storage() -> Result<()> {
    assert_eq!(Q8FfnConfig::native().activation_bytes(2048)?, 0);
    assert_eq!(Q8FfnConfig::packed_f32(128)?.activation_bytes(2048)?, 0);
    assert_eq!(Q8FfnConfig::split_f32x4().activation_bytes(2048)?, 0);
    assert!(!Q8FfnConfig::split_f32x4().has_quantization());
    let q = Q8FfnConfig::q8_1(128)?;
    assert!(q.has_quantization());
    assert_eq!(q.activation_bytes(64)?, 72);
    assert_eq!(q.activation_bytes(2048)?, 2304);
    for width in [0, 31, 33, usize::MAX] {
        assert!(q.activation_bytes(width).is_err());
    }
    Ok(())
}

/// Explicit experiments must fail before context/device work in non-CUDA or
/// NVCC-disabled builds instead of silently using the native implementation.
#[cfg(not(all(feature = "cuda", nvcc)))]
#[test]
fn test_q8_ffn_explicit_request_requires_compiled_cuda() -> Result<()> {
    for config in [
        Q8FfnConfig::packed_f32(256)?,
        Q8FfnConfig::q8_1(128)?,
        Q8FfnConfig::split_f32x4(),
    ] {
        let (gguf, bytes) = fixture::build(0);
        let before = m40_llm::profile::snapshot();
        let error = m40_llm::infer::LoadedModel::from_gguf_with_q8_backends(
            gguf,
            bytes,
            -1,
            m40_llm::infer::execution_weights::Q8ExecutionConfig::disabled(),
            config,
            m40_llm::infer::q8_prefill::Q8PrefillBackend::Native,
        )
        .err()
        .expect("unsupported request must fail");
        assert!(error.to_string().contains("M40LLM_Q8_FFN_KERNEL=native"));
        assert_eq!(
            m40_llm::profile::counter_delta(&before, &m40_llm::profile::snapshot()),
            Default::default()
        );
    }
    Ok(())
}

#[cfg(all(feature = "cuda", nvcc))]
#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[cfg(all(feature = "cuda", nvcc))]
#[path = "common/env_restore.rs"]
mod env_restore;

#[cfg(all(feature = "cuda", nvcc))]
mod hardware {
    use super::*;
    use crate::env_restore::EnvRestore;
    use m40_llm::{
        cuda::{ffi_debug_read_kv_token, CudaContext, CudaStream, DeviceBuffer},
        decode_session::DecodeSession,
        infer::{
            execution_weights::Q8ExecutionConfig,
            q8_activation::{quantize_reference, Q8ActivationShape},
            LoadedModel, Q8FfnScratch,
        },
        kv_compression::KvCompressionConfig,
        profile,
    };

    fn configure() -> Vec<EnvRestore> {
        [
            ("M40LLM_ENABLE_NVCC", "1"),
            ("M40LLM_Q8_EXECUTION_WEIGHTS", "off"),
            ("M40LLM_Q8_FFN_KERNEL", "native"),
            ("M40LLM_Q8_DECODE_STREAM", "decode"),
            ("M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK", "4"),
            ("M40LLM_Q8_DECODE_COL_TILE", "4"),
            ("M40LLM_PROJECTION_BACKEND", "large-model"),
            ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
            ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
            ("M40LLM_FUSED_Q8_QKV", "1"),
            ("M40LLM_PREFILL_CHUNK_SIZE", "0"),
            ("M40LLM_DECODE_GRAPH", "0"),
            ("M40LLM_F16_DECODE_KERNEL", "0"),
        ]
        .into_iter()
        .map(|(k, v)| EnvRestore::set(k, v))
        .collect()
    }

    fn model(config: Q8FfnConfig) -> Result<LoadedModel> {
        let (gguf, bytes) = fixture::build_with_layers(0, 2);
        LoadedModel::from_gguf_with_q8_backends(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::parse(
                "split",
                Some("6144"),
                Some(if config.kernel() == Q8FfnKernel::SplitF32x4 {
                    "gate-up,down,vocabulary"
                } else {
                    "down,vocabulary"
                }),
            )?,
            config,
            m40_llm::infer::q8_prefill::Q8PrefillBackend::Native,
        )
    }

    fn session(model: &LoadedModel) -> Result<DecodeSession> {
        DecodeSession::new_for_sequence_with_kv_config(
            model,
            0,
            KvCompressionConfig::dense_reference(),
            64,
            true,
            "q8-ffn-runtime",
            "q8-ffn-runtime:input",
            "q8-ffn-runtime:output",
        )
    }

    /// A no-logits prefix must finish reading its input before the next host
    /// embedding upload reuses that buffer, even with queued nonblocking work.
    /// One-call prefill must match independently completed token steps and KV.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_greedy_prefix_preserves_input_under_decode_backlog() -> Result<()> {
        let _scope = crate::cuda_test_scope::lock_cuda();
        let _env = configure();
        let _device = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        {
            let (gguf, weights) = fixture::build_prefill_fixture(true);
            let mut model = LoadedModel::from_gguf_with_q8_backends(
                gguf,
                weights,
                -1,
                Q8ExecutionConfig::disabled(),
                Q8FfnConfig::native(),
                m40_llm::infer::q8_prefill::Q8PrefillBackend::Native,
            )?;
            model.allocate_kv_cache_for_layers(16)?;
            let create_session = || {
                DecodeSession::new_for_sequence_with_kv_config(
                    &model,
                    0,
                    KvCompressionConfig::dense_reference(),
                    256,
                    true,
                    "prefix-reuse",
                    "prefix-reuse:in",
                    "prefix-reuse:out",
                )
            };
            let ids = [65, 66, 67, 68];
            let mut reference = create_session()?;
            let mut expected_token = 0;
            for length in 1..=ids.len() {
                expected_token = reference.greedy_token_for_ids(&ids[..length])?;
            }
            let read_hidden = |session: &DecodeSession| -> Result<Vec<f32>> {
                let mut values = vec![0.0f32; 256];
                unsafe {
                    model.cuda.memcpy_d2h(
                        values.as_mut_ptr().cast(),
                        session.d_out_ptr()?.cast_const(),
                        values.len() * 4,
                    )?;
                }
                Ok(values)
            };
            let read_keys = || -> Result<Vec<u8>> {
                let cache = model.kv_cache.as_ref().unwrap();
                let mut result = Vec::new();
                for layer in 0..2 {
                    let slot = model.kv_cache_physical_slot_for_layer_sequence(layer, 0)?;
                    for position in 0..ids.len() {
                        let mut key = vec![0u8; 256];
                        let mut value = key.clone();
                        let rc = unsafe {
                            ffi_debug_read_kv_token(
                                &model.cuda,
                                cache,
                                slot,
                                position as u32,
                                key.as_mut_ptr(),
                                value.as_mut_ptr(),
                            )
                        };
                        anyhow::ensure!(rc == 0, "KV read failed: {rc}");
                        result.extend(key);
                        result.extend(value);
                    }
                }
                Ok(result)
            };
            let expected_hidden = read_hidden(&reference)?;
            let expected_kv = read_keys()?;
            drop(reference);
            model.reset_kv_cache()?;
            let mut actual = create_session()?;
            let elements = 1 << 20;
            let host = vec![1.0f32; elements];
            let a = DeviceBuffer::new_tagged(&model.cuda, elements * 4, "prefix-reuse:delay-a")?;
            let b = DeviceBuffer::new_tagged(&model.cuda, elements * 4, "prefix-reuse:delay-b")?;
            let out =
                DeviceBuffer::new_tagged(&model.cuda, elements * 4, "prefix-reuse:delay-out")?;
            unsafe {
                model
                    .cuda
                    .memcpy_h2d(a.as_mut_ptr(), host.as_ptr().cast(), elements * 4)?;
                model
                    .cuda
                    .memcpy_h2d(b.as_mut_ptr(), host.as_ptr().cast(), elements * 4)?;
                for _ in 0..256 {
                    model.cuda.residual_add_f32_async(
                        a.as_ptr(),
                        b.as_ptr(),
                        out.as_mut_ptr(),
                        elements,
                    )?;
                }
            }
            assert_eq!(actual.greedy_token_for_ids(&ids)?, expected_token);
            let actual_hidden = read_hidden(&actual)?;
            let maximum = actual_hidden
                .iter()
                .zip(&expected_hidden)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                maximum <= 1e-5,
                "prefix hidden drift under backlog: {maximum}"
            );
            assert_eq!(
                read_keys()?,
                expected_kv,
                "prefix physical KV differs under backlog"
            );
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }

    fn compare_kv(reference: &LoadedModel, candidate: &LoadedModel, tokens: u32) -> Result<()> {
        for model in [reference, candidate] {
            model.cuda.synchronize_stream(CudaStream::Decode)?;
            model.cuda.synchronize_stream(CudaStream::Prefill)?;
        }
        for layer in 0..2 {
            for position in 0..tokens {
                let mut payloads = Vec::new();
                for model in [reference, candidate] {
                    let bytes = model.model_config.attention_head_count_kv as usize
                        * model.model_config.attention_key_length as usize
                        * 2;
                    let mut k = vec![0u8; bytes];
                    let mut v = vec![0u8; bytes];
                    assert_eq!(
                        unsafe {
                            ffi_debug_read_kv_token(
                                &model.cuda,
                                model.kv_cache.as_ref().unwrap(),
                                layer,
                                position,
                                k.as_mut_ptr(),
                                v.as_mut_ptr(),
                            )
                        },
                        0
                    );
                    payloads.push((k, v));
                }
                assert_eq!(
                    payloads[0], payloads[1],
                    "layer={layer} position={position}"
                );
            }
        }
        Ok(())
    }

    fn direct(model: &LoadedModel) -> Result<Vec<f32>> {
        let mut input = fixture::projection::make_input(1, 64);
        let d_input = DeviceBuffer::new_tagged(&model.cuda, 256, "q8-ffn-runtime:direct-input")?;
        let d_output =
            DeviceBuffer::new_tagged(&model.cuda, 128 * 4, "q8-ffn-runtime:direct-output")?;
        let scratch = if model.q8_ffn.has_quantization() {
            Some(DeviceBuffer::new_tagged(
                &model.cuda,
                model.q8_ffn.activation_bytes(64)?,
                "q8-ffn-runtime:direct-quantized",
            )?)
        } else {
            None
        };
        unsafe {
            model
                .cuda
                .memcpy_h2d(d_input.as_mut_ptr(), input.as_ptr().cast(), 256)?;
        }
        // A queued decode producer must precede the experimental consumer even
        // when the legacy projection scheduling flag requests prefill.
        unsafe {
            model.cuda.residual_add_f32_async(
                d_input.as_ptr(),
                d_input.as_ptr(),
                d_input.as_mut_ptr(),
                64,
            )?;
        }
        input.iter_mut().for_each(|v| *v += *v);
        let mapped = model.map_standard_layer(0)?;
        let before = profile::snapshot();
        assert!(unsafe {
            model.try_mlp_gate_up_swiglu_f32xf16_gguf_decode_async(
                d_input.as_ptr(),
                1,
                64,
                mapped.w_gate.dptr,
                mapped.w_up.dptr,
                128,
                d_output.as_mut_ptr(),
                scratch.as_ref().map(|b| Q8FfnScratch {
                    pointer: b.as_mut_ptr(),
                    bytes: b.bytes(),
                }),
            )?
        });
        let delta = profile::counter_delta(&before, &profile::snapshot());
        assert_eq!(
            delta.launches,
            if model.q8_ffn.has_quantization() {
                2
            } else {
                1
            }
        );
        assert_eq!(
            delta.device_allocs
                + delta.device_frees
                + delta.h2d_copies
                + delta.d2h_copies
                + delta.stream_syncs,
            0
        );
        model.cuda.synchronize_stream(CudaStream::Decode)?;
        let reference_input = if model.q8_ffn.has_quantization() {
            quantize_reference(&input, Q8ActivationShape::new(1, 64, 64)?)?
                .iter()
                .flat_map(|b| b.reconstruct())
                .collect()
        } else {
            input
        };
        let bytes = |view: &m40_llm::infer::DeviceTensorView| {
            &model.host_weights[view.byte_offset as usize..view.byte_offset as usize + view.nbytes]
        };
        let mut actual = vec![0.0f32; 128];
        unsafe {
            model
                .cuda
                .memcpy_d2h(actual.as_mut_ptr().cast(), d_output.as_ptr(), 512)?;
        }
        for (col, &a) in actual.iter().enumerate() {
            let g = fixture::projection::project_cpu(
                &reference_input,
                bytes(&mapped.w_gate),
                0,
                col,
                64,
            );
            let u =
                fixture::projection::project_cpu(&reference_input, bytes(&mapped.w_up), 0, col, 64);
            let expected = g / (1.0 + (-g).exp()) * u;
            assert!(
                a.is_finite() && (f64::from(a) - expected).abs() < 1e-3 + 2e-5 * expected.abs()
            );
        }
        Ok(actual)
    }

    /// The environment is captured once, explicit native references ignore it,
    /// and selected dispatch survives later flag changes without new GPU work.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_q8_ffn_selection_is_model_local_not_per_token_environment() -> Result<()> {
        let _lock = crate::cuda_test_scope::lock_cuda();
        let _env = configure();
        let _hardware = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        {
            let _selected = EnvRestore::set("M40LLM_Q8_FFN_KERNEL", "packed-f32");
            let (gguf, bytes) = fixture::build(0);
            let selected = LoadedModel::from_gguf(gguf, bytes, -1)?;
            assert_eq!(selected.q8_ffn, Q8FfnConfig::packed_f32(256)?);
            let _changed = EnvRestore::set("M40LLM_Q8_FFN_KERNEL", "q8-1");
            let (gguf, bytes) = fixture::build(0);
            let reference = LoadedModel::from_gguf_with_q8_execution(
                gguf,
                bytes,
                -1,
                Q8ExecutionConfig::disabled(),
            )?;
            assert_eq!(reference.q8_ffn, Q8FfnConfig::native());
            let _legacy = EnvRestore::set("M40LLM_FUSED_Q8_MLP_SWIGLU", "0");
            let _stream = EnvRestore::set("M40LLM_Q8_DECODE_STREAM", "prefill");
            direct(&selected)?;
            assert_eq!(selected.q8_ffn, Q8FfnConfig::packed_f32(256)?);
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }

    /// Both screened FP32/Q8_1 geometries reach configured dispatch with the
    /// prepared down/vocabulary cache fixed. Rejected model types clean up all
    /// uploads rather than returning a misleading native fallback.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_q8_ffn_runtime_dispatch_preserves_cpu_reference_and_cache() -> Result<()> {
        let _lock = crate::cuda_test_scope::lock_cuda();
        let _env = configure();
        let _hardware = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        for config in [
            Q8FfnConfig::packed_f32(128)?,
            Q8FfnConfig::packed_f32(256)?,
            Q8FfnConfig::q8_1(128)?,
            Q8FfnConfig::q8_1(256)?,
            Q8FfnConfig::split_f32x4(),
        ] {
            let model = model(config)?;
            let before = serde_json::to_value(model.q8_execution_stats())?;
            direct(&model)?;
            assert_eq!(before, serde_json::to_value(model.q8_execution_stats())?);
        }
        for config in [Q8FfnConfig::packed_f32(256)?, Q8FfnConfig::q8_1(128)?] {
            let (gguf, bytes) = fixture::build_f16();
            let error = LoadedModel::from_gguf_with_q8_backends(
                gguf,
                bytes,
                -1,
                Q8ExecutionConfig::disabled(),
                config,
                m40_llm::infer::q8_prefill::Q8PrefillBackend::Native,
            )
            .err()
            .expect("unsupported F16 must fail");
            assert!(error.to_string().contains("M40LLM_Q8_FFN_KERNEL=native"));
            assert_eq!(CudaContext::total_device_bytes(), initial);
        }
        Ok(())
    }

    /// Full-layer session steps reuse one owned Q8_1 buffer, preserve native
    /// logits/KV bits for packed256, and keep both producer-stream schedules correct.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_q8_ffn_full_forward_reuses_workspace_and_keeps_native_bits() -> Result<()> {
        let _lock = crate::cuda_test_scope::lock_cuda();
        let _env = configure();
        let _hardware = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        for stream in ["decode", "prefill"] {
            let _stream = EnvRestore::set("M40LLM_Q8_DECODE_STREAM", stream);
            let mut reference = model(Q8FfnConfig::native())?;
            reference.allocate_kv_cache_for_layers(16)?;
            let mut sr = session(&reference)?;
            let prefixes = [vec![2, 3, 4], vec![2, 3, 4, 7], vec![2, 3, 4, 7, 8]];
            let expected: Vec<_> = prefixes
                .iter()
                .map(|ids| sr.logits_for_ids(ids, |_| {}))
                .collect::<Result<_>>()?;
            // This flag does not move Q8 Q/K/V/down onto decode. Experimental
            // FFN must not let it erase actual prefill producer dependencies.
            let _unrelated_kernel = EnvRestore::set("M40LLM_F16_DECODE_KERNEL", "1");
            let _unrelated_stream = EnvRestore::set("M40LLM_F16_DECODE_STREAM", "decode");
            for config in [
                Q8FfnConfig::packed_f32(256)?,
                Q8FfnConfig::q8_1(128)?,
                Q8FfnConfig::q8_1(256)?,
            ] {
                let mut candidate = model(config)?;
                candidate.allocate_kv_cache_for_layers(16)?;
                let mut sc = session(&candidate)?;
                let actual = sc.logits_for_ids(&[2, 3, 4], |_| {})?;
                assert!(actual.iter().all(|v| v.is_finite()));
                if config.kernel() == Q8FfnKernel::PackedF32 {
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
                let pointer = candidate
                    .forward_workspace
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .ptrs()
                    .q8_activation;
                assert_eq!(
                    pointer.map(|v| v.bytes),
                    if config.has_quantization() {
                        Some(72)
                    } else {
                        None
                    }
                );
                let resident = CudaContext::total_device_bytes();
                for (index, ids) in prefixes[1..].iter().enumerate() {
                    let before = profile::snapshot();
                    let logits = sc.logits_for_ids(ids, |_| {})?;
                    let after_profile = profile::snapshot();
                    let delta = profile::counter_delta(&before, &after_profile);
                    assert_eq!(delta.device_allocs + delta.device_frees, 0);
                    let op = match config.kernel() {
                        Q8FfnKernel::PackedF32 => "ffn_q8_f32_packed_warp_256",
                        Q8FfnKernel::Q81 if config.threads() == Some(128) => "ffn_q8_1_threads128",
                        Q8FfnKernel::Q81 => "ffn_q8_1_threads256",
                        Q8FfnKernel::SplitF32x4 => "ffn_q8_split_f32x4_128",
                        Q8FfnKernel::Native => unreachable!("candidate list excludes native"),
                    };
                    assert_eq!(
                        after_profile.by_op.get(op).map_or(0, |c| c.launches)
                            - before.by_op.get(op).map_or(0, |c| c.launches),
                        2
                    );
                    if config.kernel() == Q8FfnKernel::PackedF32 {
                        assert_eq!(
                            logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            expected[index + 1]
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>()
                        );
                    }
                    assert!(logits.iter().all(|v| v.is_finite()));
                    assert_eq!(CudaContext::total_device_bytes(), resident);
                    let after = candidate
                        .forward_workspace
                        .lock()
                        .unwrap()
                        .as_ref()
                        .unwrap()
                        .ptrs()
                        .q8_activation;
                    assert_eq!(
                        after.map(|v| (v.pointer, v.bytes)),
                        pointer.map(|v| (v.pointer, v.bytes))
                    );
                }
                if config.kernel() == Q8FfnKernel::PackedF32 {
                    compare_kv(&reference, &candidate, 5)?;
                }
            }
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }

    /// Packed-prefix projections remain native; only the final one-row prompt
    /// step dispatches the selected FFN. Packed256 keeps native logits/KV bits.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_q8_ffn_packed_prefix_changes_only_final_prompt_step() -> Result<()> {
        let _lock = crate::cuda_test_scope::lock_cuda();
        let _env = configure();
        let _hardware = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        {
            let ids = [2, 3, 4, 7, 8];
            let mut reference = model(Q8FfnConfig::native())?;
            reference.allocate_kv_cache_for_layers(16)?;
            let expected = session(&reference)?.logits_for_packed_prefix_then_ids(&ids, |_| {})?;
            for config in [Q8FfnConfig::packed_f32(256)?, Q8FfnConfig::q8_1(128)?] {
                let mut candidate = model(config)?;
                candidate.allocate_kv_cache_for_layers(16)?;
                let mut session = session(&candidate)?;
                let before = profile::snapshot();
                let logits = session.logits_for_packed_prefix_then_ids(&ids, |_| {})?;
                let after = profile::snapshot();
                let op = if config.has_quantization() {
                    "ffn_q8_1_threads128"
                } else {
                    "ffn_q8_f32_packed_warp_256"
                };
                assert_eq!(
                    after.by_op.get(op).map_or(0, |v| v.launches)
                        - before.by_op.get(op).map_or(0, |v| v.launches),
                    2
                );
                assert!(logits.iter().all(|v| v.is_finite()));
                if !config.has_quantization() {
                    assert_eq!(
                        logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    compare_kv(&reference, &candidate, ids.len() as u32)?;
                }
            }
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }

    /// Lossless split planes are model-owned, leave down/vocabulary unchanged,
    /// and cannot divert the SGEMM prefix. Both head64 and biased head128/Qwen
    /// retain numerical logits/KV parity with the qualified packed128 control.
    /// Warm decode reuses storage without quantization, copies or allocation.
    #[cfg(have_cublas)]
    #[test]
    #[ignore = "requires M40/cuBLAS; run with --include-ignored --test-threads=1"]
    fn test_split_ffn_preserves_sgemm_prefix_and_reuses_model_weights() -> Result<()> {
        use m40_llm::infer::{q8_prefill::Q8PrefillBackend, ForwardPrefillSequence};
        let _lock = crate::cuda_test_scope::lock_cuda();
        let _env = configure();
        let _hardware = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        let close = |actual: &[f32], expected: &[f32]| {
            assert_eq!(actual.len(), expected.len());
            for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-3,
                    "index={index} actual={a} expected={b}"
                );
            }
        };
        for is_qwen in [false, true] {
            let load = |ffn: Q8FfnConfig, families: &str| -> Result<LoadedModel> {
                let (gguf, bytes) = fixture::build_prefill_fixture(is_qwen);
                let mut loaded = LoadedModel::from_gguf_with_q8_backends(
                    gguf,
                    bytes,
                    -1,
                    Q8ExecutionConfig::parse("split", Some("6144"), Some(families))?,
                    ffn,
                    Q8PrefillBackend::DequantSgemm,
                )?;
                loaded.allocate_kv_cache_for_layers(16)?;
                Ok(loaded)
            };
            let reference = load(Q8FfnConfig::packed_f32(128)?, "down,vocabulary")?;
            let candidate = load(Q8FfnConfig::split_f32x4(), "gate-up,down,vocabulary")?;
            let old_stats = reference.q8_execution_stats();
            let stats = candidate.q8_execution_stats();
            let gate_up_bytes: usize = candidate
                .gguf
                .tensors
                .iter()
                .filter(|t| {
                    t.name.ends_with("feed_forward.w1.weight")
                        || t.name.ends_with("feed_forward.w3.weight")
                })
                .map(|t| (t.shape[0] * t.shape[1]) as usize / 32 * 34)
                .sum();
            assert!(gate_up_bytes > 0);
            assert_eq!(
                stats.resident_bytes,
                old_stats.resident_bytes + gate_up_bytes
            );
            assert_eq!(
                stats.total_weight_payload_bytes,
                stats.source_weights_bytes + stats.resident_bytes
            );
            assert_eq!(
                stats.single_row_down_split_launch,
                old_stats.single_row_down_split_launch
            );
            for old in &old_stats.tensors {
                let new = stats.tensors.iter().find(|t| t.key == old.key).unwrap();
                assert_eq!(new.resident_bytes, old.resident_bytes);
                assert_eq!(new.source_bytes, old.source_bytes);
            }
            // Later ambient changes must not replace the captured backend.
            let _changed = EnvRestore::set("M40LLM_Q8_FFN_KERNEL", "native");
            let _weights = EnvRestore::set("M40LLM_Q8_EXECUTION_WEIGHTS", "off");
            let width = candidate.model_config.embedding_length as usize;
            let ids = [2, 3, 4, 7, 8];
            let mut hidden = Vec::new();
            for model in [&reference, &candidate] {
                let output = DeviceBuffer::new_tagged(&model.cuda, width * 4, "split-prefix-out")?;
                let before = profile::snapshot();
                unsafe {
                    model.forward_prefill_all_layers_varlen_for_sequences(&[
                        ForwardPrefillSequence {
                            token_ids: &ids[..4],
                            sequence_id: 0,
                            d_out_f32: output.as_mut_ptr(),
                        },
                    ])?;
                }
                let after = profile::snapshot();
                let delta = profile::counter_delta(&before, &after);
                assert_eq!(delta.cublas_calls, 14);
                let launches = |op: &str| {
                    after.by_op.get(op).map_or(0, |p| p.launches)
                        - before.by_op.get(op).map_or(0, |p| p.launches)
                };
                assert_eq!(launches("dequantize_q8_0_to_f32_colmajor"), 14);
                assert_eq!(launches("ffn_q8_split_f32x4_128"), 0);
                let workspace = model.forward_workspace.lock().unwrap();
                let pointers = workspace.as_ref().unwrap().ptrs();
                assert!(pointers.q8_activation.is_none());
                assert_eq!(
                    pointers.q8_prefill_weight.unwrap().bytes,
                    width * model.model_config.feed_forward_length as usize * 4
                );
                let mut values = vec![0.0f32; width];
                unsafe {
                    model.cuda.memcpy_d2h(
                        values.as_mut_ptr().cast(),
                        output.as_ptr(),
                        width * 4,
                    )?;
                }
                hidden.push(values);
            }
            close(&hidden[0], &hidden[1]);
            // All prefix projections/attention are identical, including biases
            // and absolute RoPE positions, regardless of cached gate/up planes.
            compare_kv(&reference, &candidate, 4)?;
            let mut reference_session = DecodeSession::new_for_sequence_with_kv_config(
                &reference,
                0,
                KvCompressionConfig::dense_reference(),
                width,
                true,
                "split-reference",
                "split-reference:in",
                "split-reference:out",
            )?;
            let mut candidate_session = DecodeSession::new_for_sequence_with_kv_config(
                &candidate,
                0,
                KvCompressionConfig::dense_reference(),
                width,
                true,
                "split-candidate",
                "split-candidate:in",
                "split-candidate:out",
            )?;
            reference_session.mark_processed_through(4);
            candidate_session.mark_processed_through(4);
            close(
                &candidate_session.logits_for_ids(&ids, |_| {})?,
                &reference_session.logits_for_ids(&ids, |_| {})?,
            );
            assert!(candidate
                .forward_workspace
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .ptrs()
                .q8_prefill_weight
                .is_none());
            let resident = CudaContext::total_device_bytes();
            let before = profile::snapshot();
            let actual = candidate_session.logits_for_ids(&[2, 3, 4, 7, 8, 9], |_| {})?;
            let after = profile::snapshot();
            let delta = profile::counter_delta(&before, &after);
            assert_eq!(delta.device_allocs + delta.device_frees, 0);
            assert_eq!(
                after.by_op["ffn_q8_split_f32x4_128"].launches
                    - before.by_op["ffn_q8_split_f32x4_128"].launches,
                2
            );
            assert_eq!(CudaContext::total_device_bytes(), resident);
            assert_eq!(
                serde_json::to_value(candidate.q8_execution_stats())?,
                serde_json::to_value(&stats)?
            );
            close(
                &actual,
                &reference_session.logits_for_ids(&[2, 3, 4, 7, 8, 9], |_| {})?,
            );
            assert!(candidate.materialized_weights.lock().unwrap().is_empty());
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }

    /// Missing gate/up preparation, a zero budget, or a different model's
    /// source pointers must fail without silently selecting a native kernel.
    /// Allocation failures leave no prepared/source storage behind.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_split_ffn_rejects_unprepared_weights_and_budget_without_leaks() -> Result<()> {
        let _lock = crate::cuda_test_scope::lock_cuda();
        let _env = configure();
        let _hardware = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        for execution in [
            Q8ExecutionConfig::disabled(),
            Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
            Q8ExecutionConfig::parse("split", Some("0"), Some("gate-up,down,vocabulary"))?,
        ] {
            let (gguf, bytes) = fixture::build_with_layers(0, 2);
            let before = profile::snapshot();
            let error = LoadedModel::from_gguf_with_q8_backends(
                gguf,
                bytes,
                -1,
                execution,
                Q8FfnConfig::split_f32x4(),
                m40_llm::infer::q8_prefill::Q8PrefillBackend::Native,
            )
            .err()
            .expect("unprepared/budget-limited split FFN must fail");
            assert!(error.to_string().contains("gate-up") || error.to_string().contains("budget"));
            assert_eq!(CudaContext::total_device_bytes(), initial);
            if execution.budget_bytes != 0 {
                assert_eq!(
                    profile::counter_delta(&before, &profile::snapshot()),
                    Default::default()
                );
            }
        }
        {
            let candidate = model(Q8FfnConfig::split_f32x4())?;
            let other = model(Q8FfnConfig::split_f32x4())?;
            let foreign = other.map_standard_layer(0)?;
            let mapped = candidate.map_standard_layer(0)?;
            let input = DeviceBuffer::new_tagged(&candidate.cuda, 256, "split-invalid:in")?;
            let output = DeviceBuffer::new_tagged(&candidate.cuda, 512, "split-invalid:out")?;
            let before = profile::snapshot();
            for (sources, rows) in [
                ([foreign.w_gate.dptr, foreign.w_up.dptr], 1),
                ([mapped.w_gate.dptr, mapped.w_gate.dptr], 1),
                ([mapped.w_gate.dptr, mapped.w_up.dptr], 4),
            ] {
                assert!(unsafe {
                    candidate.try_mlp_gate_up_swiglu_f32xf16_gguf_decode_async(
                        input.as_ptr(),
                        rows,
                        64,
                        sources[0],
                        sources[1],
                        128,
                        output.as_mut_ptr(),
                        None,
                    )
                }
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

    /// Unsupported graph/batch and missing/short Q8_1 scratch requests fail
    /// before device work, rather than running a different FFN implementation.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_q8_ffn_unsupported_paths_fail_before_launch() -> Result<()> {
        let _lock = crate::cuda_test_scope::lock_cuda();
        let _env = configure();
        let _hardware = CudaContext::new_m40()?;
        let candidate = model(Q8FfnConfig::q8_1(128)?)?;
        let mapped = candidate.map_standard_layer(0)?;
        let input = DeviceBuffer::new_tagged(&candidate.cuda, 256, "q8-ffn-runtime:invalid-input")?;
        let output =
            DeviceBuffer::new_tagged(&candidate.cuda, 512, "q8-ffn-runtime:invalid-output")?;
        let scratch =
            DeviceBuffer::new_tagged(&candidate.cuda, 72, "q8-ffn-runtime:invalid-scratch")?;
        let before = profile::snapshot();
        for view in [
            None,
            Some(Q8FfnScratch {
                pointer: scratch.as_mut_ptr(),
                bytes: 71,
            }),
            Some(Q8FfnScratch {
                pointer: input.as_mut_ptr(),
                bytes: 256,
            }),
        ] {
            assert!(unsafe {
                candidate.try_mlp_gate_up_swiglu_f32xf16_gguf_decode_async(
                    input.as_ptr(),
                    1,
                    64,
                    mapped.w_gate.dptr,
                    mapped.w_up.dptr,
                    128,
                    output.as_mut_ptr(),
                    view,
                )
            }
            .is_err());
        }
        assert!(
            unsafe { candidate.forward_one_token_all_layers_batched_for_sequences(&[]) }.is_err()
        );
        {
            let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "1");
            assert!(session(&candidate).is_err());
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()),
            Default::default()
        );
        Ok(())
    }
}
