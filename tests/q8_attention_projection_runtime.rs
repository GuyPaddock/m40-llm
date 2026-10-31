use anyhow::Result;
use m40_llm::infer::{
    execution_weights::Q8ExecutionConfig, q8_prefill::Q8PrefillBackend, LoadedModel,
    Q8AttentionProjectionBackend as Backend, Q8FfnConfig,
};
#[path = "common/q8_model.rs"]
#[allow(dead_code)]
mod fixture;

/// The explicit combined candidate is distinct from the native default and
/// unsupported builds reject it before context/device work, not by fallback.
#[test]
fn test_attention_projection_config_is_opt_in() -> Result<()> {
    assert!(Backend::default().is_native());
    assert_eq!(Backend::parse("warp-256")?, Backend::Warp256);
    assert_eq!(Backend::parse("packed-warp-256")?, Backend::PackedWarp256);
    assert_ne!(
        Backend::Warp256.get_operations(),
        Backend::PackedWarp256.get_operations()
    );
    for value in [
        "",
        "1",
        "warp",
        "warp-128",
        "warp-256 ",
        "packed",
        "packed-warp-128",
    ] {
        assert!(Backend::parse(value).is_err());
    }
    #[cfg(not(all(feature = "cuda", nvcc)))]
    {
        for backend in [Backend::Warp256, Backend::PackedWarp256] {
            let (gguf, bytes) = fixture::build(0);
            let before = m40_llm::profile::snapshot();
            let error = LoadedModel::from_gguf_with_q8_runtime(
                gguf,
                bytes,
                -1,
                Q8ExecutionConfig::disabled(),
                Q8FfnConfig::native(),
                Q8PrefillBackend::Native,
                backend,
            )
            .err()
            .expect("unsupported request");
            assert!(error
                .to_string()
                .contains("M40LLM_Q8_ATTN_PROJECTIONS=native"));
            assert_eq!(
                m40_llm::profile::counter_delta(&before, &m40_llm::profile::snapshot()),
                Default::default()
            );
        }
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
    use env_restore::EnvRestore;
    use m40_llm::{
        cuda::{CudaContext, CudaStream, DeviceBuffer},
        decode_session::DecodeSession,
        kv_compression::KvCompressionConfig,
        profile,
    };

    fn configure() -> Vec<EnvRestore> {
        [
            ("M40LLM_ENABLE_NVCC", "1"),
            ("M40LLM_Q8_FFN_KERNEL", "native"),
            ("M40LLM_Q8_PREFILL_BACKEND", "native"),
            ("M40LLM_Q8_EXECUTION_WEIGHTS", "off"),
            ("M40LLM_Q8_DECODE_STREAM", "decode"),
            ("M40LLM_Q8_DECODE_COL_TILE", "4"),
            ("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4"),
            ("M40LLM_FUSED_Q8_QKV", "1"),
            ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
            ("M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK", "4"),
            ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
            ("M40LLM_PROJECTION_BACKEND", "large-model"),
            ("M40LLM_F16_DECODE_KERNEL", "0"),
            ("M40LLM_DECODE_GRAPH", "0"),
        ]
        .into_iter()
        .map(|(k, v)| EnvRestore::set(k, v))
        .collect()
    }

    fn model(backend: Backend, is_qwen: bool) -> Result<LoadedModel> {
        let (gguf, bytes) = fixture::build_prefill_fixture(is_qwen);
        LoadedModel::from_gguf_with_q8_runtime(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::parse("split", Some("16"), Some("gate-up,down,vocabulary"))?,
            Q8FfnConfig::split_f32x4(),
            Q8PrefillBackend::Native,
            backend,
        )
    }

    fn session(model: &LoadedModel) -> Result<DecodeSession> {
        DecodeSession::new_for_sequence_with_kv_config(
            model,
            0,
            KvCompressionConfig::dense_reference(),
            model.model_config.embedding_length as usize,
            true,
            "q8-projection-runtime",
            "q8-projection:input",
            "q8-projection:output",
        )
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// Both QKV and output dispatch are model-local after environment changes.
    /// Real two-layer Qwen biases/RoPE and Llama forward preserve final logits
    /// and cache identities while reusing scratch without per-token allocation.
    /// Packed loads are compared with the qualified warp path and must not
    /// dispatch either legacy projection after selection is captured.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_attention_projection_runtime_preserves_logits_and_dispatch() -> Result<()> {
        let _lock = cuda_test_scope::lock_cuda();
        let _env = configure();
        let _hardware = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        for backend in [Backend::Warp256, Backend::PackedWarp256] {
            for is_qwen in [false, true] {
                {
                    let _flag = EnvRestore::set("M40LLM_Q8_ATTN_PROJECTIONS", backend.as_str());
                    let reference_backend = if backend == Backend::PackedWarp256 {
                        Backend::Warp256
                    } else {
                        Backend::Native
                    };
                    let mut native = model(reference_backend, is_qwen)?;
                    let mut candidate = model(backend, is_qwen)?;
                    native.allocate_kv_cache_for_layers(32)?;
                    candidate.allocate_kv_cache_for_layers(32)?;
                    let a = serde_json::to_value(native.q8_execution_stats())?;
                    let b = serde_json::to_value(candidate.q8_execution_stats())?;
                    assert_eq!(a["resident_bytes"], b["resident_bytes"]);
                    assert_eq!(a["tensors"], b["tensors"]);
                    let expected =
                        session(&native)?.logits_for_ids(&[65, 66, 67, 68, 69], |_| {})?;
                    let mut candidate_session = session(&candidate)?;
                    candidate_session.logits_for_ids(&[65, 66, 67, 68], |_| {})?;
                    let _changed =
                        EnvRestore::set("M40LLM_Q8_ATTN_PROJECTIONS", "invalid-after-load");
                    let _fusion = EnvRestore::set("M40LLM_FUSED_Q8_QKV", "0");
                    let before = profile::snapshot();
                    let resident = CudaContext::total_device_bytes();
                    let actual = candidate_session.logits_for_ids(&[65, 66, 67, 68, 69], |_| {})?;
                    assert_eq!(bits(&expected), bits(&actual));
                    let after = profile::snapshot();
                    for op in backend.get_operations().unwrap() {
                        let count = after.by_op.get(op).map_or(0, |v| v.launches)
                            - before.by_op.get(op).map_or(0, |v| v.launches);
                        assert_eq!(count, 2);
                    }
                    let delta = profile::counter_delta(&before, &after);
                    assert_eq!(delta.device_allocs + delta.device_frees, 0);
                    assert_eq!(CudaContext::total_device_bytes(), resident);
                    assert_eq!(candidate.q8_attention_projection.backend(), backend);
                    assert_eq!(native.q8_attention_projection.backend(), reference_backend);
                    if backend == Backend::PackedWarp256 {
                        for op in Backend::Warp256.get_operations().unwrap() {
                            assert_eq!(
                                after.by_op.get(op).map_or(0, |v| v.launches)
                                    - before.by_op.get(op).map_or(0, |v| v.launches),
                                0
                            );
                        }
                    }
                    assert_eq!(
                        serde_json::to_value(candidate.q8_execution_stats())?["tensors"],
                        b["tensors"]
                    );
                }
                assert_eq!(CudaContext::total_device_bytes(), initial);
            }
        }
        Ok(())
    }

    /// Generic loading captures the flag once; explicit references ignore it.
    /// Conflicting prepared projections fail before device work, unsupported
    /// F16 candidates fail clearly, and all rejected loads release their storage.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_attention_projection_load_captures_and_validates_selection() -> Result<()> {
        let _lock = cuda_test_scope::lock_cuda();
        let _env = configure();
        let _hardware = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        for backend in [Backend::Warp256, Backend::PackedWarp256] {
            {
                let _flag = EnvRestore::set("M40LLM_Q8_ATTN_PROJECTIONS", backend.as_str());
                let (gguf, bytes) = fixture::build(0);
                let mut loaded = LoadedModel::from_gguf(gguf, bytes, -1)?;
                loaded.allocate_kv_cache_for_layers(16)?;
                let _changed = EnvRestore::set("M40LLM_Q8_ATTN_PROJECTIONS", "native");
                session(&loaded)?.logits_for_ids(&[2, 3, 4], |_| {})?;
                assert_eq!(loaded.q8_attention_projection.backend(), backend);
            }
            assert_eq!(CudaContext::total_device_bytes(), initial);
            for family in ["qkv", "output"] {
                let (gguf, bytes) = fixture::build(0);
                let before = profile::snapshot();
                let error = LoadedModel::from_gguf_with_q8_runtime(
                    gguf,
                    bytes,
                    -1,
                    Q8ExecutionConfig::parse("split", Some("16"), Some(family))?,
                    Q8FfnConfig::native(),
                    Q8PrefillBackend::Native,
                    backend,
                )
                .err()
                .expect("raw-only candidate");
                assert!(error.to_string().contains("raw Q8_0"));
                assert_eq!(
                    profile::counter_delta(&before, &profile::snapshot()),
                    Default::default()
                );
            }
            let (gguf, bytes) = fixture::build_f16();
            let error = LoadedModel::from_gguf_with_q8_runtime(
                gguf,
                bytes,
                -1,
                Q8ExecutionConfig::disabled(),
                Q8FfnConfig::native(),
                Q8PrefillBackend::Native,
                backend,
            )
            .err()
            .expect("Q8-only candidate");
            assert!(error.to_string().contains("Q8_0 QKV/output"));
            assert_eq!(CudaContext::total_device_bytes(), initial);
        }
        Ok(())
    }

    /// One-row wrappers reject foreign or wrong-family weights and shapes
    /// without work; multi-row prefill retains the original implementation.
    /// Explicit old constructors ignore the ambient warp-selection flag.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_attention_projection_rejects_wrong_spans_and_keeps_prefill() -> Result<()> {
        let _lock = cuda_test_scope::lock_cuda();
        let _env = configure();
        let _hardware = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        for backend in [Backend::Warp256, Backend::PackedWarp256] {
            {
                let model = model(backend, false)?;
                let w = model.map_standard_layer(0)?;
                let x = DeviceBuffer::new_tagged(
                    &model.cuda,
                    4 * 64 * 4,
                    "attn-projection:test-input",
                )?;
                let y = DeviceBuffer::new_tagged(
                    &model.cuda,
                    4 * 128 * 4,
                    "attn-projection:test-output",
                )?;
                let z =
                    DeviceBuffer::new_tagged(&model.cuda, 128 * 4, "attn-projection:test-output")?;
                let t =
                    DeviceBuffer::new_tagged(&model.cuda, 128 * 4, "attn-projection:test-output")?;
                let input = fixture::projection::make_input(4, 64);
                unsafe {
                    model
                        .cuda
                        .memcpy_h2d(x.as_mut_ptr(), input.as_ptr().cast(), x.bytes())?;
                }
                let before = profile::snapshot();
                unsafe {
                    if backend == Backend::PackedWarp256 {
                        assert!(model
                            .out_proj_f32xf16_gguf_f32_async(
                                x.as_ptr().cast::<u8>().add(4).cast(),
                                w.wo.dptr,
                                y.as_mut_ptr(),
                                1,
                                64,
                                64
                            )
                            .is_err());
                    }
                    assert!(model
                        .out_proj_f32xf16_gguf_f32_async(
                            x.as_ptr(),
                            w.w_gate.dptr,
                            y.as_mut_ptr(),
                            1,
                            64,
                            64
                        )
                        .is_err());
                    assert!(model
                        .out_proj_f32xf16_gguf_f32_async(
                            x.as_ptr(),
                            w.wo.dptr,
                            y.as_mut_ptr(),
                            1,
                            63,
                            64
                        )
                        .is_err());
                    assert!(model
                        .try_qkv_project_f32xf16_gguf_f32_decode_async(
                            x.as_ptr(),
                            1,
                            64,
                            w.wo.dptr,
                            64,
                            w.wk.dptr,
                            64,
                            w.wv.dptr,
                            64,
                            None,
                            None,
                            None,
                            y.as_mut_ptr(),
                            z.as_mut_ptr(),
                            t.as_mut_ptr()
                        )
                        .is_err());
                }
                assert_eq!(
                    profile::counter_delta(&before, &profile::snapshot()),
                    Default::default()
                );
                unsafe {
                    assert!(!model.try_qkv_project_f32xf16_gguf_f32_decode_async(
                        x.as_ptr(),
                        4,
                        64,
                        w.wq.dptr,
                        64,
                        w.wk.dptr,
                        64,
                        w.wv.dptr,
                        64,
                        None,
                        None,
                        None,
                        y.as_mut_ptr(),
                        z.as_mut_ptr(),
                        t.as_mut_ptr()
                    )?);
                    model.out_proj_f32xf16_gguf_f32_async(
                        x.as_ptr(),
                        w.wo.dptr,
                        y.as_mut_ptr(),
                        4,
                        64,
                        64,
                    )?;
                }
                model.cuda.synchronize_stream(CudaStream::Prefill)?;
                let mut actual = vec![0.0f32; 4 * 64];
                unsafe {
                    model.cuda.memcpy_d2h(
                        actual.as_mut_ptr().cast(),
                        y.as_ptr(),
                        actual.len() * 4,
                    )?;
                }
                let source = &model.host_weights
                    [w.wo.byte_offset as usize..w.wo.byte_offset as usize + w.wo.nbytes];
                for row in 0..4 {
                    for col in 0..64 {
                        let cpu = fixture::projection::project_cpu(&input, source, row, col, 64);
                        assert!(
                            (f64::from(actual[row * 64 + col]) - cpu).abs()
                                < 1e-3 + 2e-5 * cpu.abs()
                        );
                    }
                }
                let _flag = EnvRestore::set("M40LLM_Q8_ATTN_PROJECTIONS", backend.as_str());
                let (gguf, bytes) = fixture::build(0);
                let reference = LoadedModel::from_gguf_with_q8_execution(
                    gguf,
                    bytes,
                    -1,
                    Q8ExecutionConfig::disabled(),
                )?;
                assert_eq!(reference.q8_attention_projection.backend(), Backend::Native);
            }
            assert_eq!(CudaContext::total_device_bytes(), initial);
        }
        Ok(())
    }
}
