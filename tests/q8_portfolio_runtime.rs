#![cfg(all(feature = "cuda", nvcc))]
#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/env_restore.rs"]
mod env_restore;
#[path = "common/q8_model.rs"]
#[allow(dead_code)]
mod fixture;

use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{ffi_debug_read_kv_token, CudaContext, CudaStream, DeviceBuffer},
    decode_session::DecodeSession,
    infer::{
        execution_weights::{Q8ExecutionConfig, Q8PortfolioArm},
        q8_prefill::Q8PrefillBackend,
        LoadedModel, Q8AttentionProjectionBackend, Q8FfnConfig,
    },
    kv_compression::KvCompressionConfig,
    profile,
};

fn configure() -> Vec<env_restore::EnvRestore> {
    [
        ("M40LLM_PROJECTION_BACKEND", "large-model"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_Q8_DECODE_COL_TILE", "4"),
        ("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4"),
        ("M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK", "4"),
        ("M40LLM_ATTN_HEAD128_KERNEL", "native"),
        ("M40LLM_FUSED_Q8_QKV", "1"),
        ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
        ("M40LLM_CUDA_GREEDY_ARGMAX", "1"),
        ("M40LLM_DECODE_GRAPH", "0"),
        ("M40LLM_SERVER_BATCH_DECODE", "0"),
    ]
    .into_iter()
    .map(|(k, v)| env_restore::EnvRestore::set(k, v))
    .collect()
}

fn model(budget: usize) -> Result<LoadedModel> {
    let (mut gguf, mut bytes) = fixture::build_prefill_fixture(true);
    gguf.tensors.push(m40_llm::gguf::GgufTensor {
        name: "output_norm.weight".into(),
        shape: vec![256],
        dtype: m40_llm::gguf::GgmlDType::F32,
        offset: bytes.len() as u64,
    });
    for index in 0..256 {
        bytes.extend_from_slice(&(0.5 + index as f32 / 512.).to_le_bytes());
    }
    let mut config =
        Q8ExecutionConfig::parse("split", Some("6144"), Some("gate-up,down,vocabulary"))?;
    config.budget_bytes = budget;
    let mut model = LoadedModel::from_gguf_with_q8_runtime(
        gguf,
        bytes,
        -1,
        config,
        Q8FfnConfig::joint_shared_scales(),
        if cfg!(have_cublas) {
            Q8PrefillBackend::DequantSgemm
        } else {
            Q8PrefillBackend::Native
        },
        Q8AttentionProjectionBackend::PackedWarp256,
    )?;
    model.allocate_kv_cache_for_layers(32)?;
    Ok(model)
}

fn session(model: &LoadedModel) -> Result<DecodeSession> {
    DecodeSession::new_for_sequence_with_kv_config(
        model,
        0,
        KvCompressionConfig::dense_reference(),
        256,
        true,
        "portfolio-test",
        "portfolio-test:x",
        "portfolio-test:out",
    )
}

fn launches(before: &profile::ProfileSnapshot, op: &str) -> u64 {
    profile::snapshot().by_op.get(op).map_or(0, |v| v.launches)
        - before.by_op.get(op).map_or(0, |v| v.launches)
}

fn kv_bits(model: &LoadedModel, length: u32) -> Result<Vec<u16>> {
    let kv = model.kv_cache.as_ref().unwrap();
    let mut bits = Vec::new();
    for layer in 0..2 {
        assert_eq!(kv.debug_sequence_length(&model.cuda, layer)?, length);
        for position in 0..length {
            let mut k = vec![0u16; kv.elems_per_token()];
            let mut v = k.clone();
            let rc = unsafe {
                ffi_debug_read_kv_token(
                    &model.cuda,
                    kv,
                    layer,
                    position,
                    k.as_mut_ptr().cast(),
                    v.as_mut_ptr().cast(),
                )
            };
            ensure!(rc == 0, "KV read failed: {rc}");
            bits.extend(k);
            bits.extend(v);
        }
    }
    Ok(bits)
}

/// Five model-local arms retain equal packet storage, preserve full-forward
/// logits/greedy IDs/valid KV, and leave packed prefix dispatch unchanged.
#[test]
#[ignore = "requires M40 hardware"]
fn test_portfolio_runtime_preserves_bits_and_dispatch() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let initial = CudaContext::total_device_bytes();
    {
        let baseline = model(6 * 1024 * 1024 * 1024)?;
        assert!(baseline.q8_execution.portfolio_stats().is_none());
        for arm in [
            Q8PortfolioArm::Control,
            Q8PortfolioArm::Down,
            Q8PortfolioArm::Vocabulary,
            Q8PortfolioArm::Argmax,
            Q8PortfolioArm::All,
        ] {
            let mut candidate = model(6 * 1024 * 1024 * 1024)?;
            let originals = candidate.q8_execution_stats();
            let before = CudaContext::total_device_bytes();
            candidate.prepare_q8_portfolio(arm)?;
            let stats = candidate.q8_execution.portfolio_stats().unwrap();
            assert_eq!(stats.tensors.len(), 3);
            assert_eq!(stats.resident_bytes, 365568);
            assert_eq!(
                CudaContext::total_device_bytes() - before,
                stats.resident_bytes
            );
            assert_eq!(
                stats.original_prepared_bytes,
                candidate.q8_ffn_pair_stats().combined_prepared_bytes
            );
            assert_eq!(
                serde_json::to_value(candidate.q8_execution_stats())?,
                serde_json::to_value(originals)?
            );
            let before = CudaContext::total_device_bytes();
            assert!(candidate.prepare_q8_portfolio(Q8PortfolioArm::All).is_err());
            assert_eq!(CudaContext::total_device_bytes(), before);
            for (is_packed, stream) in [(false, "decode"), (true, "decode"), (false, "prefill")] {
                let _stream = env_restore::EnvRestore::set("M40LLM_Q8_DECODE_STREAM", stream);
                baseline.reset_kv_cache()?;
                candidate.reset_kv_cache()?;
                let before = CudaContext::total_device_bytes();
                let mut a = session(&baseline)?;
                let baseline_session_bytes = CudaContext::total_device_bytes() - before;
                let before = CudaContext::total_device_bytes();
                let mut b = session(&candidate)?;
                assert_eq!(
                    CudaContext::total_device_bytes() - before,
                    baseline_session_bytes
                        + CudaContext::argmax_f32_partitioned_scratch_bytes(320)?
                );
                let ids = [1, 7, 3, 4, 5, 8, 9, 11];
                if is_packed {
                    let before = profile::snapshot();
                    a.prefill_packed_prefix_ids(&ids[..5])?;
                    let control = profile::counter_delta(&before, &profile::snapshot());
                    let before = profile::snapshot();
                    b.prefill_packed_prefix_ids(&ids[..5])?;
                    assert_eq!(
                        profile::counter_delta(&before, &profile::snapshot()),
                        control
                    );
                    assert_eq!(launches(&before, "gemm_f32xq8_warp_packet"), 0);
                    assert_eq!(kv_bits(&baseline, 4)?, kv_bits(&candidate, 4)?);
                }
                for length in 5..=ids.len() {
                    let expected = a.greedy_token_for_ids(&ids[..length])?;
                    let before = profile::snapshot();
                    let actual = b.greedy_token_for_ids(&ids[..length])?;
                    assert_eq!(actual, expected);
                    if length > 5 || is_packed {
                        assert_eq!(
                            launches(&before, "gemm_f32xq8_warp_packet"),
                            2 * u64::from(arm.has_packet_down())
                                + u64::from(arm.has_packet_vocabulary())
                        );
                        assert_eq!(
                            launches(&before, "argmax_f32_partition_partials"),
                            u64::from(arm.has_partitioned_argmax())
                        );
                        assert_eq!(
                            launches(&before, "argmax_f32_partition_reduce"),
                            u64::from(arm.has_partitioned_argmax())
                        );
                        assert_eq!(
                            launches(&before, "argmax_f32"),
                            u64::from(!arm.has_partitioned_argmax())
                        );
                    }
                    if length > 5 {
                        let counters = profile::counter_delta(&before, &profile::snapshot());
                        assert_eq!(counters.device_allocs, 0);
                        assert_eq!(counters.device_frees, 0);
                    }
                    let control = unsafe { baseline.logits_from_hidden(a.d_out_ptr()?)? };
                    let actual = unsafe { candidate.logits_from_hidden(b.d_out_ptr()?)? };
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        control.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    assert_eq!(
                        kv_bits(&baseline, length as u32)?,
                        kv_bits(&candidate, length as u32)?
                    );
                }
            }
            candidate.q8_execution.verify_portfolio_payloads()?;
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Insufficient combined budget leaves the live model unchanged. Missing
/// argmax scratch fails before launches, rather than silently using control.
#[test]
#[ignore = "requires M40 hardware"]
fn test_portfolio_rejects_budget_and_missing_scratch() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let initial = CudaContext::total_device_bytes();
    {
        let mut limited = model(1_000_000)?;
        let resident = CudaContext::total_device_bytes();
        assert!(limited.prepare_q8_portfolio(Q8PortfolioArm::All).is_err());
        assert!(limited.q8_execution.portfolio_stats().is_none());
        assert_eq!(CudaContext::total_device_bytes(), resident);
        let mut candidate = model(6 * 1024 * 1024 * 1024)?;
        candidate.prepare_q8_portfolio(Q8PortfolioArm::Argmax)?;
        let input = DeviceBuffer::new_tagged(&candidate.cuda, 256 * 4, "portfolio:missing-input")?;
        let logits =
            DeviceBuffer::new_tagged(&candidate.cuda, 320 * 4, "portfolio:missing-logits")?;
        let norm = DeviceBuffer::new_tagged(&candidate.cuda, 256 * 4, "portfolio:missing-norm")?;
        let token = DeviceBuffer::new_tagged(&candidate.cuda, 4, "portfolio:missing-token")?;
        let before = profile::snapshot();
        assert!(unsafe {
            candidate.greedy_token_from_hidden_into(
                input.as_ptr(),
                logits.as_mut_ptr(),
                Some(norm.as_mut_ptr()),
                token.as_mut_ptr(),
            )
        }
        .is_err());
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()).launches,
            0
        );
        assert!(DecodeSession::new_for_sequence_with_kv_config(
            &candidate,
            0,
            KvCompressionConfig::default(),
            256,
            true,
            "portfolio-invalid",
            "portfolio-invalid:x",
            "portfolio-invalid:out"
        )
        .is_err());
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// The inconclusive four-row packet experiment is not admitted: explicit
/// multi-row projection keeps the original prepared consumer and exact bits.
#[test]
#[ignore = "requires M40 hardware"]
fn test_portfolio_keeps_multirow_projection_control() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let initial = CudaContext::total_device_bytes();
    {
        let baseline = model(6 * 1024 * 1024 * 1024)?;
        let mut candidate = model(6 * 1024 * 1024 * 1024)?;
        candidate.prepare_q8_portfolio(Q8PortfolioArm::All)?;
        for (control_weight, candidate_weight) in [
            (
                baseline.map_standard_layer(0)?.w_down,
                candidate.map_standard_layer(0)?.w_down,
            ),
            (baseline.map_lm_head()?.1, candidate.map_lm_head()?.1),
        ] {
            let (k, n) = (
                control_weight.shape[0] as usize,
                control_weight.shape[1] as usize,
            );
            let input = fixture::projection::make_input(4, k);
            let mut outputs = Vec::new();
            for (model, weight) in [(&baseline, control_weight), (&candidate, candidate_weight)] {
                let d_input = DeviceBuffer::new_tagged(
                    &model.cuda,
                    input.len() * 4,
                    "portfolio:multirow-input",
                )?;
                let output =
                    DeviceBuffer::new_tagged(&model.cuda, 4 * n * 4, "portfolio:multirow-output")?;
                unsafe {
                    model.cuda.memcpy_h2d(
                        d_input.as_mut_ptr(),
                        input.as_ptr().cast(),
                        d_input.bytes(),
                    )?;
                }
                let before = profile::snapshot();
                unsafe {
                    model.matmul_f32xf16_gguf_f32_async(
                        d_input.as_ptr(),
                        weight.dptr,
                        output.as_mut_ptr(),
                        4,
                        n as i32,
                        k as i32,
                    )?;
                }
                model.cuda.synchronize_stream(CudaStream::Prefill)?;
                assert_eq!(launches(&before, "gemm_f32xq8_warp_packet"), 0);
                assert_eq!(launches(&before, "gemm_f32xq8_separate_f32"), 1);
                let mut bits = vec![0u32; 4 * n];
                unsafe {
                    model.cuda.memcpy_d2h(
                        bits.as_mut_ptr().cast(),
                        output.as_ptr(),
                        output.bytes(),
                    )?;
                }
                outputs.push(bits);
            }
            assert_eq!(outputs[0], outputs[1]);
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}
