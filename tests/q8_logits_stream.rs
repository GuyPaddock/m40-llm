#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/env_restore.rs"]
mod env_restore;
#[path = "common/q8_model.rs"]
#[allow(dead_code)]
mod fixture;

use anyhow::Result;
use env_restore::EnvRestore;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    infer::{execution_weights::Q8ExecutionConfig, LoadedModel},
    profile,
};

/// Q8 logits must follow the actual Q8 producer, regardless of unrelated F16
/// decode flags. The returned stream, dependency on queued decode input and
/// complete CPU-reference logits must agree with and without prepared weights.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_q8_logits_stream_ignores_unrelated_f16_flags() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _nvcc = EnvRestore::set("M40LLM_ENABLE_NVCC", "1");
    let _backend = EnvRestore::set("M40LLM_PROJECTION_BACKEND", "large-model");
    let _materialize = EnvRestore::set("M40LLM_MATERIALIZE_F32_WEIGHTS", "0");
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for cache in ["off", "split"] {
        let (gguf, bytes) = fixture::build(0);
        let model = LoadedModel::from_gguf_with_q8_execution(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::parse(cache, Some("1"), Some("vocabulary"))?,
        )?;
        let (name, head, d_model, vocab, _) = model.map_lm_head()?;
        assert_eq!(d_model, 64);
        let input = fixture::projection::make_input(1, d_model);
        let added: Vec<_> = input
            .iter()
            .enumerate()
            .map(|(i, v)| v + ((i % 7) as f32) * 0.25)
            .collect();
        let expected_input: Vec<_> = input.iter().zip(&added).map(|(a, b)| a + b).collect();
        let a = DeviceBuffer::new_tagged(&model.cuda, d_model * 4, "q8-logits:producer-a")?;
        let b = DeviceBuffer::new_tagged(&model.cuda, d_model * 4, "q8-logits:producer-b")?;
        let hidden = DeviceBuffer::new_tagged(&model.cuda, d_model * 4, "q8-logits:hidden")?;
        let output = DeviceBuffer::new_tagged(&model.cuda, vocab * 4, "q8-logits:output")?;
        unsafe {
            model
                .cuda
                .memcpy_h2d(a.as_mut_ptr(), input.as_ptr().cast(), a.bytes())?;
            model
                .cuda
                .memcpy_h2d(b.as_mut_ptr(), added.as_ptr().cast(), b.bytes())?;
        }
        for stream in ["decode", "prefill"] {
            let _stream = EnvRestore::set("M40LLM_Q8_DECODE_STREAM", stream);
            for legacy in ["0", "1"] {
                let _kernel = EnvRestore::set("M40LLM_F16_DECODE_KERNEL", legacy);
                let _legacy_stream = EnvRestore::set("M40LLM_F16_DECODE_STREAM", "decode");
                unsafe {
                    model.cuda.residual_add_f32_async(
                        a.as_ptr(),
                        b.as_ptr(),
                        hidden.as_mut_ptr(),
                        d_model,
                    )?;
                }
                let before = profile::snapshot();
                let (actual_vocab, producer) = unsafe {
                    model.enqueue_logits_from_hidden_into(
                        hidden.as_ptr(),
                        output.as_mut_ptr(),
                        None,
                    )?
                };
                let delta = profile::counter_delta(&before, &profile::snapshot());
                model.cuda.synchronize_stream(CudaStream::Decode)?;
                model.cuda.synchronize_stream(CudaStream::Prefill)?;
                assert_eq!(actual_vocab, vocab);
                assert_eq!(
                    producer,
                    if stream == "decode" {
                        CudaStream::Decode
                    } else {
                        CudaStream::Prefill
                    },
                    "cache={cache} legacy={legacy}"
                );
                assert_eq!(
                    delta.device_allocs
                        + delta.device_frees
                        + delta.h2d_copies
                        + delta.d2h_copies
                        + delta.stream_syncs,
                    0
                );
                let mut actual = vec![0.0f32; vocab];
                unsafe {
                    model.cuda.memcpy_d2h(
                        actual.as_mut_ptr().cast(),
                        output.as_ptr(),
                        output.bytes(),
                    )?;
                }
                let start = head.byte_offset as usize;
                let bytes = &model.host_weights[start..start + head.nbytes];
                for (column, &value) in actual.iter().enumerate() {
                    let expected = fixture::projection::project_cpu(
                        &expected_input,
                        bytes,
                        0,
                        column,
                        d_model,
                    );
                    assert!(
                        value.is_finite()
                            && (f64::from(value) - expected).abs() < 1e-3 + 2e-5 * expected.abs(),
                        "{name} column={column}"
                    );
                }
            }
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}
