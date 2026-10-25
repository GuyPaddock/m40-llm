#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/env_restore.rs"]
mod env_restore;
#[path = "common/q8_model.rs"]
mod fixture;
#[path = "common/q8_greedy.rs"]
mod greedy;

use anyhow::{ensure, Result};
use env_restore::EnvRestore;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    infer::{
        execution_weights::{
            Q8ExecutionConfig, Q8ExecutionFamilies, Q8ExecutionMode, Q8ProjectionGroup,
        },
        LoadedModel,
    },
    profile,
};

fn projection_group(name: &str, vocabulary_name: &str) -> Q8ProjectionGroup {
    if name == vocabulary_name {
        return Q8ProjectionGroup::Vocabulary;
    }
    match name {
        "layers.0.attention.wq.weight"
        | "layers.0.attention.wk.weight"
        | "layers.0.attention.wv.weight" => Q8ProjectionGroup::Qkv,
        "layers.0.attention.wo.weight" => Q8ProjectionGroup::Output,
        "layers.0.feed_forward.w1.weight" | "layers.0.feed_forward.w3.weight" => {
            Q8ProjectionGroup::GateUp
        }
        "layers.0.feed_forward.w2.weight" => Q8ProjectionGroup::Down,
        _ => panic!("unexpected fixture tensor {name}"),
    }
}

fn project(model: &LoadedModel, name: &str, is_prepared: bool) -> Result<Vec<f32>> {
    let tensor = &model.device_tensors[name];
    let (k, n) = (tensor.shape[0] as usize, tensor.shape[1] as usize);
    let input = fixture::projection::make_input(1, k);
    let d_input = DeviceBuffer::new_tagged(&model.cuda, k * 4, "families:input")?;
    let d_output = DeviceBuffer::new_tagged(&model.cuda, n * 4, "families:output")?;
    unsafe {
        model
            .cuda
            .memcpy_h2d(d_input.as_mut_ptr(), input.as_ptr().cast(), d_input.bytes())?;
    }
    let before = profile::snapshot();
    unsafe {
        model.matmul_f32xf16_gguf_f32_async(
            d_input.as_ptr(),
            tensor.dptr,
            d_output.as_mut_ptr(),
            1,
            n as i32,
            k as i32,
        )?;
    }
    let after = profile::snapshot();
    let delta = profile::counter_delta(&before, &after);
    assert_eq!(delta.launches, 1);
    assert_eq!(delta.stream_syncs, 0);
    assert_eq!(delta.device_allocs, 0);
    assert_eq!(delta.device_frees, 0);
    assert_eq!(delta.h2d_copies, 0);
    let op = if is_prepared {
        "gemm_f32xq8_separate_f32"
    } else {
        "gemm_f32xq8_0_gguf_f32_decode"
    };
    assert_eq!(
        after.by_op.get(op).map_or(0, |c| c.launches)
            - before.by_op.get(op).map_or(0, |c| c.launches),
        1,
        "wrong family dispatch: {name}"
    );
    model.cuda.synchronize_stream(CudaStream::Decode)?;
    let mut output = vec![0.0_f32; n];
    unsafe {
        model.cuda.memcpy_d2h(
            output.as_mut_ptr().cast(),
            d_output.as_ptr(),
            d_output.bytes(),
        )?;
    }
    let source = &model.host_weights
        [tensor.byte_offset as usize..tensor.byte_offset as usize + tensor.nbytes];
    for (col, &actual) in output.iter().enumerate() {
        let expected = fixture::projection::project_cpu(&input, source, 0, col, k);
        ensure!(
            actual.is_finite()
                && (f64::from(actual) - expected).abs() < 1e-3 + expected.abs() * 1e-5,
            "family projection CPU mismatch"
        );
    }
    Ok(output)
}

fn fused_mlp(model: &LoadedModel, is_prepared: bool) -> Result<Vec<f32>> {
    let input = fixture::projection::make_input(1, 64);
    let d_input = DeviceBuffer::new_tagged(&model.cuda, 64 * 4, "families:mlp-input")?;
    let d_output = DeviceBuffer::new_tagged(&model.cuda, 128 * 4, "families:mlp-output")?;
    unsafe {
        model
            .cuda
            .memcpy_h2d(d_input.as_mut_ptr(), input.as_ptr().cast(), d_input.bytes())?;
    }
    let before = profile::snapshot();
    ensure!(
        unsafe {
            model.try_mlp_gate_up_swiglu_f32xf16_gguf_decode_async(
                d_input.as_ptr(),
                1,
                64,
                model.device_tensors["layers.0.feed_forward.w3.weight"].dptr,
                model.device_tensors["layers.0.feed_forward.w1.weight"].dptr,
                128,
                d_output.as_mut_ptr(),
            )?
        },
        "fused MLP must remain enabled"
    );
    let after = profile::snapshot();
    let op = if is_prepared {
        "mlp_gate_up_swiglu_f32xq8_separate"
    } else {
        "mlp_gate_up_swiglu_f32xq8_0_gguf_decode"
    };
    assert_eq!(
        after.by_op.get(op).map_or(0, |c| c.launches)
            - before.by_op.get(op).map_or(0, |c| c.launches),
        1
    );
    assert_eq!(profile::counter_delta(&before, &after).launches, 1);
    model.cuda.synchronize_stream(CudaStream::Decode)?;
    let mut output = vec![0.0_f32; 128];
    unsafe {
        model.cuda.memcpy_d2h(
            output.as_mut_ptr().cast(),
            d_output.as_ptr(),
            d_output.bytes(),
        )?;
    }
    Ok(output)
}

/// Selected families alone consume prepared memory and dispatch; omitted
/// families retain native math and useful fused MLP/greedy paths. Separate
/// immutable models must coexist without toggling global execution flags.
#[test]
fn test_family_selection_preserves_native_consumers_and_accounting() -> Result<()> {
    let _stream = EnvRestore::set("M40LLM_Q8_DECODE_STREAM", "decode");
    let _mlp = EnvRestore::set("M40LLM_FUSED_Q8_MLP_SWIGLU", "1");
    let _head = EnvRestore::set("M40LLM_Q8_LM_HEAD_ARGMAX", "1");
    let initial = CudaContext::total_device_bytes();
    {
        let (gguf, bytes) = fixture::build(0);
        let native = LoadedModel::from_gguf_with_q8_execution(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::disabled(),
        )?;
        let props = native.cuda.current_device_props()?;
        ensure!(
            props.major == 5 && props.minor == 2,
            "family selection requires M40"
        );
        let (vocabulary, _, _, _, _) = native.map_lm_head()?;
        for selected in [
            "qkv",
            "output",
            "gate-up",
            "down",
            "vocabulary",
            "qkv,output,down,vocabulary",
        ] {
            let families = Q8ExecutionFamilies::parse(selected)?;
            let (gguf, bytes) = fixture::build(0);
            let model = LoadedModel::from_gguf_with_q8_execution(
                gguf,
                bytes,
                -1,
                Q8ExecutionConfig {
                    mode: Q8ExecutionMode::Split,
                    budget_bytes: 1024 * 1024,
                    families,
                },
            )?;
            let stats = model.q8_execution_stats();
            assert_eq!(stats.families, families);
            let expected_bytes: usize = native
                .device_tensors
                .iter()
                .filter(|(name, _)| families.contains(projection_group(name, &vocabulary)))
                .map(|(_, tensor)| tensor.nbytes)
                .sum();
            assert_eq!(stats.resident_bytes, expected_bytes);
            assert_eq!(stats.estimated_bytes, expected_bytes);
            assert_eq!(
                stats.has_prepared_vocabulary,
                families.contains(Q8ProjectionGroup::Vocabulary)
            );
            assert_eq!(
                stats.total_weight_payload_bytes,
                stats.source_weights_bytes + expected_bytes
            );
            for name in native.device_tensors.keys() {
                let expected = project(&native, name, false)?;
                let actual = project(
                    &model,
                    name,
                    families.contains(projection_group(name, &vocabulary)),
                )?;
                let (max, mean) = fixture::projection::measure_error(&actual, &expected);
                ensure!(max < 1e-3 && mean < 1e-4, "native/family mismatch");
            }
            let expected = fused_mlp(&native, false)?;
            let actual = fused_mlp(&model, families.contains(Q8ProjectionGroup::GateUp))?;
            let (max, mean) = fixture::projection::measure_error(&actual, &expected);
            ensure!(max < 1e-3 && mean < 1e-4, "family fused MLP mismatch");
            let input = fixture::projection::make_input(1, 64);
            assert_eq!(
                greedy::compare_greedy_with_logits(&native, &input)?.0,
                greedy::compare_greedy_with_logits(&model, &input)?.0
            );
            assert!(native.q8_execution_stats().tensors.is_empty());
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}
