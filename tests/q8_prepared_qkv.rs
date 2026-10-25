#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/env_restore.rs"]
mod env_restore;
#[path = "common/q8_model.rs"]
mod fixture;

use anyhow::{ensure, Result};
use env_restore::EnvRestore;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    infer::{
        execution_weights::{Q8ExecutionConfig, Q8ExecutionMode},
        LoadedModel,
    },
    profile,
};

fn make_model(mode: Q8ExecutionMode, widths: [usize; 3]) -> Result<LoadedModel> {
    let (gguf, bytes) = if widths == [64; 3] {
        fixture::build(0)
    } else {
        fixture::build_with_qkv_widths(0, widths)
    };
    LoadedModel::from_gguf_with_q8_execution(
        gguf,
        bytes,
        -1,
        Q8ExecutionConfig {
            mode,
            budget_bytes: 1024 * 1024,
        },
    )
}

struct QkvBuffers {
    input: DeviceBuffer,
    biases: Vec<DeviceBuffer>,
    outputs: Vec<DeviceBuffer>,
    widths: [usize; 3],
}

impl QkvBuffers {
    fn new(model: &LoadedModel, widths: [usize; 3]) -> Result<Self> {
        let input = fixture::projection::make_input(1, 64);
        let device = DeviceBuffer::new_tagged(&model.cuda, input.len() * 4, "qkv:input")?;
        unsafe {
            model
                .cuda
                .memcpy_h2d(device.as_mut_ptr(), input.as_ptr().cast(), device.bytes())?;
        }
        let mut biases = Vec::new();
        let mut outputs = Vec::new();
        for (index, width) in widths.into_iter().enumerate() {
            let bias: Vec<f32> = (0..width)
                .map(|col| ((col + index * 7) % 17) as f32 * 0.015625 - 0.125)
                .collect();
            let values = vec![12345.0_f32; width + 4];
            let d_bias = DeviceBuffer::new_tagged(&model.cuda, bias.len() * 4, "qkv:bias")?;
            let d_output = DeviceBuffer::new_tagged(&model.cuda, values.len() * 4, "qkv:output")?;
            unsafe {
                model
                    .cuda
                    .memcpy_h2d(d_bias.as_mut_ptr(), bias.as_ptr().cast(), d_bias.bytes())?;
                model.cuda.memcpy_h2d(
                    d_output.as_mut_ptr(),
                    values.as_ptr().cast(),
                    d_output.bytes(),
                )?;
            }
            biases.push(d_bias);
            outputs.push(d_output);
        }
        Ok(Self {
            input: device,
            biases,
            outputs,
            widths,
        })
    }

    unsafe fn enqueue(
        &self,
        model: &LoadedModel,
        rows: i32,
        k: i32,
        has_bias: [bool; 3],
    ) -> Result<bool> {
        let weights = fixture::QKV_NAMES.map(|name| model.device_tensors[name].dptr);
        let biases: [_; 3] = std::array::from_fn(|i| has_bias[i].then(|| self.biases[i].as_ptr()));
        model.try_qkv_project_f32xf16_gguf_f32_decode_async(
            self.input.as_ptr(),
            rows,
            k,
            weights[0],
            self.widths[0] as i32,
            weights[1],
            self.widths[1] as i32,
            weights[2],
            self.widths[2] as i32,
            biases[0],
            biases[1],
            biases[2],
            self.outputs[0].as_mut_ptr(),
            self.outputs[1].as_mut_ptr(),
            self.outputs[2].as_mut_ptr(),
        )
    }

    fn read(&self, model: &LoadedModel, has_bias: [bool; 3]) -> Result<Vec<Vec<f32>>> {
        let input = fixture::projection::make_input(1, 64);
        let mut result = Vec::new();
        for (index, name) in fixture::QKV_NAMES.into_iter().enumerate() {
            let mut output = vec![0.0_f32; self.widths[index] + 4];
            unsafe {
                model.cuda.memcpy_d2h(
                    output.as_mut_ptr().cast(),
                    self.outputs[index].as_ptr(),
                    output.len() * 4,
                )?;
            }
            assert_eq!(&output[self.widths[index]..], &[12345.0; 4]);
            output.truncate(self.widths[index]);
            let tensor = &model.device_tensors[name];
            let source = &model.host_weights
                [tensor.byte_offset as usize..tensor.byte_offset as usize + tensor.nbytes];
            for (col, &actual) in output.iter().enumerate() {
                let bias = if has_bias[index] {
                    ((col + index * 7) % 17) as f64 * 0.015625 - 0.125
                } else {
                    0.0
                };
                let expected = fixture::projection::project_cpu(&input, source, 0, col, 64) + bias;
                ensure!(
                    actual.is_finite()
                        && (f64::from(actual) - expected).abs() < 1e-3 + expected.abs() * 1e-5,
                    "QKV CPU mismatch: {name} column={col}"
                );
            }
            result.push(output);
        }
        Ok(result)
    }
}

fn run_qkv(
    model: &LoadedModel,
    buffers: &QkvBuffers,
    stream: CudaStream,
    has_bias: [bool; 3],
) -> Result<Vec<Vec<f32>>> {
    let before = profile::snapshot();
    ensure!(
        unsafe { buffers.enqueue(model, 1, 64, has_bias)? },
        "fused QKV should dispatch"
    );
    let after = profile::snapshot();
    let delta = profile::counter_delta(&before, &after);
    assert_eq!(delta.launches, 1);
    assert_eq!(delta.stream_syncs, 0);
    assert_eq!(delta.device_allocs, 0);
    assert_eq!(delta.device_frees, 0);
    assert_eq!(delta.h2d_copies, 0);
    assert_eq!(delta.d2h_copies, 0);
    let op = if model.q8_execution_stats().mode == Q8ExecutionMode::Off {
        "qkv_f32xq8_0_gguf_decode"
    } else {
        "qkv_f32xq8_separate"
    };
    let old = before.by_op.get(op).map_or(0, |v| v.launches);
    let new = after.by_op.get(op).map_or(0, |v| v.launches);
    assert_eq!(new - old, 1, "wrong QKV execution path");
    model.cuda.synchronize_stream(stream)?;
    buffers.read(model, has_bias)
}

/// Prepared fused QKV preserves biases, unequal widths, tails, and both producer
/// streams while using one launch and no per-call allocation/copy/sync. Native
/// GPU and CPU references agree; invalid shapes and non-decode calls do not launch.
#[test]
fn test_prepared_fused_qkv_matches_native_without_splitting_launches() -> Result<()> {
    let _fused = EnvRestore::set("M40LLM_FUSED_Q8_QKV", "1");
    let initial = CudaContext::total_device_bytes();
    for widths in [[64, 64, 64], [65, 7, 11], [256, 32, 32]] {
        let native = make_model(Q8ExecutionMode::Off, widths)?;
        let props = native.cuda.current_device_props()?;
        ensure!(
            props.major == 5 && props.minor == 2,
            "QKV test requires sm_52"
        );
        let native_buffers = QkvBuffers::new(&native, widths)?;
        for mode in [Q8ExecutionMode::Split, Q8ExecutionMode::Mixed] {
            let model = make_model(mode, widths)?;
            let buffers = QkvBuffers::new(&model, widths)?;
            for stream in [CudaStream::Prefill, CudaStream::Decode] {
                let _stream = EnvRestore::set(
                    "M40LLM_Q8_DECODE_STREAM",
                    if stream == CudaStream::Decode {
                        "decode"
                    } else {
                        "prefill"
                    },
                );
                for has_bias in [[false; 3], [true, false, true], [true; 3]] {
                    let expected = run_qkv(&native, &native_buffers, stream, has_bias)?;
                    let actual = run_qkv(&model, &buffers, stream, has_bias)?;
                    for (actual, expected) in actual.iter().zip(&expected) {
                        let (maximum, mean) = fixture::projection::measure_error(actual, expected);
                        ensure!(maximum < 1e-3 && mean < 1e-4, "native fused QKV mismatch");
                    }
                }
            }
            let before = profile::snapshot();
            assert!(unsafe { buffers.enqueue(&model, 1, 32, [false; 3]) }.is_err());
            assert!(!unsafe { buffers.enqueue(&model, 2, 64, [false; 3])? });
            {
                let _disabled = EnvRestore::set("M40LLM_FUSED_Q8_QKV", "0");
                assert!(!unsafe { buffers.enqueue(&model, 1, 64, [false; 3])? });
            }
            assert_eq!(
                profile::counter_delta(&before, &profile::snapshot()).launches,
                0
            );
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}
