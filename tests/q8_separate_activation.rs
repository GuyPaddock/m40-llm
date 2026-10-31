#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/q8_projection.rs"]
mod fixture;

use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    infer::execution_weights::{
        DeviceQ8ExecutionWeight, HostQ8ExecutionWeight, Q8MatrixShape, Q8SeparateLaunch,
        Q8SourceIdentity, Q8SourceView,
    },
    profile,
};

const GUARD: usize = 8;
const GUARD_BITS: u32 = 0x4b12_3456;

fn read_f32(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
    let mut values = vec![0.0; buffer.bytes() / 4];
    unsafe { ctx.memcpy_d2h(values.as_mut_ptr().cast(), buffer.as_ptr(), buffer.bytes())? };
    Ok(values)
}

/// Vector activation loads retain exact shared-tree control bits and CPU
/// tolerance for distinct rows, odd output tails and real down widths. Async
/// producer/consumer dependencies, guards, input/weight ownership and cleanup
/// must remain intact without hidden allocations, copies or synchronizations.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_packed_activations_preserve_control_bits_and_ownership() -> Result<()> {
    let _scope = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for (rows, k, n) in [
        (1, 32, 1),
        (4, 64, 3),
        (4, 96, 7),
        (1, 2048, 257),
        (4, 2048, 33),
        (1, 11008, 2048),
        (4, 11008, 2048),
    ] {
        let id = Q8SourceIdentity::from_column_blocks(
            [3; 32],
            "activation-probe".into(),
            103,
            Q8MatrixShape::new(k, n)?,
        )?;
        let bytes = fixture::make_column_blocks(k, n);
        let source = Q8SourceView::from_tensor_bytes(&id, &bytes)?;
        let host = HostQ8ExecutionWeight::separate_scales_values(source)?;
        let weight = DeviceQ8ExecutionWeight::upload(&ctx, &host)?;
        let input: Vec<_> = fixture::make_input(rows, k)
            .into_iter()
            .enumerate()
            .map(|(i, v)| v + (i as f32 * 0.017).sin() * 0.11)
            .collect();
        let device = DeviceBuffer::new_tagged(&ctx, input.len() * 4, "activation-probe:input")?;
        let producer =
            DeviceBuffer::new_tagged(&ctx, input.len() * 4, "activation-probe:producer")?;
        let zero = DeviceBuffer::new_tagged(&ctx, input.len() * 4, "activation-probe:zero")?;
        let control = DeviceBuffer::new_tagged(&ctx, rows * n * 4, "activation-probe:control")?;
        let output =
            DeviceBuffer::new_tagged(&ctx, (rows * n + 2 * GUARD) * 4, "activation-probe:output")?;
        let consumer = DeviceBuffer::new_tagged(&ctx, rows * n * 4, "activation-probe:consumer")?;
        let guards = vec![f32::from_bits(GUARD_BITS); output.bytes() / 4];
        let zeros = vec![0.0f32; input.len()];
        unsafe {
            ctx.memcpy_h2d(device.as_mut_ptr(), input.as_ptr().cast(), device.bytes())?;
            ctx.memcpy_h2d(zero.as_mut_ptr(), zeros.as_ptr().cast(), zero.bytes())?;
        }
        let out = unsafe { output.as_mut_ptr().cast::<f32>().add(GUARD).cast() };
        for threads in [128, 256] {
            let launch = Q8SeparateLaunch::new(4, threads)?;
            unsafe { weight.project_separate_async(&device, &control, rows, launch)? };
            ctx.synchronize_stream(CudaStream::Decode)?;
            let expected = read_f32(&ctx, &control)?;
            for stream in [CudaStream::Decode, CudaStream::Prefill] {
                unsafe {
                    ctx.memcpy_h2d(output.as_mut_ptr(), guards.as_ptr().cast(), output.bytes())?;
                    ctx.residual_add_f32_async(
                        device.as_ptr(),
                        zero.as_ptr(),
                        producer.as_mut_ptr(),
                        input.len(),
                    )?;
                }
                if stream == CudaStream::Prefill {
                    ctx.stream_wait_for_stream(
                        stream,
                        CudaStream::Decode,
                        "activation-probe:producer",
                    )?;
                }
                let before = profile::snapshot();
                unsafe {
                    weight.enqueue_separate_packed_activation_async(
                        producer.as_ptr(),
                        out,
                        rows,
                        launch,
                        stream,
                    )?
                };
                let delta = profile::counter_delta(&before, &profile::snapshot());
                assert_eq!(delta.launches, 1);
                assert_eq!(delta.stream_syncs, 0);
                assert_eq!(delta.device_allocs, 0);
                assert_eq!(delta.device_frees, 0);
                assert_eq!(delta.h2d_copies, 0);
                assert_eq!(delta.d2h_copies, 0);
                if stream == CudaStream::Prefill {
                    ctx.stream_wait_for_stream(
                        CudaStream::Decode,
                        stream,
                        "activation-probe:consumer",
                    )?;
                }
                unsafe { ctx.residual_add_f32_async(out, out, consumer.as_mut_ptr(), rows * n)? };
                ctx.synchronize_stream(CudaStream::Decode)?;
                ctx.synchronize_stream(CudaStream::Prefill)?;
                let actual = read_f32(&ctx, &output)?;
                assert!(actual[..GUARD]
                    .iter()
                    .chain(&actual[GUARD + rows * n..])
                    .all(|v| v.to_bits() == GUARD_BITS));
                let actual = &actual[GUARD..GUARD + rows * n];
                let (max, mean) = fixture::measure_error(actual, &expected);
                assert_eq!(
                    max, 0.0,
                    "rows={rows} K={k} N={n} threads={threads} mean={mean}"
                );
                assert!(actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
                assert!(read_f32(&ctx, &consumer)?
                    .iter()
                    .zip(actual)
                    .all(|(a, b)| a.to_bits() == (b + b).to_bits()));
                for row in [0, rows - 1] {
                    for col in [0, n / 2, n - 1] {
                        let cpu = fixture::project_cpu(&input, &bytes, row, col, k);
                        ensure!(
                            (f64::from(actual[row * n + col]) - cpu).abs()
                                < 1e-3 + cpu.abs() * 2e-5,
                            "CPU mismatch row={row} col={col}"
                        );
                    }
                }
            }
        }
        assert_eq!(unsafe { weight.read_planes()? }, host.planes());
        assert_eq!(read_f32(&ctx, &device)?, input);
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Incomplete Q8 blocks and pointers that are only scalar-aligned must fail
/// without enqueueing work, changing output or allocating fallback storage.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_packed_activations_reject_invalid_calls_without_effects() -> Result<()> {
    let _scope = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let input = DeviceBuffer::new_tagged(&ctx, 256, "activation-invalid:input")?;
    let output = DeviceBuffer::new_tagged(&ctx, 16, "activation-invalid:output")?;
    let prior = [17.0f32; 4];
    unsafe { ctx.memcpy_h2d(output.as_mut_ptr(), prior.as_ptr().cast(), output.bytes())? };
    for k in [32, 35] {
        let id = Q8SourceIdentity::from_column_blocks(
            [3; 32],
            "invalid-activation".into(),
            0,
            Q8MatrixShape::new(k, 4)?,
        )?;
        let bytes = fixture::make_column_blocks(k, 4);
        let source = Q8SourceView::from_tensor_bytes(&id, &bytes)?;
        let split = DeviceQ8ExecutionWeight::upload(
            &ctx,
            &HostQ8ExecutionWeight::separate_scales_values(source)?,
        )?;
        let raw = DeviceQ8ExecutionWeight::upload(
            &ctx,
            &HostQ8ExecutionWeight::copy_column_blocks(source)?,
        )?;
        let launch = Q8SeparateLaunch::new(4, 256)?;
        let before = profile::snapshot();
        for rows in [0, 65536] {
            assert!(unsafe {
                split.enqueue_separate_packed_activation_async(
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    rows,
                    launch,
                    CudaStream::Decode,
                )
            }
            .is_err());
        }
        for (weight, inp, out, config) in [
            (&raw, input.as_ptr(), output.as_mut_ptr(), launch),
            (&split, std::ptr::null(), output.as_mut_ptr(), launch),
            (&split, input.as_ptr(), std::ptr::null_mut(), launch),
            (&split, input.as_ptr(), input.as_mut_ptr(), launch),
            (
                &split,
                input.as_ptr(),
                unsafe { output.as_mut_ptr().cast::<u8>().add(1).cast() },
                launch,
            ),
            (
                &split,
                input.as_ptr(),
                output.as_mut_ptr(),
                Q8SeparateLaunch::new(1, 128)?,
            ),
        ] {
            assert!(unsafe {
                weight.enqueue_separate_packed_activation_async(
                    inp,
                    out,
                    1,
                    config,
                    CudaStream::Decode,
                )
            }
            .is_err());
        }
        for offset in [1, 4, 8, 12] {
            let inp = unsafe { input.as_ptr().cast::<u8>().add(offset).cast() };
            assert!(unsafe {
                split.enqueue_separate_packed_activation_async(
                    inp,
                    output.as_mut_ptr(),
                    1,
                    launch,
                    CudaStream::Decode,
                )
            }
            .is_err());
        }
        if k % 32 != 0 {
            assert!(unsafe {
                split.enqueue_separate_packed_activation_async(
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    1,
                    launch,
                    CudaStream::Decode,
                )
            }
            .is_err());
        }
        let delta = profile::counter_delta(&before, &profile::snapshot());
        assert_eq!(delta.launches, 0);
        assert_eq!(delta.device_allocs, 0);
        assert_eq!(delta.stream_syncs, 0);
        assert_eq!(read_f32(&ctx, &output)?, prior);
    }
    Ok(())
}
