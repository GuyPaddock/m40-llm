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

fn read(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
    let mut values = vec![0.0; buffer.bytes() / 4];
    unsafe { ctx.memcpy_d2h(values.as_mut_ptr().cast(), buffer.as_ptr(), buffer.bytes())? };
    Ok(values)
}

/// The fixed warp finish must preserve every control bit, CPU tolerance,
/// guards and source planes on both streams, including ragged and real down
/// shapes. A dependent decode consumer must observe prefill-produced output.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_warp_projection_preserves_control_bits_and_ownership() -> Result<()> {
    let _scope = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for (rows, k, n) in [
        (1, 1, 1),
        (4, 35, 3),
        (4, 67, 7),
        (1, 2048, 257),
        (4, 2048, 33),
        (1, 11008, 2048),
        (4, 11008, 2048),
    ] {
        let identity = Q8SourceIdentity::from_column_blocks(
            [3; 32],
            "warp-fixture".into(),
            103,
            Q8MatrixShape::new(k, n)?,
        )?;
        let bytes = fixture::make_column_blocks(k, n);
        let source = Q8SourceView::from_tensor_bytes(&identity, &bytes)?;
        let host = HostQ8ExecutionWeight::separate_scales_values(source)?;
        let weight = DeviceQ8ExecutionWeight::upload(&ctx, &host)?;
        let input: Vec<_> = fixture::make_input(rows, k)
            .into_iter()
            .enumerate()
            .map(|(i, v)| v + (i as f32 * 0.017).sin() * 0.11)
            .collect();
        let d_input = DeviceBuffer::new_tagged(&ctx, input.len() * 4, "warp-test:input")?;
        let control = DeviceBuffer::new_tagged(&ctx, rows * n * 4, "warp-test:control")?;
        let output =
            DeviceBuffer::new_tagged(&ctx, (rows * n + 2 * GUARD) * 4, "warp-test:output")?;
        let consumer = DeviceBuffer::new_tagged(&ctx, rows * n * 4, "warp-test:consumer")?;
        let guards = vec![f32::from_bits(GUARD_BITS); output.bytes() / 4];
        unsafe { ctx.memcpy_h2d(d_input.as_mut_ptr(), input.as_ptr().cast(), d_input.bytes())? };
        let out = unsafe { output.as_mut_ptr().cast::<f32>().add(GUARD).cast() };
        for threads in [128, 256] {
            let config = Q8SeparateLaunch::new(4, threads)?;
            unsafe { weight.project_separate_async(&d_input, &control, rows, config)? };
            ctx.synchronize_stream(CudaStream::Decode)?;
            let expected = read(&ctx, &control)?;
            for stream in [CudaStream::Decode, CudaStream::Prefill] {
                unsafe {
                    ctx.memcpy_h2d(output.as_mut_ptr(), guards.as_ptr().cast(), output.bytes())?
                };
                let before = profile::snapshot();
                unsafe {
                    weight.enqueue_separate_warp_async(
                        d_input.as_ptr(),
                        out,
                        rows,
                        config,
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
                    ctx.stream_wait_for_stream(CudaStream::Decode, stream, "warp-test:consumer")?;
                }
                unsafe { ctx.residual_add_f32_async(out, out, consumer.as_mut_ptr(), rows * n)? };
                ctx.synchronize_stream(CudaStream::Decode)?;
                ctx.synchronize_stream(CudaStream::Prefill)?;
                let actual = read(&ctx, &output)?;
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
                let consumed = read(&ctx, &consumer)?;
                assert!(consumed
                    .iter()
                    .zip(actual)
                    .all(|(a, b)| a.to_bits() == (b + b).to_bits()));
                for row in [0, rows - 1] {
                    for column in [0, n / 2, n - 1] {
                        let cpu = fixture::project_cpu(&input, &bytes, row, column, k);
                        ensure!(
                            (f64::from(actual[row * n + column]) - cpu).abs()
                                < 1e-3 + cpu.abs() * 2e-5,
                            "CPU mismatch at row={row} col={column}"
                        );
                    }
                }
            }
        }
        assert_eq!(unsafe { weight.read_planes()? }, host.planes());
        assert_eq!(read(&ctx, &d_input)?, input);
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Malformed metadata, wrong layout, null/aliased/misaligned pointers must be
/// rejected before any enqueue or allocation; the previous output is retained.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_warp_projection_rejects_invalid_calls_without_effects() -> Result<()> {
    let _scope = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let identity = Q8SourceIdentity::from_column_blocks(
        [3; 32],
        "invalid-warp".into(),
        0,
        Q8MatrixShape::new(32, 4)?,
    )?;
    let bytes = fixture::make_column_blocks(32, 4);
    let source = Q8SourceView::from_tensor_bytes(&identity, &bytes)?;
    let split = DeviceQ8ExecutionWeight::upload(
        &ctx,
        &HostQ8ExecutionWeight::separate_scales_values(source)?,
    )?;
    let original =
        DeviceQ8ExecutionWeight::upload(&ctx, &HostQ8ExecutionWeight::copy_column_blocks(source)?)?;
    let input = DeviceBuffer::new_tagged(&ctx, 128, "invalid-warp:input")?;
    let output = DeviceBuffer::new_tagged(&ctx, 16, "invalid-warp:output")?;
    let prior = [17.0f32; 4];
    unsafe { ctx.memcpy_h2d(output.as_mut_ptr(), prior.as_ptr().cast(), output.bytes())? };
    let config = Q8SeparateLaunch::new(4, 256)?;
    let before = profile::snapshot();
    for rows in [0, 65536] {
        assert!(unsafe {
            split.enqueue_separate_warp_async(
                input.as_ptr(),
                output.as_mut_ptr(),
                rows,
                config,
                CudaStream::Decode,
            )
        }
        .is_err());
    }
    for (weight, inp, out, launch) in [
        (&original, input.as_ptr(), output.as_mut_ptr(), config),
        (&split, std::ptr::null(), output.as_mut_ptr(), config),
        (&split, input.as_ptr(), std::ptr::null_mut(), config),
        (&split, input.as_ptr(), input.as_mut_ptr(), config),
        (
            &split,
            unsafe { input.as_ptr().cast::<u8>().add(1).cast() },
            output.as_mut_ptr(),
            config,
        ),
        (
            &split,
            input.as_ptr(),
            output.as_mut_ptr(),
            Q8SeparateLaunch::new(1, 128)?,
        ),
    ] {
        assert!(unsafe {
            weight.enqueue_separate_warp_async(inp, out, 1, launch, CudaStream::Decode)
        }
        .is_err());
    }
    let delta = profile::counter_delta(&before, &profile::snapshot());
    assert_eq!(delta.launches, 0);
    assert_eq!(delta.device_allocs, 0);
    assert_eq!(delta.stream_syncs, 0);
    assert_eq!(read(&ctx, &output)?, prior);
    Ok(())
}
