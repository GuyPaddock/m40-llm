#![cfg(all(feature = "cuda", nvcc))]
#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/q8_projection.rs"]
#[allow(dead_code)]
mod fixture;

use anyhow::Result;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    infer::execution_weights::{
        DeviceQ8ExecutionWeight, DeviceQ8WarpPacket, HostQ8ExecutionWeight, HostQ8WarpPacket,
        Q8MatrixShape, Q8SeparateLaunch, Q8SourceIdentity, Q8SourceView,
    },
    profile,
};
const GUARD: u32 = 0x4b123456;

fn upload(ctx: &CudaContext, values: &[f32]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new_tagged(ctx, values.len() * 4, "packet_test:scratch")?;
    unsafe {
        ctx.memcpy_h2d(buffer.as_mut_ptr(), values.as_ptr().cast(), buffer.bytes())?;
    }
    Ok(buffer)
}
fn read(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
    let mut values = vec![0.0; buffer.bytes() / 4];
    unsafe {
        ctx.memcpy_d2h(values.as_mut_ptr().cast(), buffer.as_ptr(), buffer.bytes())?;
    }
    Ok(values)
}

fn check_case(ctx: &CudaContext, k: usize, n: usize, rows: usize) -> Result<()> {
    let source = Q8SourceIdentity::from_column_blocks(
        [7; 32],
        "fixture".into(),
        0,
        Q8MatrixShape::new(k, n)?,
    )?;
    let mut raw = fixture::make_column_blocks(k, n);
    for (index, block) in raw.chunks_exact_mut(34).enumerate() {
        if index % 7 == 0 {
            block[..2].copy_from_slice(&0x8000u16.to_le_bytes());
        }
        if index % 11 == 0 {
            block[2..].fill(0);
        }
    }
    let view = Q8SourceView::from_tensor_bytes(&source, &raw)?;
    let host = HostQ8WarpPacket::pack(view)?;
    assert_eq!(host.unpack_column_blocks()?, raw);
    let packet = DeviceQ8WarpPacket::upload(ctx, &host)?;
    let control = DeviceQ8ExecutionWeight::upload(
        ctx,
        &HostQ8ExecutionWeight::separate_scales_values(view)?,
    )?;
    let input: Vec<_> = fixture::make_input(rows, k)
        .iter()
        .enumerate()
        .map(|(i, v)| v * 0.10003 + ((i % 7) as f32 - 3.0) * 0.000017)
        .collect();
    let producer = upload(ctx, &input)?;
    let mut guarded_input = vec![f32::from_bits(GUARD); rows * k + 18];
    guarded_input[9..9 + rows * k].fill(0.0);
    let candidate_input = upload(ctx, &guarded_input)?;
    let input_pointer = unsafe { candidate_input.as_mut_ptr().cast::<f32>().add(9).cast() };
    assert_eq!((input_pointer as usize) % 16, 4);
    let output = upload(ctx, &vec![f32::from_bits(GUARD); rows * n + 16])?;
    let out = unsafe { output.as_mut_ptr().cast::<f32>().add(8).cast() };
    let control_output = upload(ctx, &vec![f32::from_bits(GUARD); rows * n + 8])?;
    for threads in [128, 256] {
        let launch = Q8SeparateLaunch::new(4, threads)?;
        unsafe {
            control.project_separate_async(&producer, &control_output, rows, launch)?;
        }
        ctx.synchronize_stream(CudaStream::Decode)?;
        let expected = read(ctx, &control_output)?;
        assert!(expected[rows * n..].iter().all(|v| v.to_bits() == GUARD));
        for stream in [CudaStream::Decode, CudaStream::Prefill] {
            for _ in 0..3 {
                unsafe {
                    ctx.memcpy_h2d(
                        candidate_input.as_mut_ptr(),
                        guarded_input.as_ptr().cast(),
                        candidate_input.bytes(),
                    )?;
                }
                let other = if stream == CudaStream::Decode {
                    CudaStream::Prefill
                } else {
                    CudaStream::Decode
                };
                unsafe {
                    ctx.memcpy_d2d_async(
                        input_pointer,
                        producer.as_ptr(),
                        producer.bytes(),
                        other,
                    )?;
                }
                ctx.stream_wait_for_stream(stream, other, "packet_test_input_producer")?;
                let resident = CudaContext::total_device_bytes();
                let before = profile::snapshot();
                unsafe {
                    packet.enqueue_async(input_pointer.cast_const(), out, rows, launch, stream)?;
                }
                let after = profile::snapshot();
                let delta = profile::counter_delta(&before, &after);
                assert_eq!(delta.launches, 1);
                assert_eq!(
                    delta.device_allocs
                        + delta.device_frees
                        + delta.h2d_copies
                        + delta.d2h_copies
                        + delta.stream_waits
                        + delta.stream_syncs
                        + delta.cublas_calls,
                    0
                );
                assert_eq!(CudaContext::total_device_bytes(), resident);
                ctx.synchronize_stream(stream)?;
                let actual = read(ctx, &output)?;
                assert!(actual[..8]
                    .iter()
                    .chain(&actual[8 + rows * n..])
                    .all(|v| v.to_bits() == GUARD));
                for row in 0..rows {
                    for col in 0..n {
                        let value = actual[8 + row * n + col];
                        assert_eq!(
                            value.to_bits(),
                            expected[row * n + col].to_bits(),
                            "control bits K={k} N={n} rows={rows} threads={threads}"
                        );
                        let cpu = fixture::project_cpu(&input, &raw, row, col, k);
                        assert!((f64::from(value) - cpu).abs() < 1e-3 + 2e-5 * cpu.abs());
                    }
                }
            }
        }
    }
    let actual_input = read(ctx, &candidate_input)?;
    assert!(actual_input[..9]
        .iter()
        .chain(&actual_input[9 + rows * k..])
        .all(|v| v.to_bits() == GUARD));
    assert!(actual_input[9..9 + rows * k]
        .iter()
        .zip(&input)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    for (device, original) in packet.planes().into_iter().zip(host.planes()) {
        let mut actual = vec![0u8; device.bytes()];
        unsafe {
            ctx.memcpy_d2h(actual.as_mut_ptr().cast(), device.as_ptr(), actual.len())?;
        }
        assert_eq!(actual, original);
    }
    Ok(())
}

/// Packet layout preserves original prepared output bits and CPU bounds for
/// genuine distinct rows, nonbinary input, signed weights, zero scales,
/// scalar-aligned input, short/long tasks, ordered streams and reused scratch.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_packet_preserves_control_bits_and_scratch_ownership() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for (k, n) in [(256, 4), (512, 12), (11008, 8)] {
        for rows in [1, 4] {
            check_case(&ctx, k, n, rows)?;
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Unsupported launches, partial aliases, null/unaligned inputs and address
/// overflow fail before any kernel or buffer mutation, without fallback.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_packet_rejects_invalid_calls_without_gpu_work() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let source = Q8SourceIdentity::from_column_blocks(
            [7; 32],
            "fixture".into(),
            0,
            Q8MatrixShape::new(256, 4)?,
        )?;
        let raw = fixture::make_column_blocks(256, 4);
        let host = HostQ8WarpPacket::pack(Q8SourceView::from_tensor_bytes(&source, &raw)?)?;
        let packet = DeviceQ8WarpPacket::upload(&ctx, &host)?;
        let input = upload(&ctx, &[1.25; 256])?;
        let output = upload(&ctx, &[3.75; 32])?;
        let launch = Q8SeparateLaunch::new(4, 128)?;
        let before = profile::snapshot();
        unsafe {
            for rows in [0, 2, 65536] {
                assert!(packet
                    .enqueue_async(
                        input.as_ptr(),
                        output.as_mut_ptr(),
                        rows,
                        launch,
                        CudaStream::Decode
                    )
                    .is_err());
            }
            assert!(packet
                .enqueue_async(
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    1,
                    Q8SeparateLaunch::new(1, 128)?,
                    CudaStream::Decode
                )
                .is_err());
            for bad in [
                std::ptr::null(),
                input.as_ptr().cast::<u8>().add(1).cast(),
                (usize::MAX - 3) as *const std::ffi::c_void,
            ] {
                assert!(packet
                    .enqueue_async(bad, output.as_mut_ptr(), 1, launch, CudaStream::Decode)
                    .is_err());
            }
            for bad in [
                std::ptr::null_mut(),
                output.as_mut_ptr().cast::<u8>().add(2).cast(),
                input.as_mut_ptr().cast::<u8>().add(4).cast(),
                packet.planes()[0].as_mut_ptr(),
            ] {
                assert!(packet
                    .enqueue_async(input.as_ptr(), bad, 1, launch, CudaStream::Decode)
                    .is_err());
            }
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()),
            Default::default()
        );
        assert_eq!(read(&ctx, &output)?, vec![3.75; 32]);
        assert_eq!(read(&ctx, &input)?, vec![1.25; 256]);
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}
