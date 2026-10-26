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
        DeviceQ8ExecutionView, DeviceQ8ExecutionWeight, DeviceQ8FfnTaskPair, HostQ8ExecutionWeight,
        HostQ8FfnTaskPair, Q8MatrixShape, Q8SourceIdentity, Q8SourceView,
    },
    profile,
};
use std::ffi::c_void;
const GUARD: u32 = 0x4b123456;

struct Guarded {
    buffer: DeviceBuffer,
    count: usize,
}
impl Guarded {
    fn new(ctx: &CudaContext, values: &[f32]) -> Result<Self> {
        let mut all = vec![f32::from_bits(GUARD); values.len() + 16];
        all[8..8 + values.len()].copy_from_slice(values);
        let buffer = DeviceBuffer::new_tagged(ctx, all.len() * 4, "task_pair_test:scratch")?;
        unsafe {
            ctx.memcpy_h2d(buffer.as_mut_ptr(), all.as_ptr().cast(), buffer.bytes())?;
        }
        Ok(Self {
            buffer,
            count: values.len(),
        })
    }
    fn pointer(&self) -> *mut c_void {
        unsafe { self.buffer.as_mut_ptr().cast::<f32>().add(8).cast() }
    }
    fn read(&self, ctx: &CudaContext) -> Result<Vec<f32>> {
        let mut all = vec![0.0f32; self.count + 16];
        unsafe {
            ctx.memcpy_d2h(
                all.as_mut_ptr().cast(),
                self.buffer.as_ptr(),
                self.buffer.bytes(),
            )?;
        }
        assert!(all[..8]
            .iter()
            .chain(&all[8 + self.count..])
            .all(|v| v.to_bits() == GUARD));
        Ok(all[8..8 + self.count].to_vec())
    }
}
fn identity(name: &str, k: usize, n: usize) -> Result<Q8SourceIdentity> {
    Q8SourceIdentity::from_column_blocks([7; 32], name.into(), 0, Q8MatrixShape::new(k, n)?)
}
fn planes(weight: &DeviceQ8ExecutionWeight) -> (*const c_void, *const c_void) {
    match weight.view() {
        DeviceQ8ExecutionView::SeparateScalesValues { scales, values, .. } => {
            (scales.as_ptr(), values.as_ptr())
        }
        _ => panic!("expected original separate planes"),
    }
}
fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

fn check_case(ctx: &CudaContext, k: usize, n: usize, rows: usize) -> Result<()> {
    let gi = identity("gate", k, n)?;
    let ui = identity("up", k, n)?;
    let mut gate = fixture::make_column_blocks(k, n);
    let mut up = gate.clone();
    for (index, block) in gate.chunks_exact_mut(34).enumerate() {
        if index % 7 == 0 {
            block[..2].copy_from_slice(&0x8000u16.to_le_bytes());
        }
    }
    for (index, block) in up.chunks_exact_mut(34).enumerate() {
        for byte in &mut block[2..] {
            *byte = byte.wrapping_add(3);
        }
        if index % 11 == 0 {
            block[2..].fill(0);
        }
    }
    let gv = Q8SourceView::from_tensor_bytes(&gi, &gate)?;
    let uv = Q8SourceView::from_tensor_bytes(&ui, &up)?;
    let host = HostQ8FfnTaskPair::pack(gv, uv)?;
    assert_eq!(host.unpack_column_blocks()?, [gate.clone(), up.clone()]);
    let packet = DeviceQ8FfnTaskPair::upload(ctx, &host)?;
    let g =
        DeviceQ8ExecutionWeight::upload(ctx, &HostQ8ExecutionWeight::separate_scales_values(gv)?)?;
    let u =
        DeviceQ8ExecutionWeight::upload(ctx, &HostQ8ExecutionWeight::separate_scales_values(uv)?)?;
    let (gs, gq) = planes(&g);
    let (us, uq) = planes(&u);
    let values: Vec<_> = fixture::make_input(rows, k)
        .iter()
        .enumerate()
        .map(|(i, v)| v * 0.10003 + ((i % 7) as f32 - 3.0) * 0.000017)
        .collect();
    let producer = Guarded::new(ctx, &values)?;
    let input = Guarded::new(ctx, &vec![0.0; values.len()])?;
    let reference = Guarded::new(ctx, &vec![0.0; rows * n])?;
    let reference_gate = Guarded::new(ctx, &vec![0.0; rows * n])?;
    let reference_up = Guarded::new(ctx, &vec![0.0; rows * n])?;
    let output = Guarded::new(ctx, &vec![0.0; rows * n])?;
    let capture_gate = Guarded::new(ctx, &vec![0.0; rows * n])?;
    let capture_up = Guarded::new(ctx, &vec![0.0; rows * n])?;
    unsafe {
        ctx.ffn_q8_split_f32_async(
            producer.pointer(),
            [gs, us],
            [gq, uq],
            reference.pointer(),
            rows as i32,
            n as i32,
            k as i32,
            Some((reference_gate.pointer(), reference_up.pointer())),
        )?;
    }
    ctx.synchronize_stream(CudaStream::Decode)?;
    let expected = reference.read(ctx)?;
    let expected_g = reference_gate.read(ctx)?;
    let expected_u = reference_up.read(ctx)?;
    for _ in 0..3 {
        unsafe {
            ctx.memcpy_d2d_async(
                input.pointer(),
                producer.pointer(),
                values.len() * 4,
                CudaStream::Prefill,
            )?;
        }
        ctx.stream_wait_for_stream(
            CudaStream::Decode,
            CudaStream::Prefill,
            "task_pair_test_input_producer",
        )?;
        let resident = CudaContext::total_device_bytes();
        let before = profile::snapshot();
        unsafe {
            packet.enqueue_async(input.pointer(), output.pointer(), rows as i32, None)?;
        }
        let delta = profile::counter_delta(&before, &profile::snapshot());
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
        ctx.synchronize_stream(CudaStream::Decode)?;
        let ordinary = output.read(ctx)?;
        assert_bits(&ordinary, &expected);
        unsafe {
            packet.enqueue_async(
                input.pointer(),
                output.pointer(),
                rows as i32,
                Some((capture_gate.pointer(), capture_up.pointer())),
            )?;
        }
        ctx.synchronize_stream(CudaStream::Decode)?;
        assert_bits(&output.read(ctx)?, &ordinary);
        assert_bits(&capture_gate.read(ctx)?, &expected_g);
        assert_bits(&capture_up.read(ctx)?, &expected_u);
        for row in 0..rows {
            for col in 0..n {
                let gcpu = fixture::project_cpu(&values, &gate, row, col, k);
                let ucpu = fixture::project_cpu(&values, &up, row, col, k);
                let cpu = gcpu / (1.0 + (-gcpu).exp()) * ucpu;
                for (actual, expected) in [
                    (expected_g[row * n + col], gcpu),
                    (expected_u[row * n + col], ucpu),
                    (ordinary[row * n + col], cpu),
                ] {
                    assert!((f64::from(actual) - expected).abs() < 1e-3 + 2e-5 * expected.abs());
                }
            }
        }
    }
    assert_bits(&input.read(ctx)?, &values);
    assert_bits(&producer.read(ctx)?, &values);
    for (device, original) in packet.planes().into_iter().zip(host.planes()) {
        let mut actual = vec![0u8; device.bytes()];
        unsafe {
            ctx.memcpy_d2h(actual.as_mut_ptr().cast(), device.as_ptr(), actual.len())?;
        }
        assert_eq!(actual, original);
    }
    Ok(())
}

/// Task-pair payloads preserve original control/capture bits and CPU bounds
/// for dynamic-K bundles, odd widths, nonbinary distinct rows and ordered
/// producers, with guarded immutable planes and reusable output scratch.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_task_pairs_preserve_control_bits_and_scratch_ownership() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for (k, n) in [(1024, 1), (2048, 37), (4096, 3)] {
        for rows in [1, 4] {
            check_case(&ctx, k, n, rows)?;
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Bad rows, null/alignment/address errors and partial input/output/capture
/// overlaps fail before GPU work or mutation rather than using a fallback.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_task_pairs_reject_invalid_spans_without_gpu_work() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let g = identity("gate", 1024, 3)?;
        let u = identity("up", 1024, 3)?;
        let raw = fixture::make_column_blocks(1024, 3);
        let packet = DeviceQ8FfnTaskPair::upload(
            &ctx,
            &HostQ8FfnTaskPair::pack(
                Q8SourceView::from_tensor_bytes(&g, &raw)?,
                Q8SourceView::from_tensor_bytes(&u, &raw)?,
            )?,
        )?;
        let input = Guarded::new(&ctx, &[1.25; 1024])?;
        let out = Guarded::new(&ctx, &[3.75; 3])?;
        let cap = Guarded::new(&ctx, &[2.5; 3])?;
        let before = profile::snapshot();
        unsafe {
            for rows in [0, 2, 65536] {
                assert!(packet
                    .enqueue_async(input.pointer(), out.pointer(), rows, None)
                    .is_err());
            }
            for bad in [
                std::ptr::null(),
                input.pointer().cast::<u8>().add(4).cast(),
                (usize::MAX - 15) as *const c_void,
            ] {
                assert!(packet.enqueue_async(bad, out.pointer(), 1, None).is_err());
            }
            for bad in [
                std::ptr::null_mut(),
                out.pointer().cast::<u8>().add(2).cast(),
                input.pointer().cast::<u8>().add(16).cast(),
                packet.planes()[0].as_mut_ptr(),
            ] {
                assert!(packet.enqueue_async(input.pointer(), bad, 1, None).is_err());
            }
            for captures in [
                (out.pointer(), cap.pointer()),
                (cap.pointer(), cap.pointer()),
                (cap.pointer(), std::ptr::null_mut()),
                (out.pointer().cast::<u8>().add(4).cast(), cap.pointer()),
            ] {
                assert!(packet
                    .enqueue_async(input.pointer(), out.pointer(), 1, Some(captures))
                    .is_err());
            }
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()),
            Default::default()
        );
        assert_eq!(out.read(&ctx)?, vec![3.75; 3]);
        assert_eq!(cap.read(&ctx)?, vec![2.5; 3]);
        assert_eq!(input.read(&ctx)?, vec![1.25; 1024]);
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}
