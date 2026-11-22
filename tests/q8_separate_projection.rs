#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/env_restore.rs"]
mod env_restore;
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

fn read_output(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
    let mut output = vec![0.0; buffer.bytes() / 4];
    unsafe {
        ctx.memcpy_d2h(output.as_mut_ptr().cast(), buffer.as_ptr(), buffer.bytes())?;
    }
    Ok(output)
}

/// All four launch shapes must match the GGUF/CPU projections for ragged K/N,
/// negative/zero scales, signed extremes, and distinct rows. Invalid calls must
/// not enqueue work or change the prior output.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_separate_projection_matches_source_and_rejects_invalid_buffers() -> Result<()> {
    let _scope = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new(-1)?;
    let props = ctx.current_device_props()?;
    ensure!(
        props.major == 5 && props.minor == 2,
        "separate Q8 test requires M40"
    );
    for (rows, k, n) in [
        (1, 1, 1),
        (1, 35, 3),
        (3, 67, 7),
        (4, 129, 31),
        (64, 65, 17),
        (1, 2048, 2048),
    ] {
        let shape = Q8MatrixShape::new(k, n)?;
        let identity = Q8SourceIdentity::from_column_blocks([1; 32], "fixture".into(), 103, shape)?;
        let bytes = fixture::make_column_blocks(k, n);
        let source = Q8SourceView::from_tensor_bytes(&identity, &bytes)?;
        let host = HostQ8ExecutionWeight::separate_scales_values(source)?;
        let split = DeviceQ8ExecutionWeight::upload(&ctx, &host)?;
        let original = DeviceQ8ExecutionWeight::upload(
            &ctx,
            &HostQ8ExecutionWeight::copy_column_blocks(source)?,
        )?;
        let input = fixture::make_input(rows, k);
        let d_input = DeviceBuffer::new_tagged(&ctx, input.len() * 4, "split_test:input")?;
        let d_output = DeviceBuffer::new_tagged(&ctx, rows * n * 4, "split_test:output")?;
        let d_reference = DeviceBuffer::new_tagged(&ctx, d_output.bytes(), "split_test:reference")?;
        let m40_llm::infer::execution_weights::DeviceQ8ExecutionView::ColumnBlocks {
            blocks, ..
        } = original.view()
        else {
            unreachable!()
        };
        unsafe {
            ctx.memcpy_h2d(d_input.as_mut_ptr(), input.as_ptr().cast(), d_input.bytes())?;
            ctx.gemm_f32xq8_0_gguf_f32_async(
                d_input.as_ptr(),
                blocks.as_ptr(),
                d_reference.as_mut_ptr(),
                rows as i32,
                n as i32,
                k as i32,
            )?;
        }
        ctx.synchronize_stream(CudaStream::Prefill)?;
        ctx.synchronize_stream(CudaStream::Decode)?;
        let expected = read_output(&ctx, &d_reference)?;
        for columns in [1, 4] {
            for threads in [128, 256] {
                let config = Q8SeparateLaunch::new(columns, threads)?;
                let before = profile::snapshot();
                unsafe {
                    split.project_separate_async(&d_input, &d_output, rows, config)?;
                }
                let delta = profile::counter_delta(&before, &profile::snapshot());
                assert_eq!(delta.stream_syncs, 0);
                assert_eq!(delta.device_allocs, 0);
                ctx.synchronize_stream(CudaStream::Decode)?;
                let actual = read_output(&ctx, &d_output)?;
                let (maximum, mean) = fixture::measure_error(&actual, &expected);
                ensure!(
                    maximum < 1e-3 && mean < 1e-4,
                    "rows={rows} K={k} N={n} config={config:?} max={maximum} mean={mean}"
                );
                for (index, &value) in actual.iter().enumerate() {
                    let cpu = fixture::project_cpu(&input, &bytes, index / n, index % n, k);
                    ensure!(
                        (f64::from(value) - cpu).abs() < 1e-3 + cpu.abs() * 1e-5,
                        "CPU projection mismatch"
                    );
                }
            }
        }
        let config = Q8SeparateLaunch::new(4, 256)?;
        let tiny = DeviceBuffer::new_tagged(&ctx, 4, "split_test:tiny")?;
        let prior = read_output(&ctx, &d_output)?;
        let before = profile::snapshot();
        unsafe {
            assert!(split
                .project_separate_async(&d_input, &d_output, 0, config)
                .is_err());
            assert!(split
                .project_separate_async(&d_input, &d_output, 65536, config)
                .is_err());
            assert!(original
                .project_separate_async(&d_input, &d_output, rows, config)
                .is_err());
            if rows * k > 1 {
                assert!(split
                    .project_separate_async(&tiny, &d_output, rows, config)
                    .is_err());
            }
            if rows * n > 1 {
                assert!(split
                    .project_separate_async(&d_input, &tiny, rows, config)
                    .is_err());
            }
            assert!(split
                .project_separate_async(&d_input, &d_input, rows, config)
                .is_err());
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()).launches,
            0
        );
        assert_eq!(read_output(&ctx, &d_output)?, prior);
    }
    Ok(())
}

/// The existing 4-column/256-thread split-plane launch must preserve the
/// selected native coltile4/split4 accumulation order at aligned decode shapes.
/// Non-binary activations expose the 128-thread order difference; matching
/// payload values alone is not enough to establish bitwise native parity.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_separate_256_preserves_native_decode_accumulation_bits() -> Result<()> {
    let _scope = cuda_test_scope::lock_cuda();
    let _env: Vec<_> = [
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_Q8_DECODE_COL_TILE", "4"),
        ("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4"),
    ]
    .into_iter()
    .map(|(key, value)| env_restore::EnvRestore::set(key, value))
    .collect();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    let mut different_128_outputs = 0;
    for (k, n) in [(2048, 33), (11008, 2048)] {
        let shape = Q8MatrixShape::new(k, n)?;
        let identity =
            Q8SourceIdentity::from_column_blocks([1; 32], "native-order".into(), 0, shape)?;
        let bytes = fixture::make_column_blocks(k, n);
        let source = Q8SourceView::from_tensor_bytes(&identity, &bytes)?;
        let native = DeviceQ8ExecutionWeight::upload(
            &ctx,
            &HostQ8ExecutionWeight::copy_column_blocks(source)?,
        )?;
        let split = DeviceQ8ExecutionWeight::upload(
            &ctx,
            &HostQ8ExecutionWeight::separate_scales_values(source)?,
        )?;
        let input: Vec<f32> = (0..k)
            .map(|i| (i as f32 * 0.017).sin() * 0.73 + (i as f32 * 0.031).cos() * 0.11)
            .collect();
        let d_input = DeviceBuffer::new_tagged(&ctx, k * 4, "native_order:input")?;
        let expected = DeviceBuffer::new_tagged(&ctx, n * 4, "native_order:reference")?;
        let output = DeviceBuffer::new_tagged(&ctx, n * 4, "native_order:output")?;
        let m40_llm::infer::execution_weights::DeviceQ8ExecutionView::ColumnBlocks {
            blocks, ..
        } = native.view()
        else {
            unreachable!()
        };
        unsafe {
            ctx.memcpy_h2d(d_input.as_mut_ptr(), input.as_ptr().cast(), d_input.bytes())?;
            ctx.gemm_f32xq8_0_gguf_f32_decode_async(
                d_input.as_ptr(),
                blocks.as_ptr(),
                expected.as_mut_ptr(),
                1,
                n as i32,
                k as i32,
            )?;
        }
        ctx.synchronize_stream(CudaStream::Decode)?;
        ctx.synchronize_stream(CudaStream::Prefill)?;
        let expected = read_output(&ctx, &expected)?;
        for threads in [128, 256] {
            unsafe {
                split.project_separate_async(
                    &d_input,
                    &output,
                    1,
                    Q8SeparateLaunch::new(4, threads)?,
                )?;
            }
            ctx.synchronize_stream(CudaStream::Decode)?;
            let actual = read_output(&ctx, &output)?;
            let different = actual
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let (maximum, mean) = fixture::measure_error(&actual, &expected);
            eprintln!("native-order K={k} N={n} threads={threads} different={different}/{n} max={maximum:e} mean={mean:e}");
            if threads == 256 {
                assert_eq!(
                    different, 0,
                    "same native thread/task/reduction order must preserve every FP32 output bit"
                );
            } else {
                different_128_outputs += different;
            }
        }
    }
    assert!(
        different_128_outputs > 0,
        "non-binary inputs must exercise a real reduction-order difference"
    );
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Weight reuse must preserve each serial output bit and leave the following
/// guard words intact when the last quantization block, output tile and input
/// row group are all partial. Include exact-zero scales with nonzero codes.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_small_row_weight_reuse_preserves_serial_bits_and_guards() -> Result<()> {
    let _scope = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let (k, n) = (2083usize, 7usize);
    let shape = Q8MatrixShape::new(k, n)?;
    let identity = Q8SourceIdentity::from_column_blocks([3; 32], "small-rows".into(), 0, shape)?;
    let mut bytes = fixture::make_column_blocks(k, n);
    for block in bytes.chunks_exact_mut(34).step_by(3) {
        block[..2].fill(0);
    }
    let split = DeviceQ8ExecutionWeight::upload(
        &ctx,
        &HostQ8ExecutionWeight::separate_scales_values(Q8SourceView::from_tensor_bytes(
            &identity, &bytes,
        )?)?,
    )?;
    for rows in [1, 2, 3, 4, 5, 7, 8, 16] {
        let mut input = fixture::make_input(rows, k);
        input[..k].fill(0.);
        for row in input.chunks_exact_mut(k) {
            row[64..96].fill(0.);
        }
        let expected = DeviceBuffer::new_tagged(&ctx, rows * n * 4, "small_rows:reference")?;
        let input_device = DeviceBuffer::new_tagged(&ctx, input.len() * 4, "small_rows:inputs")?;
        let sentinel = 1337.25_f32;
        let initial = vec![sentinel; rows * n + 15];
        let output = DeviceBuffer::new_tagged(&ctx, initial.len() * 4, "small_rows:guarded")?;
        unsafe {
            ctx.memcpy_h2d(
                input_device.as_mut_ptr(),
                input.as_ptr().cast(),
                input_device.bytes(),
            )?;
            ctx.memcpy_h2d(output.as_mut_ptr(), initial.as_ptr().cast(), output.bytes())?;
            split.project_serial_rows_async(&input_device, &expected, rows)?;
            split.project_small_rows_async(&input_device, &output, rows)?;
        }
        ctx.synchronize_stream(CudaStream::Decode)?;
        let actual = read_output(&ctx, &output)?;
        let reference = read_output(&ctx, &expected)?;
        assert_eq!(&actual[rows * n..], &initial[rows * n..]);
        assert!(actual[..n].iter().all(|v| *v == 0.));
        for (index, (a, b)) in actual[..rows * n].iter().zip(&reference).enumerate() {
            assert!(a.is_finite());
            assert_eq!(a.to_bits(), b.to_bits(), "rows={rows} element={index}");
            let cpu = fixture::project_cpu(&input, &bytes, index / n, index % n, k);
            assert!((f64::from(*a) - cpu).abs() < 1e-3 + cpu.abs() * 2e-5);
        }
    }
    Ok(())
}
