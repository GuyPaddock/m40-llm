#![cfg(all(feature = "cuda", nvcc))]

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
