#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/env_restore.rs"]
mod env_restore;
#[path = "common/q8_projection.rs"]
mod fixture;

use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    infer::{
        execution_weights::{Q8MatrixShape, Q8SourceIdentity, Q8SourceView},
        sparse_weights::{DeviceSparseQ8Weight, HostSparseQ8Weight, SparseQ8Launch},
    },
    profile,
};

fn read(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
    let mut values = vec![0.0; buffer.bytes() / 4];
    unsafe {
        ctx.memcpy_d2h(values.as_mut_ptr().cast(), buffer.as_ptr(), buffer.bytes())?;
    }
    Ok(values)
}

/// All bounded sparse launch variants must match the same zeroed dense matrix
/// and a scalar oracle for ragged K/N, signed data and multiple activation rows.
/// Invalid calls must leave the previous output and launch counts unchanged.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_sparse_q8_matches_pruned_dense_and_scalar() -> Result<()> {
    let _stream = env_restore::EnvRestore::set("M40LLM_Q8_DECODE_STREAM", "decode");
    let _split = env_restore::EnvRestore::set("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4");
    let _tile = env_restore::EnvRestore::set("M40LLM_Q8_DECODE_COL_TILE", "4");
    let ctx = CudaContext::new(-1)?;
    let props = ctx.current_device_props()?;
    ensure!(
        (props.major, props.minor) == (5, 2),
        "sparse Q8 probe requires M40"
    );
    for (rows, k, n) in [
        (1, 1, 1),
        (1, 35, 3),
        (3, 67, 7),
        (4, 129, 31),
        (64, 65, 17),
        (1, 2048, 2048),
        (16, 11008, 33),
        (64, 11008, 17),
    ] {
        let source = Q8SourceIdentity::from_column_blocks(
            [3; 32],
            "sparse-fixture".into(),
            7,
            Q8MatrixShape::new(k, n)?,
        )?;
        let bytes = fixture::make_column_blocks(k, n);
        let masks: Vec<_> = (0..source.shape().block_count()?)
            .map(|i| 0x3333_3333u32.rotate_left(i as u32))
            .collect();
        let host =
            HostSparseQ8Weight::pack(Q8SourceView::from_tensor_bytes(&source, &bytes)?, &masks)?;
        let dense = host.reconstruct_column_blocks();
        let sparse = DeviceSparseQ8Weight::upload(&ctx, &host)?;
        assert_eq!(sparse.storage(), &host.storage());
        let d_dense = DeviceBuffer::new_tagged(&ctx, dense.len(), "sparse-test:dense")?;
        let input = fixture::make_input(rows, k);
        let d_input = DeviceBuffer::new_tagged(&ctx, input.len() * 4, "sparse-test:input")?;
        let output = DeviceBuffer::new_tagged(&ctx, rows * n * 4, "sparse-test:output")?;
        let reference = DeviceBuffer::new_tagged(&ctx, rows * n * 4, "sparse-test:reference")?;
        unsafe {
            ctx.memcpy_h2d(d_dense.as_mut_ptr(), dense.as_ptr().cast(), dense.len())?;
            ctx.memcpy_h2d(d_input.as_mut_ptr(), input.as_ptr().cast(), d_input.bytes())?;
        }
        // Finish the default-stream pageable upload before launching either
        // nonblocking consumer stream. Readback also validates the whole span.
        assert_eq!(read(&ctx, &d_input)?, input);
        unsafe {
            ctx.gemm_f32xq8_0_gguf_f32_async(
                d_input.as_ptr(),
                d_dense.as_ptr(),
                reference.as_mut_ptr(),
                rows as i32,
                n as i32,
                k as i32,
            )?;
        }
        ctx.synchronize_stream(CudaStream::Prefill)?;
        ctx.synchronize_stream(CudaStream::Decode)?;
        let expected = read(&ctx, &reference)?;
        for columns in [1, 4] {
            for threads in [128, 256] {
                let launch = SparseQ8Launch::new(columns, threads)?;
                let before = profile::snapshot();
                unsafe {
                    sparse.project_async(&d_input, &output, rows, launch)?;
                }
                let delta = profile::counter_delta(&before, &profile::snapshot());
                assert_eq!(delta.launches, 1);
                assert_eq!(delta.device_allocs, 0);
                assert_eq!(delta.stream_syncs, 0);
                ctx.synchronize_stream(CudaStream::Decode)?;
                let actual = read(&ctx, &output)?;
                let (maximum, mean) = fixture::measure_error(&actual, &expected);
                ensure!(
                    maximum < 1e-3 && mean < 1e-4,
                    "rows={rows} K={k} N={n} launch={launch:?} max={maximum} mean={mean}"
                );
                for (i, &value) in actual.iter().enumerate() {
                    let cpu = fixture::project_cpu(&input, &dense, i / n, i % n, k);
                    ensure!(
                        (f64::from(value) - cpu).abs() < 1e-3 + cpu.abs() * 1e-5,
                        "sparse scalar mismatch"
                    );
                }
            }
        }
        let small = DeviceBuffer::new_tagged(&ctx, 4, "sparse-test:small")?;
        let launch = SparseQ8Launch::new(4, 256)?;
        let prior = read(&ctx, &output)?;
        let before = profile::snapshot();
        unsafe {
            assert!(sparse.project_async(&d_input, &output, 0, launch).is_err());
            assert!(sparse
                .project_async(&d_input, &output, 65536, launch)
                .is_err());
            assert!(sparse
                .project_async(&d_input, &d_input, rows, launch)
                .is_err());
            if input.len() > 1 {
                assert!(sparse.project_async(&small, &output, rows, launch).is_err());
            }
            if rows * n > 1 {
                assert!(sparse
                    .project_async(&d_input, &small, rows, launch)
                    .is_err());
            }
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()).launches,
            0
        );
        assert_eq!(read(&ctx, &output)?, prior);
        assert!(SparseQ8Launch::new(2, 128).is_err());
        assert!(SparseQ8Launch::new(1, 64).is_err());
    }
    Ok(())
}
