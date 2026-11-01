#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/q8_projection.rs"]
mod fixture;

use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    infer::execution_weights::{
        DeviceQ8ExecutionView, DeviceQ8ExecutionWeight, HostQ8ExecutionWeight, Q8MatrixShape,
        Q8SourceIdentity, Q8SourceView, Q8TiledLaunch,
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

/// Tiled projections preserve source math through warp/CTA reductions for all
/// six retained vector configurations, including K/N tails and multiple rows. Guard outputs
/// catch padded-column writes; invalid calls must leave prior output unchanged.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_input_tiled_projection_matches_cpu_and_gguf_with_guarded_tails() -> Result<()> {
    let ctx = CudaContext::new(-1)?;
    let props = ctx.current_device_props()?;
    ensure!(
        props.major == 5 && props.minor == 2,
        "input-tiled Q8 parity requires M40"
    );
    for (rows, k, n) in [
        (1, 1, 1),
        (1, 35, 3),
        (3, 67, 7),
        (4, 129, 31),
        (64, 65, 33),
        (1, 2048, 2048),
    ] {
        let shape = Q8MatrixShape::new(k, n)?;
        let identity = Q8SourceIdentity::from_column_blocks([2; 32], "fixture".into(), 103, shape)?;
        let bytes = fixture::make_column_blocks(k, n);
        let source = Q8SourceView::from_tensor_bytes(&identity, &bytes)?;
        let original = DeviceQ8ExecutionWeight::upload(
            &ctx,
            &HostQ8ExecutionWeight::copy_column_blocks(source)?,
        )?;
        let DeviceQ8ExecutionView::ColumnBlocks { blocks, .. } = original.view() else {
            unreachable!()
        };
        let input = fixture::make_input(rows, k);
        let d_input = DeviceBuffer::new_tagged(&ctx, input.len() * 4, "tiled_test:input")?;
        let d_reference = DeviceBuffer::new_tagged(&ctx, rows * n * 4, "tiled_test:reference")?;
        let d_output = DeviceBuffer::new_tagged(&ctx, (rows * n + 16) * 4, "tiled_test:output")?;
        let poison = vec![f32::from_bits(0x7fc0_1234); rows * n + 16];
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
        for columns in [4, 8, 16] {
            let host = HostQ8ExecutionWeight::tile_inputs(source, columns)?;
            let tiled = DeviceQ8ExecutionWeight::upload(&ctx, &host)?;
            assert_eq!(unsafe { tiled.read_planes()? }, host.planes());
            drop(host);
            for threads in [128, 256] {
                let config = Q8TiledLaunch::new(columns as u32, threads)?;
                unsafe {
                    ctx.memcpy_h2d(
                        d_output.as_mut_ptr(),
                        poison.as_ptr().cast(),
                        d_output.bytes(),
                    )?;
                }
                let before = profile::snapshot();
                unsafe {
                    tiled.project_tiled_async(&d_input, &d_output, rows, config)?;
                }
                let delta = profile::counter_delta(&before, &profile::snapshot());
                assert_eq!(delta.launches, 1);
                assert_eq!(delta.stream_syncs, 0);
                assert_eq!(delta.device_allocs, 0);
                ctx.synchronize_stream(CudaStream::Decode)?;
                let actual = read_output(&ctx, &d_output)?;
                assert!(actual[rows * n..]
                    .iter()
                    .all(|v| v.to_bits() == poison[0].to_bits()));
                let (maximum, mean) = fixture::measure_error(&actual[..rows * n], &expected);
                ensure!(
                    maximum < 1e-3 && mean < 1e-4,
                    "rows={rows} K={k} N={n} config={config:?} max={maximum} mean={mean}"
                );
                for (index, &value) in actual[..rows * n].iter().enumerate() {
                    let cpu = fixture::project_cpu(&input, &bytes, index / n, index % n, k);
                    ensure!(
                        (f64::from(value) - cpu).abs() < 1e-3 + cpu.abs() * 1e-5,
                        "CPU tiled projection mismatch"
                    );
                }
            }
            let tiny = DeviceBuffer::new_tagged(&ctx, 4, "tiled_test:tiny")?;
            let prior = read_output(&ctx, &d_output)?;
            let before = profile::snapshot();
            let config = Q8TiledLaunch::new(columns as u32, 128)?;
            unsafe {
                assert!(tiled
                    .project_tiled_async(&d_input, &d_output, 0, config)
                    .is_err());
                assert!(tiled
                    .project_tiled_async(&d_input, &d_output, 65536, config)
                    .is_err());
                assert!(original
                    .project_tiled_async(&d_input, &d_output, rows, config)
                    .is_err());
                assert!(tiled
                    .project_tiled_async(
                        &d_input,
                        &d_output,
                        rows,
                        Q8TiledLaunch::new(if columns == 4 { 8 } else { 4 }, 128)?
                    )
                    .is_err());
                if rows * k > 1 {
                    assert!(tiled
                        .project_tiled_async(&tiny, &d_output, rows, config)
                        .is_err());
                }
                if rows * n > 1 {
                    assert!(tiled
                        .project_tiled_async(&d_input, &tiny, rows, config)
                        .is_err());
                }
                assert!(tiled
                    .project_tiled_async(&d_input, &d_input, rows, config)
                    .is_err());
            }
            assert_eq!(
                profile::counter_delta(&before, &profile::snapshot()).launches,
                0
            );
            let after = read_output(&ctx, &d_output)?;
            assert_eq!(
                after.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                prior.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}
