#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/q8_projection.rs"]
mod fixture;

use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    infer::{
        activation_sparsity::{
            ActivationThreshold, DeviceSparseInputWorkspace, SparseInputSelection,
        },
        execution_weights::{
            DeviceQ8ExecutionView, DeviceQ8ExecutionWeight, HostQ8ExecutionWeight, Q8MatrixShape,
            Q8SourceIdentity, Q8SourceView, Q8TiledLaunch,
        },
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

/// Sparse decode matches the same thresholded dense inputs at all four bounded
/// geometries, including zero/full omission, K/N tails and real down dimensions.
/// Reusing scratch adds two launches, no sync/copy/allocation; guards and rejected
/// calls protect output spans. CPU samples independently validate GPU math.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_sparse_activation_projection_matches_zeroed_dense_and_cpu() -> Result<()> {
    let ctx = CudaContext::new_m40()?;
    for (k, n) in [(1, 1), (35, 7), (259, 33), (11008, 2048)] {
        let shape = Q8MatrixShape::new(k, n)?;
        let identity =
            Q8SourceIdentity::from_column_blocks([9; 32], "sparse_input".into(), 107, shape)?;
        let bytes = fixture::make_column_blocks(k, n);
        let source = Q8SourceView::from_tensor_bytes(&identity, &bytes)?;
        let original = DeviceQ8ExecutionWeight::upload(
            &ctx,
            &HostQ8ExecutionWeight::copy_column_blocks(source)?,
        )?;
        let DeviceQ8ExecutionView::ColumnBlocks { blocks, .. } = original.view() else {
            unreachable!()
        };
        let input = fixture::make_input(1, k);
        let d_input = DeviceBuffer::new_tagged(&ctx, k * 4, "sparse_test:input")?;
        let d_matched = DeviceBuffer::new_tagged(&ctx, k * 4, "sparse_test:matched")?;
        let d_reference = DeviceBuffer::new_tagged(&ctx, n * 4, "sparse_test:reference")?;
        let d_output = DeviceBuffer::new_tagged(&ctx, (n + 16) * 4, "sparse_test:output")?;
        let tiny = DeviceBuffer::new_tagged(&ctx, 4, "sparse_test:tiny")?;
        let poison = vec![f32::from_bits(0x7fc0_1234); n + 16];
        unsafe {
            ctx.memcpy_h2d(d_input.as_mut_ptr(), input.as_ptr().cast(), k * 4)?;
        }
        let before_bytes = CudaContext::total_device_bytes();
        {
            let mut scratch = DeviceSparseInputWorkspace::new(&ctx, k)?;
            assert_eq!(scratch.bytes(), k * 8 + 4);
            assert_eq!(
                CudaContext::total_device_bytes(),
                before_bytes + scratch.bytes()
            );
            assert!(scratch.read_selection_sync().is_err());
            let mut thresholds: Vec<_> = [0.0, 0.25, 0.4, 0.5, 0.6]
                .into_iter()
                .map(|fraction| ActivationThreshold::calibrate(&input, fraction))
                .collect::<Result<_>>()?;
            thresholds.push(ActivationThreshold::new(f32::MAX)?);
            for columns in [8, 16] {
                let tiled = DeviceQ8ExecutionWeight::upload(
                    &ctx,
                    &HostQ8ExecutionWeight::tile_inputs(source, columns)?,
                )?;
                for &threshold in &thresholds {
                    let selected = SparseInputSelection::select(&input, threshold)?;
                    let matched = selected.reconstruct();
                    unsafe {
                        ctx.memcpy_h2d(d_matched.as_mut_ptr(), matched.as_ptr().cast(), k * 4)?;
                        ctx.gemm_f32xq8_0_gguf_f32_async(
                            d_matched.as_ptr(),
                            blocks.as_ptr(),
                            d_reference.as_mut_ptr(),
                            1,
                            n as i32,
                            k as i32,
                        )?;
                    }
                    ctx.synchronize_stream(CudaStream::Decode)?;
                    ctx.synchronize_stream(CudaStream::Prefill)?;
                    let expected = read_output(&ctx, &d_reference)?;
                    for threads in [128, 256] {
                        unsafe {
                            ctx.memcpy_h2d(
                                d_output.as_mut_ptr(),
                                poison.as_ptr().cast(),
                                d_output.bytes(),
                            )?;
                        }
                        let config = Q8TiledLaunch::new(columns as u32, threads)?;
                        let before = profile::snapshot();
                        unsafe {
                            scratch.select_async(&d_input, threshold)?;
                            scratch.project_tiled_async(&tiled, &d_output, config)?;
                        }
                        let delta = profile::counter_delta(&before, &profile::snapshot());
                        assert_eq!(delta.launches, 2);
                        assert_eq!(delta.stream_syncs, 0);
                        assert_eq!(delta.stream_waits, 0);
                        assert_eq!(delta.device_allocs, 0);
                        assert_eq!(delta.device_frees, 0);
                        assert_eq!(delta.h2d_copies, 0);
                        assert_eq!(delta.d2h_copies, 0);
                        let actual_selection = scratch.read_selection_sync()?;
                        assert_eq!(actual_selection.indices(), selected.indices());
                        assert_eq!(actual_selection.values(), selected.values());
                        let actual = read_output(&ctx, &d_output)?;
                        assert!(actual[n..]
                            .iter()
                            .all(|v| v.to_bits() == poison[0].to_bits()));
                        let (max, mean) = fixture::measure_error(&actual[..n], &expected);
                        ensure!(max < 1e-3 && mean < 1e-4,
                            "K={k} N={n} config={config:?} threshold={threshold:?} max={max} mean={mean}");
                        for sample in 0..n.min(16) {
                            let column = sample * n / n.min(16);
                            let cpu = fixture::project_cpu(&matched, &bytes, 0, column, k);
                            ensure!(
                                (f64::from(actual[column]) - cpu).abs() < 1e-3 + cpu.abs() * 1e-5,
                                "sparse input CPU mismatch K={k} N={n} column={column}"
                            );
                        }
                        let before = profile::snapshot();
                        unsafe {
                            assert!(scratch
                                .project_tiled_async(&original, &d_output, config)
                                .is_err());
                            assert!(scratch
                                .project_tiled_async(
                                    &tiled,
                                    &d_output,
                                    Q8TiledLaunch::new(if columns == 8 { 16 } else { 8 }, threads)?
                                )
                                .is_err());
                            assert!(scratch
                                .project_tiled_async(
                                    &tiled,
                                    &d_output,
                                    Q8TiledLaunch::new(4, threads)?
                                )
                                .is_err());
                            if k > 1 {
                                assert!(scratch.select_async(&tiny, threshold).is_err());
                            }
                            if n > 1 {
                                assert!(scratch
                                    .project_tiled_async(&tiled, &tiny, config)
                                    .is_err());
                            }
                        }
                        assert_eq!(
                            profile::counter_delta(&before, &profile::snapshot()).launches,
                            0
                        );
                        assert_eq!(
                            read_output(&ctx, &d_output)?
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        );
                    }
                }
            }
            ctx.synchronize_stream(CudaStream::Decode)?;
        }
        assert_eq!(CudaContext::total_device_bytes(), before_bytes);
    }
    Ok(())
}

/// Stable compaction retains nonfinite payload bits, strict cutoff ties and
/// original indices without consulting host data; an all-discarded reuse
/// must clear the old count rather than consume stale selected entries.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_sparse_selection_preserves_nonfinite_bits_and_resets_count() -> Result<()> {
    let ctx = CudaContext::new_m40()?;
    let input = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        f32::INFINITY,
        f32::from_bits(0x7fc0_1234),
    ];
    let d_input = DeviceBuffer::new_tagged(&ctx, input.len() * 4, "sparse_test:nonfinite")?;
    let mut scratch = DeviceSparseInputWorkspace::new(&ctx, input.len())?;
    assert!(DeviceSparseInputWorkspace::new(&ctx, 0).is_err());
    unsafe {
        ctx.memcpy_h2d(d_input.as_mut_ptr(), input.as_ptr().cast(), d_input.bytes())?;
    }
    let threshold = ActivationThreshold::new(1.0)?;
    unsafe {
        scratch.select_async(&d_input, threshold)?;
    }
    let selected = scratch.read_selection_sync()?;
    let expected = SparseInputSelection::select(&input, threshold)?;
    assert_eq!(selected.indices(), expected.indices());
    assert_eq!(
        selected
            .values()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected
            .values()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    let zeros = [0.0f32; 7];
    unsafe {
        ctx.memcpy_h2d(d_input.as_mut_ptr(), zeros.as_ptr().cast(), d_input.bytes())?;
        scratch.select_async(&d_input, threshold)?;
    }
    assert!(scratch.read_selection_sync()?.indices().is_empty());
    Ok(())
}
