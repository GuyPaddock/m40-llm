#![cfg(all(feature = "cuda", nvcc))]

use std::sync::Mutex;

use anyhow::{ensure, Result};
use half::f16;
use m40_llm::{
    cuda::{CudaContext, DeviceBuffer},
    infer::execution_weights::{
        DeviceQ8ExecutionView, DeviceQ8ExecutionWeight, HostQ8ExecutionWeight, Q8ExecutionKey,
        Q8ExecutionLayout, Q8MatrixShape, Q8SourceIdentity, Q8SourceView,
    },
    profile,
};

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn context() -> Result<CudaContext> {
    let ctx = CudaContext::new(-1)?;
    let props = ctx.current_device_props()?;
    ensure!(
        props.major == 5 && props.minor == 2,
        "Q8 ownership tests require sm_52"
    );
    Ok(ctx)
}

/// Every typed layout owns its requested planes independently of host/source
/// lifetime, preserves bytes on upload, and releases tracked device payload.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_device_execution_planes_round_trip_and_release() -> Result<()> {
    let _lock = TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let ctx = context()?;
    let initial_bytes = CudaContext::total_device_bytes();
    for layout in [
        Q8ExecutionLayout::ColumnBlocks,
        Q8ExecutionLayout::SeparateScalesValues,
        Q8ExecutionLayout::InputTiled { columns: 4 },
        Q8ExecutionLayout::InputTiled { columns: 8 },
    ] {
        let shape = Q8MatrixShape::new(35, 3)?;
        let source =
            Q8SourceIdentity::from_column_blocks([9; 32], "round_trip.weight".into(), 5, shape)?;
        let key = Q8ExecutionKey { source, layout };
        let plan = layout.allocation_plan(shape)?;
        let expected: Vec<Vec<u8>> = plan
            .planes()
            .iter()
            .enumerate()
            .map(|(plane, spec)| {
                (0..spec.bytes)
                    .map(|index| (index * 23 + plane) as u8)
                    .collect()
            })
            .collect();
        let host = HostQ8ExecutionWeight::new(key.clone(), expected.clone())?;
        let before = profile::snapshot();
        let device = DeviceQ8ExecutionWeight::upload(&ctx, &host)?;
        drop(host);
        assert_eq!(device.key(), &key);
        assert_eq!(
            CudaContext::total_device_bytes() - initial_bytes,
            plan.resident_bytes()
        );
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()).device_allocs,
            plan.planes().len() as u64
        );
        let buffers = match device.view() {
            DeviceQ8ExecutionView::ColumnBlocks { blocks, .. } => vec![blocks],
            DeviceQ8ExecutionView::SeparateScalesValues { scales, values, .. } => {
                vec![scales, values]
            }
            DeviceQ8ExecutionView::InputTiled { tiles, .. } => vec![tiles],
        };
        assert_eq!(
            buffers.iter().map(|buffer| buffer.bytes()).sum::<usize>(),
            plan.resident_bytes()
        );
        assert_eq!(unsafe { device.read_planes()? }, expected);
        drop(device);
        assert_eq!(CudaContext::total_device_bytes(), initial_bytes);
    }
    Ok(())
}

/// A copied column-block view remains valid after host storage is dropped and
/// drives the real Q8 projection, including K/N tails and a nonzero source offset.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_owned_column_blocks_drive_existing_projection() -> Result<()> {
    let _lock = TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let ctx = context()?;
    let shape = Q8MatrixShape::new(35, 3)?;
    let identity =
        Q8SourceIdentity::from_column_blocks([3; 32], "projection.weight".into(), 5, shape)?;
    let mut blob = vec![0xa5; 5 + shape.source_bytes()?];
    for column in 0..3 {
        for block in 0..2 {
            let base = 5 + (column * 2 + block) * 34;
            let scale = f16::from_f32((column + block + 1) as f32 * 0.125)
                .to_bits()
                .to_le_bytes();
            blob[base..base + 2].copy_from_slice(&scale);
            for index in 0..32 {
                blob[base + 2 + index] = ((index * 7 + column * 13) % 255) as u8;
            }
        }
    }
    let source = Q8SourceView::new(&identity, &blob)?;
    let source_bytes = source.blocks().to_vec();
    let host = HostQ8ExecutionWeight::copy_column_blocks(source)?;
    let device = DeviceQ8ExecutionWeight::upload(&ctx, &host)?;
    drop(host);
    drop(blob);
    let DeviceQ8ExecutionView::ColumnBlocks { blocks, .. } = device.view() else {
        panic!("wrong typed layout");
    };
    let input: Vec<f32> = (0..70).map(|i| (i as f32 - 17.0) * 0.03125).collect();
    let d_input = DeviceBuffer::new_tagged(&ctx, input.len() * 4, "q8_owned:input")?;
    let d_output = DeviceBuffer::new_tagged(&ctx, 6 * 4, "q8_owned:output")?;
    let mut output = [0.0_f32; 6];
    unsafe {
        ctx.memcpy_h2d(d_input.as_mut_ptr(), input.as_ptr().cast(), input.len() * 4)?;
        ctx.gemm_f32xq8_0_gguf_f32(
            d_input.as_ptr(),
            blocks.as_ptr(),
            d_output.as_mut_ptr(),
            2,
            3,
            35,
        )?;
        ctx.memcpy_d2h(
            output.as_mut_ptr().cast(),
            d_output.as_ptr(),
            output.len() * 4,
        )?;
    }
    for row in 0..2 {
        for column in 0..3 {
            let mut expected = 0.0;
            for k in 0..35 {
                let base = (column * 2 + k / 32) * 34;
                let scale = f16::from_bits(u16::from_le_bytes([
                    source_bytes[base],
                    source_bytes[base + 1],
                ]))
                .to_f32();
                expected +=
                    input[row * 35 + k] * scale * f32::from(source_bytes[base + 2 + k % 32] as i8);
            }
            assert!(
                (output[row * 3 + column] - expected).abs() < 1e-4,
                "row={row} column={column} got={} expected={expected}",
                output[row * 3 + column]
            );
        }
    }
    Ok(())
}
