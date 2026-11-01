#![cfg(all(feature = "cuda", nvcc))]
#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;

use anyhow::Result;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer, DeviceQ4SelectedFfn, Q4DownLayout},
    infer::q4_channels::Q4ChannelWeights,
    profile,
};

const GUARD: f32 = 12345.25;

fn make_weights(n: usize, k: usize, seed: usize) -> Vec<u8> {
    let mut raw = Vec::with_capacity(n * k / 32 * 18);
    for block in 0..n * k / 32 {
        let scale = ((block + seed) % 9) as f32 / 256.0 - 0.015625;
        raw.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        for pair in 0..16 {
            raw.push(((block * 13 + pair * 37 + seed * 19) % 256) as u8);
        }
    }
    raw
}

fn read_weight(raw: &[u8], output: usize, input: usize, k: usize) -> f64 {
    let block = (output * (k / 32) + input / 32) * 18;
    let scale = half::f16::from_le_bytes([raw[block], raw[block + 1]]).to_f64();
    let byte = raw[block + 2 + input % 16];
    let q = if input % 32 < 16 {
        byte & 15
    } else {
        byte >> 4
    };
    scale * f64::from(i32::from(q) - 8)
}

fn upload(ctx: &CudaContext, values: &[f32]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new_tagged(ctx, values.len() * 4, "q4_test:buffer")?;
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

fn check_case(ctx: &CudaContext, k: usize, h: usize, rows: usize, count: usize) -> Result<()> {
    let gate = make_weights(h, k, 1);
    let up = make_weights(h, k, 7);
    let down = make_weights(k, h, 13);
    let packed = Q4ChannelWeights::pack(&down, h, k)?;
    let mut indices = Vec::new();
    for row in 0..rows {
        // Ascending irregular selections differ between rows, except full retention.
        let mut selected: Vec<_> = (0..count)
            .map(|i| ((i * h / count + row) % h) as u32)
            .collect();
        selected.sort_unstable();
        indices.extend(selected);
    }
    let owner = DeviceQ4SelectedFfn::upload(ctx, &gate, &up, &packed, &indices, rows)?;
    let input: Vec<_> = (0..rows * k)
        .map(|i| ((i * 31 + i / k * 7) % 59) as f32 / 113.0 - 0.25)
        .collect();
    let mut expected = vec![0.0f64; rows * k];
    for row in 0..rows {
        for &channel in &indices[row * count..(row + 1) * count] {
            let mut g = 0.0;
            let mut u = 0.0;
            for i in 0..k {
                let x = f64::from(input[row * k + i]);
                g += x * read_weight(&gate, channel as usize, i, k);
                u += x * read_weight(&up, channel as usize, i, k);
            }
            let hidden = g / (1.0 + (-g).exp()) * u;
            for output in 0..k {
                expected[row * k + output] +=
                    hidden * read_weight(&down, output, channel as usize, h);
            }
        }
    }
    let producer = upload(ctx, &input)?;
    let input_device = upload(ctx, &vec![GUARD; rows * k + 16])?;
    let input_pointer = unsafe { input_device.as_mut_ptr().cast::<f32>().add(8).cast() };
    let output = upload(ctx, &vec![GUARD; rows * k + 16])?;
    let output_pointer = unsafe { output.as_mut_ptr().cast::<f32>().add(8).cast() };
    let mut baseline = None;
    for stream in [CudaStream::Decode, CudaStream::Prefill] {
        let producer_stream = if stream == CudaStream::Decode {
            CudaStream::Prefill
        } else {
            CudaStream::Decode
        };
        unsafe {
            ctx.memcpy_d2d_async(
                input_pointer,
                producer.as_ptr(),
                producer.bytes(),
                producer_stream,
            )?;
        }
        ctx.stream_wait_for_stream(stream, producer_stream, "q4_test:producer")?;
        for layout in [Q4DownLayout::Native, Q4DownLayout::ChannelMajor] {
            for _ in 0..2 {
                let before = profile::snapshot();
                let resident = CudaContext::total_device_bytes();
                unsafe {
                    owner.enqueue_async(input_pointer, output_pointer, layout, stream)?;
                }
                let delta = profile::counter_delta(&before, &profile::snapshot());
                assert_eq!(delta.launches, 3);
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
                assert_eq!(resident, CudaContext::total_device_bytes());
                ctx.synchronize_stream(stream)?;
                let actual = read(ctx, &output)?;
                assert!(actual[..8]
                    .iter()
                    .chain(&actual[rows * k + 8..])
                    .all(|&v| v == GUARD));
                let bits: Vec<_> = actual[8..rows * k + 8]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                if let Some(ref control) = baseline {
                    assert_eq!(&bits, control);
                } else {
                    baseline = Some(bits);
                }
                for (index, (&value, &reference)) in
                    actual[8..rows * k + 8].iter().zip(&expected).enumerate()
                {
                    assert!((f64::from(value) - reference).abs() < 1e-3 + 2e-5 * reference.abs(),
                        "K={k} H={h} rows={rows} count={count} index={index}: {value} vs {reference}");
                }
            }
        }
    }
    let actual_input = read(ctx, &input_device)?;
    assert_eq!(&actual_input[8..rows * k + 8], &input);
    assert!(actual_input[..8]
        .iter()
        .chain(&actual_input[rows * k + 8..])
        .all(|&v| v == GUARD));
    let index_bytes: Vec<_> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    let [scales, codes] = packed.planes();
    for (buffer, original) in owner.immutable_buffers().into_iter().zip([
        gate.as_slice(),
        up.as_slice(),
        down.as_slice(),
        scales,
        codes,
        index_bytes.as_slice(),
    ]) {
        let mut actual = vec![0u8; buffer.bytes()];
        unsafe {
            ctx.memcpy_d2h(actual.as_mut_ptr().cast(), buffer.as_ptr(), actual.len())?;
        }
        assert_eq!(actual, original);
    }
    Ok(())
}

/// Both layouts match FP64 masked FFN results and each other at every output,
/// preserve guards/owners, order cross-stream producers and reuse scratch.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_selected_ffn_matches_masked_reference() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let before = CudaContext::total_device_bytes();
    for rows in [1, 2, 4, 8] {
        for count in [1, 17, 96] {
            check_case(&ctx, 64, 96, rows, count)?;
        }
    }
    check_case(&ctx, 2048, 11008, 1, 1101)?;
    check_case(&ctx, 2048, 11008, 1, 11008)?;
    assert_eq!(before, CudaContext::total_device_bytes());
    Ok(())
}

/// Invalid selections, spans and aliases fail before allocations or launches.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_selected_ffn_rejects_invalid_inputs() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let before_bytes = CudaContext::total_device_bytes();
    {
        let raw = make_weights(32, 32, 1);
        let down = Q4ChannelWeights::pack(&raw, 32, 32)?;
        let before = profile::snapshot();
        for selection in [vec![], vec![32], vec![0, 0], vec![2, 1]] {
            assert!(DeviceQ4SelectedFfn::upload(&ctx, &raw, &raw, &down, &selection, 1).is_err());
        }
        for rows in [0, 3, 9] {
            assert!(DeviceQ4SelectedFfn::upload(&ctx, &raw, &raw, &down, &[0], rows).is_err());
        }
        assert!(DeviceQ4SelectedFfn::upload(&ctx, &raw[..10], &raw, &down, &[0], 1).is_err());
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()).device_allocs,
            0
        );
        let owner = DeviceQ4SelectedFfn::upload(&ctx, &raw, &raw, &down, &[0], 1)?;
        let input = upload(&ctx, &[1.0; 32])?;
        let output = upload(&ctx, &[GUARD; 32])?;
        let before = profile::snapshot();
        for pointer in [
            std::ptr::null(),
            unsafe { input.as_ptr().cast::<u8>().add(1).cast() },
            (usize::MAX - 3) as *const _,
        ] {
            assert!(unsafe {
                owner.enqueue_async(
                    pointer,
                    output.as_mut_ptr(),
                    Q4DownLayout::Native,
                    CudaStream::Decode,
                )
            }
            .is_err());
        }
        assert!(unsafe {
            owner.enqueue_async(
                input.as_ptr(),
                input.as_mut_ptr(),
                Q4DownLayout::ChannelMajor,
                CudaStream::Decode,
            )
        }
        .is_err());
        assert!(unsafe {
            owner.enqueue_async(
                input.as_ptr(),
                owner.immutable_buffers()[0].as_mut_ptr(),
                Q4DownLayout::Native,
                CudaStream::Decode,
            )
        }
        .is_err());
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()).launches,
            0
        );
        assert_eq!(read(&ctx, &output)?, vec![GUARD; 32]);
    }
    assert_eq!(before_bytes, CudaContext::total_device_bytes());
    Ok(())
}
