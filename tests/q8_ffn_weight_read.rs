#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;

use anyhow::Result;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    profile,
};
use std::ffi::c_void;

const GUARD: u32 = 0x5a12_3498;

fn upload(ctx: &CudaContext, bytes: &[u8]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), "ffn-read:test")?;
    unsafe {
        ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
    }
    Ok(buffer)
}

fn download(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; buffer.bytes()];
    unsafe {
        ctx.memcpy_d2h(bytes.as_mut_ptr().cast(), buffer.as_ptr(), bytes.len())?;
    }
    Ok(bytes)
}

fn check_case(ctx: &CudaContext, k: usize, hidden: usize, rows: usize) -> Result<()> {
    let mut hosts = Vec::new();
    let mut expected = Vec::new();
    for seed in [0usize, 3] {
        let mut scales = Vec::new();
        let mut values = Vec::new();
        let mut sums = Vec::new();
        for column in 0..hidden {
            let mut sum = 0u32;
            for block in 0..k / 32 {
                let scale = if (column + block) % 3 == 0 {
                    0
                } else {
                    0xf123u16.wrapping_add((column + block + seed) as u16)
                };
                scales.extend(scale.to_le_bytes());
                for word in 0..8 {
                    let payload = if column == 0 {
                        u32::MAX
                    } else {
                        u32::from_le_bytes(
                            [0x80, 0x7f, 0xff, 0]
                                .map(|v: u8| v.wrapping_add((column + block + word + seed) as u8)),
                        )
                    };
                    values.extend(payload.to_le_bytes());
                    sum = sum.wrapping_add(payload).wrapping_add(u32::from(scale));
                }
            }
            sums.push(sum);
        }
        hosts.push((scales, values));
        expected.push(sums);
    }
    let devices: Vec<_> = hosts
        .iter()
        .map(|(s, v)| Ok((upload(ctx, s)?, upload(ctx, v)?)))
        .collect::<Result<_>>()?;
    let scales = [devices[0].0.as_ptr(), devices[1].0.as_ptr()];
    let values = [devices[0].1.as_ptr(), devices[1].1.as_ptr()];
    let guard_bytes: Vec<_> = vec![GUARD; rows * hidden * 2 + 16]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let output = upload(ctx, &guard_bytes)?;
    let out = unsafe { output.as_mut_ptr().cast::<u32>().add(8).cast() };
    for repeat in 0..3 {
        // Exercise a real asynchronous producer and explicit dependency before
        // the measured enqueue, without relying on a synchronous upload.
        let replacement: Vec<_> = hosts
            .iter()
            .map(|(_, v)| upload(ctx, v))
            .collect::<Result<_>>()?;
        for (index, source) in replacement.iter().enumerate() {
            unsafe {
                ctx.memcpy_d2d_async(
                    values[index].cast_mut(),
                    source.as_ptr(),
                    source.bytes(),
                    CudaStream::Prefill,
                )?;
            }
        }
        ctx.stream_wait_for_stream(
            CudaStream::Decode,
            CudaStream::Prefill,
            "ffn_read_test_producer",
        )?;
        let resident = CudaContext::total_device_bytes();
        let before = profile::snapshot();
        unsafe {
            ctx.ffn_q8_weight_checksum_async(
                scales,
                values,
                out,
                rows as i32,
                hidden as i32,
                k as i32,
            )?;
        }
        let after = profile::snapshot();
        let delta = profile::counter_delta(&before, &after);
        assert_eq!(delta.launches, 1);
        assert_eq!(
            delta.device_allocs
                + delta.device_frees
                + delta.h2d_copies
                + delta.d2h_copies
                + delta.stream_syncs
                + delta.stream_waits
                + delta.cublas_calls,
            0
        );
        assert_eq!(CudaContext::total_device_bytes(), resident);
        ctx.synchronize_stream(CudaStream::Decode)?;
        let actual = download(ctx, &output)?;
        let words: Vec<_> = actual
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert!(words[..8]
            .iter()
            .chain(&words[words.len() - 8..])
            .all(|&v| v == GUARD));
        for row in 0..rows {
            for col in 0..hidden {
                assert_eq!(
                    words[8 + (row * hidden + col) * 2],
                    expected[0][col],
                    "gate repeat {repeat}"
                );
                assert_eq!(
                    words[9 + (row * hidden + col) * 2],
                    expected[1][col],
                    "up repeat {repeat}"
                );
            }
        }
    }
    for (device, host) in devices.iter().zip(&hosts) {
        assert_eq!(download(ctx, &device.0)?, host.0);
        assert_eq!(download(ctx, &device.1)?, host.1);
    }
    Ok(())
}

/// Actual payload words and arbitrary scale bits produce exact modular CPU
/// sums for short/long loops, odd widths and rows 1/4 without scratch churn.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_weight_reads_preserve_checksums_guards_and_ownership() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for (k, hidden) in [(32, 1), (160, 7), (2048, 17)] {
        for rows in [1, 4] {
            check_case(&ctx, k, hidden, rows)?;
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Invalid dimensions, null/misaligned spans, partial overlap and address
/// overflow fail before any CUDA work or output mutation.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_weight_reads_reject_invalid_spans_before_enqueue() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let plane = upload(&ctx, &[0x23; 512])?;
        let output = upload(&ctx, &[0x45; 128])?;
        let s = [plane.as_ptr(); 2];
        let v = [plane.as_ptr(); 2];
        let o = output.as_mut_ptr();
        let before = profile::snapshot();
        unsafe {
            for (r, h, k) in [(0, 1, 32), (2, 1, 32), (1, 0, 32), (1, 1, 0), (1, 1, 31)] {
                assert!(ctx.ffn_q8_weight_checksum_async(s, v, o, r, h, k).is_err());
            }
            for offset in [1usize, 2] {
                assert!(ctx
                    .ffn_q8_weight_checksum_async(
                        s,
                        [plane.as_ptr().cast::<u8>().add(offset).cast(); 2],
                        o,
                        1,
                        1,
                        32
                    )
                    .is_err());
            }
            assert!(ctx
                .ffn_q8_weight_checksum_async([std::ptr::null(); 2], v, o, 1, 1, 32)
                .is_err());
            assert!(ctx
                .ffn_q8_weight_checksum_async(s, v, std::ptr::null_mut(), 1, 1, 32)
                .is_err());
            assert!(ctx
                .ffn_q8_weight_checksum_async(
                    s,
                    v,
                    plane.as_mut_ptr().cast::<u8>().add(4).cast(),
                    1,
                    1,
                    32
                )
                .is_err());
            assert!(ctx
                .ffn_q8_weight_checksum_async(
                    s,
                    v,
                    plane.as_mut_ptr().cast::<u8>().add(28).cast(),
                    1,
                    1,
                    32
                )
                .is_err());
            assert!(ctx
                .ffn_q8_weight_checksum_async(
                    [(usize::MAX - 1) as *const c_void; 2],
                    v,
                    o,
                    1,
                    1,
                    32
                )
                .is_err());
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()),
            Default::default()
        );
        assert_eq!(download(&ctx, &output)?, vec![0x45; 128]);
        assert_eq!(download(&ctx, &plane)?, vec![0x23; 512]);
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}
