#![cfg(all(feature = "cuda", nvcc))]
#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/env_restore.rs"]
mod env_restore;
#[path = "common/q8_projection.rs"]
#[allow(dead_code)]
mod fixture;
use anyhow::Result;
use env_restore::EnvRestore;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    profile,
};

struct Guarded {
    device: DeviceBuffer,
    original: Vec<f32>,
}

impl Guarded {
    fn new(ctx: &CudaContext, values: &[f32]) -> Result<Self> {
        let mut original = vec![12345.0; values.len() + 16];
        original[8..8 + values.len()].copy_from_slice(values);
        let device = DeviceBuffer::new_tagged(ctx, original.len() * 4, "output_warp:test")?;
        unsafe {
            ctx.memcpy_h2d(
                device.as_mut_ptr(),
                original.as_ptr().cast(),
                device.bytes(),
            )?;
        }
        Ok(Self { device, original })
    }
    fn pointer(&self) -> *mut std::ffi::c_void {
        unsafe { self.device.as_mut_ptr().cast::<f32>().add(8).cast() }
    }
    fn read(&self, ctx: &CudaContext) -> Result<Vec<f32>> {
        let mut actual = vec![0.0; self.original.len()];
        unsafe {
            ctx.memcpy_d2h(
                actual.as_mut_ptr().cast(),
                self.device.as_ptr(),
                self.device.bytes(),
            )?;
        }
        assert_eq!(&actual[..8], &self.original[..8]);
        assert_eq!(
            &actual[actual.len() - 8..],
            &self.original[self.original.len() - 8..]
        );
        Ok(actual[8..actual.len() - 8].to_vec())
    }
}

fn compare_case(ctx: &CudaContext, k: usize, n: usize) -> Result<()> {
    let mut source = fixture::make_column_blocks(k, n);
    for (i, block) in source.chunks_exact_mut(34).enumerate() {
        if i % 7 == 0 {
            block[..2].fill(0);
        }
        for (lane, q) in block[2..].iter_mut().enumerate() {
            *q = [0x80, 0x7f, 0xff, 0, 1, 0x40, 0xc0, 8][(lane + i) % 8];
        }
    }
    let mut guarded = vec![0xa5; source.len() + 18];
    guarded[2..2 + source.len()].copy_from_slice(&source);
    let weight = DeviceBuffer::new_tagged(ctx, guarded.len(), "output_warp:weights")?;
    unsafe {
        ctx.memcpy_h2d(weight.as_mut_ptr(), guarded.as_ptr().cast(), guarded.len())?;
    }
    let weight_ptr = unsafe { weight.as_ptr().cast::<u8>().add(2).cast() };
    let input = fixture::make_input(1, k);
    let x = Guarded::new(ctx, &input)?;
    let y = Guarded::new(ctx, &vec![0.0; n])?;
    let doubled: Vec<_> = input.iter().map(|v| v + v).collect();
    for stream in [CudaStream::Decode, CudaStream::Prefill] {
        let _stream = EnvRestore::set(
            "M40LLM_Q8_DECODE_STREAM",
            if stream == CudaStream::Decode {
                "decode"
            } else {
                "prefill"
            },
        );
        let mut control = Vec::new();
        for (is_native, has_packed_loads) in [(true, false), (false, false), (false, true)] {
            unsafe {
                ctx.memcpy_h2d(x.pointer(), input.as_ptr().cast(), k * 4)?;
                ctx.residual_add_f32_async(x.pointer(), x.pointer(), x.pointer(), k)?;
            }
            if stream == CudaStream::Prefill {
                ctx.stream_wait_for_stream(
                    stream,
                    CudaStream::Decode,
                    "output_warp:test_producer",
                )?;
            }
            let before = profile::snapshot();
            unsafe {
                if !is_native {
                    let enqueue = if has_packed_loads {
                        CudaContext::gemm_f32xq8_0_gguf_coltile4_packed_warp_async
                    } else {
                        CudaContext::gemm_f32xq8_0_gguf_coltile4_warp_async
                    };
                    enqueue(
                        ctx,
                        x.pointer(),
                        weight_ptr,
                        y.pointer(),
                        n as i32,
                        k as i32,
                        stream,
                    )?;
                } else {
                    ctx.gemm_f32xq8_0_gguf_f32_decode_async(
                        x.pointer(),
                        weight_ptr,
                        y.pointer(),
                        1,
                        n as i32,
                        k as i32,
                    )?;
                }
            }
            let delta = profile::counter_delta(&before, &profile::snapshot());
            assert_eq!(delta.launches, 1);
            assert_eq!(
                (
                    delta.device_allocs,
                    delta.device_frees,
                    delta.h2d_copies,
                    delta.d2h_copies,
                    delta.stream_syncs,
                    delta.stream_waits,
                    delta.cublas_calls
                ),
                (0, 0, 0, 0, 0, 0, 0)
            );
            ctx.synchronize_stream(stream)?;
            let actual = y.read(ctx)?;
            if !is_native {
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    control
                );
            } else {
                control = actual.iter().map(|v| v.to_bits()).collect();
            }
            for (col, &observed) in actual.iter().enumerate() {
                let cpu = fixture::project_cpu(&doubled, &source, 0, col, k);
                assert!(
                    observed.is_finite()
                        && (f64::from(observed) - cpu).abs() < 1e-3 + 2e-5 * cpu.abs()
                );
            }
            assert_eq!(x.read(ctx)?, doubled);
        }
    }
    let mut actual = vec![0; guarded.len()];
    unsafe {
        ctx.memcpy_d2h(actual.as_mut_ptr().cast(), weight.as_ptr(), actual.len())?;
    }
    assert_eq!(actual, guarded);
    Ok(())
}

/// Four-column warp reduction preserves original FP32 bits and CPU tolerance
/// for all column tails, zero scales and signed extremes, on both streams
/// after an asynchronous producer. Guarded spans and RAII cleanup are intact.
/// Packed input/payload loads must preserve these results with two-byte-only
/// weight bases, all column tails, and the same stream/dependency contract.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_native_output_warp_preserves_tree_tails_and_streams() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _split = EnvRestore::set("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4");
    let _columns = EnvRestore::set("M40LLM_Q8_DECODE_COL_TILE", "4");
    let _diag = EnvRestore::set("M40LLM_Q8_DECODE_EVENT_LOG", "0");
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for (k, n) in [
        (32, 1),
        (32, 2),
        (32, 3),
        (32, 4),
        (96, 5),
        (96, 33),
        (2048, 257),
        (2048, 2048),
        (4096, 37),
    ] {
        compare_case(&ctx, k, n)?;
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Invalid dimensions, missing spans, misalignment and output aliases fail
/// without enqueueing GPU work or allocating temporary storage.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_native_output_warp_rejects_invalid_spans() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let x = DeviceBuffer::new_tagged(&ctx, 512, "output_warp:invalid")?;
        let w = DeviceBuffer::new_tagged(&ctx, 512, "output_warp:invalid")?;
        let y = DeviceBuffer::new_tagged(&ctx, 512, "output_warp:invalid")?;
        let before = profile::snapshot();
        unsafe {
            for enqueue in [
                CudaContext::gemm_f32xq8_0_gguf_coltile4_warp_async,
                CudaContext::gemm_f32xq8_0_gguf_coltile4_packed_warp_async,
            ] {
                for (n, k) in [(0, 32), (1, 0), (1, 33), (-1, 32), (1, -32)] {
                    assert!(enqueue(
                        &ctx,
                        x.as_ptr(),
                        w.as_ptr(),
                        y.as_mut_ptr(),
                        n,
                        k,
                        CudaStream::Decode
                    )
                    .is_err());
                }
                for (input, weight, output) in [
                    (std::ptr::null(), w.as_ptr(), y.as_mut_ptr()),
                    (x.as_ptr(), std::ptr::null(), y.as_mut_ptr()),
                    (x.as_ptr(), w.as_ptr(), std::ptr::null_mut()),
                    (x.as_ptr(), w.as_ptr(), x.as_mut_ptr()),
                    (x.as_ptr(), w.as_ptr(), w.as_mut_ptr()),
                    (
                        x.as_ptr().cast::<u8>().add(2).cast(),
                        w.as_ptr(),
                        y.as_mut_ptr(),
                    ),
                    (
                        x.as_ptr(),
                        w.as_ptr().cast::<u8>().add(1).cast(),
                        y.as_mut_ptr(),
                    ),
                    (
                        x.as_ptr(),
                        w.as_ptr(),
                        y.as_mut_ptr().cast::<u8>().add(2).cast(),
                    ),
                ] {
                    assert!(
                        enqueue(&ctx, input, weight, output, 1, 32, CudaStream::Decode).is_err()
                    );
                }
            }
            for offset in [4, 8, 12] {
                assert!(ctx
                    .gemm_f32xq8_0_gguf_coltile4_packed_warp_async(
                        x.as_ptr().cast::<u8>().add(offset).cast(),
                        w.as_ptr(),
                        y.as_mut_ptr(),
                        1,
                        32,
                        CudaStream::Decode
                    )
                    .is_err());
            }
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()),
            profile::counter_delta(&before, &before)
        );
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}
