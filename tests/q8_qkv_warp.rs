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
    initial: Vec<f32>,
}
impl Guarded {
    fn new(ctx: &CudaContext, values: &[f32]) -> Result<Self> {
        let mut initial = vec![12345.0; values.len() + 16];
        initial[8..8 + values.len()].copy_from_slice(values);
        let device = DeviceBuffer::new_tagged(ctx, initial.len() * 4, "qkv_warp:test")?;
        unsafe {
            ctx.memcpy_h2d(device.as_mut_ptr(), initial.as_ptr().cast(), device.bytes())?;
        }
        Ok(Self { device, initial })
    }
    fn ptr(&self) -> *mut std::ffi::c_void {
        unsafe { self.device.as_mut_ptr().cast::<f32>().add(8).cast() }
    }
    fn read(&self, ctx: &CudaContext) -> Result<Vec<f32>> {
        let mut actual = vec![0.0; self.initial.len()];
        unsafe {
            ctx.memcpy_d2h(
                actual.as_mut_ptr().cast(),
                self.device.as_ptr(),
                self.device.bytes(),
            )?;
        }
        assert_eq!(&actual[..8], &self.initial[..8]);
        assert_eq!(
            &actual[actual.len() - 8..],
            &self.initial[self.initial.len() - 8..]
        );
        Ok(actual[8..actual.len() - 8].to_vec())
    }
}

fn compare_case(ctx: &CudaContext, k: usize, widths: [usize; 3]) -> Result<()> {
    let sources = widths.map(|n| {
        let mut bytes = fixture::make_column_blocks(k, n);
        for (i, block) in bytes.chunks_exact_mut(34).enumerate() {
            if i % 7 == 0 {
                block[..2].fill(0);
            }
            for (lane, q) in block[2..].iter_mut().enumerate() {
                *q = [0x80, 0x7f, 0xff, 0, 1, 0x40, 0xc0, 8][(lane + i) % 8];
            }
        }
        bytes
    });
    let mut weights = Vec::new();
    for bytes in &sources {
        let mut guarded = vec![0xa5; bytes.len() + 18];
        guarded[2..2 + bytes.len()].copy_from_slice(bytes);
        let device = DeviceBuffer::new_tagged(ctx, guarded.len(), "qkv_warp:weights")?;
        unsafe {
            ctx.memcpy_h2d(device.as_mut_ptr(), guarded.as_ptr().cast(), guarded.len())?;
        }
        weights.push((device, guarded));
    }
    let weight_ptrs =
        std::array::from_fn(|i| unsafe { weights[i].0.as_ptr().cast::<u8>().add(2).cast() });
    let biases: Vec<Vec<f32>> = widths
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            (0..n)
                .map(|j| ((j + i * 7) % 17) as f32 * 0.015625 - 0.125)
                .collect()
        })
        .collect();
    let bias_buffers = biases
        .iter()
        .map(|v| Guarded::new(ctx, v))
        .collect::<Result<Vec<_>>>()?;
    let input = fixture::make_input(1, k);
    let x = Guarded::new(ctx, &input)?;
    let outputs = widths
        .map(|n| Guarded::new(ctx, &vec![0.0; n]))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let out = std::array::from_fn(|i| outputs[i].ptr());
    for stream in [CudaStream::Decode, CudaStream::Prefill] {
        let _stream = EnvRestore::set(
            "M40LLM_Q8_DECODE_STREAM",
            if stream == CudaStream::Decode {
                "decode"
            } else {
                "prefill"
            },
        );
        for has_bias in [[false; 3], [true, false, true], [true; 3]] {
            let b =
                std::array::from_fn(|i| has_bias[i].then(|| bias_buffers[i].ptr().cast_const()));
            unsafe {
                ctx.memcpy_h2d(x.ptr(), input.as_ptr().cast(), k * 4)?;
                ctx.residual_add_f32_async(x.ptr(), x.ptr(), x.ptr(), k)?;
            }
            if stream == CudaStream::Prefill {
                ctx.stream_wait_for_stream(stream, CudaStream::Decode, "qkv_warp:test_producer")?;
            }
            unsafe {
                ctx.qkv_f32xq8_0_gguf_decode_async(
                    x.ptr(),
                    weight_ptrs[0],
                    weight_ptrs[1],
                    weight_ptrs[2],
                    b[0],
                    b[1],
                    b[2],
                    out[0],
                    out[1],
                    out[2],
                    widths[0] as i32,
                    widths[1] as i32,
                    widths[2] as i32,
                    k as i32,
                )?;
            }
            ctx.synchronize_stream(stream)?;
            let expected = outputs
                .iter()
                .map(|o| o.read(ctx))
                .collect::<Result<Vec<_>>>()?;
            for enqueue in [
                CudaContext::qkv_f32xq8_0_gguf_warp_async,
                CudaContext::qkv_f32xq8_0_gguf_packed_warp_async,
            ] {
                let before = profile::snapshot();
                unsafe {
                    enqueue(
                        ctx,
                        x.ptr(),
                        weight_ptrs,
                        b,
                        out,
                        widths.map(|n| n as i32),
                        k as i32,
                        stream,
                    )?;
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
                        delta.stream_waits
                    ),
                    (0, 0, 0, 0, 0, 0)
                );
                ctx.synchronize_stream(stream)?;
                let doubled: Vec<f32> = input.iter().map(|v| v + v).collect();
                for i in 0..3 {
                    let actual = outputs[i].read(ctx)?;
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected[i].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    for (col, &observed) in actual.iter().enumerate() {
                        let cpu = fixture::project_cpu(&doubled, &sources[i], 0, col, k)
                            + if has_bias[i] {
                                f64::from(biases[i][col])
                            } else {
                                0.0
                            };
                        assert!((f64::from(observed) - cpu).abs() < 1e-3 + 2e-5 * cpu.abs());
                    }
                }
                assert_eq!(x.read(ctx)?, doubled);
            }
        }
    }
    for ((device, original), bias) in weights.iter().zip(&bias_buffers) {
        let mut actual = vec![0; device.bytes()];
        unsafe {
            ctx.memcpy_d2h(actual.as_mut_ptr().cast(), device.as_ptr(), actual.len())?;
        }
        assert_eq!(&actual, original);
        bias.read(ctx)?;
    }
    Ok(())
}

/// A one-barrier finish preserves the original raw-QKV tree bitwise with
/// unequal/tail widths, zero scales, signed extremes and optional biases.
/// Ordered producers on both streams, guards and allocation cleanup hold.
/// The packed-load variant must preserve these bits with weight bases aligned
/// to only two bytes and unchanged task/reduction/bias order.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_native_qkv_warp_preserves_tree_biases_and_streams() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _split = EnvRestore::set("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4");
    let _diag = EnvRestore::set("M40LLM_Q8_DECODE_EVENT_LOG", "0");
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for (k, widths) in [
        (32, [1, 1, 1]),
        (96, [65, 7, 11]),
        (2048, [257, 32, 32]),
        (4096, [37, 5, 9]),
    ] {
        compare_case(&ctx, k, widths)?;
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Invalid widths, misalignment and aliases fail without any launch, copy,
/// synchronization or fallback to another QKV implementation.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_native_qkv_warp_rejects_invalid_spans() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let x = DeviceBuffer::new_tagged(&ctx, 512, "qkv_warp:invalid")?;
        let w = DeviceBuffer::new_tagged(&ctx, 512, "qkv_warp:invalid")?;
        let y = DeviceBuffer::new_tagged(&ctx, 512, "qkv_warp:invalid")?;
        let z = DeviceBuffer::new_tagged(&ctx, 512, "qkv_warp:invalid")?;
        let t = DeviceBuffer::new_tagged(&ctx, 512, "qkv_warp:invalid")?;
        let outputs = [y.as_mut_ptr(), z.as_mut_ptr(), t.as_mut_ptr()];
        let before = profile::snapshot();
        unsafe {
            for enqueue in [
                CudaContext::qkv_f32xq8_0_gguf_warp_async,
                CudaContext::qkv_f32xq8_0_gguf_packed_warp_async,
            ] {
                for (k, widths) in [
                    (0, [1; 3]),
                    (33, [1; 3]),
                    (32, [0, 1, 1]),
                    (32, [i32::MAX; 3]),
                ] {
                    assert!(enqueue(
                        &ctx,
                        x.as_ptr(),
                        [w.as_ptr(); 3],
                        [None; 3],
                        outputs,
                        widths,
                        k,
                        CudaStream::Decode
                    )
                    .is_err());
                }
                assert!(enqueue(
                    &ctx,
                    x.as_ptr().cast::<u8>().add(2).cast(),
                    [w.as_ptr(); 3],
                    [None; 3],
                    outputs,
                    [1; 3],
                    32,
                    CudaStream::Decode
                )
                .is_err());
                assert!(enqueue(
                    &ctx,
                    x.as_ptr(),
                    [w.as_ptr(); 3],
                    [None; 3],
                    [y.as_mut_ptr(); 3],
                    [1; 3],
                    32,
                    CudaStream::Decode
                )
                .is_err());
                assert!(enqueue(
                    &ctx,
                    x.as_ptr(),
                    [w.as_ptr(); 3],
                    [Some(std::ptr::null()); 3],
                    outputs,
                    [1; 3],
                    32,
                    CudaStream::Decode
                )
                .is_err());
            }
            for offset in [4, 8, 12] {
                assert!(ctx
                    .qkv_f32xq8_0_gguf_packed_warp_async(
                        x.as_ptr().cast::<u8>().add(offset).cast(),
                        [w.as_ptr(); 3],
                        [None; 3],
                        outputs,
                        [1; 3],
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
