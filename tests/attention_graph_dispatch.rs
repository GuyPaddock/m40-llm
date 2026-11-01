#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/env_restore.rs"]
mod env_restore;

use anyhow::{ensure, Result};
use env_restore::EnvRestore;
use half::f16;
use m40_llm::cuda::{ffi_debug_read_kv_token, CudaContext, CudaStream, DeviceBuffer, KVCache};
use std::ffi::c_void;

const GUARD: f32 = -12345.0;
const Q_HEADS: u32 = 4;
const KV_HEADS: u32 = 2;

struct FloatBuffer {
    device: DeviceBuffer,
    length: usize,
}

impl FloatBuffer {
    fn new(ctx: &CudaContext, values: &[f32]) -> Result<Self> {
        let buffer = Self {
            device: DeviceBuffer::new_tagged(ctx, values.len() * 4, "graph-dispatch:buffer")?,
            length: values.len(),
        };
        buffer.upload(ctx, values)?;
        Ok(buffer)
    }

    fn upload(&self, ctx: &CudaContext, values: &[f32]) -> Result<()> {
        ensure!(values.len() == self.length, "buffer length changed");
        // All previous consumers finish before host updates; allocation owns
        // exactly values.len() FP32 elements on this context.
        unsafe {
            ctx.memcpy_h2d(
                self.device.as_mut_ptr(),
                values.as_ptr().cast(),
                self.length * 4,
            )
        }
    }

    fn read(&self, ctx: &CudaContext) -> Result<Vec<f32>> {
        let mut values = vec![0.0; self.length];
        // Callers drain the producer stream before readback.
        unsafe {
            ctx.memcpy_d2h(
                values.as_mut_ptr().cast(),
                self.device.as_ptr(),
                self.length * 4,
            )?;
        }
        Ok(values)
    }

    fn pointer(&self, offset: usize) -> *mut c_void {
        assert!(offset < self.length);
        self.device.as_mut_ptr().wrapping_byte_add(offset * 4)
    }
}

fn compare_dispatch(capacity: u32, lengths: &[u32]) -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    let mut differences = 0;
    let mut comparisons = 0;
    for (head_dim, kernel, cache) in [
        (64, "native", "off"),
        (64, "native", "ldg_kv"),
        (128, "native", "off"),
        (128, "shared", "off"),
        (128, "warp32", "off"),
        (7, "native", "off"),
    ] {
        let _kernel = EnvRestore::set("M40LLM_ATTN_HEAD128_KERNEL", kernel);
        let _cache = EnvRestore::set("M40LLM_CACHE_EXPERIMENT", cache);
        let kv = KVCache::new_with_context(&ctx, capacity, 3, KV_HEADS, head_dim)?;
        let token_width = (KV_HEADS * head_dim) as usize;
        let query_width = (Q_HEADS * head_dim) as usize;
        let keys: Vec<f32> = (0..capacity as usize * token_width * 2)
            .map(|i| ((i * 31 + i / 17 + 13) % 521) as f32 / 173.0 - 1.5)
            .collect();
        let values: Vec<f32> = (0..keys.len())
            .map(|i| ((i * 19 + i / 23 + 7) % 503) as f32 / 167.0 - 1.5)
            .collect();
        let input_k = FloatBuffer::new(&ctx, &keys)?;
        let input_v = FloatBuffer::new(&ctx, &values)?;
        for (index, slot) in [0, 2].into_iter().enumerate() {
            for token in 0..capacity {
                let offset = (index * capacity as usize + token as usize) * token_width;
                // Inputs remain owned until all appends finish on decode.
                // The ordinary append preserves these already prepared keys.
                unsafe {
                    kv.append_token_f32_async(
                        &ctx,
                        slot,
                        input_k.pointer(offset),
                        input_v.pointer(offset),
                    )?;
                }
            }
        }
        ctx.synchronize_stream(CudaStream::Decode)?;
        let query = FloatBuffer::new(&ctx, &vec![0.0; query_width])?;
        let reference = FloatBuffer::new(&ctx, &vec![GUARD; query_width + 8])?;
        let candidate = FloatBuffer::new(&ctx, &vec![GUARD; query_width + 8])?;
        let device_length = DeviceBuffer::new_tagged(&ctx, 4, "graph-dispatch:length")?;
        for slot in [0, 2] {
            // One graph per physical slot; every length and query below reuses
            // that executable and its stable owned addresses.
            let graph = ctx.capture_graph(CudaStream::Decode, || unsafe {
                kv.attention_last_token_f32_gqa_seq_len_dev_async(
                    &ctx,
                    slot,
                    query.pointer(0),
                    Q_HEADS,
                    device_length.as_ptr().cast(),
                    candidate.pointer(4),
                )
            })?;
            for &length in lengths.iter().chain([0, capacity + 1].iter()) {
                let q: Vec<f32> = (0..query_width)
                    .map(|i| {
                        ((i * 29 + length as usize * 3 + slot as usize) % 257) as f32 / 71.0 - 1.8
                    })
                    .collect();
                query.upload(&ctx, &q)?;
                candidate.upload(&ctx, &vec![GUARD; query_width + 8])?;
                reference.upload(&ctx, &vec![GUARD; query_width + 8])?;
                // Host parameter upload completes before replay. Both output
                // regions have four FP32 guards on each side.
                unsafe {
                    ctx.memcpy_h2d(
                        device_length.as_mut_ptr(),
                        (&length as *const u32).cast(),
                        4,
                    )?;
                    if length > 0 && length <= capacity {
                        kv.attention_last_token_f32_gqa_async(
                            &ctx,
                            slot,
                            query.pointer(0),
                            Q_HEADS,
                            length,
                            reference.pointer(4),
                        )?;
                    }
                }
                graph.launch(CudaStream::Decode)?;
                ctx.synchronize_stream(CudaStream::Decode)?;
                let expected = reference.read(&ctx)?;
                let actual = candidate.read(&ctx)?;
                assert_eq!(&actual[..4], &[GUARD; 4]);
                assert_eq!(&actual[query_width + 4..], &[GUARD; 4]);
                assert_eq!(query.read(&ctx)?, q, "query mutated");
                let mut read_length = 0_u32;
                unsafe {
                    ctx.memcpy_d2h(
                        (&mut read_length as *mut u32).cast(),
                        device_length.as_ptr(),
                        4,
                    )?;
                }
                assert_eq!(read_length, length, "length mutated");
                let mismatch = actual
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert!(actual.iter().all(|x| x.is_finite()));
                if mismatch > 0 {
                    eprintln!("dispatch mismatch head={head_dim} kernel={kernel} cache={cache} slot={slot} length={length} elements={mismatch}");
                }
                differences += mismatch;
                comparisons += 1;
            }
        }
        // Verify every initialized cache element remains the input FP16 value,
        // including the other slot and lengths, after all graph executions.
        for (index, slot) in [0, 2].into_iter().enumerate() {
            assert_eq!(kv.debug_sequence_length(&ctx, slot)?, capacity);
            for token in 0..capacity {
                let mut k = vec![0_u16; token_width];
                let mut v = vec![0_u16; token_width];
                unsafe {
                    ensure!(
                        ffi_debug_read_kv_token(
                            &ctx,
                            &kv,
                            slot,
                            token,
                            k.as_mut_ptr().cast(),
                            v.as_mut_ptr().cast()
                        ) == 0,
                        "KV read failed"
                    );
                }
                let offset = (index * capacity as usize + token as usize) * token_width;
                for d in 0..token_width {
                    assert_eq!(
                        k[d],
                        f16::from_f32(keys[offset + d]).to_bits(),
                        "key changed"
                    );
                    assert_eq!(
                        v[d],
                        f16::from_f32(values[offset + d]).to_bits(),
                        "value changed"
                    );
                }
            }
        }
        assert_eq!(kv.debug_sequence_length(&ctx, 1)?, 0, "unused slot changed");
    }
    assert_eq!(
        CudaContext::total_device_bytes(),
        initial,
        "allocation leak"
    );
    eprintln!("graph dispatch capacity={capacity} comparisons={comparisons} unequal_elements={differences} guards_inputs_cleanup=pass");
    ensure!(
        differences == 0,
        "graph dispatch must preserve exact FP32 outputs"
    );
    Ok(())
}

/// Replay changing lengths and queries through each ordinary reduction choice,
/// preserving exact outputs, cache/input bytes, guards and allocation ownership.
#[test]
#[ignore = "serial M40 graph dispatch qualification"]
fn test_graph_attention_short_dispatch() -> Result<()> {
    compare_dispatch(129, &[1, 2, 3, 31, 32, 33, 127, 128, 129, 3])
}

/// One captured graph must cross the 8192-token specialization limit in both
/// directions without retaining stale output or changing the ordinary reduction.
#[test]
#[ignore = "serial M40 graph dispatch boundary qualification"]
fn test_graph_attention_dispatch_boundary() -> Result<()> {
    compare_dispatch(8193, &[3, 8191, 8192, 8193, 33])
}

/// An unsupported parallel backend fails before writing output or submitting
/// graph work; it must not silently execute the native reduction instead.
#[test]
#[ignore = "serial M40 graph dispatch compatibility"]
fn test_graph_attention_rejects_parallel_dispatch() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _kernel = EnvRestore::set("M40LLM_ATTN_HEAD128_KERNEL", "parallel");
    let _partitions = EnvRestore::set("M40LLM_ATTN_HEAD128_PARTITIONS", "2");
    let _threads = EnvRestore::set("M40LLM_ATTN_HEAD128_THREADS", "128");
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let kv = KVCache::new_with_context(&ctx, 4, 1, 1, 128)?;
        let query = FloatBuffer::new(&ctx, &[0.0; 128])?;
        let output = FloatBuffer::new(&ctx, &[GUARD; 128])?;
        let length = DeviceBuffer::new_tagged(&ctx, 4, "graph-dispatch:length")?;
        let value = 1_u32;
        unsafe {
            ctx.memcpy_h2d(length.as_mut_ptr(), (&value as *const u32).cast(), 4)?;
        }
        let result = ctx.capture_graph(CudaStream::Decode, || unsafe {
            kv.attention_last_token_f32_gqa_seq_len_dev_async(
                &ctx,
                0,
                query.pointer(0),
                1,
                length.as_ptr().cast(),
                output.pointer(0),
            )
        });
        let error = match result {
            Ok(_) => anyhow::bail!("parallel device-length capture unexpectedly succeeded"),
            Err(error) => error,
        };
        ensure!(
            error.to_string().contains("parallel attention"),
            "wrong rejection: {error}"
        );
        ctx.synchronize_stream(CudaStream::Decode)?;
        assert_eq!(output.read(&ctx)?, vec![GUARD; 128]);
        assert_eq!(kv.debug_sequence_length(&ctx, 0)?, 0);
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}
