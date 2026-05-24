// benches/stream_overlap.rs
// Run with: cargo bench --features cuda --bench stream_overlap

#[cfg(all(feature = "cuda", nvcc))]
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(not(all(feature = "cuda", nvcc)))]
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(all(feature = "cuda", nvcc))]
use m40_llm::cuda::{CudaStream, KVCache};
#[cfg(all(feature = "cuda", nvcc))]
use m40_llm::infer::{BatchMetadata, BatchSequence, VarlenPrefillPlan};
#[cfg(all(feature = "cuda", nvcc))]
use std::ffi::c_void;

#[cfg(all(feature = "cuda", nvcc))]
#[path = "../tests/cuda_env.rs"]
mod cuda_env;

#[cfg(all(feature = "cuda", nvcc))]
fn f32s_to_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(all(feature = "cuda", nvcc))]
struct DeviceVec<'a> {
    ctx: &'a m40_llm::cuda::CudaContext,
    ptr: *mut c_void,
}

#[cfg(all(feature = "cuda", nvcc))]
impl<'a> DeviceVec<'a> {
    fn new(ctx: &'a m40_llm::cuda::CudaContext, bytes: usize) -> anyhow::Result<Self> {
        Ok(Self {
            ctx,
            ptr: ctx.device_malloc(bytes)?,
        })
    }
}

#[cfg(all(feature = "cuda", nvcc))]
impl Drop for DeviceVec<'_> {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device_free(self.ptr);
        }
    }
}

fn bench_stream_overlap(c: &mut Criterion) {
    let mut group = c.benchmark_group("stream_overlap_prefill_decode_attention");

    #[cfg(all(feature = "cuda", nvcc))]
    {
        let ctx = cuda_env::ctx_m40().expect("cuda context");
        cuda_env::require_sm52(&ctx).expect("sm_52 device");

        let q_heads = 8u32;
        let kv_heads = 2u32;
        let head_dim = 64u32;

        let prefill_meta = BatchMetadata::new(vec![
            BatchSequence {
                seq_len: 384,
                query_len: 384,
                kv_len: 384,
            },
            BatchSequence {
                seq_len: 512,
                query_len: 512,
                kv_len: 512,
            },
            BatchSequence {
                seq_len: 640,
                query_len: 640,
                kv_len: 640,
            },
            BatchSequence {
                seq_len: 768,
                query_len: 768,
                kv_len: 768,
            },
        ])
        .expect("prefill metadata");
        let prefill_plan =
            VarlenPrefillPlan::new(&ctx, prefill_meta.clone(), head_dim).expect("prefill plan");
        let q_elems = prefill_meta.total_q_tokens() as usize * q_heads as usize * head_dim as usize;
        let kv_elems =
            prefill_meta.total_kv_tokens() as usize * kv_heads as usize * head_dim as usize;
        let out_elems = q_elems;
        let q: Vec<f32> = (0..q_elems)
            .map(|i| ((i * 17 % 251) as f32) * 0.0011 - 0.13)
            .collect();
        let k: Vec<f32> = (0..kv_elems)
            .map(|i| ((i * 23 % 263) as f32) * 0.0009 - 0.09)
            .collect();
        let v: Vec<f32> = (0..kv_elems)
            .map(|i| ((i * 29 % 269) as f32) * 0.0013 - 0.17)
            .collect();
        let d_prefill_q =
            DeviceVec::new(&ctx, q.len() * std::mem::size_of::<f32>()).expect("device allocation");
        let d_prefill_k =
            DeviceVec::new(&ctx, k.len() * std::mem::size_of::<f32>()).expect("device allocation");
        let d_prefill_v =
            DeviceVec::new(&ctx, v.len() * std::mem::size_of::<f32>()).expect("device allocation");
        let d_prefill_out = DeviceVec::new(&ctx, out_elems * std::mem::size_of::<f32>())
            .expect("device allocation");

        unsafe {
            ctx.memcpy_h2d(
                d_prefill_q.ptr,
                f32s_to_bytes(&q).as_ptr() as *const c_void,
                q.len() * std::mem::size_of::<f32>(),
            )
            .expect("cuda setup");
            ctx.memcpy_h2d(
                d_prefill_k.ptr,
                f32s_to_bytes(&k).as_ptr() as *const c_void,
                k.len() * std::mem::size_of::<f32>(),
            )
            .expect("cuda setup");
            ctx.memcpy_h2d(
                d_prefill_v.ptr,
                f32s_to_bytes(&v).as_ptr() as *const c_void,
                v.len() * std::mem::size_of::<f32>(),
            )
            .expect("cuda setup");
        }

        let decode_batch = 4u32;
        let decode_q_heads = 32u32;
        let decode_kv_heads = 4u32;
        let max_seq_len = 1024u32;
        let seq_ids = [0u32, 1, 2, 3];
        let seq_lens = [384u32, 512, 640, 768];
        let decode_q_dim = (decode_q_heads * head_dim) as usize;
        let decode_kv_dim = (decode_kv_heads * head_dim) as usize;
        let kv =
            KVCache::new_with_context(&ctx, max_seq_len, decode_batch, decode_kv_heads, head_dim)
                .expect("cuda setup");
        let d_k = DeviceVec::new(&ctx, decode_kv_dim * std::mem::size_of::<f32>())
            .expect("device allocation");
        let d_v = DeviceVec::new(&ctx, decode_kv_dim * std::mem::size_of::<f32>())
            .expect("device allocation");
        for (seq_idx, &seq_len) in seq_lens.iter().enumerate() {
            for t in 0..seq_len as usize {
                let k: Vec<f32> = (0..decode_kv_dim)
                    .map(|i| ((seq_idx * 97 + t * 13 + i) as f32) * 0.0007 - 0.35)
                    .collect();
                let v: Vec<f32> = (0..decode_kv_dim)
                    .map(|i| ((seq_idx * 71 + t * 19 + decode_kv_dim - i) as f32) * 0.0005 - 0.2)
                    .collect();
                unsafe {
                    ctx.memcpy_h2d(
                        d_k.ptr,
                        f32s_to_bytes(&k).as_ptr() as *const c_void,
                        decode_kv_dim * std::mem::size_of::<f32>(),
                    )
                    .expect("cuda setup");
                    ctx.memcpy_h2d(
                        d_v.ptr,
                        f32s_to_bytes(&v).as_ptr() as *const c_void,
                        decode_kv_dim * std::mem::size_of::<f32>(),
                    )
                    .expect("cuda setup");
                    kv.append_token_f32(&ctx, seq_idx as u32, d_k.ptr, d_v.ptr)
                        .expect("cuda setup");
                }
            }
        }
        let decode_q: Vec<f32> = (0..decode_batch as usize * decode_q_dim)
            .map(|i| (i as f32) * 0.0009 - 0.15)
            .collect();
        let d_decode_q = DeviceVec::new(&ctx, decode_q.len() * std::mem::size_of::<f32>())
            .expect("device allocation");
        let d_decode_out = DeviceVec::new(&ctx, decode_q.len() * std::mem::size_of::<f32>())
            .expect("device allocation");
        let d_seq_ids = DeviceVec::new(&ctx, seq_ids.len() * std::mem::size_of::<u32>())
            .expect("device allocation");
        let d_seq_lens = DeviceVec::new(&ctx, seq_lens.len() * std::mem::size_of::<u32>())
            .expect("device allocation");
        unsafe {
            ctx.memcpy_h2d(
                d_decode_q.ptr,
                f32s_to_bytes(&decode_q).as_ptr() as *const c_void,
                decode_q.len() * std::mem::size_of::<f32>(),
            )
            .expect("cuda setup");
            ctx.memcpy_h2d(
                d_seq_ids.ptr,
                seq_ids.as_ptr() as *const c_void,
                seq_ids.len() * std::mem::size_of::<u32>(),
            )
            .expect("cuda setup");
            ctx.memcpy_h2d(
                d_seq_lens.ptr,
                seq_lens.as_ptr() as *const c_void,
                seq_lens.len() * std::mem::size_of::<u32>(),
            )
            .expect("cuda setup");
        }

        group.bench_function("sequential_sync", |b| {
            b.iter(|| unsafe {
                prefill_plan
                    .dispatch_head64(
                        d_prefill_q.ptr,
                        d_prefill_k.ptr,
                        d_prefill_v.ptr,
                        q_heads,
                        kv_heads,
                        d_prefill_out.ptr,
                    )
                    .expect("prefill attention");
                kv.attention_last_token_f32_gqa_batched(
                    &ctx,
                    d_seq_ids.ptr as *const u32,
                    d_seq_lens.ptr as *const u32,
                    decode_batch,
                    d_decode_q.ptr,
                    decode_q_heads,
                    d_decode_out.ptr,
                )
                .expect("decode attention");
            })
        });

        group.bench_function("split_async_final_sync", |b| {
            b.iter(|| unsafe {
                prefill_plan
                    .dispatch_head64_async(
                        d_prefill_q.ptr,
                        d_prefill_k.ptr,
                        d_prefill_v.ptr,
                        q_heads,
                        kv_heads,
                        d_prefill_out.ptr,
                    )
                    .expect("enqueue prefill attention");
                kv.attention_last_token_f32_gqa_batched_async(
                    &ctx,
                    d_seq_ids.ptr as *const u32,
                    d_seq_lens.ptr as *const u32,
                    decode_batch,
                    d_decode_q.ptr,
                    decode_q_heads,
                    d_decode_out.ptr,
                )
                .expect("enqueue decode attention");
                ctx.synchronize_stream(CudaStream::Prefill)
                    .expect("sync prefill stream");
                ctx.synchronize_stream(CudaStream::Decode)
                    .expect("sync decode stream");
            })
        });
    }

    #[cfg(not(all(feature = "cuda", nvcc)))]
    {
        group.bench_function("cuda_unavailable", |b| b.iter(|| ()));
    }

    group.finish();
}

criterion_group!(benches, bench_stream_overlap);
criterion_main!(benches);
