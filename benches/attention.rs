// benches/attention.rs
// Run with: cargo bench --features cuda --bench attention

#[cfg(all(feature = "cuda", nvcc))]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(not(all(feature = "cuda", nvcc)))]
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(all(feature = "cuda", nvcc))]
use m40_llm::cuda::KVCache;
#[cfg(all(feature = "cuda", nvcc))]
use std::ffi::c_void;

#[cfg(all(feature = "cuda", nvcc))]
#[path = "../tests/cuda_env.rs"]
mod cuda_env;

fn bench_attention_last_token_gqa(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_last_token_f32_gqa");

    #[cfg(all(feature = "cuda", nvcc))]
    {
        let ctx = cuda_env::ctx_m40().expect("cuda context");
        cuda_env::require_sm52(&ctx).expect("sm_52 device");

        let q_heads = 32u32;
        let kv_heads = 4u32;
        let head_dim = 64u32;
        let q_dim = (q_heads * head_dim) as usize;
        let kv_dim = (kv_heads * head_dim) as usize;
        let max_seq_len = 1024u32;
        let seq_lens = [1u32, 16, 128, 512, 1024];

        for seq_len in seq_lens {
            let kv = KVCache::new_with_context(&ctx, max_seq_len, 1, kv_heads, head_dim)
                .expect("kv cache");

            let bytes_kv = kv_dim * std::mem::size_of::<f32>();
            let d_k = ctx.device_malloc(bytes_kv).expect("d_k");
            let d_v = ctx.device_malloc(bytes_kv).expect("d_v");
            for t in 0..seq_len as usize {
                let k: Vec<f32> = (0..kv_dim)
                    .map(|i| ((t * kv_dim + i) as f32) * 0.0003 - 0.25)
                    .collect();
                let v: Vec<f32> = (0..kv_dim)
                    .map(|i| ((t * kv_dim + (kv_dim - 1 - i)) as f32) * 0.0002 - 0.1)
                    .collect();
                unsafe {
                    ctx.memcpy_h2d(d_k, k.as_ptr() as *const c_void, bytes_kv)
                        .expect("copy k");
                    ctx.memcpy_h2d(d_v, v.as_ptr() as *const c_void, bytes_kv)
                        .expect("copy v");
                    kv.append_token_f32(&ctx, 0, d_k as *const c_void, d_v as *const c_void)
                        .expect("append kv");
                }
            }

            let q: Vec<f32> = (0..q_dim).map(|i| (i as f32) * 0.0004 - 0.2).collect();
            let bytes_q = q_dim * std::mem::size_of::<f32>();
            let d_q = ctx.device_malloc(bytes_q).expect("d_q");
            let d_out = ctx.device_malloc(bytes_q).expect("d_out");
            unsafe {
                ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)
                    .expect("copy q");
            }

            group.throughput(Throughput::Elements(seq_len as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "q{q_heads}_kv{kv_heads}_d{head_dim}_s{seq_len}"
                )),
                &seq_len,
                |b, &seq_len| {
                    b.iter(|| unsafe {
                        kv.attention_last_token_f32_gqa(
                            &ctx,
                            0,
                            d_q as *const c_void,
                            q_heads,
                            seq_len,
                            d_out,
                        )
                        .expect("attention")
                    })
                },
            );

            unsafe {
                ctx.device_free(d_q).expect("free d_q");
                ctx.device_free(d_out).expect("free d_out");
                ctx.device_free(d_k).expect("free d_k");
                ctx.device_free(d_v).expect("free d_v");
            }
        }
    }

    #[cfg(not(all(feature = "cuda", nvcc)))]
    {
        group.bench_function("noop", |b| b.iter(|| {}));
    }

    group.finish();
}

#[cfg(all(feature = "cuda", nvcc))]
fn seed_kv_cache(ctx: &m40_llm::cuda::CudaContext, kv: &KVCache, seq_lens: &[u32], kv_dim: usize) {
    let bytes_kv = kv_dim * std::mem::size_of::<f32>();
    let d_k = ctx.device_malloc(bytes_kv).expect("d_k");
    let d_v = ctx.device_malloc(bytes_kv).expect("d_v");
    for (seq_idx, &seq_len) in seq_lens.iter().enumerate() {
        for t in 0..seq_len as usize {
            let k: Vec<f32> = (0..kv_dim)
                .map(|i| ((seq_idx * 97 + t * 13 + i) as f32) * 0.0003 - 0.25)
                .collect();
            let v: Vec<f32> = (0..kv_dim)
                .map(|i| ((seq_idx * 71 + t * 19 + kv_dim - i) as f32) * 0.0002 - 0.1)
                .collect();
            unsafe {
                ctx.memcpy_h2d(d_k, k.as_ptr() as *const c_void, bytes_kv)
                    .expect("copy k");
                ctx.memcpy_h2d(d_v, v.as_ptr() as *const c_void, bytes_kv)
                    .expect("copy v");
                kv.append_token_f32(&ctx, seq_idx as u32, d_k as *const c_void, d_v)
                    .expect("append kv");
            }
        }
    }
    unsafe {
        ctx.device_free(d_k).expect("free d_k");
        ctx.device_free(d_v).expect("free d_v");
    }
}

fn bench_attention_last_token_gqa_batched_varlen(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_last_token_f32_gqa_batched_varlen");

    #[cfg(all(feature = "cuda", nvcc))]
    {
        let ctx = cuda_env::ctx_m40().expect("cuda context");
        cuda_env::require_sm52(&ctx).expect("sm_52 device");

        let q_heads = 32u32;
        let kv_heads = 4u32;
        let head_dim = 64u32;
        let q_dim = (q_heads * head_dim) as usize;
        let kv_dim = (kv_heads * head_dim) as usize;
        let max_seq_len = 1024u32;
        let distributions: &[(&str, &[u32])] = &[
            ("avg_0p6_max", &[384, 512, 640, 768]),
            ("skewed", &[16, 64, 256, 1024]),
            ("near_uniform", &[896, 960, 1000, 1024]),
        ];

        for (name, seq_lens) in distributions {
            let batch_size = seq_lens.len() as u32;
            let kv = KVCache::new_with_context(&ctx, max_seq_len, batch_size, kv_heads, head_dim)
                .expect("kv cache");
            seed_kv_cache(&ctx, &kv, seq_lens, kv_dim);

            let q: Vec<f32> = (0..batch_size as usize * q_dim)
                .map(|i| (i as f32) * 0.0004 - 0.2)
                .collect();
            let seq_ids: Vec<u32> = (0..batch_size).collect();
            let bytes_q = q.len() * std::mem::size_of::<f32>();
            let bytes_out = bytes_q;
            let d_q = ctx.device_malloc(bytes_q).expect("d_q");
            let d_out = ctx.device_malloc(bytes_out).expect("d_out");
            let d_seq_ids = ctx
                .device_malloc(seq_ids.len() * std::mem::size_of::<u32>())
                .expect("d_seq_ids");
            let d_seq_lens = ctx
                .device_malloc(seq_lens.len() * std::mem::size_of::<u32>())
                .expect("d_seq_lens");
            unsafe {
                ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)
                    .expect("copy q");
                ctx.memcpy_h2d(
                    d_seq_ids,
                    seq_ids.as_ptr() as *const c_void,
                    seq_ids.len() * std::mem::size_of::<u32>(),
                )
                .expect("copy seq ids");
                ctx.memcpy_h2d(
                    d_seq_lens,
                    seq_lens.as_ptr() as *const c_void,
                    seq_lens.len() * std::mem::size_of::<u32>(),
                )
                .expect("copy seq lens");
            }

            let total_valid: u32 = seq_lens.iter().sum();
            group.throughput(Throughput::Elements(total_valid as u64));
            group.bench_function(format!("{name}_individual_dispatch"), |b| {
                b.iter(|| {
                    for (seq_idx, &seq_len) in seq_lens.iter().enumerate() {
                        let q_offset = seq_idx * q_dim * std::mem::size_of::<f32>();
                        let out_offset = seq_idx * q_dim * std::mem::size_of::<f32>();
                        unsafe {
                            kv.attention_last_token_f32_gqa(
                                &ctx,
                                seq_idx as u32,
                                (d_q as usize + q_offset) as *const c_void,
                                q_heads,
                                seq_len,
                                (d_out as usize + out_offset) as *mut c_void,
                            )
                            .expect("individual attention");
                        }
                    }
                })
            });
            group.bench_function(format!("{name}_batched_varlen"), |b| {
                b.iter(|| unsafe {
                    kv.attention_last_token_f32_gqa_batched(
                        &ctx,
                        d_seq_ids as *const u32,
                        d_seq_lens as *const u32,
                        batch_size,
                        d_q as *const c_void,
                        q_heads,
                        d_out,
                    )
                    .expect("batched attention")
                })
            });

            unsafe {
                ctx.device_free(d_q).expect("free d_q");
                ctx.device_free(d_out).expect("free d_out");
                ctx.device_free(d_seq_ids).expect("free d_seq_ids");
                ctx.device_free(d_seq_lens).expect("free d_seq_lens");
            }
        }
    }

    #[cfg(not(all(feature = "cuda", nvcc)))]
    {
        group.bench_function("noop", |b| b.iter(|| {}));
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_attention_last_token_gqa,
    bench_attention_last_token_gqa_batched_varlen
);
criterion_main!(benches);
