// benches/attention.rs
// Run with: cargo bench --features cuda --bench attention

#[cfg(all(feature = "cuda", nvcc))]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(not(all(feature = "cuda", nvcc)))]
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(all(feature = "cuda", nvcc))]
use m40_llm::cuda::{ExactOldBacking, KVCache};
#[cfg(all(feature = "cuda", nvcc))]
use m40_llm::infer::{BatchMetadata, BatchSequence};
#[cfg(all(feature = "cuda", nvcc))]
use m40_llm::kv_compression::KvRepresentativePolicy;
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
            std::env::remove_var("M40LLM_CACHE_EXPERIMENT");
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
            std::env::set_var("M40LLM_CACHE_EXPERIMENT", "ldg_kv");
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "q{q_heads}_kv{kv_heads}_d{head_dim}_s{seq_len}_ldg_kv"
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
                        .expect("attention ldg_kv")
                    })
                },
            );
            std::env::remove_var("M40LLM_CACHE_EXPERIMENT");

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
                kv.append_token_f32(ctx, seq_idx as u32, d_k as *const c_void, d_v)
                    .expect("append kv");
            }
        }
    }
    unsafe {
        ctx.device_free(d_k).expect("free d_k");
        ctx.device_free(d_v).expect("free d_v");
    }
}

#[cfg(all(feature = "cuda", nvcc))]
fn seed_kv_cache_with_explicit_positions(
    ctx: &m40_llm::cuda::CudaContext,
    kv: &KVCache,
    seq_len: u32,
    kv_dim: usize,
) {
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
            kv.append_token_f32_rope_k_at_async(
                ctx,
                0,
                d_k as *const c_void,
                d_v as *const c_void,
                t as u32,
                t as u32,
                10_000.0,
                1.0,
            )
            .expect("append kv");
            ctx.synchronize_stream(m40_llm::cuda::CudaStream::Decode)
                .expect("sync append kv");
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
            let seq_ids_bytes = std::mem::size_of_val(&seq_ids[..]);
            let seq_lens_bytes = std::mem::size_of_val(*seq_lens);
            let d_seq_ids = ctx.device_malloc(seq_ids_bytes).expect("d_seq_ids");
            let d_seq_lens = ctx.device_malloc(seq_lens_bytes).expect("d_seq_lens");
            unsafe {
                ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)
                    .expect("copy q");
                ctx.memcpy_h2d(d_seq_ids, seq_ids.as_ptr() as *const c_void, seq_ids_bytes)
                    .expect("copy seq ids");
                ctx.memcpy_h2d(
                    d_seq_lens,
                    seq_lens.as_ptr() as *const c_void,
                    seq_lens_bytes,
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
            std::env::remove_var("M40LLM_CACHE_EXPERIMENT");
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
            std::env::set_var("M40LLM_CACHE_EXPERIMENT", "ldg_kv");
            group.bench_function(format!("{name}_batched_varlen_ldg_kv"), |b| {
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
                    .expect("batched attention ldg_kv")
                })
            });
            std::env::remove_var("M40LLM_CACHE_EXPERIMENT");

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

fn bench_attention_prefill_gqa_varlen(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_prefill_f32_gqa_varlen");

    #[cfg(all(feature = "cuda", nvcc))]
    {
        let ctx = cuda_env::ctx_m40().expect("cuda context");
        cuda_env::require_sm52(&ctx).expect("sm_52 device");

        let q_heads = 32u32;
        let kv_heads = 4u32;
        let head_dim = 64u32;
        let q_dim = (q_heads * head_dim) as usize;
        let kv_dim = (kv_heads * head_dim) as usize;
        let distributions: &[(&str, &[(u32, u32)])] = &[
            (
                "avg_0p6_max",
                &[(384, 384), (512, 512), (640, 640), (768, 768)],
            ),
            ("skewed", &[(16, 16), (64, 64), (256, 256), (1024, 1024)]),
            (
                "near_uniform",
                &[(896, 896), (960, 960), (1000, 1000), (1024, 1024)],
            ),
            (
                "prefix_query",
                &[(16, 512), (32, 640), (64, 768), (128, 1024)],
            ),
        ];

        for (name, lens) in distributions {
            let sequences: Vec<BatchSequence> = lens
                .iter()
                .map(|&(query_len, kv_len)| BatchSequence {
                    seq_len: kv_len,
                    query_len,
                    kv_len,
                })
                .collect();
            let meta = BatchMetadata::new(sequences).expect("batch metadata");
            let q_lens: Vec<u32> = meta.sequences().iter().map(|seq| seq.query_len).collect();
            let kv_lens: Vec<u32> = meta.sequences().iter().map(|seq| seq.kv_len).collect();
            let max_q_len = q_lens.iter().copied().max().expect("max q");
            let max_kv_len = kv_lens.iter().copied().max().expect("max kv");
            let q: Vec<f32> = (0..meta.total_q_tokens() as usize * q_dim)
                .map(|i| ((i * 17 % 251) as f32) * 0.0011 - 0.13)
                .collect();
            let k: Vec<f32> = (0..meta.total_kv_tokens() as usize * kv_dim)
                .map(|i| ((i * 23 % 263) as f32) * 0.0009 - 0.09)
                .collect();
            let v: Vec<f32> = (0..meta.total_kv_tokens() as usize * kv_dim)
                .map(|i| ((i * 29 % 269) as f32) * 0.0013 - 0.17)
                .collect();
            let bytes_q = q.len() * std::mem::size_of::<f32>();
            let bytes_kv = k.len() * std::mem::size_of::<f32>();
            let bytes_out = bytes_q;
            let bytes_offsets = meta.sequences().len() * std::mem::size_of::<u32>();
            let d_q = ctx.device_malloc(bytes_q).expect("d_q");
            let d_k = ctx.device_malloc(bytes_kv).expect("d_k");
            let d_v = ctx.device_malloc(bytes_kv).expect("d_v");
            let d_q_offsets = ctx.device_malloc(bytes_offsets).expect("d_q_offsets");
            let d_kv_offsets = ctx.device_malloc(bytes_offsets).expect("d_kv_offsets");
            let d_q_lens = ctx.device_malloc(bytes_offsets).expect("d_q_lens");
            let d_kv_lens = ctx.device_malloc(bytes_offsets).expect("d_kv_lens");
            let d_out = ctx.device_malloc(bytes_out).expect("d_out");

            let batch_size = meta.sequences().len();
            let padded_q: Vec<f32> = (0..batch_size * max_q_len as usize * q_dim)
                .map(|i| ((i * 17 % 251) as f32) * 0.0011 - 0.13)
                .collect();
            let padded_k: Vec<f32> = (0..batch_size * max_kv_len as usize * kv_dim)
                .map(|i| ((i * 23 % 263) as f32) * 0.0009 - 0.09)
                .collect();
            let padded_v: Vec<f32> = (0..batch_size * max_kv_len as usize * kv_dim)
                .map(|i| ((i * 29 % 269) as f32) * 0.0013 - 0.17)
                .collect();
            let padded_q_offsets: Vec<u32> =
                (0..batch_size).map(|idx| idx as u32 * max_q_len).collect();
            let padded_kv_offsets: Vec<u32> =
                (0..batch_size).map(|idx| idx as u32 * max_kv_len).collect();
            let padded_q_lens = vec![max_q_len; batch_size];
            let padded_kv_lens = vec![max_kv_len; batch_size];
            let bytes_padded_q = padded_q.len() * std::mem::size_of::<f32>();
            let bytes_padded_kv = padded_k.len() * std::mem::size_of::<f32>();
            let bytes_padded_out = bytes_padded_q;
            let d_padded_q = ctx.device_malloc(bytes_padded_q).expect("d_padded_q");
            let d_padded_k = ctx.device_malloc(bytes_padded_kv).expect("d_padded_k");
            let d_padded_v = ctx.device_malloc(bytes_padded_kv).expect("d_padded_v");
            let d_padded_q_offsets = ctx
                .device_malloc(bytes_offsets)
                .expect("d_padded_q_offsets");
            let d_padded_kv_offsets = ctx
                .device_malloc(bytes_offsets)
                .expect("d_padded_kv_offsets");
            let d_padded_q_lens = ctx.device_malloc(bytes_offsets).expect("d_padded_q_lens");
            let d_padded_kv_lens = ctx.device_malloc(bytes_offsets).expect("d_padded_kv_lens");
            let d_padded_out = ctx.device_malloc(bytes_padded_out).expect("d_padded_out");
            unsafe {
                ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)
                    .expect("copy q");
                ctx.memcpy_h2d(d_k, k.as_ptr() as *const c_void, bytes_kv)
                    .expect("copy k");
                ctx.memcpy_h2d(d_v, v.as_ptr() as *const c_void, bytes_kv)
                    .expect("copy v");
                ctx.memcpy_h2d(
                    d_q_offsets,
                    meta.q_offsets().as_ptr() as *const c_void,
                    bytes_offsets,
                )
                .expect("copy q offsets");
                ctx.memcpy_h2d(
                    d_kv_offsets,
                    meta.kv_offsets().as_ptr() as *const c_void,
                    bytes_offsets,
                )
                .expect("copy kv offsets");
                ctx.memcpy_h2d(d_q_lens, q_lens.as_ptr() as *const c_void, bytes_offsets)
                    .expect("copy q lens");
                ctx.memcpy_h2d(d_kv_lens, kv_lens.as_ptr() as *const c_void, bytes_offsets)
                    .expect("copy kv lens");
                ctx.memcpy_h2d(
                    d_padded_q,
                    padded_q.as_ptr() as *const c_void,
                    bytes_padded_q,
                )
                .expect("copy padded q");
                ctx.memcpy_h2d(
                    d_padded_k,
                    padded_k.as_ptr() as *const c_void,
                    bytes_padded_kv,
                )
                .expect("copy padded k");
                ctx.memcpy_h2d(
                    d_padded_v,
                    padded_v.as_ptr() as *const c_void,
                    bytes_padded_kv,
                )
                .expect("copy padded v");
                ctx.memcpy_h2d(
                    d_padded_q_offsets,
                    padded_q_offsets.as_ptr() as *const c_void,
                    bytes_offsets,
                )
                .expect("copy padded q offsets");
                ctx.memcpy_h2d(
                    d_padded_kv_offsets,
                    padded_kv_offsets.as_ptr() as *const c_void,
                    bytes_offsets,
                )
                .expect("copy padded kv offsets");
                ctx.memcpy_h2d(
                    d_padded_q_lens,
                    padded_q_lens.as_ptr() as *const c_void,
                    bytes_offsets,
                )
                .expect("copy padded q lens");
                ctx.memcpy_h2d(
                    d_padded_kv_lens,
                    padded_kv_lens.as_ptr() as *const c_void,
                    bytes_offsets,
                )
                .expect("copy padded kv lens");
            }

            struct BucketDispatch {
                batch_size: u32,
                d_q_offsets: *mut c_void,
                d_kv_offsets: *mut c_void,
                d_q_lens: *mut c_void,
                d_kv_lens: *mut c_void,
            }

            let mut bucket_dispatches = Vec::new();
            for bucket in meta.buckets() {
                let q_offsets: Vec<u32> = bucket
                    .sequence_indices
                    .iter()
                    .map(|&idx| meta.offsets()[idx].q_offset)
                    .collect();
                let kv_offsets: Vec<u32> = bucket
                    .sequence_indices
                    .iter()
                    .map(|&idx| meta.offsets()[idx].kv_offset)
                    .collect();
                let q_lens: Vec<u32> = bucket
                    .sequence_indices
                    .iter()
                    .map(|&idx| meta.sequences()[idx].query_len)
                    .collect();
                let kv_lens: Vec<u32> = bucket
                    .sequence_indices
                    .iter()
                    .map(|&idx| meta.sequences()[idx].kv_len)
                    .collect();
                let bytes = q_offsets.len() * std::mem::size_of::<u32>();
                let dispatch = BucketDispatch {
                    batch_size: q_offsets.len() as u32,
                    d_q_offsets: ctx.device_malloc(bytes).expect("bucket q offsets"),
                    d_kv_offsets: ctx.device_malloc(bytes).expect("bucket kv offsets"),
                    d_q_lens: ctx.device_malloc(bytes).expect("bucket q lens"),
                    d_kv_lens: ctx.device_malloc(bytes).expect("bucket kv lens"),
                };
                unsafe {
                    ctx.memcpy_h2d(
                        dispatch.d_q_offsets,
                        q_offsets.as_ptr() as *const c_void,
                        bytes,
                    )
                    .expect("copy bucket q offsets");
                    ctx.memcpy_h2d(
                        dispatch.d_kv_offsets,
                        kv_offsets.as_ptr() as *const c_void,
                        bytes,
                    )
                    .expect("copy bucket kv offsets");
                    ctx.memcpy_h2d(dispatch.d_q_lens, q_lens.as_ptr() as *const c_void, bytes)
                        .expect("copy bucket q lens");
                    ctx.memcpy_h2d(dispatch.d_kv_lens, kv_lens.as_ptr() as *const c_void, bytes)
                        .expect("copy bucket kv lens");
                }
                bucket_dispatches.push(dispatch);
            }

            group.throughput(Throughput::Elements(
                batch_size as u64 * max_q_len as u64 * max_kv_len as u64,
            ));
            group.bench_function(format!("{name}_padded_max"), |b| {
                b.iter(|| unsafe {
                    ctx.attention_prefill_f32_gqa_varlen(
                        d_padded_q as *const c_void,
                        d_padded_k as *const c_void,
                        d_padded_v as *const c_void,
                        d_padded_q_offsets as *const u32,
                        d_padded_kv_offsets as *const u32,
                        d_padded_q_lens as *const u32,
                        d_padded_kv_lens as *const u32,
                        batch_size as u32,
                        q_heads,
                        kv_heads,
                        head_dim,
                        d_padded_out,
                    )
                    .expect("padded prefill attention")
                })
            });

            group.throughput(Throughput::Elements(meta.total_attention_cells()));
            group.bench_function(format!("{name}_packed_varlen"), |b| {
                b.iter(|| unsafe {
                    ctx.attention_prefill_f32_gqa_varlen(
                        d_q as *const c_void,
                        d_k as *const c_void,
                        d_v as *const c_void,
                        d_q_offsets as *const u32,
                        d_kv_offsets as *const u32,
                        d_q_lens as *const u32,
                        d_kv_lens as *const u32,
                        meta.sequences().len() as u32,
                        q_heads,
                        kv_heads,
                        head_dim,
                        d_out,
                    )
                    .expect("prefill attention")
                })
            });
            group.bench_function(format!("{name}_bucketed_varlen"), |b| {
                b.iter(|| unsafe {
                    for dispatch in &bucket_dispatches {
                        ctx.attention_prefill_f32_gqa_varlen(
                            d_q as *const c_void,
                            d_k as *const c_void,
                            d_v as *const c_void,
                            dispatch.d_q_offsets as *const u32,
                            dispatch.d_kv_offsets as *const u32,
                            dispatch.d_q_lens as *const u32,
                            dispatch.d_kv_lens as *const u32,
                            dispatch.batch_size,
                            q_heads,
                            kv_heads,
                            head_dim,
                            d_out,
                        )
                        .expect("bucketed prefill attention");
                    }
                })
            });

            unsafe {
                ctx.device_free(d_q).expect("free d_q");
                ctx.device_free(d_k).expect("free d_k");
                ctx.device_free(d_v).expect("free d_v");
                ctx.device_free(d_q_offsets).expect("free d_q_offsets");
                ctx.device_free(d_kv_offsets).expect("free d_kv_offsets");
                ctx.device_free(d_q_lens).expect("free d_q_lens");
                ctx.device_free(d_kv_lens).expect("free d_kv_lens");
                ctx.device_free(d_out).expect("free d_out");
                ctx.device_free(d_padded_q).expect("free d_padded_q");
                ctx.device_free(d_padded_k).expect("free d_padded_k");
                ctx.device_free(d_padded_v).expect("free d_padded_v");
                ctx.device_free(d_padded_q_offsets)
                    .expect("free d_padded_q_offsets");
                ctx.device_free(d_padded_kv_offsets)
                    .expect("free d_padded_kv_offsets");
                ctx.device_free(d_padded_q_lens)
                    .expect("free d_padded_q_lens");
                ctx.device_free(d_padded_kv_lens)
                    .expect("free d_padded_kv_lens");
                ctx.device_free(d_padded_out).expect("free d_padded_out");
                for dispatch in bucket_dispatches {
                    ctx.device_free(dispatch.d_q_offsets)
                        .expect("free bucket q offsets");
                    ctx.device_free(dispatch.d_kv_offsets)
                        .expect("free bucket kv offsets");
                    ctx.device_free(dispatch.d_q_lens)
                        .expect("free bucket q lens");
                    ctx.device_free(dispatch.d_kv_lens)
                        .expect("free bucket kv lens");
                }
            }
        }
    }

    #[cfg(not(all(feature = "cuda", nvcc)))]
    {
        group.bench_function("noop", |b| b.iter(|| {}));
    }

    group.finish();
}

fn bench_attention_qwen_head128(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_qwen_head128");

    #[cfg(all(feature = "cuda", nvcc))]
    {
        let ctx = cuda_env::ctx_m40().expect("cuda context");
        cuda_env::require_sm52(&ctx).expect("sm_52 device");

        let q_heads = 16u32;
        let kv_heads = 2u32;
        let head_dim = 128u32;
        let q_dim = (q_heads * head_dim) as usize;
        let kv_dim = (kv_heads * head_dim) as usize;

        for target in [128u32, 256, 512] {
            let meta = BatchMetadata::new(vec![BatchSequence {
                seq_len: target,
                query_len: target,
                kv_len: target,
            }])
            .expect("qwen prefill metadata");
            let q: Vec<f32> = (0..meta.total_q_tokens() as usize * q_dim)
                .map(|i| ((i * 17 % 251) as f32) * 0.0011 - 0.13)
                .collect();
            let k: Vec<f32> = (0..meta.total_kv_tokens() as usize * kv_dim)
                .map(|i| ((i * 23 % 263) as f32) * 0.0009 - 0.09)
                .collect();
            let v: Vec<f32> = (0..meta.total_kv_tokens() as usize * kv_dim)
                .map(|i| ((i * 29 % 269) as f32) * 0.0013 - 0.17)
                .collect();
            let bytes_q = q.len() * std::mem::size_of::<f32>();
            let bytes_kv = k.len() * std::mem::size_of::<f32>();
            let bytes_offsets = std::mem::size_of::<u32>();
            let d_q = ctx.device_malloc(bytes_q).expect("d_q");
            let d_k = ctx.device_malloc(bytes_kv).expect("d_k");
            let d_v = ctx.device_malloc(bytes_kv).expect("d_v");
            let d_q_offsets = ctx.device_malloc(bytes_offsets).expect("d_q_offsets");
            let d_kv_offsets = ctx.device_malloc(bytes_offsets).expect("d_kv_offsets");
            let d_q_lens = ctx.device_malloc(bytes_offsets).expect("d_q_lens");
            let d_kv_lens = ctx.device_malloc(bytes_offsets).expect("d_kv_lens");
            let d_out = ctx.device_malloc(bytes_q).expect("d_out");
            unsafe {
                ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)
                    .expect("copy q");
                ctx.memcpy_h2d(d_k, k.as_ptr() as *const c_void, bytes_kv)
                    .expect("copy k");
                ctx.memcpy_h2d(d_v, v.as_ptr() as *const c_void, bytes_kv)
                    .expect("copy v");
                ctx.memcpy_h2d(
                    d_q_offsets,
                    meta.q_offsets().as_ptr() as *const c_void,
                    bytes_offsets,
                )
                .expect("copy q offsets");
                ctx.memcpy_h2d(
                    d_kv_offsets,
                    meta.kv_offsets().as_ptr() as *const c_void,
                    bytes_offsets,
                )
                .expect("copy kv offsets");
                ctx.memcpy_h2d(
                    d_q_lens,
                    &target as *const u32 as *const c_void,
                    bytes_offsets,
                )
                .expect("copy q lens");
                ctx.memcpy_h2d(
                    d_kv_lens,
                    &target as *const u32 as *const c_void,
                    bytes_offsets,
                )
                .expect("copy kv lens");
            }

            group.throughput(Throughput::Elements(target as u64 * target as u64));
            group.bench_function(format!("prefill_q16_kv2_d128_s{target}"), |b| {
                b.iter(|| unsafe {
                    ctx.attention_prefill_f32_gqa_varlen(
                        d_q as *const c_void,
                        d_k as *const c_void,
                        d_v as *const c_void,
                        d_q_offsets as *const u32,
                        d_kv_offsets as *const u32,
                        d_q_lens as *const u32,
                        d_kv_lens as *const u32,
                        1,
                        q_heads,
                        kv_heads,
                        head_dim,
                        d_out,
                    )
                    .expect("qwen prefill attention")
                })
            });

            unsafe {
                ctx.device_free(d_q).expect("free d_q");
                ctx.device_free(d_k).expect("free d_k");
                ctx.device_free(d_v).expect("free d_v");
                ctx.device_free(d_q_offsets).expect("free d_q_offsets");
                ctx.device_free(d_kv_offsets).expect("free d_kv_offsets");
                ctx.device_free(d_q_lens).expect("free d_q_lens");
                ctx.device_free(d_kv_lens).expect("free d_kv_lens");
                ctx.device_free(d_out).expect("free d_out");
            }
        }

        let recent_window = 128u32;
        let block_size = 32u32;
        for seq_len in [512u32, 2048] {
            let kv = KVCache::new_compressed_with_context(
                &ctx,
                seq_len,
                1,
                kv_heads,
                head_dim,
                recent_window,
                block_size,
                16,
                0,
                KvRepresentativePolicy::Last,
                ExactOldBacking::Fp16KQ4V,
            )
            .expect("qwen mixed kv cache");
            seed_kv_cache_with_explicit_positions(&ctx, &kv, seq_len, kv_dim);
            let q: Vec<f32> = (0..q_dim)
                .map(|i| ((i * 11 % 257) as f32) * 0.0004 - 0.2)
                .collect();
            let bytes_q = q_dim * std::mem::size_of::<f32>();
            let d_q = ctx.device_malloc(bytes_q).expect("d_q");
            let d_out = ctx.device_malloc(bytes_q).expect("d_out");
            unsafe {
                ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)
                    .expect("copy q");
            }

            for top_blocks in [4u32, 8, 16] {
                group.throughput(Throughput::Elements(
                    (recent_window + top_blocks * block_size) as u64,
                ));
                group.bench_function(
                    format!("direct_fp16_k_q4_v_q16_kv2_d128_s{seq_len}_top{top_blocks}"),
                    |b| {
                        b.iter(|| unsafe {
                            kv.attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct_async(
                                &ctx,
                                0,
                                d_q as *const c_void,
                                q_heads,
                                seq_len,
                                recent_window,
                                block_size,
                                top_blocks,
                                d_out,
                            )
                            .expect("qwen direct fp16-k/q4-v attention");
                            ctx.synchronize_stream(m40_llm::cuda::CudaStream::Decode)
                                .expect("sync qwen direct attention");
                        })
                    },
                );
            }

            unsafe {
                ctx.device_free(d_q).expect("free d_q");
                ctx.device_free(d_out).expect("free d_out");
            }
        }
    }

    #[cfg(not(all(feature = "cuda", nvcc)))]
    {
        group.bench_function("noop", |b| b.iter(|| {}));
    }

    group.finish();
}

fn bench_attention_kv_compression_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_kv_compression_modes");

    #[cfg(all(feature = "cuda", nvcc))]
    {
        let ctx = cuda_env::ctx_m40().expect("cuda context");
        cuda_env::require_sm52(&ctx).expect("sm_52 device");

        let q_heads = 32u32;
        let kv_heads = 4u32;
        let head_dim = 64u32;
        let q_dim = (q_heads * head_dim) as usize;
        let kv_dim = (kv_heads * head_dim) as usize;
        let recent_window = 1024u32;
        let block_size = 32u32;
        let top_blocks = 16u32;
        let seq_lens = [4096u32, 8192, 16384, 32768];

        for seq_len in seq_lens {
            let dense_kv =
                KVCache::new_with_context(&ctx, seq_len, 1, kv_heads, head_dim).expect("kv cache");
            seed_kv_cache_with_explicit_positions(&ctx, &dense_kv, seq_len, kv_dim);
            let compressed_kv = KVCache::new_compressed_with_context(
                &ctx,
                seq_len,
                1,
                kv_heads,
                head_dim,
                recent_window,
                block_size,
                top_blocks,
                0,
                KvRepresentativePolicy::Last,
                ExactOldBacking::Dense,
            )
            .expect("compressed kv cache");
            seed_kv_cache_with_explicit_positions(&ctx, &compressed_kv, seq_len, kv_dim);
            let q: Vec<f32> = (0..q_dim)
                .map(|i| ((i * 11 % 257) as f32) * 0.0004 - 0.2)
                .collect();
            let bytes_q = q_dim * std::mem::size_of::<f32>();
            let d_q = ctx.device_malloc(bytes_q).expect("d_q");
            let d_out = ctx.device_malloc(bytes_q).expect("d_out");
            unsafe {
                ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)
                    .expect("copy q");
            }

            let dense_kv_bytes = seq_len as usize * kv_dim * std::mem::size_of::<half::f16>() * 2;
            let summary_count = seq_len.saturating_sub(recent_window).div_ceil(block_size);
            let block_summary_bytes = (recent_window as usize + summary_count as usize)
                * kv_dim
                * std::mem::size_of::<half::f16>()
                * 2;
            let block_select_lossy_bytes = (recent_window as usize
                + top_blocks.min(summary_count) as usize)
                * kv_dim
                * std::mem::size_of::<half::f16>()
                * 2;
            let actual_compressed_bytes = compressed_kv.actual_bytes();
            eprintln!(
                "[kv-compress-bench] seq_len={seq_len} dense_kv_bytes={dense_kv_bytes} actual_compressed_kv_bytes={actual_compressed_bytes} block_summary_equiv_bytes={block_summary_bytes} block_select_lossy_equiv_bytes={block_select_lossy_bytes}"
            );

            group.throughput(Throughput::Elements(seq_len as u64));
            group.bench_function(format!("s{seq_len}_dense"), |b| {
                b.iter(|| unsafe {
                    dense_kv
                        .attention_last_token_f32_gqa(
                            &ctx,
                            0,
                            d_q as *const c_void,
                            q_heads,
                            seq_len,
                            d_out,
                        )
                        .expect("dense attention")
                })
            });
            group.bench_function(format!("s{seq_len}_block_select_exact"), |b| {
                b.iter(|| unsafe {
                    dense_kv
                        .attention_last_token_f32_gqa_block_select_exact_async(
                            &ctx,
                            0,
                            d_q as *const c_void,
                            q_heads,
                            seq_len,
                            recent_window,
                            block_size,
                            top_blocks,
                            d_out,
                        )
                        .expect("block-select-exact attention");
                    ctx.synchronize_stream(m40_llm::cuda::CudaStream::Decode)
                        .expect("sync block-select-exact");
                })
            });
            group.bench_function(format!("s{seq_len}_block_summary"), |b| {
                b.iter(|| unsafe {
                    compressed_kv
                        .attention_last_token_f32_gqa_block_summary_lossy_async(
                            &ctx,
                            0,
                            d_q as *const c_void,
                            q_heads,
                            seq_len,
                            recent_window,
                            block_size,
                            0,
                            d_out,
                        )
                        .expect("block-summary attention");
                    ctx.synchronize_stream(m40_llm::cuda::CudaStream::Decode)
                        .expect("sync block-summary");
                })
            });
            group.bench_function(format!("s{seq_len}_block_select_lossy"), |b| {
                b.iter(|| unsafe {
                    compressed_kv
                        .attention_last_token_f32_gqa_block_summary_lossy_async(
                            &ctx,
                            0,
                            d_q as *const c_void,
                            q_heads,
                            seq_len,
                            recent_window,
                            block_size,
                            top_blocks,
                            d_out,
                        )
                        .expect("block-select-lossy attention");
                    ctx.synchronize_stream(m40_llm::cuda::CudaStream::Decode)
                        .expect("sync block-select-lossy");
                })
            });

            unsafe {
                ctx.device_free(d_q).expect("free d_q");
                ctx.device_free(d_out).expect("free d_out");
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
    bench_attention_last_token_gqa_batched_varlen,
    bench_attention_prefill_gqa_varlen,
    bench_attention_qwen_head128,
    bench_attention_kv_compression_modes
);
criterion_main!(benches);
