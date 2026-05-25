// Benchmark GGUF Q8_0 fused projection against existing F16 and materialized
// FP32 projection paths. Run with:
// M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo bench --features cuda --bench q8_projection

#[cfg(all(feature = "cuda", nvcc))]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(not(all(feature = "cuda", nvcc)))]
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(all(feature = "cuda", nvcc))]
use m40_llm::cuda::CudaStream;
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
fn f32s_to_halves_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = half::f16::from_f32(v).to_bits();
        out.push((bits & 0xff) as u8);
        out.push((bits >> 8) as u8);
    }
    out
}

#[cfg(all(feature = "cuda", nvcc))]
fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect()
}

#[cfg(all(feature = "cuda", nvcc))]
fn cpu_gguf_gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[row * k + kk] * b[col * k + kk];
            }
            c[row * n + col] = acc;
        }
    }
    c
}

#[cfg(all(feature = "cuda", nvcc))]
fn q8_0_gguf_bytes_from_dequantized(vals: &[f32], n: usize, k: usize) -> Vec<u8> {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let blocks_per_col = k.div_ceil(QK);
    let mut out = vec![0u8; n * blocks_per_col * BLOCK_BYTES];
    for col in 0..n {
        for block in 0..blocks_per_col {
            let start = block * QK;
            let end = (start + QK).min(k);
            let scale = vals[col * k + start..col * k + end]
                .iter()
                .fold(0.0f32, |acc, v| acc.max(v.abs()))
                / 127.0;
            let scale = if scale == 0.0 { 1.0 } else { scale };
            let base = (col * blocks_per_col + block) * BLOCK_BYTES;
            let scale_bits = half::f16::from_f32(scale).to_bits();
            out[base] = (scale_bits & 0xff) as u8;
            out[base + 1] = (scale_bits >> 8) as u8;
            for idx in 0..QK {
                let k_idx = start + idx;
                let q = if k_idx < k {
                    (vals[col * k + k_idx] / scale).round().clamp(-128.0, 127.0) as i8
                } else {
                    0
                };
                out[base + 2 + idx] = q as u8;
            }
        }
    }
    out
}

#[cfg(all(feature = "cuda", nvcc))]
fn q8_0_gguf_dequantize(bytes: &[u8], n: usize, k: usize) -> Vec<f32> {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let blocks_per_col = k.div_ceil(QK);
    let mut out = vec![0f32; n * k];
    for col in 0..n {
        for block in 0..blocks_per_col {
            let base = (col * blocks_per_col + block) * BLOCK_BYTES;
            let scale_bits = u16::from_le_bytes([bytes[base], bytes[base + 1]]);
            let scale = half::f16::from_bits(scale_bits).to_f32();
            for idx in 0..QK {
                let k_idx = block * QK + idx;
                if k_idx < k {
                    out[col * k + k_idx] = f32::from(bytes[base + 2 + idx] as i8) * scale;
                }
            }
        }
    }
    out
}

#[cfg(all(feature = "cuda", nvcc))]
fn max_mean_abs_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_diff = 0.0f32;
    let mut sum = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = (x - y).abs();
        max_diff = max_diff.max(diff);
        sum += diff;
    }
    (max_diff, sum / a.len().max(1) as f32)
}

fn bench_q8_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("q8_projection");

    #[cfg(all(feature = "cuda", nvcc))]
    {
        let ctx = cuda_env::ctx_m40().expect("cuda context");
        let cases = vec![
            ("tiny", 2, 3, 35),
            ("llama_decode_q", 1, 2048, 2048),
            ("llama_decode_mlp", 1, 2048, 8192),
            ("qwen_decode_q", 1, 2048, 2048),
            ("qwen_decode_mlp", 1, 2048, 11008),
            ("qwen_prefill64_q", 64, 2048, 2048),
        ];

        for (label, m, k, n) in cases {
            let a: Vec<f32> = (0..m * k)
                .map(|idx| ((idx % 17) as f32 - 8.0) * 0.03125)
                .collect();
            let b: Vec<f32> = (0..n * k)
                .map(|idx| ((idx % 23) as f32 - 11.0) * 0.015625)
                .collect();
            let a_bytes = f32s_to_bytes(&a);
            let b_f16 = f32s_to_halves_bytes(&b);
            let b_q8 = q8_0_gguf_bytes_from_dequantized(&b, n, k);
            let b_deq = q8_0_gguf_dequantize(&b_q8, n, k);

            let bytes_a = m * k * std::mem::size_of::<f32>();
            let bytes_c = m * n * std::mem::size_of::<f32>();
            let bytes_bt = n * k * std::mem::size_of::<f32>();

            let da = ctx.device_malloc(bytes_a).unwrap();
            let db_f16 = ctx.device_malloc(b_f16.len()).unwrap();
            let db_q8 = ctx.device_malloc(b_q8.len()).unwrap();
            let db_f32 = ctx.device_malloc(bytes_bt).unwrap();
            let dc = ctx.device_malloc(bytes_c).unwrap();

            unsafe {
                ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)
                    .unwrap();
                ctx.memcpy_h2d(db_f16, b_f16.as_ptr() as *const c_void, b_f16.len())
                    .unwrap();
                ctx.memcpy_h2d(db_q8, b_q8.as_ptr() as *const c_void, b_q8.len())
                    .unwrap();
                ctx.materialize_gguf_f16_to_f32_colmajor_nt(
                    db_f16 as *const c_void,
                    db_f32,
                    n as i32,
                    k as i32,
                )
                .unwrap();
            }

            if m * n <= 8192 {
                let expect = cpu_gguf_gemm_f32(&a, &b_deq, m, n, k);
                unsafe {
                    ctx.gemm_f32xq8_0_gguf_f32(
                        da as *const c_void,
                        db_q8 as *const c_void,
                        dc,
                        m as i32,
                        n as i32,
                        k as i32,
                    )
                    .unwrap();
                }
                let mut got_bytes = vec![0u8; bytes_c];
                unsafe {
                    ctx.memcpy_d2h(
                        got_bytes.as_mut_ptr() as *mut c_void,
                        dc as *const c_void,
                        got_bytes.len(),
                    )
                    .unwrap();
                }
                let got = bytes_to_f32s(&got_bytes);
                let (max_diff, mean_diff) = max_mean_abs_diff(&got, &expect);
                eprintln!(
                    "[q8_projection] correctness label={label} max_abs_diff={max_diff:.6} mean_abs_diff={mean_diff:.6}"
                );
            }

            let moved_q8 = bytes_a + b_q8.len() + bytes_c;
            let moved_f16 = bytes_a + b_f16.len() + bytes_c;
            let moved_f32 = bytes_a + bytes_bt + bytes_c;
            let ops = 2u64 * m as u64 * n as u64 * k as u64;

            group.throughput(Throughput::Bytes(moved_q8 as u64));
            group.bench_with_input(
                BenchmarkId::new("q8_generic", format!("{label}_{m}x{k}x{n}")),
                &(m, n, k),
                |bch, &(m, n, k)| {
                    bch.iter(|| unsafe {
                        ctx.gemm_f32xq8_0_gguf_f32_generic_async(
                            da as *const c_void,
                            db_q8 as *const c_void,
                            dc,
                            m as i32,
                            n as i32,
                            k as i32,
                        )
                        .unwrap();
                        ctx.synchronize_stream(CudaStream::Prefill).unwrap();
                    })
                },
            );

            if m == 1 && k % 32 == 0 {
                group.throughput(Throughput::Bytes(moved_q8 as u64));
                group.bench_with_input(
                    BenchmarkId::new("q8_decode_tiled", format!("{label}_{m}x{k}x{n}")),
                    &(m, n, k),
                    |bch, &(m, n, k)| {
                        bch.iter(|| unsafe {
                            ctx.gemm_f32xq8_0_gguf_f32_decode_async(
                                da as *const c_void,
                                db_q8 as *const c_void,
                                dc,
                                m as i32,
                                n as i32,
                                k as i32,
                            )
                            .unwrap();
                            ctx.synchronize_stream(CudaStream::Prefill).unwrap();
                        })
                    },
                );
            }

            group.throughput(Throughput::Bytes(moved_f16 as u64));
            group.bench_with_input(
                BenchmarkId::new("f16_gguf_kernel", format!("{label}_{m}x{k}x{n}")),
                &(m, n, k),
                |bch, &(m, n, k)| {
                    bch.iter(|| unsafe {
                        ctx.gemm_f32xf16_gguf_f32(
                            da as *const c_void,
                            db_f16 as *const c_void,
                            dc,
                            m as i32,
                            n as i32,
                            k as i32,
                        )
                        .unwrap()
                    })
                },
            );

            group.throughput(Throughput::Bytes(moved_f32 as u64));
            group.bench_with_input(
                BenchmarkId::new("materialized_f32_cublas", format!("{label}_{m}x{k}x{n}")),
                &(m, n, k),
                |bch, &(m, n, k)| {
                    bch.iter(|| unsafe {
                        ctx.gemm_f32xf32_f32(
                            da as *const c_void,
                            db_f32 as *const c_void,
                            dc,
                            m as i32,
                            n as i32,
                            k as i32,
                        )
                        .unwrap()
                    })
                },
            );

            eprintln!(
                "[q8_projection] shape={label} M={m} K={k} N={n} ops={ops} bytes_q8={moved_q8} bytes_f16={moved_f16} bytes_f32={moved_f32}"
            );

            unsafe {
                ctx.device_free(da).unwrap();
                ctx.device_free(db_f16).unwrap();
                ctx.device_free(db_q8).unwrap();
                ctx.device_free(db_f32).unwrap();
                ctx.device_free(dc).unwrap();
            }
        }
    }

    #[cfg(not(all(feature = "cuda", nvcc)))]
    {
        group.bench_function("noop", |b| b.iter(|| {}));
    }

    group.finish();
}

criterion_group!(benches, bench_q8_projection);
criterion_main!(benches);
