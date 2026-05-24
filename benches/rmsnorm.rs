// benches/rmsnorm.rs
// Run with: cargo bench --features cuda --bench rmsnorm

#[cfg(all(feature = "cuda", nvcc))]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(not(all(feature = "cuda", nvcc)))]
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(all(feature = "cuda", nvcc))]
use std::ffi::c_void;

#[cfg(all(feature = "cuda", nvcc))]
#[path = "../tests/cuda_env.rs"]
mod cuda_env;

fn bench_weighted_rmsnorm_cache_experiment(c: &mut Criterion) {
    let mut group = c.benchmark_group("rms_norm_f32_weighted_cache");

    #[cfg(all(feature = "cuda", nvcc))]
    {
        let ctx = cuda_env::ctx_m40().expect("cuda context");
        cuda_env::require_sm52(&ctx).expect("sm_52 device");
        let eps = 1e-5f32;
        let shapes = [(1usize, 2048usize), (4, 2048), (1, 4096), (4, 4096)];

        for (rows, dim) in shapes {
            let input: Vec<f32> = (0..rows * dim)
                .map(|i| ((i * 17 % 251) as f32) * 0.01 - 1.25)
                .collect();
            let weight: Vec<f32> = (0..dim).map(|i| 0.75 + i as f32 * 0.0003).collect();
            let input_bytes = input.len() * std::mem::size_of::<f32>();
            let weight_bytes = weight.len() * std::mem::size_of::<f32>();
            let d_in = ctx.device_malloc(input_bytes).expect("d_in");
            let d_weight = ctx.device_malloc(weight_bytes).expect("d_weight");
            let d_out = ctx.device_malloc(input_bytes).expect("d_out");
            unsafe {
                ctx.memcpy_h2d(d_in, input.as_ptr() as *const c_void, input_bytes)
                    .expect("copy input");
                ctx.memcpy_h2d(d_weight, weight.as_ptr() as *const c_void, weight_bytes)
                    .expect("copy weight");
            }

            group.throughput(Throughput::Elements((rows * dim) as u64));
            group.bench_with_input(
                BenchmarkId::new("default", format!("rows{rows}_dim{dim}")),
                &(rows, dim),
                |b, &(rows, dim)| {
                    b.iter(|| unsafe {
                        std::env::remove_var("M40LLM_CACHE_EXPERIMENT");
                        ctx.rms_norm_f32_weighted(
                            d_in,
                            d_weight as *const c_void,
                            d_out,
                            rows as u32,
                            dim as u32,
                            eps,
                            1,
                        )
                        .expect("default rmsnorm")
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new("ldg", format!("rows{rows}_dim{dim}")),
                &(rows, dim),
                |b, &(rows, dim)| {
                    b.iter(|| unsafe {
                        std::env::set_var("M40LLM_CACHE_EXPERIMENT", "ldg");
                        let result = ctx.rms_norm_f32_weighted(
                            d_in,
                            d_weight as *const c_void,
                            d_out,
                            rows as u32,
                            dim as u32,
                            eps,
                            1,
                        );
                        std::env::remove_var("M40LLM_CACHE_EXPERIMENT");
                        result.expect("ldg rmsnorm")
                    })
                },
            );

            unsafe {
                ctx.device_free(d_in).expect("free d_in");
                ctx.device_free(d_weight).expect("free d_weight");
                ctx.device_free(d_out).expect("free d_out");
            }
        }
        std::env::remove_var("M40LLM_CACHE_EXPERIMENT");
    }

    #[cfg(not(all(feature = "cuda", nvcc)))]
    {
        group.bench_function("noop", |b| b.iter(|| {}));
    }

    group.finish();
}

criterion_group!(benches, bench_weighted_rmsnorm_cache_experiment);
criterion_main!(benches);
