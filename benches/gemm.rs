// benches/gemm.rs
// Run with: cargo bench --features cuda [and ensure nvcc present].
// If nvcc is not present, this benchmark will be compiled but will immediately return.

#[cfg(all(feature = "cuda", nvcc))]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(not(all(feature = "cuda", nvcc)))]
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(all(feature = "cuda", nvcc))]
use std::ffi::c_void;

fn bench_gemm_impl(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_f16xf16_f32");

    // Only meaningful under CUDA + nvcc. Otherwise, early return.
    #[cfg(all(feature = "cuda", nvcc))]
    {
        use m40_llm::cuda::CudaContext;

        // Prefer Tesla M40 (sm_52) using the same helper pattern as tests
        fn ctx_m40() -> CudaContext {
            CudaContext::new(-1).expect("cuda context")
        }
        let ctx = ctx_m40();
        // Shapes to test
        let cases = vec![
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
        ];
        for (m, n, k) in cases {
            let bytes_a = (m * k * 2) as usize;
            let bytes_b = (k * n * 2) as usize;
            let bytes_c = (m * n * 4) as usize; // f32 output
            let da = ctx.device_malloc(bytes_a).unwrap();
            let db = ctx.device_malloc(bytes_b).unwrap();
            let dc = ctx.device_malloc(bytes_c).unwrap();

            // Fill with deterministic data on host
            let mut ha = vec![0u8; bytes_a];
            let mut hb = vec![0u8; bytes_b];
            for i in 0..(bytes_a / 2) {
                let v = half::f16::from_f32(((i % 13) as f32) * 0.1);
                let b = v.to_bits();
                ha[2 * i] = (b & 0xFF) as u8;
                ha[2 * i + 1] = (b >> 8) as u8;
            }
            for i in 0..(bytes_b / 2) {
                let v = half::f16::from_f32(((i % 17) as f32) * 0.1);
                let b = v.to_bits();
                hb[2 * i] = (b & 0xFF) as u8;
                hb[2 * i + 1] = (b >> 8) as u8;
            }

            unsafe {
                ctx.memcpy_h2d(da, ha.as_ptr() as *const c_void, bytes_a)
                    .unwrap();
                ctx.memcpy_h2d(db, hb.as_ptr() as *const c_void, bytes_b)
                    .unwrap();
            }

            group.throughput(Throughput::Bytes((bytes_a + bytes_b + bytes_c) as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{m}x{k}x{n}")),
                &(m, n, k),
                |b, &(m, n, k)| {
                    b.iter(|| unsafe {
                        ctx.gemm_f16xf16_f32(da as *const c_void, db as *const c_void, dc, m, n, k)
                            .unwrap()
                    })
                },
            );

            // Cleanup
            unsafe {
                ctx.device_free(da).unwrap();
                ctx.device_free(db).unwrap();
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

criterion_group!(benches, bench_gemm_impl);
criterion_main!(benches);
