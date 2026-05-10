// benches/persistent_decode.rs
// Run with: cargo bench --features cuda --bench persistent_decode

#[cfg(all(feature = "cuda", nvcc))]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(not(all(feature = "cuda", nvcc)))]
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(all(feature = "cuda", nvcc))]
use m40_llm::cuda::PersistentDecodeStatus;
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

fn bench_persistent_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistent_decode_vec");

    #[cfg(all(feature = "cuda", nvcc))]
    {
        let ctx = cuda_env::ctx_m40().expect("cuda context");
        cuda_env::require_sm52(&ctx).expect("sm_52 device");
        let n = 2048usize;
        let iterations = 4u32;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.001 - 1.0).collect();
        let bytes = input.len() * std::mem::size_of::<f32>();
        let d_in = ctx.device_malloc(bytes).expect("d_in");
        let d_tmp = ctx.device_malloc(bytes).expect("d_tmp");
        let d_out = ctx.device_malloc(bytes).expect("d_out");

        unsafe {
            ctx.memcpy_h2d(d_in, f32s_to_bytes(&input).as_ptr() as *const c_void, bytes)
                .expect("copy input");
        }

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("launch_residual_add", n), &n, |b, _| {
            b.iter(|| unsafe {
                for _ in 0..iterations {
                    ctx.residual_add_f32(d_in as *const c_void, d_in as *const c_void, d_tmp, n)
                        .expect("residual add");
                }
            })
        });

        ctx.start_persistent_decode()
            .expect("start persistent decode");
        group.bench_with_input(BenchmarkId::new("persistent_worker", n), &n, |b, _| {
            b.iter(|| unsafe {
                let command_id = ctx
                    .persistent_decode_submit_vec(
                        d_in as *const c_void,
                        d_out,
                        n as u32,
                        1.0001,
                        0.001,
                        iterations,
                    )
                    .expect("submit persistent vec");
                loop {
                    let poll = ctx.persistent_decode_poll().expect("poll persistent vec");
                    if poll.status == PersistentDecodeStatus::Done && poll.command_id == command_id
                    {
                        break;
                    }
                    std::hint::spin_loop();
                }
            })
        });
        ctx.stop_persistent_decode()
            .expect("stop persistent decode");

        unsafe {
            ctx.device_free(d_in).expect("free d_in");
            ctx.device_free(d_tmp).expect("free d_tmp");
            ctx.device_free(d_out).expect("free d_out");
        }
    }

    #[cfg(not(all(feature = "cuda", nvcc)))]
    {
        group.bench_function("cuda_unavailable", |b| b.iter(|| ()));
    }

    group.finish();
}

criterion_group!(benches, bench_persistent_decode);
criterion_main!(benches);
