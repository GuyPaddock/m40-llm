#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/q8_projection.rs"]
#[allow(dead_code)]
mod fixture;

use anyhow::Result;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    infer::execution_weights::{
        DeviceQ8FfnPair, HostQ8FfnPair, Q8MatrixShape, Q8SourceIdentity, Q8SourceView,
    },
    profile,
};
use std::ffi::c_void;

const K: usize = 2048;
const N: usize = 11008;
const GUARD: u32 = 0x4b123456;

/// Own a guarded device vector with a 32-byte-aligned logical origin.
struct Buffer(DeviceBuffer, usize);
impl Buffer {
    fn new(ctx: &CudaContext, values: &[f32]) -> Result<Self> {
        let buffer = Self(
            DeviceBuffer::new_tagged(ctx, (values.len() + 16) * 4, "ilp4:test")?,
            values.len(),
        );
        let mut host = vec![f32::from_bits(GUARD); values.len() + 16];
        host[8..8 + values.len()].copy_from_slice(values);
        unsafe {
            ctx.memcpy_h2d(buffer.0.as_mut_ptr(), host.as_ptr().cast(), host.len() * 4)?;
        }
        Ok(buffer)
    }
    fn ptr(&self) -> *mut c_void {
        unsafe { self.0.as_mut_ptr().cast::<f32>().add(8).cast() }
    }
    fn read(&self, ctx: &CudaContext) -> Result<Vec<f32>> {
        let mut host = vec![0f32; self.1 + 16];
        unsafe {
            ctx.memcpy_d2h(host.as_mut_ptr().cast(), self.0.as_ptr(), host.len() * 4)?;
        }
        assert!(host[..8]
            .iter()
            .chain(&host[8 + self.1..])
            .all(|v| v.to_bits() == GUARD));
        Ok(host[8..8 + self.1].to_vec())
    }
}

/// Exercise the one admitted shape against independent FP64 arithmetic for
/// signed/zero, cancellation and dynamic-range inputs. Optional capture must
/// not change output bits; guards, producer dependencies and owners survive.
#[test]
#[ignore = "requires M40; run with --include-ignored --test-threads=1"]
fn test_ilp4_numerics_and_ownership() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    let mut numerical_failures = 0usize;
    {
        let ids = ["gate", "up"].map(|name| {
            Q8SourceIdentity::from_column_blocks(
                [1; 32],
                name.into(),
                0,
                Q8MatrixShape::new(K, N).unwrap(),
            )
            .unwrap()
        });
        let mut bytes = [
            fixture::make_column_blocks(K, N),
            fixture::make_column_blocks(K, N),
        ];
        for (side, source) in bytes.iter_mut().enumerate() {
            for (block_index, block) in source.chunks_exact_mut(34).enumerate() {
                if block_index % 7 == 0 {
                    block[..2].copy_from_slice(
                        &(if block_index % 14 == 0 { 0u16 } else { 0x8000 }).to_le_bytes(),
                    );
                }
                for (lane, value) in block[2..].iter_mut().enumerate() {
                    *value = [0x80, 0x7f, 0xff, 0, 1, 0x40, 0xc0, 8]
                        [(lane + block_index + side * 3) % 8];
                }
            }
        }
        let host = HostQ8FfnPair::pack(
            Q8SourceView::from_tensor_bytes(&ids[0], &bytes[0])?,
            Q8SourceView::from_tensor_bytes(&ids[1], &bytes[1])?,
        )?;
        let pair = DeviceQ8FfnPair::upload(&ctx, &host)?;
        for (case, input) in [
            ("signed", fixture::make_input(1, K)),
            ("zero", vec![0.; K]),
            (
                "cancellation",
                (0..K)
                    .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
                    .collect(),
            ),
            (
                "dynamic-range",
                (0..K).map(|i| [8., -8., 1e-5, -1e-5][i % 4]).collect(),
            ),
        ] {
            let x = Buffer::new(&ctx, &vec![0.; K])?;
            let producer = Buffer::new(&ctx, &input)?;
            let outputs = [
                Buffer::new(&ctx, &vec![0.; N])?,
                Buffer::new(&ctx, &vec![0.; N])?,
                Buffer::new(&ctx, &vec![0.; N])?,
            ];
            unsafe {
                ctx.memcpy_d2d_async(x.ptr(), producer.ptr(), K * 4, CudaStream::Prefill)?;
            }
            ctx.stream_wait_for_stream(CudaStream::Decode, CudaStream::Prefill, "ilp4:producer")?;
            let mut control: Option<[Vec<f32>; 3]> = None;
            for candidate in [false, true] {
                let before = profile::snapshot();
                unsafe {
                    if candidate {
                        ctx.ffn_q8_joint_ilp4_f32_async(
                            x.ptr(),
                            pair.planes()[0].as_ptr(),
                            pair.planes()[1].as_ptr(),
                            outputs[0].ptr(),
                            1,
                            N as i32,
                            K as i32,
                            Some((outputs[1].ptr(), outputs[2].ptr())),
                        )?;
                    } else {
                        pair.enqueue_shared_scales_async(
                            x.ptr(),
                            outputs[0].ptr(),
                            1,
                            Some((outputs[1].ptr(), outputs[2].ptr())),
                        )?;
                    }
                }
                assert_eq!(
                    profile::counter_delta(&before, &profile::snapshot()),
                    profile::OpCounters {
                        launches: 1,
                        ..Default::default()
                    }
                );
                ctx.synchronize_stream(CudaStream::Decode)?;
                let actual = [
                    outputs[0].read(&ctx)?,
                    outputs[1].read(&ctx)?,
                    outputs[2].read(&ctx)?,
                ];
                let mut max_cpu_error = [0f64; 3];
                let mut sum_cpu_error = [0f64; 3];
                let mut failures = [0usize; 3];
                let mut first_failure = None;
                for (column, _) in actual[0].iter().enumerate() {
                    let g = fixture::project_cpu(&input, &bytes[0], 0, column, K);
                    let u = fixture::project_cpu(&input, &bytes[1], 0, column, K);
                    for (index, expected) in
                        [g / (1. + (-g).exp()) * u, g, u].into_iter().enumerate()
                    {
                        let error = (f64::from(actual[index][column]) - expected).abs();
                        if !actual[index][column].is_finite()
                            || error >= 1e-3 + 2e-5 * expected.abs()
                        {
                            failures[index] += 1;
                            first_failure.get_or_insert(serde_json::json!({"stage":index,"column":column,"error":error,"expected":expected,"actual":actual[index][column]}));
                        }
                        max_cpu_error[index] = max_cpu_error[index].max(error);
                        sum_cpu_error[index] += error;
                    }
                }
                eprintln!("ILP4 case={case} candidate={candidate} max_cpu_error={max_cpu_error:?} mean_cpu_error={:?}",sum_cpu_error.map(|s|s/N as f64));
                numerical_failures += failures.iter().sum::<usize>();
                eprintln!(
                    "ILP4_JSON {}",
                    serde_json::json!({"case":case,"candidate":candidate,"stage_order":["swiglu","gate","up"],"max_cpu_error":max_cpu_error,"mean_cpu_error":sum_cpu_error.map(|s|s/N as f64),"failed_values":failures,"first_failure":first_failure})
                );
                if let Some(reference) = &control {
                    for (a, b) in actual.iter().zip(reference) {
                        eprintln!(
                            "ILP4 case={case} control_drift={:?}",
                            fixture::measure_error(a, b)
                        );
                    }
                } else {
                    control = Some(actual.clone());
                }
                unsafe {
                    if candidate {
                        ctx.ffn_q8_joint_ilp4_f32_async(
                            x.ptr(),
                            pair.planes()[0].as_ptr(),
                            pair.planes()[1].as_ptr(),
                            outputs[0].ptr(),
                            1,
                            N as i32,
                            K as i32,
                            None,
                        )?;
                    } else {
                        pair.enqueue_shared_scales_async(x.ptr(), outputs[0].ptr(), 1, None)?;
                    }
                }
                ctx.synchronize_stream(CudaStream::Decode)?;
                for (buffer, expected) in outputs.iter().zip(&actual) {
                    assert_eq!(
                        buffer
                            .read(&ctx)?
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
            assert_eq!(x.read(&ctx)?, input);
            assert_eq!(producer.read(&ctx)?, input);
        }
        for (plane, expected) in pair.planes().into_iter().zip(host.planes()) {
            let mut actual = vec![0; plane.bytes()];
            unsafe {
                ctx.memcpy_d2h(actual.as_mut_ptr().cast(), plane.as_ptr(), actual.len())?;
            }
            assert_eq!(actual, expected);
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    eprintln!(
        "ILP4_JSON {}",
        serde_json::json!({"kind":"cleanup","tracked_before":initial,"tracked_after":CudaContext::total_device_bytes(),"numerical_failures":numerical_failures,"is_passed":numerical_failures==0})
    );
    anyhow::ensure!(numerical_failures==0,"ILP4 numerical admission failed for {numerical_failures} values; do not run performance or integrate");
    Ok(())
}

/// Invalid shapes, null/misaligned and overlapping spans must fail before any
/// device work; no fallback is allowed for this fixed research operator.
#[test]
#[ignore = "requires M40; run with --include-ignored --test-threads=1"]
fn test_ilp4_rejects_invalid_spans_without_launch() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    let before = profile::snapshot();
    for (rows, n, k) in [(0, N, K), (4, N, K), (1, N - 1, K), (1, N, K - 32)] {
        assert!(unsafe {
            ctx.ffn_q8_joint_ilp4_f32_async(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                rows,
                n as i32,
                k as i32,
                None,
            )
        }
        .is_err());
    }
    // Rejected before dereference; these deliberately invalid address ranges
    // isolate host-side validation without allocating full tensor planes.
    for (input, output, capture) in [
        (0, 0x4000000, None),
        (17, 0x4000000, None),
        (16, 32, None),
        (16, 0x4000000, Some((0x4000004, 0x5000000))),
        (16, 0x4000000, Some((0, 0x5000000))),
    ] {
        assert!(unsafe {
            ctx.ffn_q8_joint_ilp4_f32_async(
                input as *const c_void,
                0x100000 as *const c_void,
                0x500000 as *const c_void,
                output as *mut c_void,
                1,
                N as i32,
                K as i32,
                capture.map(|(g, u)| (g as *mut c_void, u as *mut c_void)),
            )
        }
        .is_err());
    }
    assert_eq!(
        profile::counter_delta(&before, &profile::snapshot()),
        profile::OpCounters::default()
    );
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}
