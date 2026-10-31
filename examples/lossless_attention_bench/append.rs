//! Price the implemented CPU replacement path, including new GPU-token readback.
use super::{
    benchmark::{self, Measurement, Report},
    capture::KV_HEADS,
    fixture::{Encoded, Fixture, DIMENSION},
    pipeline::{Arm, Case},
};
use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream, LosslessKvDevice},
    profile,
};
use serde_json::{json, Value};
use std::time::Instant;

pub fn validate(case: &Case, batch: usize) -> Result<()> {
    let expected = [
        case.read_arm(Arm::Native, batch)?,
        case.read_arm(Arm::Fused, batch)?,
    ];
    for is_compressed in [false, true] {
        measure(
            case,
            batch,
            is_compressed,
            &expected[usize::from(is_compressed)],
        )?;
    }
    Ok(())
}

pub fn compare(case: &Case, batch: usize, key: &Value, report: &mut Report) -> Result<()> {
    let expected = [
        case.read_arm(Arm::Native, batch)?,
        case.read_arm(Arm::Fused, batch)?,
    ];
    benchmark::compare(
        key,
        "native_append_attention_vs_cpu_replacement_attention",
        report,
        |is_compressed| {
            measure(
                case,
                batch,
                is_compressed,
                &expected[usize::from(is_compressed)],
            )
        },
    )
}

fn measure(case: &Case, batch: usize, is_compressed: bool, expected: &[u8]) -> Result<Measurement> {
    let ctx = &case.ctx;
    ctx.synchronize_stream(CudaStream::Decode)?;
    let initial_live = CudaContext::total_device_bytes();
    let prepared = Instant::now();
    case.reset_output()?;
    // Recreate the pre-existing state before each independent observation.
    // Only length-1 words enter the CPU compressor's host history.
    let mut host_keys = if is_compressed {
        case.fixture.get_prefix(batch, true)?
    } else {
        Vec::new()
    };
    let mut host_values = if is_compressed {
        case.fixture.get_prefix(batch, false)?
    } else {
        Vec::new()
    };
    let old = if is_compressed {
        let prefix = Encoded::encode(&host_keys, &host_values)?;
        Some((
            LosslessKvDevice::upload(ctx, &prefix.keys)?,
            LosslessKvDevice::upload(ctx, &prefix.values)?,
        ))
    } else {
        unsafe {
            case.native.truncate_dense_async(
                ctx,
                0,
                batch as u32,
                (case.fixture.length - 1) as u32,
                case.fixture.length as u32,
            )?;
        }
        ctx.synchronize_stream(CudaStream::Decode)?;
        for sequence in 0..batch {
            ensure!(
                case.native.debug_sequence_length(ctx, sequence as u32)?
                    == (case.fixture.length - 1) as u32,
                "native append did not start at prefix length"
            );
        }
        None
    };
    let prepare_ms = prepared.elapsed().as_secs_f64() * 1000.0;
    let old_device_bytes = old
        .as_ref()
        .map(|(k, v)| k.get_bytes() + v.get_bytes())
        .unwrap_or(0);
    let baseline = profile::snapshot();
    let began = Instant::now();
    case.start.record(CudaStream::Decode)?;
    let mut phase = Value::Null;
    let replacement = if is_compressed {
        let phase_start = Instant::now();
        let mut next_keys = vec![0.0f32; batch * KV_HEADS * DIMENSION];
        let mut next_values = vec![0.0f32; next_keys.len()];
        unsafe {
            ctx.memcpy_d2h(
                next_keys.as_mut_ptr().cast(),
                case.last_keys.as_ptr(),
                next_keys.len() * 4,
            )?;
            ctx.memcpy_d2h(
                next_values.as_mut_ptr().cast(),
                case.last_values.as_ptr(),
                next_values.len() * 4,
            )?;
        }
        let readback_ms = phase_start.elapsed().as_secs_f64() * 1000.0;
        let phase_start = Instant::now();
        Fixture::append_tokens(&mut host_keys, &next_keys)?;
        Fixture::append_tokens(&mut host_values, &next_values)?;
        let encoded = Encoded::encode(&host_keys, &host_values)?;
        let encode_ms = phase_start.elapsed().as_secs_f64() * 1000.0;
        let phase_start = Instant::now();
        let keys = LosslessKvDevice::upload(ctx, &encoded.keys)?;
        let values = LosslessKvDevice::upload(ctx, &encoded.values)?;
        let upload_ms = phase_start.elapsed().as_secs_f64() * 1000.0;
        let peak = CudaContext::total_device_bytes();
        let phase_start = Instant::now();
        drop(old);
        let retirement_ms = phase_start.elapsed().as_secs_f64() * 1000.0;
        phase = json!({"gpu_token_readback_ms":readback_ms,"host_append_encode_validate_ms":encode_ms,
            "new_device_allocation_upload_ms":upload_ms,"old_device_retirement_ms":retirement_ms,
            "old_device_bytes":old_device_bytes,"new_device_bytes":keys.get_bytes()+values.get_bytes(),
            "peak_tracked_device_bytes":peak,"old_raw_host_prefix_bytes":2*batch*KV_HEADS*(case.fixture.length-1)*DIMENSION*2,
            "raw_host_history_after_bytes":2*batch*KV_HEADS*case.fixture.length*DIMENSION*2,
            "new_fp32_token_readback_bytes":next_keys.len()*8});
        case.enqueue_encoded(&keys, &values, batch)?;
        Some((keys, values))
    } else {
        for sequence in 0..batch {
            unsafe {
                case.native.append_token_f32_rope_k_at_async(
                    ctx,
                    sequence as u32,
                    case.last_keys
                        .as_ptr()
                        .cast::<f32>()
                        .add(sequence * KV_HEADS * DIMENSION)
                        .cast(),
                    case.last_values
                        .as_ptr()
                        .cast::<f32>()
                        .add(sequence * KV_HEADS * DIMENSION)
                        .cast(),
                    (case.fixture.length - 1) as u32,
                    (case.fixture.length - 1) as u32,
                    10000.0,
                    0.0,
                )?;
            }
        }
        case.enqueue(Arm::Native, batch)?;
        None
    };
    case.stop.record(CudaStream::Decode)?;
    let cuda_timeline_ms = f64::from(
        case.start
            .elapsed_sync(&case.stop, "lossless_append_timing")?,
    );
    let wall_ms = began.elapsed().as_secs_f64() * 1000.0;
    let counters = profile::counter_delta(&baseline, &profile::snapshot());
    ensure!(
        case.read_output(batch)? == expected,
        "append-plus-attention output differs from its full-prefix reference"
    );
    if let Some((keys, values)) = &replacement {
        case.verify_encoded(keys, values, batch)?;
    } else {
        for sequence in 0..batch {
            ensure!(
                case.native.debug_sequence_length(ctx, sequence as u32)?
                    == case.fixture.length as u32,
                "native append length differs"
            );
        }
        ensure!(
            counters.h2d_copies == 0
                && counters.d2h_copies == 0
                && counters.device_allocs == 0
                && counters.device_frees == 0,
            "native append allocated/copied host memory"
        );
    }
    drop(replacement);
    ensure!(
        CudaContext::total_device_bytes() == initial_live,
        "append observation leaked device buffers"
    );
    case.verify_storage()?;
    Ok(Measurement {
        wall_ms,
        cuda_timeline_ms,
        detail: json!({"is_cpu_replacement":is_compressed,"iterations":1,
        "wrapper_counters":counters,"excluded_prefix_recreation_ms":prepare_ms,"phases":phase,
        "boundary":"new GPU-resident token to completed full-context attention; CPU replacement includes D2H, host history growth, full encoding/validation, allocation/upload and old cache retirement",
        "cuda_timeline_scope":"elapsed device timeline includes CPU preparation gaps; not GPU active kernel time",
        "host_history_assumption":"CPU compressor retains original FP16 history; memory is charged separately; no incremental GPU encoder assumed"}),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        capture::{LayerSample, QUERY_HEADS},
        pipeline::Case,
    };
    use half::f16;

    /// The real append runner starts from shorter caches, restores the captured
    /// final token, and preserves guarded outputs and tracked ownership across
    /// distinct batch sizes. Device prefixes exclude inactive sequence rows.
    #[test]
    #[ignore = "requires idle M40 hardware; run serially with --include-ignored"]
    fn test_append_runner_preserves_history_and_outputs() -> Result<()> {
        benchmark::validate_environment()?;
        benchmark::check_gpu_processes()?;
        let ctx = CudaContext::new_m40()?;
        let props = ctx.current_device_props()?;
        ensure!(props.major == 5 && props.minor == 2, "requires sm_52");
        let baseline = CudaContext::total_device_bytes();
        let length = 19;
        let rows = |salt: usize| {
            (0..8)
                .map(|sequence| {
                    (0..length * KV_HEADS * DIMENSION)
                        .map(|i| {
                            f16::from_f32(
                                ((i * 17 + sequence * 71 + salt) % 257) as f32 / 256.0 - 0.5,
                            )
                            .to_f32()
                        })
                        .collect()
                })
                .collect()
        };
        let sample = LayerSample {
            keys: rows(3),
            values: rows(29),
            queries: (0..8)
                .map(|sequence| {
                    (0..QUERY_HEADS * DIMENSION)
                        .map(|i| ((i + sequence) % 17) as f32 / 128.0 - 0.0625)
                        .collect()
                })
                .collect(),
        };
        let case = Case::new(&ctx, Fixture::from_sample(&sample, length)?)?;
        case.validate()?;
        for batch in [1, 2, 4, 8] {
            validate(&case, batch)?;
        }
        drop(case);
        ensure!(
            CudaContext::total_device_bytes() == baseline,
            "append runner leaked memory"
        );
        Ok(())
    }
}
