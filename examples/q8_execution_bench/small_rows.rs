use super::*;
use m40_llm::infer::q8_prefill::Q8PrefillWeightShape;

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Arm {
    Serial,
    ExistingMultiRow,
    Shared,
    Native,
    Sgemm,
    DequantSgemm,
}

const ARMS: [Arm; 6] = [
    Arm::Serial,
    Arm::ExistingMultiRow,
    Arm::Shared,
    Arm::Native,
    Arm::Sgemm,
    Arm::DequantSgemm,
];

pub(super) fn measure(
    projection: &Projection<'_>,
    file: &mut File,
    args: &Args,
    bytes: &[u8],
    input: &[f32],
    tensor: &GgufTensor,
) -> Result<()> {
    let shape = projection.execution.key().source.shape();
    let (k, n) = (shape.input_dim(), shape.output_dim());
    let rows = projection.rows;
    let launch = Q8SeparateLaunch::new(4, 256)?;
    let numerical = projection.check(bytes, input, ProjectionLaunch::Separate(launch))?;
    let expected = read_output(&projection.ctx, &projection.output)?;
    let weight = Q8PrefillWeightShape::new(k, n)?;
    let materialized = DeviceBuffer::new_tagged(
        &projection.ctx,
        weight.f32_bytes(),
        "small_rows:sgemm_weight",
    )?;
    let DeviceQ8ExecutionView::ColumnBlocks { blocks, .. } = projection.original.view() else {
        anyhow::bail!("missing GGUF source")
    };
    let dequant = || unsafe {
        projection.ctx.dequantize_q8_0_to_f32_colmajor_async(
            blocks.as_ptr(),
            materialized.as_mut_ptr(),
            materialized.bytes(),
            weight,
        )
    };
    dequant()?;
    projection.ctx.synchronize_stream(CudaStream::Prefill)?;
    let enqueue = |arm: Arm| unsafe {
        match arm {
            Arm::Native => projection.enqueue(None),
            Arm::Serial => projection.execution.project_serial_rows_async(
                &projection.input,
                &projection.output,
                rows,
            ),
            Arm::ExistingMultiRow => projection.enqueue(Some(ProjectionLaunch::Separate(launch))),
            Arm::Shared => projection.execution.project_small_rows_async(
                &projection.input,
                &projection.output,
                rows,
            ),
            Arm::Sgemm | Arm::DequantSgemm => {
                if matches!(arm, Arm::DequantSgemm) {
                    dequant()?;
                }
                projection.ctx.gemm_f32xf32_f32_async(
                    projection.input.as_ptr(),
                    materialized.as_ptr(),
                    projection.output.as_mut_ptr(),
                    rows as i32,
                    n as i32,
                    k as i32,
                )
            }
        }
    };
    let stream = |arm: Arm| match arm {
        Arm::Native => projection.stream(None),
        Arm::Sgemm | Arm::DequantSgemm => CudaStream::Prefill,
        _ => CudaStream::Decode,
    };
    for arm in ARMS {
        for _ in 0..3 {
            enqueue(arm)?;
        }
        projection.ctx.synchronize_stream(stream(arm))?;
        let buffer = if matches!(arm, Arm::Native) {
            &projection.reference
        } else {
            &projection.output
        };
        let actual = read_output(&projection.ctx, buffer)?;
        ensure!(actual.len() == expected.len(), "incomplete projection");
        let mut max_error = 0.0_f32;
        let mut bit_differences = 0;
        for (&a, &b) in actual.iter().zip(&expected) {
            let delta = (a - b).abs();
            ensure!(
                a.is_finite() && b.is_finite() && delta < 1e-3 + b.abs() * 2e-5,
                "{arm:?} numerical drift {delta}"
            );
            max_error = max_error.max(delta);
            bit_differences += usize::from(a.to_bits() != b.to_bits());
        }
        if matches!(arm, Arm::Serial | Arm::ExistingMultiRow | Arm::Shared) {
            ensure!(
                bit_differences == 0,
                "{arm:?} mismatched {bit_differences} outputs"
            );
        }
        write_record(
            file,
            json!({"kind":"small_rows_parity","tensor":tensor.name,"rows":rows,"arm":arm,
            "outputs":actual.len(),"different_bits":bit_differences,"max_abs":max_error,"control":numerical,
            "sgemm_temporary_bytes":materialized.bytes(),"shared_temporary_bytes":0}),
        )?;
    }
    for trial in 0..args.trials {
        let snapshot = inspect_gpu()?;
        for index in 0..ARMS.len() {
            let slot = if trial % 2 == 0 {
                index
            } else {
                ARMS.len() - 1 - index
            };
            let arm = ARMS[(slot + trial / 2) % ARMS.len()];
            let timing =
                projection.time_operation(stream(arm), args.iterations, || enqueue(arm))?;
            write_record(
                file,
                json!({"kind":"small_rows_timing","tensor":tensor.name,"rows":rows,"arm":arm,
                "trial":trial,"order":index,"iterations":args.iterations,"timing":timing,"gpu_before_round":snapshot}),
            )?;
        }
    }
    Ok(())
}
