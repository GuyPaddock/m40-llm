use super::super::pipeline::GuardedF32;
use super::{
    reference::{self, Reference},
    weights::Weights,
};
use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{CudaContext, CudaEvent, CudaStream},
    infer::execution_weights::Q8SeparateLaunch,
    profile,
};
use serde_json::{json, Value};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Arm {
    Shiftadd,
    Q4,
    Q8,
    PreparedQ8,
}

impl Arm {
    pub const ALL: [Self; 4] = [Self::Shiftadd, Self::Q4, Self::Q8, Self::PreparedQ8];
    pub fn get_operation(self) -> &'static str {
        match self {
            Self::Shiftadd => "gemv_f32xsa4_128",
            Self::Q4 => "gemm_f32xq4_0_gguf_f32_decode",
            Self::Q8 => "gemm_f32xq8_0_gguf_f32_decode",
            Self::PreparedQ8 => "gemm_f32xq8_separate_packed_activation",
        }
    }
}

pub(super) struct Probe<'a> {
    ctx: &'a CudaContext,
    weights: &'a Weights,
    rows: usize,
    host_input: Vec<f32>,
    input: GuardedF32,
    output: GuardedF32,
    references: [Reference; 3],
    start: CudaEvent,
    stop: CudaEvent,
    pub report: Value,
}

impl<'a> Probe<'a> {
    pub fn new(ctx: &'a CudaContext, weights: &'a Weights, input: &[f32]) -> Result<Self> {
        let k = weights.sa.get_input_width();
        let n = weights.sa.get_output_width();
        ensure!(
            input.len().is_multiple_of(k) && matches!(input.len() / k, 1 | 2 | 4 | 8),
            "invalid probe rows"
        );
        let rows = input.len() / k;
        let began = Instant::now();
        let references = [
            reference::project(&weights.sa_values, input, k, Some(&weights.sa))?,
            reference::project(&weights.q4_values, input, k, None)?,
            reference::project(&weights.q8_values, input, k, None)?,
        ];
        let reference_ms = began.elapsed().as_secs_f64() * 1000.0;
        let began = Instant::now();
        let device_input = GuardedF32::new(ctx, input, "research:shiftadd-input")?;
        let output = GuardedF32::new(ctx, &vec![f32::NAN; rows * n], "research:shiftadd-output")?;
        let start = ctx.create_event()?;
        let stop = ctx.create_event()?;
        let report = json!({"rows":rows,"input_sha256":m40_llm::evaluation::identity::sha256(&input.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?,
            "allocation_upload_events_ms":began.elapsed().as_secs_f64()*1000.0,"reference_ms":reference_ms,
            "scratch_guarded_bytes":device_input.bytes()+output.bytes(),"hidden_scratch_bytes":0,
            "projection_fit_error":{"shiftadd":reference::fit_error(&references[2].values,&references[0].values)?,
                "q4":reference::fit_error(&references[2].values,&references[1].values)?}});
        Ok(Self {
            ctx,
            weights,
            rows,
            host_input: input.to_vec(),
            input: device_input,
            output,
            references,
            start,
            stop,
            report,
        })
    }

    fn enqueue(&self, arm: Arm) -> Result<()> {
        let k = self.weights.sa.get_input_width();
        let n = self.weights.sa.get_output_width();
        // Every pointer comes from a live owner; producers are synchronous
        // uploads or ordered decode work. Output is disjoint from every read.
        unsafe {
            match arm {
                Arm::Shiftadd => self.ctx.gemv_f32xsa4_128_async(
                    self.input.pointer(),
                    self.weights.sa_device.as_ptr(),
                    self.output.pointer(),
                    self.rows as i32,
                    n as i32,
                    k as i32,
                    CudaStream::Decode,
                ),
                Arm::PreparedQ8 => self
                    .weights
                    .prepared
                    .enqueue_separate_packed_activation_async(
                        self.input.pointer(),
                        self.output.pointer(),
                        self.rows,
                        Q8SeparateLaunch::new(4, 128)?,
                        CudaStream::Decode,
                    ),
                Arm::Q4 | Arm::Q8 => {
                    for row in 0..self.rows {
                        let input = self.input.pointer().cast::<f32>().add(row * k).cast();
                        let output = self.output.pointer().cast::<f32>().add(row * n).cast();
                        match arm {
                            Arm::Q4 => self.ctx.gemm_f32xq4_0_gguf_f32_decode_async(
                                input,
                                self.weights.q4_device.as_ptr(),
                                output,
                                1,
                                n as i32,
                                k as i32,
                            )?,
                            Arm::Q8 => self.ctx.gemm_f32xq8_0_gguf_f32_decode_async(
                                input,
                                self.weights.q8_device.as_ptr(),
                                output,
                                1,
                                n as i32,
                                k as i32,
                            )?,
                            _ => unreachable!(),
                        }
                    }
                    Ok(())
                }
            }
        }
    }

    pub fn validate(&self) -> Result<Value> {
        let mut checks = Vec::new();
        for arm in Arm::ALL {
            self.enqueue(arm)?;
            self.ctx.synchronize_stream(CudaStream::Decode)?;
            let actual = self.output.read(self.ctx)?;
            let reference = &self.references[match arm {
                Arm::Shiftadd => 0,
                Arm::Q4 => 1,
                _ => 2,
            }];
            checks.push(json!({"arm":arm,"arithmetic":reference.check(&actual)?}));
        }
        let actual = self.input.read(self.ctx)?;
        ensure!(
            actual
                .iter()
                .zip(&self.host_input)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "input modified"
        );
        self.weights.verify(self.ctx)?;
        Ok(json!({"checks":checks,"guards":"pass","immutable":"pass"}))
    }

    pub fn time(&self, arm: Arm, iterations: usize) -> Result<Value> {
        ensure!(iterations > 0, "empty timing batch");
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        let before = profile::snapshot();
        let resident = CudaContext::total_device_bytes();
        let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let began = Instant::now();
        self.start.record(CudaStream::Decode)?;
        for _ in 0..iterations {
            self.enqueue(arm)?;
        }
        self.stop.record(CudaStream::Decode)?;
        let submission_ms = began.elapsed().as_secs_f64() * 1000.0;
        let gpu_ms = f64::from(self.start.elapsed_sync(&self.stop, "shiftadd-projection")?)
            / iterations as f64;
        let wall_ms = began.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        let after = profile::snapshot();
        let delta = profile::counter_delta(&before, &after);
        let operation = arm.get_operation();
        let launches = after.by_op.get(operation).map_or(0, |v| v.launches)
            - before.by_op.get(operation).map_or(0, |v| v.launches);
        let expected = iterations
            * if matches!(arm, Arm::Q4 | Arm::Q8) {
                self.rows
            } else {
                1
            };
        ensure!(
            launches == expected as u64 && delta.launches == expected as u64,
            "unexpected projection dispatch: {operation} {launches}/{expected}"
        );
        ensure!(
            delta.device_allocs == 0
                && delta.device_frees == 0
                && CudaContext::total_device_bytes() == resident,
            "timing changed allocations"
        );
        ensure!(gpu_ms.is_finite() && gpu_ms > 0.0, "invalid event duration");
        Ok(
            json!({"arm":arm,"iterations":iterations,"gpu_ms":gpu_ms,"wall_ms":wall_ms,"submission_ms":submission_ms/iterations as f64,
            "started_unix_us":started,"completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros(),
            "operation":operation,"launches":launches,"tracked_device_bytes":resident,"allocation_free_enqueue":true}),
        )
    }
}
