//! Fixed standalone operator measurements on already-validated actual inputs.
use super::*;
use crate::protocol::{compare_pairs, Pair};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy)]
enum Operator {
    Vocabulary,
    Argmax,
}

impl Operator {
    fn name(self) -> &'static str {
        match self {
            Self::Vocabulary => "vocabulary",
            Self::Argmax => "argmax",
        }
    }
}

impl LogitProbe<'_> {
    fn enqueue_operator(&self, operator: Operator, is_candidate: bool) -> Result<()> {
        unsafe {
            match (operator, is_candidate) {
                (Operator::Vocabulary, true) => self.packet.device.enqueue_async(
                    self.normalized.pointer(),
                    self.candidate.pointer(),
                    1,
                    Q8SeparateLaunch::new(4, 128)?,
                    CudaStream::Decode,
                ),
                (Operator::Vocabulary, false) => {
                    let (_, head, width, _, _) = self.model.map_lm_head()?;
                    self.model.matmul_f32xf16_gguf_f32_async(
                        self.normalized.pointer(),
                        head.dptr,
                        self.candidate.pointer(),
                        1,
                        self.vocab as i32,
                        width as i32,
                    )
                }
                (Operator::Argmax, true) => self.model.cuda.argmax_f32_partitioned_async(
                    self.control.pointer(),
                    self.vocab as u32,
                    self.partials.pointer(),
                    self.scratch_bytes,
                    self.token.pointer(),
                    CudaStream::Decode,
                ),
                (Operator::Argmax, false) => self.model.cuda.argmax_f32_async(
                    self.control.pointer(),
                    self.vocab as u32,
                    self.token.pointer(),
                    CudaStream::Decode,
                ),
            }
        }
    }

    pub(super) fn measure_operators(
        &self,
        report: &mut Report,
        step: usize,
        expected_token: u32,
    ) -> Result<()> {
        let ctx = &self.model.cuda;
        let original_input = self.normalized.read(ctx)?;
        let original_logits = self.control.read(ctx)?;
        let start = ctx.create_event()?;
        let stop = ctx.create_event()?;
        let resident = CudaContext::total_device_bytes();
        for operator in [Operator::Vocabulary, Operator::Argmax] {
            let time = |is_candidate, iterations: u64| -> Result<Value> {
                ctx.synchronize_stream(CudaStream::Decode)?;
                let began = Instant::now();
                let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
                let before = profile::snapshot();
                start.record(CudaStream::Decode)?;
                for _ in 0..iterations {
                    self.enqueue_operator(operator, is_candidate)?;
                }
                stop.record(CudaStream::Decode)?;
                let counters = profile::counter_delta(&before, &profile::snapshot());
                let launches = if matches!(operator, Operator::Argmax) && is_candidate {
                    2
                } else {
                    1
                };
                ensure!(
                    counters.launches == launches * iterations
                        && counters.device_allocs == 0
                        && counters.device_frees == 0
                        && counters.h2d_copies == 0
                        && counters.d2h_copies == 0
                        && counters.stream_syncs == 0
                        && counters.cublas_calls == 0,
                    "operator timing changed launch/resource contract"
                );
                let gpu_ms = f64::from(start.elapsed_sync(&stop, "portfolio_operator:completion")?)
                    / iterations as f64;
                ensure!(
                    CudaContext::total_device_bytes() == resident,
                    "operator residency changed"
                );
                Ok(
                    json!({"is_candidate":is_candidate,"iterations":iterations,"gpu_ms":gpu_ms,
                    "host_enqueue_and_completion_ms":began.elapsed().as_secs_f64()*1000.0/iterations as f64,
                    "started_unix_us":started,"completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros(),
                    "enqueue_counters":counters}),
                )
            };
            for is_candidate in [false, true] {
                let mut total = 0.0;
                let bound = if is_candidate { 100.0 } else { 1000.0 };
                for batch in 0..512 {
                    total += time(is_candidate, 32)?["gpu_ms"]
                        .as_f64()
                        .context("warmup latency")?
                        * 32.0;
                    if total >= bound {
                        break;
                    }
                    ensure!(batch < 511, "operator warmup exceeded bound");
                }
                report.record(json!({"kind":"portfolio_operator_warmup","step":step,
                    "operator":operator.name(),"is_candidate":is_candidate,"elapsed_gpu_ms":total}))?;
            }
            let mut pairs = Vec::new();
            for trial in 0..5 {
                let gpu_before = run::gpu()?;
                let mut measurements = Vec::new();
                for is_candidate in if trial % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    measurements.push(time(is_candidate, 100)?);
                    match operator {
                        Operator::Vocabulary => ensure!(
                            self.candidate.read(ctx)? == original_logits,
                            "timed vocabulary bits differ"
                        ),
                        Operator::Argmax => {
                            ensure!(
                                self.token.read(ctx)? == [expected_token],
                                "timed argmax differs"
                            );
                            self.partials.read(ctx)?;
                        }
                    }
                }
                let ms = |candidate| {
                    measurements
                        .iter()
                        .find(|v| v["is_candidate"] == candidate)
                        .and_then(|v| v["gpu_ms"].as_f64())
                        .context("operator pair latency")
                };
                pairs.push(Pair {
                    control_tps: 1000.0 / ms(false)?,
                    candidate_tps: 1000.0 / ms(true)?,
                });
                report.record(json!({"kind":"portfolio_operator_pair","operator":operator.name(),"step":step,
                    "trial":trial,"measurements":measurements,"gpu_before":gpu_before,"gpu_after":run::gpu()?,
                    "resident_bytes":resident}))?;
            }
            let comparison = compare_pairs(&pairs)?;
            // Reuse paired statistics, never the HTTP promotion/target labels.
            report.record(json!({"kind":"portfolio_operator_summary","operator":operator.name(),"step":step,
                "pairs":comparison.pairs,"control_operators_per_sec":comparison.control,
                "candidate_operators_per_sec":comparison.candidate,"throughput_gain":comparison.gain,
                "bootstrap_gain_ci95":comparison.bootstrap_gain_ci95,
                "bootstrap_resamples":comparison.bootstrap_resamples,"bootstrap_seed":comparison.bootstrap_seed,
                "needs_variation_extension":comparison.needs_variation_extension,
                "scope":"isolated actual-input operator, not whole-token or HTTP throughput",
                "runtime_integration_allowed":false}))?;
        }
        ensure!(
            self.normalized.read(ctx)? == original_input
                && self.control.read(ctx)? == original_logits,
            "timed operators mutated their actual inputs"
        );
        Ok(())
    }
}
