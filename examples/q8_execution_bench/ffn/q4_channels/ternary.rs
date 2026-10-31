use super::super::{
    fixture::{source_bytes, FfnInputs},
    pipeline::GuardedF32,
    validation, warmup,
};
use super::timing;
use crate::run::{command_output, inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaEvent, CudaStream, DeviceTernaryScreen},
    evaluation::{
        identity::sha256,
        quantized_fixture::{decode_q4_values, requantize_q8_to_q4},
        ternary_screen::{generate_signs, rotate_reference, select_coordinates, TernarySketch},
    },
    infer::LoadedModel,
    profile,
};
use serde_json::{json, Value};
use std::{
    fs::File,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

struct Probe<'a> {
    ctx: &'a CudaContext,
    owner: &'a DeviceTernaryScreen,
    input: GuardedF32,
    output: GuardedF32,
    start: CudaEvent,
    stop: CudaEvent,
}

impl Probe<'_> {
    fn enqueue(&self) -> Result<()> {
        unsafe {
            self.owner.enqueue_async(
                self.input.pointer(),
                self.output.pointer(),
                CudaStream::Decode,
            )
        }
    }

    fn time(&self, iterations: usize) -> Result<Value> {
        ensure!(iterations > 0, "empty screening interval");
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        let before = profile::snapshot();
        let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let began = Instant::now();
        self.start.record(CudaStream::Decode)?;
        for _ in 0..iterations {
            self.enqueue()?;
        }
        self.stop.record(CudaStream::Decode)?;
        let enqueue_ms = began.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        let counters = profile::counter_delta(&before, &profile::snapshot());
        ensure!(
            counters.launches == (iterations * 2) as u64,
            "screen dispatch changed"
        );
        ensure!(
            counters.device_allocs
                + counters.device_frees
                + counters.h2d_copies
                + counters.d2h_copies
                + counters.stream_syncs
                + counters.stream_waits
                + counters.cublas_calls
                == 0,
            "screening timing allocated/copied/synchronized"
        );
        let gpu_ms = f64::from(
            self.start
                .elapsed_sync(&self.stop, "ternary_probe:complete")?,
        ) / iterations as f64;
        ensure!(
            gpu_ms.is_finite() && gpu_ms > 0.0,
            "invalid screening timing"
        );
        Ok(json!({"gpu_ms":gpu_ms,"host_enqueue_ms":enqueue_ms,
            "host_enqueue_and_completion_ms":began.elapsed().as_secs_f64()*1000.0/iterations as f64,
            "iterations":iterations,"started_unix_us":started,"enqueue_counters":counters}))
    }

    fn validate(
        &self,
        input: &[f32],
        gate: &TernarySketch,
        up: &TernarySketch,
        norms: &[f32],
    ) -> Result<(Value, Vec<u32>)> {
        self.enqueue()?;
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        let (rotated, coordinates) = self.owner.read_preparation()?;
        let expected_rotation = rotate_reference(input, gate.signs())?;
        ensure!(
            rotated
                .iter()
                .zip(&expected_rotation)
                .all(|(&a, &b)| a.is_finite() && (f64::from(a) - b).abs() <= 2e-6 + 2e-6 * b.abs()),
            "screening rotation differs from Walsh reference"
        );
        ensure!(
            coordinates == select_coordinates(&rotated, 80)?,
            "coordinate ranking differs"
        );
        let g = gate.project_reference(&rotated, &coordinates, 80)?;
        let u = up.project_reference(&rotated, &coordinates, 80)?;
        let activation: Vec<_> = g
            .iter()
            .zip(&u)
            .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let scores: Vec<_> = activation
            .iter()
            .zip(norms)
            .map(|(&a, &n)| a.abs() * f64::from(n))
            .collect();
        let actual = self.output.read(self.ctx)?;
        let mut reports = Vec::new();
        for (values, expected) in
            actual
                .chunks_exact(norms.len())
                .zip([&g, &u, &activation, &scores])
        {
            validation::compare(values, expected)?;
            reports.push(validation::measure(values, expected)?);
        }
        ensure!(
            self.input
                .read(self.ctx)?
                .iter()
                .zip(input)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "screen input changed"
        );
        Ok((
            json!({"rotation":validation::measure(&rotated,&expected_rotation)?,"coordinate_order_passed":true,
            "planes":reports,"plane_order":["gate","up","activation","score"],"passed":true}),
            actual.iter().map(|x| x.to_bits()).collect(),
        ))
    }
}

fn immutable_hashes(ctx: &CudaContext, owner: &DeviceTernaryScreen) -> Result<Vec<String>> {
    owner
        .immutable_buffers()
        .iter()
        .map(|buffer| {
            let mut bytes = vec![0; buffer.bytes()];
            unsafe {
                ctx.memcpy_d2h(bytes.as_mut_ptr().cast(), buffer.as_ptr(), bytes.len())?;
            }
            sha256(&bytes)
        })
        .collect()
}

/// Observe the cost of the screen alone. Passing an absolute point budget is
/// only permission to test the remaining operations, not a speedup conclusion.
pub(super) fn benchmark(
    args: &Args,
    model: &LoadedModel,
    inputs: &FfnInputs,
    output: &mut File,
) -> Result<()> {
    let k = inputs.artifact.plan.d_model;
    let h = inputs.artifact.plan.hidden_dim;
    ensure!(
        k == 2048 && h == 11008,
        "screening requires frozen FFN shape"
    );
    let resident = CudaContext::total_device_bytes();
    for layer in &inputs.artifact.layers {
        let prepare = Instant::now();
        ensure!(
            layer.rows.len() >= 8 && layer.normalized_inputs.len() >= k * 8,
            "missing screen inputs"
        );
        for a in 0..8 {
            for b in a + 1..8 {
                ensure!(
                    layer.normalized_inputs[a * k..(a + 1) * k]
                        .iter()
                        .zip(&layer.normalized_inputs[b * k..(b + 1) * k])
                        .any(|(x, y)| x.to_bits() != y.to_bits()),
                    "duplicate screening rows"
                );
            }
        }
        let mut weights = Vec::new();
        let mut encodings = Vec::new();
        for source in [&layer.source.gate, &layer.source.up, &layer.source.down] {
            let (bytes, report) = requantize_q8_to_q4(source_bytes(model, source)?)?;
            ensure!(
                report.source_sha256 == source.source_sha256,
                "screening Q4 source mismatch"
            );
            weights.push(bytes);
            encodings.push(report);
        }
        let signs = generate_signs(k, 0)?;
        let gate = TernarySketch::encode_q4(&weights[0], k, h, &signs)?;
        let up = TernarySketch::encode_q4(&weights[1], k, h, &signs)?;
        let down = decode_q4_values(&weights[2])?;
        let mut norms = vec![0.0f64; h];
        for row in down.chunks_exact(h) {
            for (sum, &v) in norms.iter_mut().zip(row) {
                *sum += v * v;
            }
        }
        let norms: Vec<_> = norms.into_iter().map(|n| n.sqrt() as f32).collect();
        drop(down);
        let encoding_ms = prepare.elapsed().as_secs_f64() * 1000.0;
        let upload = Instant::now();
        let owner = DeviceTernaryScreen::upload(&model.cuda, &gate, &up, &norms)?;
        let upload_ms = upload.elapsed().as_secs_f64() * 1000.0;
        let hashes = immutable_hashes(&model.cuda, &owner)?;
        let immutable_bytes: usize = owner.immutable_buffers().iter().map(|b| b.bytes()).sum();
        write_record(
            output,
            json!({"kind":"ternary_screen_layer","layer":layer.source.layer_id,
            "source":layer.source,"q4_encoding":encodings,"gate_sketch":gate.report(),"up_sketch":up.report(),
            "encoding_ms":encoding_ms,"upload_ms":upload_ms,"immutable_sha256":hashes,"immutable_bytes":immutable_bytes,
            "workspace_bytes":owner.workspace_bytes(),"tracked_device_bytes":CudaContext::total_device_bytes(),
            "selected_code_logical_bytes":k/128*80*h.div_ceil(4)*2,"scale_logical_bytes":k/128*h*2*2,
            "traffic_interpretation":"unique selected code and scale payload; excludes reloads/sector effects; not measured DRAM",
            "residency_policy":"original model and prepared Q8 down/vocabulary retained; one layer screening owner; no sparse FFN or recovery owner"}),
        )?;
        for row in 0..8 {
            let activation = &layer.normalized_inputs[row * k..(row + 1) * k];
            let probe = Probe {
                ctx: &model.cuda,
                owner: &owner,
                input: GuardedF32::new(&model.cuda, activation, "ternary_probe:input")?,
                output: GuardedF32::new(&model.cuda, &vec![0.0; h * 4], "ternary_probe:output")?,
                start: model.cuda.create_event()?,
                stop: model.cuda.create_event()?,
            };
            let case = json!({"layer":layer.source.layer_id,"row":row,"source_row":layer.rows[row],"rows":1});
            let (correctness, expected_bits) = probe.validate(activation, &gate, &up, &norms)?;
            write_record(
                output,
                json!({"kind":"ternary_screen_correctness","case":case,"checks":correctness,
                "tracked_device_bytes":CudaContext::total_device_bytes(),"io_bytes":probe.input.bytes()+probe.output.bytes()}),
            )?;
            let warm = warmup::measure(1000.0, |n| {
                probe.time(n)?["gpu_ms"]
                    .as_f64()
                    .context("missing warmup duration")
            })?;
            write_record(
                output,
                json!({"kind":"ternary_screen_warmup","case":case,"warmup":warm}),
            )?;
            let mut times = Vec::new();
            let mut target = 5;
            loop {
                while times.len() < target {
                    let sample = probe.time(args.iterations)?;
                    times.push(sample["gpu_ms"].as_f64().context("missing timing")?);
                    write_record(
                        output,
                        json!({"kind":"ternary_screen_sample","case":case,"sample":times.len()-1,"timing":sample}),
                    )?;
                }
                if target == 5 && timing::variation(&times) > 0.05 {
                    target = 10;
                } else {
                    break;
                }
            }
            let median_us = timing::median(&times) * 1000.0;
            let variation = timing::variation(&times);
            let (_, actual_bits) = probe.validate(activation, &gate, &up, &norms)?;
            ensure!(
                actual_bits == expected_bits,
                "screening output changed after timing"
            );
            write_record(
                output,
                json!({"kind":"ternary_screen_observation","case":case,"samples":times.len(),
                "median_gpu_us":median_us,"throughput_mad_over_median":variation,"is_stable":variation<=0.05,
                "fits_10_percent_point_budget":variation<=0.05 && median_us<150.88,
                "fits_20_percent_point_budget":variation<=0.05 && median_us<56.89,
                "remaining_10_percent_point_budget_us":150.88-median_us,
                "remaining_20_percent_point_budget_us":56.89-median_us,
                "interpretation":"screen-only preliminary observation; excludes channel top-k, recovery and exact selected FFN",
                "ownership_passed":true,"gpu_after":inspect_gpu()?}),
            )?;
        }
        ensure!(
            immutable_hashes(&model.cuda, &owner)? == hashes,
            "screening immutable data changed"
        );
        drop(owner);
        ensure!(
            CudaContext::total_device_bytes() == resident,
            "screening allocation leak"
        );
        write_record(
            output,
            json!({"kind":"ternary_screen_layer_complete","layer":layer.source.layer_id,"ownership_passed":true}),
        )?;
    }
    write_record(
        output,
        json!({"kind":"ternary_screen_complete","is_complete":true,"observations":inputs.artifact.layers.len()*8,
        "revision_after":command_output("git", &["rev-parse","HEAD"])?,"gpu_after":inspect_gpu()?}),
    )
}
