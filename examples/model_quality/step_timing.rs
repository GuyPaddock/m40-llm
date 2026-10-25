use super::records::Report;
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaStream, decode_session::DecodeSession, evaluation::corpus::Split, infer::LoadedModel,
    kv_compression::KvCompressionConfig,
};
use serde::Serialize;
use serde_json::json;
use std::time::Instant;

pub struct StepTimingConfig {
    pub trials: usize,
    pub warm_prefix_tokens: usize,
}

#[derive(Debug, Serialize)]
struct StepTrace {
    warm_prefix_tokens: usize,
    warm_prediction_id: u32,
    warm_elapsed_ms: f64,
    timed_prefix_lengths: Vec<usize>,
    predicted_token_ids: Vec<u32>,
    timed_steps: usize,
    elapsed_ms: f64,
    steps_per_second: f64,
}

fn drain(model: &LoadedModel) -> Result<()> {
    model.cuda.synchronize_stream(CudaStream::Decode)?;
    model.cuda.synchronize_stream(CudaStream::Prefill)
}

/// Force identical observed prefixes instead of timing two divergent generations
/// with unequal token counts. Scratch, reset and warmup precede the timed region.
fn run_trace(model: &mut LoadedModel, ids: &[u32], warm_prefix_tokens: usize) -> Result<StepTrace> {
    ensure!(
        warm_prefix_tokens > 0
            && warm_prefix_tokens
                .checked_add(1)
                .is_some_and(|n| ids.len() > n),
        "step timing needs a nonempty warm prefix, at least one steady step and its target"
    );
    let cache = model
        .kv_cache
        .as_ref()
        .context("step timing requires dense KV")?;
    ensure!(
        !cache.is_compressed()
            && ids.len() <= cache.max_seq_len() as usize
            && ids.len() <= model.model_config.context_length as usize,
        "step timing input exceeds dense context"
    );
    let (_, _, _, vocabulary, _) = model.map_lm_head()?;
    ensure!(
        ids.iter().all(|&id| (id as usize) < vocabulary),
        "step timing token outside vocabulary"
    );
    let (d_model, _) = model.validate_full_layer_decode()?;
    model.reset_kv_cache()?;
    let mut session = DecodeSession::new_for_sequence_with_kv_config(
        model,
        0,
        KvCompressionConfig::dense_reference(),
        d_model,
        true,
        "layer_mask_steps",
        "layer_mask_steps:x",
        "layer_mask_steps:out",
    )?;
    let warm_start = Instant::now();
    let warm_prediction_id = session.greedy_token_for_ids(&ids[..warm_prefix_tokens])?;
    drain(model)?;
    let warm_elapsed_ms = warm_start.elapsed().as_secs_f64() * 1000.0;
    let timed_prefix_lengths: Vec<_> = (warm_prefix_tokens + 1..ids.len()).collect();
    let mut predicted_token_ids = Vec::with_capacity(timed_prefix_lengths.len());
    let start = Instant::now();
    for &length in &timed_prefix_lengths {
        predicted_token_ids.push(session.greedy_token_for_ids(&ids[..length])?);
    }
    drain(model)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    ensure!(
        elapsed_ms > 0.0,
        "steady-step clock duration must be positive"
    );
    Ok(StepTrace {
        warm_prefix_tokens,
        warm_prediction_id,
        warm_elapsed_ms,
        timed_steps: timed_prefix_lengths.len(),
        steps_per_second: 1000.0 * timed_prefix_lengths.len() as f64 / elapsed_ms,
        timed_prefix_lengths,
        predicted_token_ids,
        elapsed_ms,
    })
}

fn median(mut values: Vec<f64>) -> Result<f64> {
    ensure!(
        !values.is_empty() && values.iter().all(|v| v.is_finite()),
        "invalid timing sample"
    );
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Ok(if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    })
}

pub fn measure_pairs(
    report: &mut Report,
    split: Split,
    passage_id: &str,
    reference: &mut LoadedModel,
    candidate: &mut LoadedModel,
    ids: &[u32],
    config: StepTimingConfig,
) -> Result<()> {
    ensure!(
        (1..=32).contains(&config.trials),
        "step timing requires 1..32 trials"
    );
    ensure!(
        !reference.layer_mask.has_skipped_layers(),
        "step timing reference must keep every layer"
    );
    let mut reference_times = Vec::new();
    let mut candidate_times = Vec::new();
    let mut reference_predictions = None;
    let mut candidate_predictions = None;
    let mut paired_gains = Vec::new();
    for trial in 0..config.trials {
        let (a, b) = if trial % 2 == 0 {
            (
                run_trace(reference, ids, config.warm_prefix_tokens)?,
                run_trace(candidate, ids, config.warm_prefix_tokens)?,
            )
        } else {
            let b = run_trace(candidate, ids, config.warm_prefix_tokens)?;
            (run_trace(reference, ids, config.warm_prefix_tokens)?, b)
        };
        ensure!(
            a.timed_prefix_lengths == b.timed_prefix_lengths,
            "timed token workloads differ"
        );
        for (trace, prior) in [
            (&a, &mut reference_predictions),
            (&b, &mut candidate_predictions),
        ] {
            if let Some(predictions) = prior.as_ref() {
                ensure!(
                    predictions == &trace.predicted_token_ids,
                    "unstable per-model forced-prefix predictions across trials"
                );
            } else {
                *prior = Some(trace.predicted_token_ids.clone());
            }
        }
        let first_prediction_divergence_prefix_tokens = a
            .predicted_token_ids
            .iter()
            .zip(&b.predicted_token_ids)
            .position(|(a, b)| a != b)
            .map(|index| a.timed_prefix_lengths[index]);
        reference_times.push(a.elapsed_ms);
        candidate_times.push(b.elapsed_ms);
        paired_gains.push(a.elapsed_ms / b.elapsed_ms - 1.0);
        report.append(json!({"kind":"model_quality_step_timing","split":split,"passage_id":passage_id,
            "trial":trial,"input_token_ids":ids,"reference":a,"candidate":b,
            "execution_order":if trial%2 == 0 {vec!["reference","candidate"]} else {vec!["candidate","reference"]},
            "first_prediction_divergence_prefix_tokens":first_prediction_divergence_prefix_tokens,
            "scope":"matched observed prefixes, greedy next-token API including GPU argmax/readback and final stream drains; excludes reset/scratch/warmup; NOT free generation or HTTP TPS"}))?;
    }
    let ref_median = median(reference_times.clone())?;
    let cand_median = median(candidate_times.clone())?;
    report.append(json!({"kind":"model_quality_step_timing_summary","split":split,"passage_id":passage_id,
        "trials":config.trials,"reference_elapsed_ms":reference_times,"candidate_elapsed_ms":candidate_times,
        "reference_median_ms":ref_median,"candidate_median_ms":cand_median,
        "paired_throughput_gains":paired_gains,"median_paired_throughput_gain":median(paired_gains.clone())?,
        "reference_mad_over_median":median(reference_times.iter().map(|v|(v-ref_median).abs()).collect())?/ref_median,
        "candidate_mad_over_median":median(candidate_times.iter().map(|v|(v-cand_median).abs()).collect())?/cand_median,
        "has_stable_predictions":true,"is_exact_engine_promotion":false}))?;
    eprintln!("{split:?} {passage_id}: {} paired forced-prefix trials, reference={ref_median:.2}ms candidate={cand_median:.2}ms; not generated/HTTP TPS",config.trials);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use m40_llm::{cuda::CudaContext, kv_compression::ScopedRuntimeConfig};

    fn model(skipped: &[usize]) -> Result<LoadedModel> {
        let (gguf, weights) =
            crate::fixture::make_tied_identity_tiny_gguf(crate::fixture::TinyGgufConfig {
                vocab: 96,
                d_model: 64,
                hidden: 128,
                head_count: 1,
                block_count: 4,
                context_length: 16,
            });
        let mut model = LoadedModel::from_gguf(gguf, weights, -1)?;
        model.set_diagnostic_layer_mask(skipped)?;
        model.allocate_kv_cache_for_layers(16)?;
        Ok(model)
    }

    /// Fixed observed tokens, not sampled continuations, drive matched steps.
    /// Reset makes a repeated trace independent of a preceding different input.
    #[test]
    fn test_step_trace_forces_prefixes_and_resets() -> Result<()> {
        let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
        let initial = CudaContext::total_device_bytes();
        {
            let mut model = model(&[1])?;
            let first = run_trace(&mut model, &[1, 2, 3, 4, 5, 6], 2)?;
            assert_eq!(first.warm_prediction_id, 2);
            assert_eq!(first.timed_prefix_lengths, vec![3, 4, 5]);
            assert_eq!(first.predicted_token_ids, vec![3, 4, 5]);
            assert_eq!(first.timed_steps, 3);
            assert!(first.elapsed_ms > 0.0 && first.steps_per_second.is_finite());
            run_trace(&mut model, &[9, 8, 7, 6, 5], 1)?;
            assert_eq!(
                run_trace(&mut model, &[1, 2, 3, 4, 5, 6], 2)?.predicted_token_ids,
                first.predicted_token_ids
            );
            assert!(run_trace(&mut model, &[1, 2, 3], 2).is_err());
            assert!(run_trace(&mut model, &[1, 2, 3, 96], 1).is_err());
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }

    /// Paired records alternate order, retain equal workloads and raw times,
    /// and keep compute-skipping measurements separate from exact promotion.
    #[test]
    fn test_paired_step_records_preserve_scope() -> Result<()> {
        let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
        let mut reference = model(&[])?;
        let mut candidate = model(&[1])?;
        let destination = crate::test_files::TestReportPath::new();
        let mut report = Report::create(destination.path())?;
        measure_pairs(
            &mut report,
            Split::Calibration,
            "passage",
            &mut reference,
            &mut candidate,
            &[1, 2, 3, 4, 5],
            StepTimingConfig {
                trials: 2,
                warm_prefix_tokens: 2,
            },
        )?;
        let records = std::fs::read_to_string(destination.path())?
            .lines()
            .map(serde_json::from_str::<serde_json::Value>)
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(records.len(), 3);
        assert_eq!(
            records[0]["execution_order"],
            json!(["reference", "candidate"])
        );
        assert_eq!(
            records[1]["execution_order"],
            json!(["candidate", "reference"])
        );
        assert_eq!(
            records[0]["reference"]["timed_steps"],
            records[0]["candidate"]["timed_steps"]
        );
        assert_eq!(records[2]["trials"], 2);
        assert_eq!(records[2]["has_stable_predictions"], true);
        assert_eq!(records[2]["is_exact_engine_promotion"], false);
        Ok(())
    }
}
