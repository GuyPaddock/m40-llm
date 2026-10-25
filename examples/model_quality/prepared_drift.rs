use super::{
    activations::ActivationSetup,
    records::{self, Report},
    research, run, Args,
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, DeviceBuffer},
    decode_session::DecodeSession,
    evaluation::loss::score_logits,
    infer::LoadedModel,
    kv_compression::KvCompressionConfig,
};
use serde_json::{json, Value};
use std::{ffi::c_void, time::Instant};

mod trace;
use trace::{candidate_config, PinnedTrace};

struct HeadProbe {
    logits: DeviceBuffer,
    norm: Option<DeviceBuffer>,
    token: DeviceBuffer,
}

impl HeadProbe {
    fn new(model: &LoadedModel) -> Result<Self> {
        let (_, _, dimension, vocabulary, _) = model.map_lm_head()?;
        Ok(Self {
            logits: DeviceBuffer::new_tagged(&model.cuda, vocabulary * 4, "prefix_probe:logits")?,
            norm: model
                .map_output_norm()?
                .map(|_| DeviceBuffer::new_tagged(&model.cuda, dimension * 4, "prefix_probe:norm"))
                .transpose()?,
            token: DeviceBuffer::new_tagged(&model.cuda, 4, "prefix_probe:token")?,
        })
    }

    /// Input comes from a completed greedy step on the same CUDA device.
    /// Recompute heads only; neither hidden state nor KV may be mutated.
    unsafe fn evaluate(
        &self,
        model: &LoadedModel,
        hidden: *const c_void,
        target: u32,
    ) -> Result<(Value, Vec<f32>)> {
        let norm = self.norm.as_ref().map(DeviceBuffer::as_mut_ptr);
        let actual = model.greedy_token_from_hidden_into(
            hidden,
            self.logits.as_mut_ptr(),
            norm,
            self.token.as_mut_ptr(),
        )?;
        let logits = model.logits_from_hidden_into(hidden, self.logits.as_mut_ptr(), norm)?;
        let prediction = score_logits(&logits, target)?;
        let actual_prediction = score_logits(&logits, actual)?;
        let top = prediction.top_indices[0];
        let margin = prediction
            .top_indices
            .get(1)
            .map(|&second| f64::from(logits[top]) - f64::from(logits[second]));
        Ok((
            json!({"actual_greedy_token_id":actual,
            "generic_expected_token":prediction,"generic_actual_token_rank":actual_prediction.target_rank,
            "generic_actual_token_logit":actual_prediction.target_logit,"generic_top_token_id":top,
            "generic_top_token_logit":logits[top],"generic_top2_margin":margin,
            "is_generic_top_equal_to_actual_greedy":actual as usize==top,
            "scope":"generic full-logit projection is NOT the native fused greedy-head arithmetic"}),
            logits,
        ))
    }
}

fn difference(a: &[f32], b: &[f32]) -> Result<Value> {
    ensure!(
        !a.is_empty() && a.len() == b.len() && a.iter().chain(b).all(|v| v.is_finite()),
        "invalid/nonfinite diagnostic vectors"
    );
    let deltas: Vec<_> = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .collect();
    Ok(
        json!({"elements":a.len(),"max_abs_diff":deltas.iter().copied().fold(0.0,f64::max),
        "mean_abs_diff":deltas.iter().sum::<f64>()/a.len() as f64}),
    )
}

fn read_hidden(model: &LoadedModel, session: &DecodeSession) -> Result<Vec<f32>> {
    let mut values = vec![0.0_f32; model.model_config.embedding_length as usize];
    // greedy_token_for_ids already drained the hidden state's actual producer.
    unsafe {
        model.cuda.memcpy_d2h(
            values.as_mut_ptr().cast(),
            session.d_out_ptr()?.cast_const(),
            values.len() * 4,
        )?;
    }
    Ok(values)
}

fn record_heads(
    report: &mut Report,
    step: usize,
    target: u32,
    models: (&LoadedModel, &LoadedModel),
    sessions: (&DecodeSession, &DecodeSession),
    probes: (&HeadProbe, &HeadProbe),
    actual: (u32, u32),
) -> Result<()> {
    let (reference, candidate) = models;
    let (a, b) = sessions;
    let hidden_a = read_hidden(reference, a)?;
    let hidden_b = read_hidden(candidate, b)?;
    let (aa, ab, ba, bb) = unsafe {
        (
            probes
                .0
                .evaluate(reference, a.d_out_ptr()?.cast_const(), target)?,
            probes
                .1
                .evaluate(candidate, a.d_out_ptr()?.cast_const(), target)?,
            probes
                .0
                .evaluate(reference, b.d_out_ptr()?.cast_const(), target)?,
            probes
                .1
                .evaluate(candidate, b.d_out_ptr()?.cast_const(), target)?,
        )
    };
    ensure!(
        aa.0["actual_greedy_token_id"] == actual.0 && bb.0["actual_greedy_token_id"] == actual.1,
        "same-hidden head recomputation changed actual greedy prediction"
    );
    for (model, session, expected) in [(reference, a, &hidden_a), (candidate, b, &hidden_b)] {
        ensure!(
            read_hidden(model, session)?
                .iter()
                .zip(expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "head diagnostics modified their input hidden vector"
        );
    }
    let top_a = score_logits(&aa.1, target)?.top_indices;
    let top_b = score_logits(&bb.1, target)?.top_indices;
    report.append(json!({"kind":"q8_prefix_head_probe","generated_step_zero_based":step,"expected_reference_token_id":target,
        "hidden_difference":difference(&hidden_a,&hidden_b)?,
        "reference_hidden_native_head":aa.0,"reference_hidden_prepared_head":ab.0,
        "prepared_hidden_native_head":ba.0,"prepared_hidden_prepared_head":bb.0,
        "generic_whole_path_difference":difference(&aa.1,&bb.1)?,
        "generic_head_difference_on_reference_hidden":difference(&aa.1,&ab.1)?,
        "generic_head_difference_on_prepared_hidden":difference(&ba.1,&bb.1)?,
        "generic_forward_difference_with_native_head":difference(&aa.1,&ba.1)?,
        "generic_forward_difference_with_prepared_head":difference(&ab.1,&bb.1)?,
        "generic_whole_path_top10_overlap":top_a.iter().filter(|id|top_b.contains(id)).count(),
        "scope":"same absolute prefix; read-only two-by-two hidden/head recomputation; no new forward or sampled continuation; diagnostic allocation/readback, NOT timing"}))
}

fn probe_models(
    report: &mut Report,
    reference: &mut LoadedModel,
    candidate: &mut LoadedModel,
    trace: &PinnedTrace,
) -> Result<Value> {
    let last = *trace.sample_steps.last().context("no probe steps")?;
    let mut ids = trace.prompt_ids.clone();
    ids.extend_from_slice(&trace.reference_ids);
    ensure!(
        last < trace.reference_ids.len() && !trace.prompt_ids.is_empty(),
        "invalid forced prefix range"
    );
    ensure!(
        reference.cuda.device_id() == candidate.cuda.device_id()
            && reference.model_config.embedding_length == candidate.model_config.embedding_length
            && reference.model_config.vocab_size == candidate.model_config.vocab_size,
        "diagnostic models must share device and hidden/vocabulary dimensions"
    );
    let a_cache = reference.kv_cache.as_ref().context("reference KV")?;
    let b_cache = candidate.kv_cache.as_ref().context("candidate KV")?;
    ensure!(
        !a_cache.has_same_storage(b_cache) && !a_cache.is_compressed() && !b_cache.is_compressed(),
        "prefix diagnostic requires independent dense KV"
    );
    ensure!(
        ids.len() <= a_cache.max_seq_len() as usize
            && ids.len() <= b_cache.max_seq_len() as usize
            && ids.iter().all(|&id| id < reference.model_config.vocab_size),
        "prefix tokens exceed KV capacity or vocabulary"
    );
    reference.reset_kv_cache()?;
    candidate.reset_kv_cache()?;
    let dimension = reference.validate_full_layer_decode()?.0;
    candidate.validate_full_layer_decode()?;
    let mut a = DecodeSession::new_for_sequence_with_kv_config(
        reference,
        0,
        KvCompressionConfig::dense_reference(),
        dimension,
        true,
        "q8_prefix_native",
        "q8_prefix:input",
        "q8_prefix:output",
    )?;
    let mut b = DecodeSession::new_for_sequence_with_kv_config(
        candidate,
        0,
        KvCompressionConfig::dense_reference(),
        dimension,
        true,
        "q8_prefix_prepared",
        "q8_prefix:input",
        "q8_prefix:output",
    )?;
    let probe_a = HeadProbe::new(reference)?;
    let probe_b = HeadProbe::new(candidate)?;
    let mut first_difference = None;
    let mut reference_mismatches = 0;
    let mut prepared_history_mismatches = 0;
    for step in 0..=last {
        let count = trace.prompt_ids.len() + step;
        let expected = ids[count];
        let actual_a = a.greedy_token_for_ids(&ids[..count])?;
        let actual_b = b.greedy_token_for_ids(&ids[..count])?;
        if actual_a != actual_b && first_difference.is_none() {
            first_difference = Some(step);
        }
        reference_mismatches += usize::from(actual_a != expected);
        let historical_prepared =
            (step <= trace.first_difference).then(|| trace.prepared_ids[step]);
        prepared_history_mismatches +=
            usize::from(historical_prepared.is_some_and(|old| old != actual_b));
        report.append(json!({"kind":"q8_forced_prefix_prediction","generated_step_zero_based":step,"prefix_tokens":count,
            "expected_reference_token_id":expected,"reference_greedy_token_id":actual_a,"prepared_greedy_token_id":actual_b,
            "historical_prepared_token_on_same_prefix":historical_prepared,
            "is_reference_equal_to_history":actual_a==expected,"has_same_actual_greedy":actual_a==actual_b,
            "scope":"recorded native continuation is forced; prepared-history expectation ends at historical first branch"}))?;
        if trace.sample_steps.contains(&step) {
            record_heads(
                report,
                step,
                expected,
                (reference, candidate),
                (&a, &b),
                (&probe_a, &probe_b),
                (actual_a, actual_b),
            )?;
            ensure!(
                a.processed_len() == count && b.processed_len() == count,
                "head diagnostics changed session position"
            );
        }
    }
    Ok(
        json!({"kind":"q8_prefix_drift_summary","predictions":last+1,"head_probes":trace.sample_steps.len(),
        "first_different_generated_step":first_difference,"historical_first_difference":trace.first_difference,
        "reference_history_mismatches":reference_mismatches,"prepared_history_mismatches_on_shared_prefix":prepared_history_mismatches,
        "reproduces_historical_first_branch":first_difference==Some(trace.first_difference) && reference_mismatches==0 && prepared_history_mismatches==0,
        "is_performance_measurement":false,"qualifies_for_promotion":false}),
    )
}

pub(super) fn diagnose(args: Args) -> Result<()> {
    ensure!(
        args.step_timing_trials == 0,
        "prefix drift is untimed; omit --step-timing-trials"
    );
    let environment = run::environment()?;
    research::validate_execution(&environment)?;
    let path = args
        .q8_prefix_drift
        .as_ref()
        .context("provide --q8-prefix-drift")?;
    let source_hash = records::file_sha256(&args.reference_model)?;
    ensure!(
        records::file_sha256(&args.model)? == source_hash,
        "prefix drift requires identical original model files"
    );
    let trace = PinnedTrace::load(path, &source_hash, args.context_tokens, &environment)?;
    let mut report = Report::create(&args.output)?;
    let before = CudaContext::total_device_bytes();
    let started = Instant::now();
    let (mut reference, tokenizer) = run::load_model(
        &args.reference_model,
        args.context_tokens,
        &[],
        ActivationSetup::Disabled,
    )?;
    let (mut candidate, candidate_tokenizer) = run::load_model_with_execution(
        &args.model,
        args.context_tokens,
        &[],
        ActivationSetup::Disabled,
        candidate_config()?,
    )?;
    ensure!(
        tokenizer.is_qwen2()
            && candidate_tokenizer.is_qwen2()
            && tokenizer.encode_with_specials(&trace.prompt, false, false)? == trace.prompt_ids
            && candidate_tokenizer.encode_with_specials(&trace.prompt, false, false)?
                == trace.prompt_ids,
        "recorded prompt tokenization differs from loaded Qwen models"
    );
    let reference_identity = run::model_identity(&args.reference_model, &reference, &tokenizer)?;
    let candidate_identity = run::model_identity(&args.model, &candidate, &candidate_tokenizer)?;
    ensure!(
        reference_identity["sha256"] == source_hash
            && candidate_identity["sha256"] == source_hash
            && reference_identity["tokenizer_metadata_sha256"]
                == candidate_identity["tokenizer_metadata_sha256"],
        "model/tokenizer identity changed during diagnostic setup"
    );
    report.append(json!({"kind":"q8_prefix_drift_manifest","variant_id":args.variant_id,
        "reference":reference_identity,"candidate":candidate_identity,"trace":trace.describe(),
        "environment":environment,"candidate_execution":candidate.q8_execution_stats(),
        "revision":records::command_text("git",&["rev-parse","HEAD"])?,"worktree":records::command_text("git",&["status","--short"])?,
        "executable_sha256":records::file_sha256(&std::env::current_exe()?)?,"build_profile":"release",
        "rustc":records::command_text("rustc",&["--version"])?,"nvcc":records::command_text("nvcc",&["--version"])?,
        "host_compiler":records::command_text("g++",&["--version"])?,"cuda_flags":research::CUDA_FLAGS,
        "setup_through_identity_ms":started.elapsed().as_secs_f64()*1000.0,
        "scope":"untimed matched-prefix diagnosis; actual greedy vs generic heads named separately; no geometry search, model change or promotion"}))?;
    let summary = probe_models(&mut report, &mut reference, &mut candidate, &trace)?;
    report.append(summary)?;
    report.append(json!({"kind":"q8_prefix_drift_memory","reference":run::memory(&reference)?,"candidate":run::memory(&candidate)?}))?;
    drop(reference);
    drop(candidate);
    ensure!(
        CudaContext::total_device_bytes() == before,
        "prefix drift leaked tracked device bytes"
    );
    report.append(json!({"kind":"q8_prefix_drift_complete","is_complete":true,"device_bytes_before":before,"device_bytes_after":CudaContext::total_device_bytes()}))
}

#[cfg(test)]
#[path = "prepared_drift_tests.rs"]
mod tests;
