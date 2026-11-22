use super::{records, run, sparse_q4_extended, tasks, Args, SplitChoice};
use anyhow::{ensure, Context, Result};
use clap::ValueEnum;
use m40_llm::{
    cuda::CudaContext,
    evaluation::{
        corpus::{Corpus, Split},
        ffn_capture::FfnTensorSource,
        teacher_forced::{compare_loaded_models, score_loaded_model},
    },
    generate::PromptPrefill,
    infer::{
        activation_sparsity::{ActivationThreshold, Q4ResearchCutoff},
        execution_weights::Q8ExecutionConfig,
        q4_channels::Q4ResearchLayout,
    },
};
use serde::Deserialize;
use serde_json::json;
use std::{fs, time::Instant};

const MODEL_SHA: &str = "670d82d0fbee6289b661eb323612c65bd2e97ec8c65cdbd9131aa44d30e9816a";
const CORPUS_SHA: &str = "2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38";
const FROZEN_CUTOFFS: &[u8] =
    include_bytes!("../../docs/experiments/2026-11-22-sparse-q4-live/cutoffs.json");

#[derive(Clone, Copy, ValueEnum)]
pub enum Layout {
    Packed,
    Zigzag,
}

#[derive(Deserialize)]
struct Cutoffs {
    model_sha256: String,
    corpus_sha256: String,
    source_weights_sha256: String,
    layers: [Layer; 3],
}

#[derive(Deserialize)]
struct Layer {
    layer: usize,
    down: FfnTensorSource,
    cutoffs: [f32; 3],
}

pub(super) fn evaluate(args: &Args) -> Result<()> {
    let is_extended = args.sparse_q4_extended_input.is_some();
    let expected_context = if is_extended { 3072 } else { 1024 };
    ensure!(
        matches!(args.split, SplitChoice::HeldOut) && args.context_tokens == expected_context,
        "sparse Q4 requires held-out split and capacity {expected_context}"
    );
    let environment = run::environment()?;
    for (key, expected) in [
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_PREFILL_CHUNK_SIZE", "0"),
        ("M40LLM_CUDA_GREEDY_ARGMAX", "1"),
    ] {
        ensure!(
            environment.get(key).map(String::as_str) == Some(expected),
            "sparse Q4 requires {key}={expected}"
        );
    }
    ensure!(
        fs::canonicalize(&args.model)? == fs::canonicalize(&args.reference_model)?
            && records::file_sha256(&args.model)? == MODEL_SHA,
        "sparse Q4 compares the same pinned native checkpoint"
    );
    let bytes = fs::read(
        args.sparse_q4_cutoffs
            .as_ref()
            .context("missing frozen thresholds")?,
    )?;
    ensure!(bytes == FROZEN_CUTOFFS, "sparse Q4 threshold bytes differ");
    let corpus_bytes = fs::read(&args.corpus)?;
    ensure!(
        records::sha256(&corpus_bytes)? == CORPUS_SHA,
        "sparse Q4 corpus differs"
    );
    let corpus = Corpus::parse(std::str::from_utf8(&corpus_bytes)?)?;
    let extended = args
        .sparse_q4_extended_input
        .as_deref()
        .map(|path| sparse_q4_extended::load_inputs(path, &args.model))
        .transpose()?;
    let Cutoffs {
        model_sha256,
        corpus_sha256,
        source_weights_sha256,
        layers,
    } = serde_json::from_slice(&bytes)?;
    ensure!(
        model_sha256 == MODEL_SHA && corpus_sha256 == CORPUS_SHA,
        "cutoff provenance differs"
    );
    let (layout, primary_cutoff) = match args.sparse_q4_layout.context("missing Q4 layout")? {
        Layout::Packed => (Q4ResearchLayout::Packed, 1),
        Layout::Zigzag => (Q4ResearchLayout::Zigzag, 2),
    };
    let level = if args.sparse_q4_thresholded {
        primary_cutoff
    } else {
        0
    };
    let cutoffs = layers
        .into_iter()
        .map(|layer| {
            Ok(Q4ResearchCutoff {
                layer_id: layer.layer,
                down: layer.down,
                threshold: ActivationThreshold::new(layer.cutoffs[level])?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let cutoffs: [Q4ResearchCutoff; 3] = cutoffs.try_into().ok().context("wrong Q4 layer count")?;
    let mut report = records::Report::create(&args.output)?;
    let initial = CudaContext::total_device_bytes();
    let start = Instant::now();
    let (mut reference, tokenizer) = run::load_model_with_setup(
        &args.model,
        args.context_tokens,
        Q8ExecutionConfig::disabled(),
        |_| Ok(()),
    )?;
    let reference_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let (mut candidate, _) = run::load_model_with_setup(
        &args.model,
        args.context_tokens,
        Q8ExecutionConfig::disabled(),
        |model| {
            model.configure_q4_research(layout, &source_weights_sha256, &cutoffs, 40 * 1024 * 1024)
        },
    )?;
    let candidate_ms = start.elapsed().as_secs_f64() * 1000.0;
    report.append(json!({
        "kind":"sparse_q4_quality_manifest","variant":args.variant_id,"layout":layout,
        "is_thresholded":args.sparse_q4_thresholded,"model_sha256":model_sha256,
        "corpus_sha256":corpus_sha256,"cutoffs_sha256":records::sha256(&bytes)?,
        "revision":records::command_text("git",&["rev-parse","HEAD"])?,
        "worktree":records::command_text("git",&["status","--short"])?,
        "executable_sha256":records::file_sha256(&std::env::current_exe()?)?,
        "environment":environment,"reference_load_ms":reference_ms,"candidate_load_ms":candidate_ms,
        "reference_memory":run::memory(&reference)?,"candidate_memory":run::memory(&candidate)?,
        "candidate_layout":candidate.q4_research_stats()?,
        "timing_scope":"held-out quality diagnostics, two models resident in one process"
    }))?;
    tasks::warmup(&reference, &tokenizer)?;
    tasks::warmup(&candidate, &tokenizer)?;
    if let Some(inputs) = extended {
        let summary = sparse_q4_extended::evaluate(
            &inputs,
            &mut reference,
            &mut candidate,
            &mut report,
            &args.output.with_extension("logits"),
        )?;
        let layout = candidate.q4_research_stats()?;
        let memory = run::memory(&candidate)?;
        drop(candidate);
        drop(reference);
        ensure!(
            CudaContext::total_device_bytes() == initial,
            "sparse Q4 leaked a device allocation"
        );
        return report.append(json!({
            "kind":"sparse_q4_extended_complete", "summary":summary,
            "candidate_layout":layout, "candidate_memory":memory,
            "device_bytes_after":CudaContext::total_device_bytes()
        }));
    }
    let split = corpus.split(Split::HeldOut);
    let mut stop_reason = None;
    let mut has_usable_dense_passages = true;
    let mut dense_passes = 0;
    let mut dense_failures = 0;
    let mut tasks_run = 0;
    for task in &split.tasks {
        let before = candidate
            .q4_research_stats()?
            .context("missing Q4 layout")?
            .projection_calls;
        let result = tasks::evaluate_task_with_prefill(
            &mut report,
            Split::HeldOut,
            task,
            &reference,
            &candidate,
            &tokenizer,
            (PromptPrefill::Sequential, PromptPrefill::Sequential),
        )?;
        let after = candidate
            .q4_research_stats()?
            .context("missing Q4 layout")?
            .projection_calls;
        ensure!(
            after.into_iter().zip(before).all(|(a, b)| a > b),
            "a sparse Q4 layer failed to execute for {}",
            task.id
        );
        tasks_run += 1;
        dense_passes += usize::from(result.reference_passed);
        dense_failures += usize::from(!result.reference_passed);
        if (result.reference_passed && !result.candidate_passed) || result.has_new_repetition {
            let reason = if result.has_new_repetition {
                "new_repetition"
            } else {
                "lost_dense_pass"
            };
            stop_reason = Some(json!({"reason":reason,"id":task.id}));
            break;
        }
    }
    let mut passages_run = 0;
    if stop_reason.is_none() {
        for passage in &split.passages {
            let ids = tasks::encode_passage(&tokenizer, &passage.text)?;
            let comparison = match compare_loaded_models(&mut reference, &mut candidate, &ids) {
                Ok(comparison) => comparison,
                Err(error) if error.to_string().contains("nonfinite") => {
                    has_usable_dense_passages = score_loaded_model(&mut reference, &ids).is_ok();
                    stop_reason = Some(json!({"reason":"nonfinite_logits","id":passage.id,
                        "message":format!("{error:#}")}));
                    break;
                }
                Err(error) => return Err(error),
            };
            let ratio = comparison
                .reference
                .perplexity
                .zip(comparison.candidate.perplexity)
                .map(|(a, b)| b / a);
            report.append(json!({
                "kind":"sparse_q4_quality_passage","id":passage.id,"token_ids":ids,
                "comparison":comparison,"perplexity_ratio":ratio
            }))?;
            passages_run += 1;
            if comparison.reference.perplexity.is_none() {
                has_usable_dense_passages = false;
                stop_reason = Some(json!({"reason":"unusable_dense_perplexity","id":passage.id}));
                break;
            }
            if !ratio.is_some_and(|ratio| ratio.is_finite() && ratio <= 1.03) {
                stop_reason = Some(json!({"reason":"perplexity","id":passage.id}));
                break;
            }
        }
    }
    let disposition = if !has_usable_dense_passages {
        "INCONCLUSIVE"
    } else if stop_reason.is_some() {
        "FAIL"
    } else if dense_passes > 0 {
        "PASS"
    } else {
        "INCONCLUSIVE"
    };
    let candidate_memory = run::memory(&candidate)?;
    let candidate_layout = candidate.q4_research_stats()?;
    drop(candidate);
    drop(reference);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "sparse Q4 leaked a device allocation"
    );
    report.append(json!({
        "kind":"sparse_q4_quality_complete","disposition":disposition,"failure":stop_reason,
        "tasks_run":tasks_run,"dense_passes":dense_passes,"dense_failures":dense_failures,
        "passages_run":passages_run,"candidate_memory":candidate_memory,"candidate_layout":candidate_layout,
        "device_bytes_after":CudaContext::total_device_bytes()
    }))
}
