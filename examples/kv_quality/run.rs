use super::{
    config::{self, CacheArm},
    evidence::{command_text, Report},
    execution,
    fixture::{self, Case, ContextKind, Manifest},
    metrics, november,
    prefix::{collect_prefixes, PrefixObservations},
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream},
    evaluation::{
        corpus::Corpus,
        identity::{file_sha256, sha256},
        scoring::TaskScore,
    },
    generate::GeneratedText,
    gguf,
    infer::LoadedModel,
    tokenizer::Tokenizer,
};
use serde_json::json;
use std::{
    collections::{BTreeMap, VecDeque},
    fs,
    io::Write,
    path::Path,
    time::Instant,
};

const INPUT_SHA256: &str = "e2e7ce56f0223bf929705dd49c33610ac01a0992837bb9c376bcea6dfc62b5de";

fn classify_quality(reference: bool, candidate: bool) -> &'static str {
    if !reference {
        "inconclusive"
    } else if candidate {
        "preserved"
    } else {
        "regression"
    }
}

fn record_generation(
    model: &mut LoadedModel,
    case: &Case,
    arm: CacheArm,
    tokenizer: &Tokenizer,
    report: &mut Report,
) -> Result<(GeneratedText, TaskScore)> {
    eprintln!("kv quality generation case={} arm={arm:?}", case.id);
    let (generated, dispatch, state) = execution::generate_case(model, case, arm)?;
    let score = case.task.scoring.score(&generated.output)?;
    let ids = &generated.token_ids[generated.prompt_token_len..];
    let repetition_ids = if ids
        .last()
        .is_some_and(|id| tokenizer.stop_ids().contains(id))
    {
        &ids[..ids.len() - 1]
    } else {
        ids
    };
    report.append(json!({"kind":"kv_quality_generation","case":case.id,"context":case.context,"arm":arm,
        "output":generated.output,"generated_ids":ids,"metrics":generated.metrics,"score":score,
        "repetition":metrics::measure_repetition(repetition_ids),"prefill_mode":generated.prefill_mode,
        "prefill_temporary_weight_bytes":generated.q8_prefill_temporary_weight_bytes,
        "dispatch":dispatch,"cache":state,"is_performance_measurement":false}))?;
    Ok((generated, score))
}

fn retain_logits(
    directory: &Path,
    case: &Case,
    arm: CacheArm,
    observations: &PrefixObservations,
) -> Result<()> {
    for (step, logits) in observations.steps.iter().zip(&observations.logits) {
        let path = directory.join(format!("{}-{arm:?}-{step}.f32le", case.id));
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)?;
        let bytes: Vec<u8> = logits
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        file.write_all(&bytes)?;
    }
    Ok(())
}

pub fn run(model_path: &Path, corpus_path: &Path, inputs: &Path, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "refuse to overwrite quality evidence");
    let environment: BTreeMap<String, String> = std::env::vars()
        .filter(|(key, _)| key.starts_with("M40LLM_") || key.starts_with("CUDA_"))
        .collect();
    config::validate_environment(&environment)?;
    ensure!(
        command_text("git", &["status", "--porcelain"])?.is_empty(),
        "commit source and frozen inputs before inference"
    );
    let revision = command_text("git", &["rev-parse", "HEAD"])?;
    ensure!(
        file_sha256(model_path)? == fixture::MODEL_SHA256,
        "wrong model source"
    );
    let corpus_bytes = fs::read(corpus_path)?;
    ensure!(
        sha256(&corpus_bytes)? == fixture::CORPUS_SHA256,
        "wrong corpus source"
    );
    let input_hash = file_sha256(inputs)?;
    let is_november =
        input_hash == "18d0c6f69dbdfa3bf976780b5b20e14d06dfec5f32980418a9d4f308da9f1cc3";
    ensure!(
        is_november || input_hash == INPUT_SHA256,
        "inputs changed after freeze"
    );
    let corpus = Corpus::parse(std::str::from_utf8(&corpus_bytes)?)?;
    let metadata = gguf::load_gguf(model_path)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&metadata.metadata)?;
    let manifest: Manifest = serde_json::from_slice(&fs::read(inputs)?)?;
    let identity = fixture::tokenizer_identity(&metadata)?;
    let logit_directory = output.with_extension("logits");
    if is_november {
        let expected = november::create(&tokenizer, identity)?;
        ensure!(
            serde_json::to_value(&manifest)? == serde_json::to_value(expected)?,
            "November input replay differs"
        );
        fs::create_dir(&logit_directory)?;
    } else {
        manifest.validate(&corpus, &tokenizer, identity)?;
    }
    let executable = std::env::current_exe()?;
    let mut report = Report::create(output)?;
    let initial = CudaContext::total_device_bytes();
    let started = Instant::now();
    let mut total_generated = 0;
    let mut generation_rows = 0;
    let mut prefix_rows = 0;
    {
        let mut model = execution::load_model(model_path)?;
        report.append(json!({"kind":"kv_quality_manifest","schema":"m40-kv-quality-results-v1",
            "source_revision":revision,"source_clean":true,"executable_sha256":file_sha256(&executable)?,
            "model_sha256":fixture::MODEL_SHA256,"corpus_sha256":manifest.corpus_sha256,
            "inputs_sha256":input_hash,"inputs":manifest,"environment":environment,
            "prepared_weights":model.q8_execution_stats(),"gpu":format!("{:?}",model.cuda.current_device_props()?),
            "capacity":fixture::CAPACITY,"recent_window":fixture::RECENT,"block_size":fixture::BLOCK,
            "arms":CacheArm::ALL,"quantized_backing":"fp16-k-q4-v","quantized_attention":"fp16-k-q4-v-direct",
            "quantized_all_top_blocks":64,"quantized_top8_top_blocks":8,"selection_policy":"plain top-k; no policy overrides",
            "initial_tracked_device_bytes":initial,"is_performance_measurement":false}))?;
        let mut dense_runs = VecDeque::new();
        if is_november {
            for case in &manifest.cases {
                dense_runs.push_back(record_generation(
                    &mut model,
                    case,
                    CacheArm::Dense,
                    &tokenizer,
                    &mut report,
                )?);
            }
            report.append(json!({"kind":"dense_grid_complete","cases":dense_runs.len()}))?;
        }
        for case in &manifest.cases {
            let mut generations = Vec::new();
            let mut scores = Vec::new();
            for arm in CacheArm::ALL {
                let (generated, score) = if is_november && arm == CacheArm::Dense {
                    dense_runs.pop_front().context("missing dense reference")?
                } else {
                    record_generation(&mut model, case, arm, &tokenizer, &mut report)?
                };
                total_generated += generated.token_ids.len() - generated.prompt_token_len;
                generation_rows += 1;
                generations.push(generated);
                scores.push(score);
            }
            let dense = &generations[0];
            let continuation = &dense.token_ids[dense.prompt_token_len..];
            eprintln!(
                "kv quality prefixes case={} dense_tokens={}",
                case.id,
                continuation.len()
            );
            let reference =
                collect_prefixes(&mut model, CacheArm::Dense, &case.prompt_ids, continuation)?;
            if is_november {
                retain_logits(&logit_directory, case, CacheArm::Dense, &reference)?;
            }
            report.append(json!({"kind":"kv_quality_prefix_reference","case":case.id,"steps":reference.steps,
                "dispatch":reference.dispatch,"cache":reference.state,"targets":reference.steps.iter().map(|&step|continuation[step]).collect::<Vec<_>>()}))?;
            for (index, arm) in [(1, CacheArm::QuantizedAll), (2, CacheArm::QuantizedTop8)] {
                let candidate = collect_prefixes(&mut model, arm, &case.prompt_ids, continuation)?;
                ensure!(
                    candidate.steps == reference.steps,
                    "mismatched observation positions"
                );
                if is_november {
                    retain_logits(&logit_directory, case, arm, &candidate)?;
                }
                let comparisons = reference.steps.iter().enumerate().map(|(row,&step)| {
                    Ok(json!({"step":step,"prefix_tokens":case.prompt_ids.len()+step,"target":continuation[step],
                        "logits":metrics::compare_logits(&reference.logits[row],&candidate.logits[row],continuation[step])?}))
                }).collect::<Result<Vec<_>>>()?;
                let actual = &generations[index];
                let first_difference = dense
                    .token_ids
                    .iter()
                    .zip(&actual.token_ids)
                    .position(|(a, b)| a != b)
                    .or_else(|| {
                        (dense.token_ids.len() != actual.token_ids.len())
                            .then_some(dense.token_ids.len().min(actual.token_ids.len()))
                    })
                    .map(|position| position - dense.prompt_token_len);
                report.append(json!({"kind":"kv_quality_comparison","case":case.id,"context":case.context,"arm":arm,
                    "quality_conclusion":classify_quality(scores[0].passed,scores[index].passed),
                    "dense_score":scores[0],"candidate_score":scores[index],
                    "identical_generated_ids":dense.token_ids==actual.token_ids,
                    "identical_output":dense.output==actual.output,"first_generated_difference":first_difference,
                    "no_old_control":case.context==ContextKind::Short,"prefix_observations":comparisons,
                    "dispatch":candidate.dispatch,"cache":candidate.state,
                    "prefix_scope":"same dense-generated history; sparse full-logit observations, not corpus perplexity"}))?;
                prefix_rows += 1;
            }
            report.append(json!({"kind":"kv_quality_case_complete","case":case.id,"generation_arms":3,"comparison_arms":2}))?;
        }
        model.cuda.synchronize_stream(CudaStream::Decode)?;
        model.cuda.synchronize_stream(CudaStream::Prefill)?;
    }
    let final_bytes = CudaContext::total_device_bytes();
    ensure!(
        final_bytes == initial,
        "tracked device allocations remain after model teardown"
    );
    ensure!(
        generation_rows == manifest.cases.len() * 3 && prefix_rows == manifest.cases.len() * 2,
        "incomplete frozen grid"
    );
    report.append(json!({"kind":"kv_quality_complete","cases":manifest.cases.len(),"generation_rows":generation_rows,
        "comparison_rows":prefix_rows,"generated_tokens":total_generated,"initial_tracked_device_bytes":initial,
        "final_tracked_device_bytes":final_bytes,"elapsed_wall_seconds":started.elapsed().as_secs_f64(),
        "is_performance_measurement":false}))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense failure never becomes a quality-preservation claim, even if the
    /// approximate arm happens to pass; dense-valid failures remain regressions.
    #[test]
    fn test_quality_requires_a_passing_dense_reference() {
        assert_eq!(classify_quality(false, false), "inconclusive");
        assert_eq!(classify_quality(false, true), "inconclusive");
        assert_eq!(classify_quality(true, false), "regression");
        assert_eq!(classify_quality(true, true), "preserved");
    }
}
