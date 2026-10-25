use super::{
    records::{QualityConclusion, Report},
    run::memory,
};
use anyhow::{ensure, Result};
use m40_llm::{
    evaluation::corpus::{Split, Task},
    generate::{generate_text, prepare_prompt, GenerateOptions, GeneratedText, PromptFormat},
    infer::LoadedModel,
    kv_compression::KvCompressionConfig,
    tokenizer::Tokenizer,
};
use serde_json::{json, Value};
use std::time::Instant;

pub fn encode_task(tokenizer: &Tokenizer, user_prompt: &str) -> Result<(String, Vec<u32>)> {
    let (prompt, add_bos) = if tokenizer.is_qwen2() {
        (format!("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"), false)
    } else {
        prepare_prompt(tokenizer, user_prompt, PromptFormat::Auto)
    };
    let ids = tokenizer.encode_with_specials(&prompt, add_bos, false)?;
    Ok((prompt, ids))
}

pub fn encode_passage(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    tokenizer.encode_with_specials(text, tokenizer.bos_id().is_some(), false)
}

pub(super) fn generate(
    model: &LoadedModel,
    prompt: &str,
    max_tokens: usize,
) -> Result<GeneratedText> {
    generate_text(
        model,
        GenerateOptions {
            prompt: prompt.to_owned(),
            prompt_format: PromptFormat::Raw,
            top_k: Some(1),
            seed: Some(0),
            max_tokens: Some(max_tokens),
            kv_compression: KvCompressionConfig::dense_reference(),
            reset_kv_cache: true,
            ..Default::default()
        },
    )
}

pub fn warmup(model: &LoadedModel, tokenizer: &Tokenizer) -> Result<()> {
    let (prompt, _) = encode_task(tokenizer, "Say ready.")?;
    generate(model, &prompt, 2)?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct TaskResult {
    pub reference_passed: bool,
    pub candidate_passed: bool,
    pub first_reference_token_id: Option<u32>,
    pub has_identical_generation: bool,
}

pub(super) fn generation_record(
    generated: &GeneratedText,
    wall_ms: f64,
    model: &LoadedModel,
) -> Result<Value> {
    Ok(json!({"output":generated.output,
        "generated_token_ids":generated.token_ids[generated.prompt_token_len..],
        "metrics":generated.metrics,"resident_generation_wall_ms":wall_ms,
        "prefill_mode":generated.prefill_mode,"prefill_chunk_size":generated.prefill_chunk_size,
        "memory":memory(model)?}))
}

pub fn evaluate_task(
    report: &mut Report,
    split: Split,
    task: &Task,
    reference: &LoadedModel,
    candidate: &LoadedModel,
    tokenizer: &Tokenizer,
) -> Result<TaskResult> {
    let (prompt, expected_ids) = encode_task(tokenizer, &task.prompt)?;
    reference.reset_activation_statistics()?;
    candidate.reset_activation_statistics()?;
    let start = Instant::now();
    let reference_generated = generate(reference, &prompt, task.max_tokens)?;
    let reference_wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let candidate_generated = generate(candidate, &prompt, task.max_tokens)?;
    let candidate_wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    for generated in [&reference_generated, &candidate_generated] {
        ensure!(
            generated.token_ids[..generated.prompt_token_len] == expected_ids,
            "generation changed task prompt tokenization: {}",
            task.id
        );
        ensure!(
            generated.metrics.generated_tokens
                == generated.token_ids.len() - generated.prompt_token_len,
            "generation token accounting changed: {}",
            task.id
        );
    }
    let reference_score = task.scoring.score(&reference_generated.output)?;
    let candidate_score = task.scoring.score(&candidate_generated.output)?;
    let first_generated_divergence = reference_generated.token_ids
        [reference_generated.prompt_token_len..]
        .iter()
        .zip(&candidate_generated.token_ids[candidate_generated.prompt_token_len..])
        .position(|(a, b)| a != b)
        .or_else(|| {
            (reference_generated.metrics.generated_tokens
                != candidate_generated.metrics.generated_tokens)
                .then_some(
                    reference_generated
                        .metrics
                        .generated_tokens
                        .min(candidate_generated.metrics.generated_tokens),
                )
        });
    let has_identical_generation = first_generated_divergence.is_none()
        && reference_generated.output == candidate_generated.output
        && reference_generated.metrics.stop_reason == candidate_generated.metrics.stop_reason;
    report.append(json!({"kind":"model_quality_task","split":split,"id":task.id,"family":task.family,
        "user_prompt":task.prompt,"rendered_prompt":prompt,"prompt_token_ids":expected_ids,
        "max_tokens":task.max_tokens,"scoring_rule":task.scoring,
        "reference":generation_record(&reference_generated, reference_wall_ms, reference)?,
        "candidate":generation_record(&candidate_generated, candidate_wall_ms, candidate)?,
        "reference_score":reference_score,"candidate_score":candidate_score,
        "quality_conclusion":QualityConclusion::compare(reference_score.passed, candidate_score.passed),
        "first_generated_divergence_step_zero_based":first_generated_divergence,
        "has_identical_generation":has_identical_generation}))?;
    eprintln!(
        "{split:?} {}: reference={}, candidate={}, prefill={:.1}/{:.1} ms decode={:.1}/{:.1} ms",
        task.id,
        reference_score.passed,
        candidate_score.passed,
        reference_generated.metrics.prefill_elapsed_ms,
        candidate_generated.metrics.prefill_elapsed_ms,
        reference_generated.metrics.decode_elapsed_ms,
        candidate_generated.metrics.decode_elapsed_ms
    );
    Ok(TaskResult {
        reference_passed: reference_score.passed,
        candidate_passed: candidate_score.passed,
        first_reference_token_id: reference_generated
            .token_ids
            .get(reference_generated.prompt_token_len)
            .copied(),
        has_identical_generation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture;
    use m40_llm::{
        cuda::CudaContext, evaluation::scoring::ScoringRule, kv_compression::ScopedRuntimeConfig,
    };
    fn model() -> Result<LoadedModel> {
        let (gguf, weights) = fixture::make_tied_identity_tiny_gguf(fixture::TinyGgufConfig {
            vocab: 256,
            d_model: 64,
            hidden: 128,
            head_count: 1,
            block_count: 2,
            context_length: 64,
        });
        let mut model = LoadedModel::from_gguf(gguf, weights, -1)?;
        model.allocate_kv_cache_for_layers(64)?;
        Ok(model)
    }

    /// Two real independent CUDA models must emit matching prompt/output IDs,
    /// explicit dense timing/memory records, and inconclusive dense-failed scores.
    #[test]
    #[ignore = "requires CUDA hardware; run with --include-ignored --test-threads=1"]
    fn test_task_records_use_explicit_dense_reference() -> Result<()> {
        let _scope = crate::cuda_test_scope::lock_cuda();
        let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
        let initial = CudaContext::total_device_bytes();
        {
            let reference = model()?;
            let candidate = model()?;
            let tokenizer = Tokenizer::byte_level();
            let task = Task {
                id: "task-record".to_owned(),
                family: m40_llm::evaluation::corpus::TaskFamily::Chat,
                prompt: "Hi".to_owned(),
                max_tokens: 2,
                scoring: ScoringRule::Exact {
                    answers: vec!["not the identity model output".to_owned()],
                },
            };
            let destination = crate::test_files::TestReportPath::new();
            let path = destination.path();
            let mut report = Report::create(path)?;
            let result = evaluate_task(
                &mut report,
                Split::HeldOut,
                &task,
                &reference,
                &candidate,
                &tokenizer,
            )?;
            assert_eq!(
                (result.reference_passed, result.candidate_passed),
                (false, false)
            );
            assert!(result.has_identical_generation);
            assert!(result.first_reference_token_id.is_some());
            let record: Value = serde_json::from_str(std::fs::read_to_string(path)?.trim())?;
            assert_eq!(record["quality_conclusion"], "inconclusive");
            assert_eq!(record["prompt_token_ids"], json!(tokenizer.encode("Hi")?));
            assert_eq!(
                record["reference"]["generated_token_ids"],
                record["candidate"]["generated_token_ids"]
            );
            assert_eq!(
                record["first_generated_divergence_step_zero_based"],
                Value::Null
            );
            assert_eq!(record["reference"]["metrics"]["generated_tokens"], 2);
            assert!(
                record["reference"]["resident_generation_wall_ms"]
                    .as_f64()
                    .unwrap()
                    > 0.0
            );
            assert!(
                record["reference"]["memory"]["resident_kv_bytes"]
                    .as_u64()
                    .unwrap()
                    > 0
            );
            let actual_materialized_bytes: usize = reference
                .materialized_weights
                .lock()
                .unwrap()
                .values()
                .map(|weight| weight.bytes)
                .sum();
            assert_eq!(
                record["reference"]["memory"]["materialized_f32_bytes"],
                actual_materialized_bytes
            );
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }
}
