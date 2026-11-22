use super::{records, tasks};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    decode_session::DecodeSession,
    evaluation::{corpus::Task, loss::score_logits},
    generate::PromptPrefill,
    gguf,
    infer::LoadedModel,
    kv_compression::KvCompressionConfig,
    tokenizer::Tokenizer,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

const INPUT_SHA: &str = "18d0c6f69dbdfa3bf976780b5b20e14d06dfec5f32980418a9d4f308da9f1cc3";

#[derive(Deserialize)]
pub(super) struct Inputs {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    context: ContextKind,
    task: Task,
    prompt: String,
    prompt_ids: Vec<u32>,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ContextKind {
    Short,
    Long,
}

pub(super) fn load_inputs(path: &Path, model: &Path) -> Result<Inputs> {
    let bytes = fs::read(path)?;
    ensure!(
        records::sha256(&bytes)? == INPUT_SHA,
        "November inputs differ"
    );
    let inputs: Inputs = serde_json::from_slice(&bytes)?;
    let gguf = gguf::load_gguf(model)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)?;
    ensure!(
        tokenizer.is_qwen2() && inputs.cases.len() == 48,
        "wrong corpus or tokenizer"
    );
    for case in &inputs.cases {
        let ids = tokenizer.encode_with_specials(&case.prompt, false, false)?;
        ensure!(
            ids == case.prompt_ids && ids.len() + case.task.max_tokens <= 3072,
            "Q4 tokenizer or context mismatch for {}",
            case.id
        );
    }
    Ok(inputs)
}

struct Observation {
    ids: Vec<u32>,
    passed: bool,
}

fn generate(
    model: &LoadedModel,
    case: &Case,
    role: &str,
    report: &mut records::Report,
) -> Result<Observation> {
    let start = Instant::now();
    let generated = tasks::generate_with_prefill(
        model,
        &case.prompt,
        case.task.max_tokens,
        PromptPrefill::Sequential,
    )?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (input, ids) = generated.token_ids.split_at(generated.prompt_token_len);
    ensure!(
        input == case.prompt_ids
            && !ids.is_empty()
            && ids.len() == generated.metrics.generated_tokens,
        "generation changed input or token accounting for {}",
        case.id
    );
    let score = case.task.scoring.score(&generated.output)?;
    report.append(json!({
        "kind":"sparse_q4_extended_generation", "case":case.id, "role":role,
        "generation":tasks::generation_record(&generated, elapsed_ms, model)?, "score":score,
        "has_repetition":tasks::has_repetition(ids)
    }))?;
    eprintln!("{} {role} passed={}", case.id, score.passed);
    Ok(Observation {
        ids: ids.to_vec(),
        passed: score.passed,
    })
}

fn first_divergence(reference: &[u32], candidate: &[u32]) -> Option<usize> {
    reference
        .iter()
        .zip(candidate)
        .position(|(a, b)| a != b)
        .or_else(|| {
            (reference.len() != candidate.len()).then_some(reference.len().min(candidate.len()))
        })
}

fn observation_steps(count: usize) -> Vec<usize> {
    let last = count - 1;
    let mut steps = vec![0, 8.min(last), last];
    steps.dedup();
    steps
}

fn observe_logits(
    model: &mut LoadedModel,
    prompt: &[u32],
    continuation: &[u32],
    steps: &[usize],
) -> Result<Vec<Vec<f32>>> {
    model.reset_kv_cache()?;
    let dimension = model.validate_full_layer_decode()?.0;
    let mut session = DecodeSession::new_for_sequence_with_kv_config(
        model,
        0,
        KvCompressionConfig::dense_reference(),
        dimension,
        true,
        "sparse_q4_extended",
        "sparse_q4_extended:input",
        "sparse_q4_extended:output",
    )?;
    let mut ids = prompt.to_vec();
    let mut processed_steps = 0;
    steps
        .iter()
        .map(|&step| {
            ids.extend_from_slice(&continuation[processed_steps..step]);
            processed_steps = step;
            let logits = session.logits_for_ids(&ids, |_| {})?;
            ensure!(
                session.processed_len() == ids.len(),
                "prefix state does not match requested history"
            );
            Ok(logits)
        })
        .collect()
}

fn write_logits(path: &Path, values: &[f32]) -> Result<()> {
    let mut file = BufWriter::new(OpenOptions::new().create_new(true).write(true).open(path)?);
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(file.flush()?)
}

// B2's full-vocabulary KL uses each vector's own stable log partition.
fn normalized_log_probs(values: &[f32]) -> Vec<f64> {
    let max = values
        .iter()
        .map(|&x| f64::from(x))
        .fold(f64::NEG_INFINITY, f64::max);
    let log_sum = values
        .iter()
        .map(|&x| (f64::from(x) - max).exp())
        .sum::<f64>()
        .ln();
    values
        .iter()
        .map(|&x| (f64::from(x) - max) - log_sum)
        .collect()
}

fn compare_logits(reference: &[f32], candidate: &[f32], target: u32) -> Result<Value> {
    ensure!(
        reference.len() == candidate.len(),
        "vocabulary sizes differ"
    );
    let a = score_logits(reference, target)?;
    let b = score_logits(candidate, target)?;
    let (max, sum) = reference
        .iter()
        .zip(candidate)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .fold((0.0_f64, 0.0), |(max, sum), diff| {
            (max.max(diff), sum + diff)
        });
    let kl = normalized_log_probs(reference)
        .into_iter()
        .zip(normalized_log_probs(candidate))
        .map(|(x, y)| x.exp() * (x - y))
        .sum::<f64>();
    ensure!(kl.is_finite() && kl >= -1e-10, "invalid KL divergence");
    let overlap = a
        .top_indices
        .iter()
        .filter(|index| b.top_indices.contains(index))
        .count();
    Ok(json!({"reference":a,"candidate":b,"max_abs_logit_diff":max,
        "mean_abs_logit_diff":sum/reference.len() as f64,"kl_reference_candidate":kl.max(0.0),
        "top10_overlap":overlap}))
}

fn record_logits(
    case: &Case,
    dense_history: &[u32],
    reference: &mut LoadedModel,
    candidate: &mut LoadedModel,
    directory: &Path,
    report: &mut records::Report,
) -> Result<Option<&'static str>> {
    let steps = observation_steps(dense_history.len());
    let dense = observe_logits(reference, &case.prompt_ids, dense_history, &steps)?;
    let sparse = observe_logits(candidate, &case.prompt_ids, dense_history, &steps)?;
    for ((&step, a), b) in steps.iter().zip(dense).zip(sparse) {
        write_logits(
            &directory.join(format!("{}-{step}-dense.f32le", case.id)),
            &a,
        )?;
        write_logits(
            &directory.join(format!("{}-{step}-candidate.f32le", case.id)),
            &b,
        )?;
        if a.iter().any(|v| !v.is_finite()) {
            return Ok(Some("nonfinite_reference"));
        }
        if b.iter().any(|v| !v.is_finite()) {
            return Ok(Some("nonfinite_candidate"));
        }
        report.append(
            json!({"kind":"sparse_q4_extended_logits","case":case.id,"step":step,
            "target":dense_history[step],"prefix_tokens":case.prompt_ids.len()+step,
            "values":compare_logits(&a,&b,dense_history[step])?}),
        )?;
    }
    Ok(None)
}

pub(super) fn evaluate(
    inputs: &Inputs,
    reference: &mut LoadedModel,
    candidate: &mut LoadedModel,
    report: &mut records::Report,
    logits: &Path,
) -> Result<Value> {
    fs::create_dir(logits)?;
    report.append(
        json!({"kind":"sparse_q4_extended_inputs","manifest_sha256":INPUT_SHA,
        "cases":inputs.cases.len(),"input_ids_match_q4_tokenizer":true, "dense_context":3072}),
    )?;
    let mut dense_passes = [0, 0];
    let mut dense_failures = [0, 0];
    let mut matched = 0;
    let mut failure = None;
    for case in &inputs.cases {
        let dense = generate(reference, case, "dense", report)?;
        let context = match case.context {
            ContextKind::Short => 0,
            ContextKind::Long => 1,
        };
        dense_passes[context] += usize::from(dense.passed);
        dense_failures[context] += usize::from(!dense.passed);
        let before = candidate
            .q4_research_stats()?
            .context("missing sparse Q4 layout")?
            .projection_calls;
        let sparse = generate(candidate, case, "candidate", report)?;
        let after = candidate
            .q4_research_stats()?
            .context("missing sparse Q4 layout")?
            .projection_calls;
        ensure!(
            after.into_iter().zip(before).all(|(b, a)| b > a),
            "candidate layer bypassed sparse Q4"
        );
        let divergence = first_divergence(&dense.ids, &sparse.ids);
        let new_repetition =
            tasks::has_repetition(&sparse.ids) && !tasks::has_repetition(&dense.ids);
        report.append(json!({"kind":"sparse_q4_extended_comparison","case":case.id,
            "first_divergence":divergence,"new_repetition":new_repetition,"dense_passed":dense.passed,
            "candidate_passed":sparse.passed,"projection_calls_before":before,"projection_calls_after":after}))?;
        matched += 1;
        let answer_failure = if new_repetition {
            Some("new_repetition")
        } else if dense.passed && !sparse.passed {
            Some("lost_dense_pass")
        } else {
            None
        };
        let logit_failure = record_logits(case, &dense.ids, reference, candidate, logits, report)?;
        if let Some(reason) = answer_failure.or(logit_failure) {
            failure = Some((case.id.as_str(), reason));
            break;
        }
    }
    let disposition = match failure {
        Some((_, "nonfinite_reference")) => "INCONCLUSIVE",
        Some(_) => "FAIL",
        None if dense_passes.contains(&0) => "INCONCLUSIVE",
        None => "PASS",
    };
    Ok(
        json!({"disposition":disposition,"failure":failure,"cases":matched,
        "dense_passes_short_long":dense_passes,"dense_failures_short_long":dense_failures}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Short generations retain distinct observation positions and identify a
    /// length-only difference after the shared continuation.
    #[test]
    fn short_continuations_retain_first_and_last() {
        assert_eq!(observation_steps(1), [0]);
        assert_eq!(observation_steps(3), [0, 2]);
        assert_eq!(observation_steps(17), [0, 8, 16]);
        assert_eq!(first_divergence(&[10, 11], &[10, 11, 12]), Some(2));
        assert_eq!(first_divergence(&[10, 11], &[10, 12]), Some(1));
        assert_eq!(first_divergence(&[10, 11], &[10, 11]), None);
    }

    /// A constant logit shift preserves the distribution despite a nonzero
    /// absolute error. A change in the preferred token changes KL and rank.
    #[test]
    fn matched_logits_keep_loss_and_distribution_separate() -> Result<()> {
        let shifted = compare_logits(&[999.0, 997.0], &[1001.0, 999.0], 0)?;
        assert_eq!(shifted["max_abs_logit_diff"], 2.0);
        assert_eq!(shifted["kl_reference_candidate"], 0.0);
        let changed = compare_logits(&[999.0, 997.0], &[997.0, 999.0], 0)?;
        assert!(changed["kl_reference_candidate"].as_f64().unwrap() > 1.5);
        assert_eq!(changed["candidate"]["target_rank"], 2);
        Ok(())
    }
}
