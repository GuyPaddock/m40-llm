use super::{expected_dispatch, load, read_rows, ARMS, FROZEN, ORIGINAL_QUALITY};
use crate::{
    dataflow, protocol, quality,
    run::{Args, Report},
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    decode_session::DecodeSession,
    evaluation::{
        corpus::Corpus,
        identity::{file_sha256, sha256},
    },
    infer::{execution_weights::Q8PortfolioArm, LoadedModel},
    kv_compression::KvCompressionConfig,
    profile,
};
use serde_json::json;
use std::{fs, path::Path};

const STEPS: [usize; 7] = [0, 1, 8, 32, 128, 256, 511];

struct Baseline {
    logits: Vec<Vec<u32>>,
    kv: Vec<u16>,
    task_ids: Vec<Vec<u32>>,
}

fn replay(model: &LoadedModel, ids: &[u32]) -> Result<Vec<Vec<u32>>> {
    model.reset_kv_cache()?;
    let mut session = DecodeSession::new_for_sequence_with_kv_config(
        model,
        0,
        KvCompressionConfig::dense_reference(),
        2048,
        true,
        "portfolio-quality",
        "portfolio-quality:x",
        "portfolio-quality:out",
    )?;
    session.prefill_packed_prefix_ids(&ids[..65])?;
    let mut vectors = Vec::new();
    for step in 0..512 {
        let before = profile::snapshot();
        ensure!(
            session.greedy_token_for_ids(&ids[..65 + step])? == ids[65 + step],
            "replay differs at {step}"
        );
        if step > 0 {
            let delta = profile::counter_delta(&before, &profile::snapshot());
            ensure!(
                delta.device_allocs == 0 && delta.device_frees == 0,
                "warm token allocates at {step}"
            );
        }
        if STEPS.contains(&step) {
            // Uses the model's selected head, not a direct original-consumer call.
            let logits = unsafe { model.logits_from_hidden(session.d_out_ptr()?.cast_const())? };
            ensure!(
                logits.len() == 151936 && logits.iter().all(|v| v.is_finite()),
                "invalid vocabulary"
            );
            vectors.push(logits.into_iter().map(f32::to_bits).collect());
        }
    }
    Ok(vectors)
}

pub(crate) fn execute(args: &Args, report: &mut Report) -> Result<()> {
    let frozen = read_rows(Path::new(FROZEN))?;
    let original = read_rows(Path::new(ORIGINAL_QUALITY))?;
    let expected = frozen
        .iter()
        .find(|r| r["kind"] == "q8_runtime_browser_generation" && r["label"] == "candidate")
        .context("frozen generation")?;
    let original_generation = original
        .iter()
        .find(|r| r["kind"] == "joint_only_generation")
        .context("original generation")?;
    let original_owners = original
        .iter()
        .find(|r| r["kind"] == "joint_only_ownership")
        .context("original ownership")?;
    let corpus = Corpus::parse(&fs::read_to_string("data/research_quality/v1.json")?)?;
    report.record(
        json!({"kind":"portfolio_quality_binding","frozen_sha256":file_sha256(Path::new(FROZEN))?,
        "original_quality_sha256":file_sha256(Path::new(ORIGINAL_QUALITY))?,
        "corpus_sha256":file_sha256(Path::new("data/research_quality/v1.json"))?,"arms":ARMS}),
    )?;
    let initial = CudaContext::total_device_bytes();
    let mut baseline: Option<Baseline> = None;
    let mut baseline_packets = None;
    for arm in ARMS {
        let model = load(&args.model, arm, report)?;
        let stats = model
            .q8_execution
            .portfolio_stats()
            .context("packet stats")?;
        let packets = serde_json::to_value(&stats.tensors)?;
        if let Some(expected) = &baseline_packets {
            ensure!(*expected == packets, "different arm packet identities");
        } else {
            baseline_packets = Some(packets);
        }
        let joint_hashes = model.q8_execution.debug_joint_payload_hashes()?;
        ensure!(
            joint_hashes.len() == 36
                && serde_json::to_value(&joint_hashes)?
                    == original_owners["candidate_payload_hashes"],
            "joint payloads differ from original qualified owners"
        );
        quality::generate(&model, &protocol::render_user_prompt("Say ready."), 2)?;
        let before = profile::snapshot();
        let generated = quality::generate(&model, &protocol::render_prompt(), 512)?;
        let dispatch = quality::dispatch(&before);
        report.record(json!({"kind":"portfolio_generation","arm":arm,"generated_ids":&generated.token_ids[generated.prompt_token_len..],
            "prompt_ids":&generated.token_ids[..generated.prompt_token_len],"output":generated.output,
            "metrics":generated.metrics,"dispatch":dispatch,"prefill_mode":generated.prefill_mode,
            "prefix_bytes":generated.q8_prefill_temporary_weight_bytes}))?;
        ensure!(
            generated.prompt_token_len == 65
                && generated.metrics.generated_tokens == 512
                && generated.metrics.stop_reason == m40_llm::decode::StopReason::Length
                && serde_json::to_value(&generated.token_ids[65..])?
                    == expected["generation"]["generated_token_ids"]
                && serde_json::to_value(&generated.token_ids[..65])?
                    == expected["prompt_token_ids"]
                && generated.output == expected["generation"]["output"],
            "actual workload differs for {arm:?}"
        );
        ensure!(
            dispatch == expected_dispatch(&original_generation["control_dispatch"], arm)?,
            "unapproved dispatch delta for {arm:?}"
        );
        ensure!(
            generated.prefill_mode == original_generation["control_prefill_mode"]
                && serde_json::to_value(generated.q8_prefill_temporary_weight_bytes)?
                    == original_generation["control_prefix_bytes"],
            "prefill changed"
        );
        let kv = dataflow::snapshot(&model, 576)?;
        let logits = replay(&model, &generated.token_ids)?;
        ensure!(dataflow::snapshot(&model, 576)? == kv, "replay changed KV");
        for (index, vector) in logits.iter().enumerate() {
            let hash = sha256(
                &vector
                    .iter()
                    .flat_map(|b| b.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let prior = original
                .iter()
                .find(|r| r["kind"] == "joint_only_logits" && r["step"] == STEPS[index])
                .context("frozen logits")?;
            let is_equal = hash == prior["control_sha256"]
                && baseline.as_ref().is_none_or(|b| b.logits[index] == *vector);
            report.record(json!({"kind":"portfolio_logits","arm":arm,"step":STEPS[index],"sha256":hash,
                "original_sha256":prior["control_sha256"],"is_bitwise_equal":is_equal,"vocabulary":vector.len()}))?;
            ensure!(
                is_equal,
                "full vocabulary differs for {arm:?} step {}",
                STEPS[index]
            );
        }
        let is_kv_equal = baseline.as_ref().is_none_or(|b| b.kv == kv);
        report.record(
            json!({"kind":"portfolio_kv","arm":arm,"valid_words":kv.len(),"length":576,
            "sha256":sha256(&kv.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?,
            "is_bitwise_equal":is_kv_equal,"control_is_reference":arm==Q8PortfolioArm::Control}),
        )?;
        ensure!(is_kv_equal, "valid KV differs for {arm:?}");
        let mut task_ids = Vec::new();
        let mut dense_valid = 0;
        for (index, task) in corpus
            .calibration
            .tasks
            .iter()
            .chain(&corpus.held_out.tasks)
            .enumerate()
        {
            let result = quality::generate(
                &model,
                &protocol::render_user_prompt(&task.prompt),
                task.max_tokens,
            )?;
            let score = task.scoring.score(&result.output)?;
            let prior = frozen
                .iter()
                .find(|r| r["kind"] == "model_quality_task" && r["id"] == task.id)
                .context("frozen task")?;
            let is_equal = result.output == prior["candidate"]["output"]
                && serde_json::to_value(&score)? == prior["candidate_score"]
                && baseline
                    .as_ref()
                    .is_none_or(|b| b.task_ids[index] == result.token_ids);
            report.record(
                json!({"kind":"portfolio_task","arm":arm,"id":task.id,"output":result.output,
                "ids":result.token_ids,"metrics":result.metrics,"score":score,"is_equal":is_equal}),
            )?;
            ensure!(is_equal, "task changed: {arm:?} {}", task.id);
            dense_valid += usize::from(score.passed);
            task_ids.push(result.token_ids);
        }
        ensure!(
            task_ids.len() == 12 && dense_valid == 7,
            "frozen coverage changed"
        );
        model.q8_execution.verify_portfolio_payloads()?;
        ensure!(
            joint_hashes == model.q8_execution.debug_joint_payload_hashes()?,
            "joint payload changed"
        );
        if baseline.is_none() {
            baseline = Some(Baseline {
                logits,
                kv,
                task_ids,
            });
        }
        drop(model);
        ensure!(
            CudaContext::total_device_bytes() == initial,
            "arm leaked allocations"
        );
        report.record(json!({"kind":"portfolio_quality_arm_complete","arm":arm,"is_gate_passed":true,
            "tasks":12,"dense_valid":7,"inconclusive":5,"device_bytes_after":CudaContext::total_device_bytes()}))?;
    }
    report.record(
        json!({"kind":"portfolio_quality_complete","is_gate_passed":true,"arms":ARMS,
        "device_bytes_after":CudaContext::total_device_bytes()}),
    )
}
