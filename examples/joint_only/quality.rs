use super::{
    protocol,
    run::{self, Args, Report},
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    decode_session::DecodeSession,
    evaluation::{corpus::Corpus, identity::sha256},
    generate::{generate_text, GenerateOptions, GeneratedText, PromptFormat, PromptPrefill},
    infer::{LoadedModel, Q8FfnOwnership},
    kv_compression::KvCompressionConfig,
    profile,
};
use serde_json::{json, Value};
use std::{fs, time::Instant};

pub fn generate(model: &LoadedModel, prompt: &str, max_tokens: usize) -> Result<GeneratedText> {
    generate_text(
        model,
        GenerateOptions {
            prompt: prompt.into(),
            prompt_format: PromptFormat::Raw,
            top_k: Some(1),
            seed: Some(0),
            max_tokens: Some(max_tokens),
            reset_kv_cache: true,
            kv_compression: KvCompressionConfig::dense_reference(),
            prompt_prefill: PromptPrefill::PackedPrefix {
                max_prompt_tokens: 128,
            },
            ..Default::default()
        },
    )
}

fn dispatch(before: &profile::ProfileSnapshot) -> Value {
    let after = profile::snapshot();
    let operations: std::collections::BTreeMap<_, _> = after
        .by_op
        .iter()
        .filter_map(|(name, c)| {
            let previous = before.by_op.get(name).copied().unwrap_or_default();
            let launches = c.launches - previous.launches;
            let cublas = c.cublas_calls - previous.cublas_calls;
            (launches > 0 || cublas > 0)
                .then_some((*name, json!({"launches":launches,"cublas_calls":cublas})))
        })
        .collect();
    json!({"totals":profile::counter_delta(before,&after),"operations":operations})
}

pub fn execute(args: &Args, report: &mut Report) -> Result<()> {
    let frozen: Vec<Value> =
        fs::read_to_string("docs/experiments/2026-10-27-maxwell-compound-runtime-quality.jsonl")?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
    let expected = frozen
        .iter()
        .find(|r| r["kind"] == "q8_runtime_browser_generation" && r["label"] == "candidate")
        .context("frozen generation")?;
    let initial = CudaContext::total_device_bytes();
    let before = profile::snapshot();
    let start = Instant::now();
    let a = run::load(&args.model, Q8FfnOwnership::RetainSplit)?;
    let load_a = start.elapsed().as_secs_f64() * 1000.;
    let allocations_a = profile::counter_delta(&before, &profile::snapshot()).device_allocs;
    let before = profile::snapshot();
    let start = Instant::now();
    let b = run::load(&args.model, Q8FfnOwnership::JointOnly)?;
    let load_b = start.elapsed().as_secs_f64() * 1000.;
    let allocations_b = profile::counter_delta(&before, &profile::snapshot()).device_allocs;
    let sa = a.q8_execution_stats();
    let sb = b.q8_execution_stats();
    ensure!(
        a.q8_execution.logical_sources() == b.q8_execution.logical_sources()
            && a.q8_execution.logical_sources().len() == 109,
        "logical source coverage differs"
    );
    ensure!(
        sa.tensors.len() == 109 && sb.tensors.len() == 37,
        "unexpected physical owners"
    );
    ensure!(
        sa.resident_bytes == 2917580800
            && sb.resident_bytes == 1192935424
            && sa.resident_bytes - sb.resident_bytes == 1724645376,
        "resident reduction differs"
    );
    ensure!(
        allocations_a - allocations_b == 144,
        "redundant split allocations occurred"
    );
    let hashes_a = a.q8_execution.debug_joint_payload_hashes()?;
    let hashes_b = b.q8_execution.debug_joint_payload_hashes()?;
    ensure!(
        hashes_a == hashes_b && hashes_a.len() == 36,
        "joint payload bytes differ"
    );
    let candidate_keys: Vec<_> = sb.tensors.iter().map(|t| &t.key).collect();
    for tensor in &sa.tensors {
        let is_joint = hashes_a
            .iter()
            .any(|(key, _)| tensor.key.source == key.gate || tensor.key.source == key.up);
        ensure!(
            candidate_keys.contains(&&tensor.key) != is_joint,
            "wrong split owner retained/removed"
        );
    }
    for layer in 0..36 {
        let mapped = b.map_standard_layer(layer)?;
        ensure!(
            b.q8_execution
                .get_separate_weight(mapped.w_gate.dptr)
                .is_none()
                && b.q8_execution
                    .get_separate_weight(mapped.w_up.dptr)
                    .is_none()
                && b.q8_execution
                    .get_separate_weight(mapped.w_down.dptr)
                    .is_some(),
            "wrong physical representation"
        );
    }
    report.record(json!({"kind":"joint_only_ownership","schema_version":1,"control":run::identity(&a),
        "candidate":run::identity(&b),"control_payload_hashes":hashes_a,"candidate_payload_hashes":hashes_b,
        "resident_reduction_bytes":sa.resident_bytes-sb.resident_bytes,"skipped_device_allocations":allocations_a-allocations_b,
        "control_load_ms":load_a,"candidate_load_ms":load_b}))?;
    // Match the qualified runner's setup boundary before comparing ordinary
    // request allocation/free counts against its frozen dispatch evidence.
    for model in [&a, &b] {
        generate(model, &protocol::render_user_prompt("Say ready."), 2)?;
    }
    let prompt = protocol::render_prompt();
    let before = profile::snapshot();
    let ga = generate(&a, &prompt, 512)?;
    let da = dispatch(&before);
    let before = profile::snapshot();
    let gb = generate(&b, &prompt, 512)?;
    let db = dispatch(&before);
    ensure!(
        ga.token_ids == gb.token_ids && ga.output == gb.output,
        "actual generated IDs/output differ"
    );
    ensure!(
        ga.prompt_token_len == 65
            && ga.metrics.generated_tokens == 512
            && ga.metrics.stop_reason == m40_llm::decode::StopReason::Length
            && gb.metrics.generated_tokens == 512
            && gb.metrics.stop_reason == m40_llm::decode::StopReason::Length,
        "wrong fixed workload"
    );
    ensure!(
        serde_json::to_value(&ga.token_ids[ga.prompt_token_len..])?
            == expected["generation"]["generated_token_ids"]
            && ga.output == expected["generation"]["output"],
        "generation changed versus immutable reference"
    );
    ensure!(da == db, "ordinary dispatch/counters differ");
    report.record(json!({"kind":"joint_only_generation","generated_ids":&ga.token_ids[ga.prompt_token_len..],
        "prompt_ids":&ga.token_ids[..ga.prompt_token_len],"output":ga.output,"candidate_output":gb.output,
        "control_metrics":ga.metrics,"candidate_metrics":gb.metrics,"control_dispatch":da,"candidate_dispatch":db,
        "control_prefill_mode":ga.prefill_mode,"candidate_prefill_mode":gb.prefill_mode,
        "control_prefix_bytes":ga.q8_prefill_temporary_weight_bytes,"candidate_prefix_bytes":gb.q8_prefill_temporary_weight_bytes}))?;
    a.reset_kv_cache()?;
    b.reset_kv_cache()?;
    {
        let mut x = DecodeSession::new_for_sequence_with_kv_config(
            &a,
            0,
            KvCompressionConfig::dense_reference(),
            2048,
            true,
            "joint-quality:a",
            "joint-quality:a-x",
            "joint-quality:a-out",
        )?;
        let mut y = DecodeSession::new_for_sequence_with_kv_config(
            &b,
            0,
            KvCompressionConfig::dense_reference(),
            2048,
            true,
            "joint-quality:b",
            "joint-quality:b-x",
            "joint-quality:b-out",
        )?;
        x.prefill_packed_prefix_ids(&ga.token_ids[..ga.prompt_token_len])?;
        y.prefill_packed_prefix_ids(&ga.token_ids[..ga.prompt_token_len])?;
        for step in 0..512 {
            let prefix = &ga.token_ids[..ga.prompt_token_len + step];
            let ta = x.greedy_token_for_ids(prefix)?;
            let tb = y.greedy_token_for_ids(prefix)?;
            ensure!(
                ta == tb && ta == ga.token_ids[ga.prompt_token_len + step],
                "greedy replay differs at {step}"
            );
            if ![0, 1, 8, 32, 128, 256, 511].contains(&step) {
                continue;
            }
            let la = unsafe { a.logits_from_hidden(x.d_out_ptr()?.cast_const())? };
            let lb = unsafe { b.logits_from_hidden(y.d_out_ptr()?.cast_const())? };
            ensure!(
                la.len() == 151936 && la.iter().zip(&lb).all(|(a, b)| a.to_bits() == b.to_bits()),
                "full-vocabulary bits differ at {step}"
            );
            let bytes: Vec<u8> = la.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
            report.record(json!({"kind":"joint_only_logits","step":step,"vocabulary":la.len(),
                "control_sha256":sha256(&bytes)?,"candidate_sha256":sha256(&lb.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect::<Vec<_>>())?,
                "max_abs_diff":0,"mean_abs_diff":0,"top10_overlap":10}))?;
        }
    }
    let corpus = Corpus::parse(&fs::read_to_string("data/research_quality/v1.json")?)?;
    let mut count = 0;
    let mut dense_valid = 0;
    for task in corpus
        .calibration
        .tasks
        .iter()
        .chain(&corpus.held_out.tasks)
    {
        let task_prompt = protocol::render_user_prompt(&task.prompt);
        let x = generate(&a, &task_prompt, task.max_tokens)?;
        let y = generate(&b, &task_prompt, task.max_tokens)?;
        let prior = frozen
            .iter()
            .find(|r| r["kind"] == "model_quality_task" && r["id"] == task.id)
            .context("frozen task")?;
        ensure!(
            x.output == y.output
                && x.token_ids == y.token_ids
                && x.output == prior["candidate"]["output"],
            "task changed: {}",
            task.id
        );
        let score = task.scoring.score(&x.output)?;
        ensure!(
            serde_json::to_value(&score)? == prior["candidate_score"],
            "score changed: {}",
            task.id
        );
        count += 1;
        dense_valid += usize::from(score.passed);
        report.record(json!({"kind":"joint_only_task","id":task.id,"output":x.output,"candidate_output":y.output,
            "score":score,"has_identical_ids":x.token_ids==y.token_ids,"control_metrics":x.metrics,"candidate_metrics":y.metrics}))?;
    }
    ensure!(
        count == 12 && dense_valid == 7,
        "frozen task coverage changed"
    );
    drop(a);
    drop(b);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "quality teardown leaked"
    );
    report.record(json!({"kind":"joint_only_quality_complete","is_gate_passed":true,"tasks":count,
        "dense_valid":dense_valid,"inconclusive":count-dense_valid,"device_bytes_after":CudaContext::total_device_bytes()}))
}
