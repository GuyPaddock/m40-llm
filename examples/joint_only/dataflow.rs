//! Profiling perturbation only; one unchanged joint-only runtime in both arms.
use super::{
    http, protocol, quality,
    run::{self, Args, Report},
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{self, CudaContext},
    decode_session::DecodeSession,
    evaluation::identity::{file_sha256, sha256},
    infer::{LoadedModel, Q8FfnOwnership},
    kv_compression::KvCompressionConfig,
};
use serde_json::json;
use std::{fs, path::Path};

struct ProfileSwitch;
impl ProfileSwitch {
    fn new(path: &Path) -> Result<Self> {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        std::env::set_var("M40LLM_STAGE_PROFILE_REPORT", path);
        std::env::set_var("M40LLM_STAGE_PROFILE_TOKENS", "0,255,511");
        Self::set(false);
        Ok(Self)
    }
    fn set(enabled: bool) {
        std::env::set_var("M40LLM_STAGE_PROFILE", if enabled { "1" } else { "0" });
    }
}
impl Drop for ProfileSwitch {
    fn drop(&mut self) {
        for key in [
            "M40LLM_STAGE_PROFILE",
            "M40LLM_STAGE_PROFILE_REPORT",
            "M40LLM_STAGE_PROFILE_TOKENS",
        ] {
            std::env::remove_var(key);
        }
    }
}

pub(super) fn snapshot(model: &LoadedModel, length: u32) -> Result<Vec<u16>> {
    let kv = model.kv_cache.as_ref().context("dense KV missing")?;
    ensure!(!kv.is_compressed(), "explicit dense required");
    let mut bits = Vec::with_capacity(36 * length as usize * 512);
    for layer in 0..36 {
        ensure!(
            kv.debug_sequence_length(&model.cuda, layer)? == length,
            "KV length differs"
        );
        for position in 0..length {
            let mut k = [0u16; 256];
            let mut v = [0u16; 256];
            let rc = unsafe {
                cuda::ffi_debug_read_kv_token(
                    &model.cuda,
                    kv,
                    layer,
                    position,
                    k.as_mut_ptr().cast(),
                    v.as_mut_ptr().cast(),
                )
            };
            ensure!(rc == 0, "KV read failed: {rc}");
            bits.extend(k);
            bits.extend(v);
        }
    }
    ensure!(
        bits.iter().all(|b| half::f16::from_bits(*b).is_finite()),
        "nonfinite KV"
    );
    Ok(bits)
}

fn replay_logits(model: &LoadedModel, ids: &[u32]) -> Result<Vec<Vec<u32>>> {
    model.reset_kv_cache()?;
    let mut session = DecodeSession::new_for_sequence_with_kv_config(
        model,
        0,
        KvCompressionConfig::dense_reference(),
        2048,
        true,
        "dataflow",
        "dataflow:x",
        "dataflow:out",
    )?;
    session.prefill_packed_prefix_ids(&ids[..65])?;
    let mut vectors = Vec::new();
    for step in 0..512 {
        ensure!(
            session.greedy_token_for_ids(&ids[..65 + step])? == ids[65 + step],
            "replay ID differs"
        );
        if [0, 1, 8, 32, 128, 256, 511].contains(&step) {
            let logits = unsafe { model.logits_from_hidden(session.d_out_ptr()?.cast_const())? };
            ensure!(
                logits.len() == 151936 && logits.iter().all(|x| x.is_finite()),
                "invalid vocabulary"
            );
            vectors.push(logits.into_iter().map(f32::to_bits).collect());
        }
    }
    Ok(vectors)
}

pub(super) async fn execute(args: &Args, report: &mut Report) -> Result<()> {
    let evidence = http::read_quality(args)?;
    let generation = evidence
        .iter()
        .find(|r| r["kind"] == "joint_only_generation")
        .context("generation")?;
    let expected = generation["output"].as_str().context("expected output")?;
    let path = args
        .profile_report
        .as_ref()
        .context("--profile-report required")?;
    let _switch = ProfileSwitch::new(path)?;
    let initial = CudaContext::total_device_bytes();
    let server = http::start(run::load(&args.model, Q8FfnOwnership::JointOnly)?).await?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;
    report.record(json!({"kind":"dataflow_binding","quality_sha256":file_sha256(args.quality_report.as_ref().unwrap())?,
        "model":run::identity(&server.state.model),"profile_report":path,"steps":[0,255,511]}))?;

    // Untimed off/on checks precede every performance decision.
    let a = quality::generate(&server.state.model, &protocol::render_prompt(), 512)?;
    let kv_a = snapshot(&server.state.model, 576)?;
    let logits_a = replay_logits(&server.state.model, &a.token_ids)?;
    ProfileSwitch::set(true);
    let b = quality::generate(&server.state.model, &protocol::render_prompt(), 512)?;
    let kv_b = snapshot(&server.state.model, 576)?;
    let logits_b = replay_logits(&server.state.model, &b.token_ids)?;
    ensure!(
        a.token_ids == b.token_ids
            && a.output == expected
            && b.output == expected
            && serde_json::to_value(&a.token_ids[65..])? == generation["generated_ids"]
            && a.metrics.stop_reason == b.metrics.stop_reason
            && a.metrics.generated_tokens == 512
            && b.metrics.generated_tokens == 512
            && kv_a == kv_b
            && logits_a == logits_b,
        "profiling changes actual tokens/logits/KV"
    );
    report.record(json!({"kind":"dataflow_correctness","actual_ids":&b.token_ids[65..],
        "stop_reason":b.metrics.stop_reason,"valid_kv_words":kv_a.len(),"is_bitwise_equal":true,
        "logit_hashes":logits_a.iter().map(|v| sha256(&v.iter().flat_map(|b|b.to_le_bytes()).collect::<Vec<_>>())).collect::<Result<Vec<_>>>()?,
        "profile_bytes_after_preflight":fs::metadata(path)?.len()}))?;
    drop((kv_a, kv_b, logits_a, logits_b));
    for enabled in [false, true] {
        ProfileSwitch::set(enabled);
        http::request(&client, &server, expected).await?;
    }
    let mut pairs = Vec::new();
    let mut phase = "screen";
    let mut target = 5;
    loop {
        for pair in pairs.len()..target {
            let mut rates = [0.; 2];
            for arm in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                ProfileSwitch::set(arm == 1);
                let before = run::gpu()?;
                http::validate_timing_gpu(&before)?;
                let event_begin = fs::metadata(path)?.len();
                let row = http::request(&client, &server, expected).await?;
                let event_end = fs::metadata(path)?.len();
                let after = run::gpu()?;
                http::validate_timing_gpu(&after)?;
                ensure!(
                    row["dispatch"] == generation["candidate_dispatch"],
                    "ordinary dispatch changed"
                );
                ensure!(
                    (event_end > event_begin) == (arm == 1),
                    "unexpected profile emission"
                );
                rates[arm] = row["http_tok_per_sec"].as_f64().context("HTTP rate")?;
                report.record(json!({"kind":"dataflow_http_trial","phase":phase,"pair":pair,"arm":arm,
                    "row":row,"gpu_before":before,"gpu_after":after,"event_begin":event_begin,"event_end":event_end}))?;
            }
            pairs.push(protocol::Pair {
                control_tps: rates[0],
                candidate_tps: rates[1],
            });
        }
        let comparison = protocol::compare_pairs(&pairs)?;
        let reverse: Vec<_> = pairs
            .iter()
            .map(|p| protocol::Pair {
                control_tps: p.candidate_tps,
                candidate_tps: p.control_tps,
            })
            .collect();
        let reverse = protocol::compare_pairs(&reverse)?;
        let latency_change = comparison.control.median / comparison.candidate.median - 1.;
        report.record(
            json!({"kind":"dataflow_http_summary","phase":phase,"comparison":comparison,
            "latency_change":latency_change,"is_attribution_allowed":latency_change.abs()<=0.03}),
        )?;
        ensure!(
            latency_change.abs() <= 0.03,
            "profiling perturbation exceeds 3%; stop attribution"
        );
        if phase == "screen" && comparison.needs_variation_extension && target == 5 {
            target = 10;
            continue;
        }
        ensure!(
            !comparison.needs_variation_extension,
            "unstable paired timing"
        );
        if phase == "screen"
            && (comparison.needs_fresh_confirmation || reverse.needs_fresh_confirmation)
        {
            pairs.clear();
            phase = "confirmation";
            target = 20;
            continue;
        }
        break;
    }
    ProfileSwitch::set(false);
    drop(client);
    let _ = server.shutdown.send(());
    tokio::time::timeout(std::time::Duration::from_secs(5), server.task).await???;
    drop(server.state);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "dataflow teardown leaked"
    );
    report.record(
        json!({"kind":"dataflow_complete","device_bytes_after":CudaContext::total_device_bytes(),
        "is_attribution_allowed":true,"is_speed_promotion":false}),
    )
}
