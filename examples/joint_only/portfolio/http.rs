use super::{load, read_rows, ARMS};
use crate::{
    http, protocol, quality,
    run::{self, Args, Report},
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext, evaluation::identity::file_sha256, infer::execution_weights::Q8PortfolioArm,
};
use serde_json::{json, Value};

fn validate_evidence(rows: &[Value], revision: &str, executable: &str) -> Result<()> {
    ensure!(
        rows.first()
            .is_some_and(|r| r["revision"] == revision && r["executable_sha256"] == executable),
        "quality source/executable differs"
    );
    ensure!(
        rows.last().is_some_and(
            |r| r["kind"] == "portfolio_quality_complete" && r["is_gate_passed"] == true
        ),
        "incomplete quality evidence"
    );
    for arm in ARMS {
        let name = serde_json::to_value(arm)?;
        ensure!(
            rows.iter()
                .filter(|r| r["kind"] == "portfolio_quality_arm_complete"
                    && r["arm"] == name
                    && r["is_gate_passed"] == true
                    && r["tasks"] == 12
                    && r["dense_valid"] == 7
                    && r["device_bytes_after"] == 0)
                .count()
                == 1,
            "missing or duplicated arm gate for {arm:?}"
        );
        ensure!(
            rows.iter()
                .filter(|r| r["kind"] == "portfolio_generation" && r["arm"] == name)
                .count()
                == 1,
            "missing or duplicate arm output"
        );
    }
    Ok(())
}

pub(crate) async fn execute(args: &Args, report: &mut Report) -> Result<()> {
    let path = args
        .quality_report
        .as_ref()
        .context("--quality-report required")?;
    let rows = read_rows(path)?;
    validate_evidence(
        &rows,
        &run::command("git", &["rev-parse", "HEAD"])?,
        &file_sha256(&std::env::current_exe()?)?,
    )?;
    report.record(json!({"kind":"portfolio_http_binding","quality_sha256":file_sha256(path)?,
        "scope":"four contemporaneous comparisons; never sum standalone gains", "historical_ollama_tps":43.194730}))?;
    let initial = CudaContext::total_device_bytes();
    let mut has_combined_passed = false;
    for candidate in &ARMS[1..] {
        let controls: Vec<_> = [Q8PortfolioArm::Control, *candidate]
            .into_iter()
            .map(|arm| {
                rows.iter()
                    .find(|r| {
                        r["kind"] == "portfolio_generation"
                            && r["arm"] == serde_json::to_value(arm).unwrap()
                    })
                    .context("arm generation")
            })
            .collect::<Result<_>>()?;
        let expected = controls[0]["output"].as_str().context("expected output")?;
        ensure!(controls[1]["output"] == expected, "quality outputs differ");
        let a = http::start(load(&args.model, Q8PortfolioArm::Control, report)?).await?;
        let b = http::start(load(&args.model, *candidate, report)?).await?;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?;
        for server in [&a, &b] {
            http::request(&client, server, expected).await?;
        }
        let mut pairs = Vec::new();
        let mut is_confirmation = false;
        let mut target = 5;
        loop {
            for pair in pairs.len()..target {
                let mut rates = [0.; 2];
                for index in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let before = run::gpu()?;
                    http::validate_timing_gpu(&before)?;
                    let row =
                        http::request(&client, if index == 0 { &a } else { &b }, expected).await?;
                    let after = run::gpu()?;
                    http::validate_timing_gpu(&after)?;
                    ensure!(
                        row["dispatch"] == controls[index]["dispatch"],
                        "HTTP dispatch differs for {candidate:?} arm {index}"
                    );
                    rates[index] = row["http_tok_per_sec"].as_f64().context("HTTP rate")?;
                    report.record(json!({"kind":"portfolio_http_trial","candidate":candidate,
                        "phase":if is_confirmation { "confirmation" } else { "screen" },
                        "pair":pair,"arm":index,"row":row,"gpu_before":before,"gpu_after":after}))?;
                }
                pairs.push(protocol::Pair {
                    control_tps: rates[0],
                    candidate_tps: rates[1],
                });
            }
            let comparison = protocol::compare_pairs(&pairs)?;
            let next = protocol::next_trial_count(&comparison, is_confirmation)?;
            let mut summary = serde_json::to_value(&comparison)?;
            // The generic comparator assumes its control is Ollama. Here the
            // control is M40-LLM, so that target field would mislabel the result.
            summary
                .as_object_mut()
                .unwrap()
                .remove("reaches_original_target");
            report.record(json!({"kind":"portfolio_http_summary","candidate":candidate,
                "phase":if is_confirmation { "confirmation" } else { "screen" },"comparison":summary,
                "next_pairs":next,"reaches_historical_http_target":comparison.candidate.median >= 56.153149,
                "requires_combined_gain_only":true}))?;
            if let Some(count) = next {
                target = count;
                if count == 20 {
                    pairs.clear();
                    is_confirmation = true;
                }
                continue;
            }
            if *candidate == Q8PortfolioArm::All {
                has_combined_passed = comparison.passes_promotion_gate;
            }
            break;
        }
        for (index, server) in [&a, &b].into_iter().enumerate() {
            let replay = quality::generate(&server.state.model, &protocol::render_prompt(), 512)?;
            ensure!(
                serde_json::to_value(&replay.token_ids[65..])? == controls[index]["generated_ids"],
                "post-HTTP actual IDs differ"
            );
            server
                .state
                .model
                .q8_execution
                .verify_portfolio_payloads()?;
        }
        drop(client);
        let _ = a.shutdown.send(());
        let _ = b.shutdown.send(());
        tokio::time::timeout(std::time::Duration::from_secs(5), a.task).await???;
        tokio::time::timeout(std::time::Duration::from_secs(5), b.task).await???;
        drop(a.state);
        drop(b.state);
        ensure!(
            CudaContext::total_device_bytes() == initial,
            "HTTP teardown leaked"
        );
        report.record(json!({"kind":"portfolio_http_arm_complete","candidate":candidate,"device_bytes_after":CudaContext::total_device_bytes()}))?;
    }
    report.record(
        json!({"kind":"portfolio_http_complete","is_gate_passed":has_combined_passed,
        "device_bytes_after":CudaContext::total_device_bytes(),"runtime_defaults_unchanged":true}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HTTP requires completed, unique quality gates for every fixed arm from
    /// this exact executable, rather than an unrelated or partial success row.
    #[test]
    fn test_quality_binding_rejects_missing_or_foreign_arms() -> Result<()> {
        let mut rows = vec![json!({"revision":"source","executable_sha256":"binary"})];
        for arm in ARMS {
            rows.push(json!({"kind":"portfolio_generation","arm":arm}));
            rows.push(
                json!({"kind":"portfolio_quality_arm_complete","arm":arm,"is_gate_passed":true,
                "tasks":12,"dense_valid":7,"device_bytes_after":0}),
            );
        }
        rows.push(json!({"kind":"portfolio_quality_complete","is_gate_passed":true}));
        validate_evidence(&rows, "source", "binary")?;
        assert!(validate_evidence(&rows, "different", "binary").is_err());
        assert!(validate_evidence(&rows, "source", "different").is_err());
        let mut missing = rows.clone();
        missing.remove(2);
        assert!(validate_evidence(&missing, "source", "binary").is_err());
        let mut duplicated = rows.clone();
        duplicated.insert(2, rows[2].clone());
        assert!(validate_evidence(&duplicated, "source", "binary").is_err());
        rows.pop();
        assert!(validate_evidence(&rows, "source", "binary").is_err());
        Ok(())
    }
}
