use super::{
    gpu_snapshot, protocol, start_server, start_server_with_prefill, ComparisonRun,
    GenerateResponse, LoadedModel, Pair, ServerVariant, CONTEXT_TOKENS,
};
use anyhow::{ensure, Context, Result};
use m40_llm::{cuda::CudaContext, gguf};
use serde_json::json;
use std::time::Instant;

fn peak_host_rss_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
}

impl ComparisonRun {
    async fn load_prepared_server(&mut self) -> Result<()> {
        let runtime = self.args.candidate_runtime_config();
        let before = CudaContext::total_device_bytes();
        let started = Instant::now();
        let model = if let Some(config) = runtime {
            config.load_model(&self.args.model, CONTEXT_TOKENS)?.0
        } else {
            let config = self
                .args
                .candidate_q8_config()?
                .context("missing Q8 candidate config")?;
            let gguf = gguf::load_gguf(&self.args.model)?;
            let bytes = std::fs::read(&self.args.model)?;
            let mut model = LoadedModel::from_gguf_with_q8_execution(gguf, bytes, -1, config)?;
            model.allocate_kv_cache_for_layers(CONTEXT_TOKENS)?;
            model
        };
        ensure!(
            model.cuda.current_device_props()?.device_id
                == self
                    .server
                    .state
                    .model
                    .cuda
                    .current_device_props()?
                    .device_id,
            "control and candidate must use the same GPU"
        );
        let stats = model.q8_execution_stats();
        if let Some(config) = runtime {
            let control = self.server.state.model.q8_execution_stats();
            super::super::q8_runtime::validate_prepared_control(
                &control,
                &stats,
                self.args.control_runtime_config(),
                config,
            )?;
        }
        let source_hash: String = stats
            .tensors
            .first()
            .context("empty prepared cache")?
            .key
            .source
            .model_sha256()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        ensure!(
            source_hash == self.model_sha256,
            "candidate source bytes differ from pinned control GGUF"
        );
        self.report.record(json!({
            "kind": "q8_execution_setup", "is_timed": false,
            "control_execution": self.server.state.model.q8_execution_stats(),
            "candidate_execution": stats, "candidate_cold_load_elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
            "candidate_runtime_config":runtime,
            "control_runtime_config":self.args.control_runtime_config(),
            "tracked_device_payload_before_candidate_bytes": before,
            "tracked_device_payload_after_candidate_bytes": CudaContext::total_device_bytes(),
            "combined_process_peak_host_rss_kib": peak_host_rss_kib(),
            "candidate_final_kv_bytes": model.kv_cache.as_ref().map(|kv| kv.actual_bytes()),
            "gpu": gpu_snapshot(self.args.ollama_pid, &self.gpu_uuid)?,
            "memory_scope": "tracked requested payload for both m40 models; GPU snapshot also includes idle Ollama; not measured peak VRAM",
        }))?;
        self.prepared_server = Some(if let Some(config) = runtime {
            start_server_with_prefill(model, config.prompt_prefill()).await?
        } else {
            start_server(model).await?
        });
        Ok(())
    }

    async fn execution_smokes(&mut self) -> Result<bool> {
        let mut dense_valid = 0;
        let mut regressions = 0;
        for (name, prompt, expected) in [
            ("arithmetic", "What is 2+2? Answer with one digit.", "4"),
            ("exact_ok", "Reply with exactly the word OK.", "OK"),
            ("config_lookup", "Configuration note: the active value is M40_BATCH_LIMIT=37. What is M40_BATCH_LIMIT? Answer with the number only.", "37"),
        ] {
            let rendered_prompt = protocol::render_user_prompt(prompt);
            let mut control_pass = false;
            for variant in [ServerVariant::Control, ServerVariant::Prepared] {
                let server = match variant {
                    ServerVariant::Control => &self.server,
                    ServerVariant::Prepared => self.prepared_server.as_ref().context("prepared server missing")?,
                };
                let response: GenerateResponse = self.client.post(&server.url)
                    .json(&json!({"prompt":rendered_prompt,"prompt_format":"raw","max_tokens":16,"top_k":1,"seed":0,"include_metrics":true}))
                    .send().await?.error_for_status()?.json().await?;
                let passed = response.output.trim() == expected;
                match variant {
                    ServerVariant::Control => { control_pass = passed; dense_valid += usize::from(passed); }
                    ServerVariant::Prepared => { regressions += usize::from(control_pass && !passed); }
                }
                self.report.record(json!({"kind":"q8_quality_smoke","is_timed":false,
                    "case":name,"model_variant":variant.as_str(),"prompt":rendered_prompt,"user_prompt":prompt,"prompt_format":"raw","expected":expected,
                    "pass":passed,"output":response.output,"metrics":response.metrics,
                    "dense_reference_passed":control_pass,
                    "quality_conclusion":if control_pass {if passed {"pass"} else {"regression"}} else {"inconclusive"}}))?;
            }
        }
        self.report.record(json!({"kind":"q8_quality_smoke_summary","dense_valid_cases":dense_valid,"regressions":regressions}))?;
        Ok(dense_valid > 0 && regressions == 0)
    }

    async fn run_execution_pairs(
        &mut self,
        phase: &str,
        start: usize,
        end: usize,
        expected_outputs: &(String, String),
        pairs: &mut Vec<Pair>,
    ) -> Result<()> {
        for pair in start..end {
            let (control, candidate) = if pair.is_multiple_of(2) {
                let control = self
                    .measure_m40_variant(phase, pair, ServerVariant::Control)
                    .await?;
                let candidate = self
                    .measure_m40_variant(phase, pair, ServerVariant::Prepared)
                    .await?;
                (control, candidate)
            } else {
                let candidate = self
                    .measure_m40_variant(phase, pair, ServerVariant::Prepared)
                    .await?;
                let control = self
                    .measure_m40_variant(phase, pair, ServerVariant::Control)
                    .await?;
                (control, candidate)
            };
            ensure!(
                control.is_fixed_workload && candidate.is_fixed_workload,
                "invalid Q8 pair {pair}; raw rows retained, no fixed-workload claim permitted"
            );
            ensure!(
                control.output == expected_outputs.0 && candidate.output == expected_outputs.1,
                "Q8 comparison output changed within a model/configuration; raw rows retained"
            );
            eprintln!("[Q8 {phase} pair {pair}] native={:.3} prepared={:.3} HTTP E2E tok/s, same_output={}", control.tps, candidate.tps, control.output == candidate.output);
            pairs.push(Pair {
                control_tps: control.tps,
                candidate_tps: candidate.tps,
            });
        }
        Ok(())
    }

    fn record_execution_summary(
        &mut self,
        phase: &str,
        comparison: &protocol::Comparison,
    ) -> Result<()> {
        let mut summary = serde_json::to_value(comparison)?;
        summary
            .as_object_mut()
            .context("comparison object")?
            .remove("reaches_original_target");
        self.report.record(
            json!({"kind":"q8_execution_summary","phase":phase,"comparison":summary,
            "candidate_runtime_config":self.args.candidate_runtime_config(),
            "control_runtime_config":self.args.control_runtime_config(),
            "is_ollama_comparison":false,"requires_quality_parity_for_promotion":true}),
        )
    }

    pub(super) async fn compare_q8_execution(&mut self) -> Result<()> {
        let mut passes_runtime_quality = true;
        let runtime = self.args.candidate_runtime_config();
        if let Some(config) = runtime {
            let evidence = super::runtime_quality::load(
                self.args
                    .q8_runtime_quality_report
                    .as_deref()
                    .context("runtime quality report")?,
                config,
                self.args.control_runtime_config(),
                &self.model_sha256,
                &self.prompt_ids,
            )?;
            passes_runtime_quality = evidence.passes_exact_gate;
            self.report.record(evidence.record.clone())?;
            if passes_runtime_quality {
                // Passing exact evidence binds both controls' timed output and
                // actual replay IDs, not merely a configuration name/summary.
                self.selected_quality = Some(evidence);
            }
        }
        self.load_prepared_server().await?;
        let smokes_pass = self.execution_smokes().await?;
        let control = self
            .measure_m40_variant("q8-warmup", 0, ServerVariant::Control)
            .await?;
        let candidate = self
            .measure_m40_variant("q8-warmup", 0, ServerVariant::Prepared)
            .await?;
        ensure!(
            control.is_fixed_workload && candidate.is_fixed_workload,
            "invalid Q8 warmup; raw rows retained"
        );
        let outputs = (control.output, candidate.output);
        self.report
            .record(json!({"kind":"q8_resident_memory","is_timed":false,
            "tracked_device_payload_after_warmup_bytes":CudaContext::total_device_bytes(),
            "combined_process_peak_host_rss_kib":peak_host_rss_kib(),
            "gpu":gpu_snapshot(self.args.ollama_pid,&self.gpu_uuid)?}))?;
        let mut pairs = Vec::new();
        self.run_execution_pairs("screen", 0, 5, &outputs, &mut pairs)
            .await?;
        let mut comparison = protocol::compare_pairs(&pairs)?;
        self.record_execution_summary("screen5", &comparison)?;
        let mut next = protocol::next_trial_count(&comparison, false)?;
        if next == Some(10) {
            self.run_execution_pairs("screen", 5, 10, &outputs, &mut pairs)
                .await?;
            comparison = protocol::compare_pairs(&pairs)?;
            self.record_execution_summary("screen10", &comparison)?;
            next = protocol::next_trial_count(&comparison, false)?;
        }
        if next == Some(20) {
            let mut confirmation = Vec::new();
            self.run_execution_pairs("confirmation", 0, 20, &outputs, &mut confirmation)
                .await?;
            comparison = protocol::compare_pairs(&confirmation)?;
            protocol::next_trial_count(&comparison, true)?;
            self.record_execution_summary("confirmation20", &comparison)?;
        }
        let control_ids = self
            .capture_m40_trace_variant(&outputs.0, ServerVariant::Control)
            .await?;
        let candidate_ids = self
            .capture_m40_trace_variant(&outputs.1, ServerVariant::Prepared)
            .await?;
        let is_equal = control_ids == candidate_ids;
        self.report.record(json!({"kind":"q8_token_comparison","is_equal":is_equal,
            "first_different_generated_step":control_ids.iter().zip(&candidate_ids).position(|(a,b)|a!=b),
            "different_id_count":control_ids.iter().zip(&candidate_ids).filter(|(a,b)|a!=b).count(),
            "control_generated_tokens":control_ids.len(),"candidate_generated_tokens":candidate_ids.len()}))?;
        self.report.record(json!({"kind":"q8_promotion_decision",
            "passes_timing_gate":comparison.passes_promotion_gate,
            "passes_smokes_without_dense_valid_regression":smokes_pass,
            "actual_generated_ids_equal":is_equal,
            "passes_completed_runtime_quality_gate":passes_runtime_quality,
            "is_numerical_track":runtime.is_some_and(|config|config.is_numerical()),
            "candidate_runtime_config":runtime,
            "qualifies_as_benchmark_candidate":comparison.passes_promotion_gate && smokes_pass && is_equal && passes_runtime_quality,
            "changes_runtime_defaults":false,"establishes_ollama_target":false}))?;
        eprintln!("Q8 comparison: native={:.3}, prepared={:.3} HTTP E2E tok/s; gain={:.3}%, timing_gate={}, token_ids_equal={}",
            comparison.control.median, comparison.candidate.median, comparison.gain * 100.0,
            comparison.passes_promotion_gate, is_equal);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::Args;
    use super::*;
    use clap::Parser;
    use m40_llm::infer::execution_weights::{Q8ExecutionFamilies, Q8ExecutionMode};

    /// Parse a complete CLI without touching the model, GPU, or filesystem.
    fn parse_args(extra: &[&str]) -> std::result::Result<Args, clap::Error> {
        let mut args = vec![
            "compare_ollama",
            "--model",
            "model.gguf",
            "--ollama-pid",
            "1",
            "--ollama-log",
            "runner.log",
            "--output",
            "result.jsonl",
        ];
        args.extend_from_slice(extra);
        Args::try_parse_from(args)
    }

    /// Explicit candidate settings must survive parsing without changing the
    /// ordinary baseline mode or silently accepting incompatible comparisons.
    #[test]
    fn test_q8_cli_selects_an_explicit_immutable_candidate() -> Result<()> {
        assert!(parse_args(&[])?.candidate_q8_config()?.is_none());
        let args = parse_args(&[
            "--compare-q8-execution",
            "split",
            "--q8-families",
            "qkv,down,vocabulary",
            "--q8-budget-mb",
            "8",
        ])?;
        let config = args.candidate_q8_config()?.unwrap();
        assert_eq!(config.mode, Q8ExecutionMode::Split);
        assert_eq!(
            config.families,
            Q8ExecutionFamilies::parse("qkv,down,vocabulary")?
        );
        assert_eq!(config.budget_bytes, 8 * 1024 * 1024);
        assert!(parse_args(&["--compare-q8-execution", "off"]).is_err());
        assert!(parse_args(&["--q8-families", "down"]).is_err());
        assert!(parse_args(&["--compare-q8-execution", "split", "--q8-families", "gate"]).is_err());
        assert!(parse_args(&[
            "--compare-q8-execution",
            "split",
            "--compare-head128-attention"
        ])
        .is_err());
        Ok(())
    }

    /// Runtime experiments require their completed quality evidence and keep
    /// independent prefill settings separate from weight-layout comparisons.
    #[test]
    fn test_runtime_cli_requires_evidence_and_preserves_prefill_choice() -> Result<()> {
        let args = parse_args(&[
            "--compare-q8-runtime",
            "packed-f32-128",
            "--q8-runtime-quality-report",
            "quality.jsonl",
            "--q8-runtime-prefill",
            "dequant-sgemm",
        ])?;
        let config = args.candidate_runtime_config().unwrap();
        assert_eq!(config.ffn_config()?.threads(), Some(128));
        assert!(!config.prefill_backend().is_native());
        assert!(args.candidate_q8_config()?.is_none());
        assert!(parse_args(&["--compare-q8-runtime", "native"]).is_err());
        assert!(parse_args(&["--q8-runtime-prefill", "dequant-sgemm"]).is_err());
        for extra in [
            vec!["--compare-q8-execution", "split"],
            vec!["--compare-stage-profile"],
            vec!["--compare-head128-attention"],
        ] {
            let values = [
                vec![
                    "--compare-q8-runtime",
                    "native",
                    "--q8-runtime-quality-report",
                    "quality.jsonl",
                ],
                extra,
            ]
            .concat();
            assert!(parse_args(&values).is_err());
        }
        Ok(())
    }

    /// Ordinary candidate pairs keep native control, while explicit Ollama
    /// timing loads the selected FFN/prefill only with matching quality input.
    #[test]
    fn test_ollama_timing_selects_primary_runtime_explicitly() -> Result<()> {
        let values = [
            "--compare-q8-runtime",
            "packed-f32-128",
            "--q8-runtime-prefill",
            "dequant-sgemm",
            "--q8-runtime-quality-report",
            "quality.jsonl",
        ];
        let paired = parse_args(&values)?;
        assert_eq!(paired.primary_runtime_config(), Some(Default::default()));
        let selected = parse_args(&[values.as_slice(), &["--time-ollama"]].concat())?;
        assert_eq!(
            selected.primary_runtime_config(),
            selected.candidate_runtime_config()
        );
        let config = selected.primary_runtime_config().unwrap();
        assert_eq!(config.ffn_config()?.threads(), Some(128));
        assert!(!config.prefill_backend().is_native());
        assert!(parse_args(&["--time-ollama"]).is_err());
        assert!(parse_args(&["--time-ollama", "--compare-q8-runtime", "native"]).is_err());
        assert!(parse_args(&[]).unwrap().primary_runtime_config().is_none());
        Ok(())
    }

    /// Incremental pairs name the current qualified control rather than the
    /// old native/sequential baseline; selected-vs-Ollama still uses candidate.
    #[test]
    fn test_split_ffn_incremental_control_is_explicit() -> Result<()> {
        let values = [
            "--compare-q8-runtime",
            "split-f32x4-128",
            "--q8-runtime-prefill",
            "dequant-sgemm",
            "--q8-runtime-control",
            "qualified-packed128-sgemm",
            "--q8-runtime-quality-report",
            "quality.jsonl",
        ];
        let args = parse_args(&values)?;
        assert_eq!(
            args.primary_runtime_config(),
            Some(super::super::super::q8_runtime::ControlChoice::QualifiedPacked128Sgemm.config())
        );
        let candidate = args.candidate_runtime_config().unwrap();
        assert_eq!(
            candidate.ffn_config()?,
            m40_llm::infer::Q8FfnConfig::split_f32x4()
        );
        assert_eq!(
            candidate.prompt_prefill(),
            args.primary_runtime_config().unwrap().prompt_prefill()
        );
        assert!(!args
            .primary_execution_config()?
            .families
            .contains(m40_llm::infer::execution_weights::Q8ProjectionGroup::GateUp));
        let args = parse_args(&[values.as_slice(), &["--time-ollama"]].concat())?;
        assert_eq!(
            args.primary_runtime_config(),
            args.candidate_runtime_config()
        );
        assert_eq!(
            args.primary_execution_config()?.families,
            candidate.execution_config()?.families
        );
        assert!(args
            .primary_execution_config()?
            .families
            .contains(m40_llm::infer::execution_weights::Q8ProjectionGroup::GateUp));
        let args = parse_args(&[values.as_slice(), &["--profile-q8-runtime"]].concat())?;
        assert_eq!(
            args.primary_execution_config()?.families,
            candidate.execution_config()?.families
        );
        assert!(parse_args(&["--q8-runtime-control", "qualified-packed128-sgemm"]).is_err());
        Ok(())
    }

    /// Projection pairs retain the current split-FFN/SGEMM control and all
    /// prepared families; selected-vs-Ollama uses the same explicit candidate.
    #[test]
    fn test_projection_incremental_control_is_explicit() -> Result<()> {
        use super::super::super::q8_runtime::{AttentionProjectionChoice, ControlChoice};
        let values = [
            "--compare-q8-runtime",
            "split-f32x4-128",
            "--q8-runtime-prefill",
            "dequant-sgemm",
            "--q8-runtime-control",
            "qualified-split-sgemm",
            "--q8-runtime-attention",
            "warp-256",
            "--q8-runtime-quality-report",
            "quality.jsonl",
        ];
        let args = parse_args(&values)?;
        let control = ControlChoice::QualifiedSplitSgemm.config();
        let candidate = args.candidate_runtime_config().unwrap();
        assert_eq!(args.primary_runtime_config(), Some(control));
        assert_eq!(
            candidate.attention_projection,
            AttentionProjectionChoice::Warp256
        );
        candidate.validate_comparison(control)?;
        assert_eq!(
            serde_json::to_value(args.primary_execution_config()?)?,
            serde_json::to_value(candidate.execution_config()?)?
        );
        let selected = parse_args(&[values.as_slice(), &["--time-ollama"]].concat())?;
        assert_eq!(selected.primary_runtime_config(), Some(candidate));
        assert!(parse_args(&["--q8-runtime-attention", "warp-256"]).is_err());
        Ok(())
    }

    /// Packed projection pairs use the current warp control, while selected
    /// primary/Ollama loading carries the same candidate through the factory.
    #[test]
    fn test_packed_projection_control_and_primary_selection_are_explicit() -> Result<()> {
        use super::super::super::q8_runtime::{AttentionProjectionChoice, ControlChoice};
        let values = [
            "--compare-q8-runtime",
            "split-f32x4-128",
            "--q8-runtime-prefill",
            "dequant-sgemm",
            "--q8-runtime-control",
            "qualified-warp-sgemm",
            "--q8-runtime-attention",
            "packed-warp-256",
            "--q8-runtime-quality-report",
            "quality.jsonl",
        ];
        let args = parse_args(&values)?;
        let control = ControlChoice::QualifiedWarpSgemm.config();
        let candidate = args.candidate_runtime_config().unwrap();
        candidate.validate_comparison(control)?;
        assert_eq!(args.primary_runtime_config(), Some(control));
        assert_eq!(
            candidate.attention_projection,
            AttentionProjectionChoice::PackedWarp256
        );
        assert_eq!(
            serde_json::to_value(candidate.execution_config()?)?,
            serde_json::to_value(control.execution_config()?)?
        );
        let selected = parse_args(&[values.as_slice(), &["--time-ollama"]].concat())?;
        assert_eq!(selected.primary_runtime_config(), Some(candidate));
        let profiled = parse_args(&[values.as_slice(), &["--profile-q8-runtime"]].concat())?;
        assert_eq!(profiled.primary_runtime_config(), Some(candidate));
        assert!(parse_args(&["--q8-runtime-attention", "packed-warp-256"]).is_err());
        Ok(())
    }

    /// Profiling must use the selected FFN/prefill, require quality evidence
    /// and remain separate from authoritative Ollama and legacy profile modes.
    #[test]
    fn test_runtime_profile_is_explicit_and_excludes_timing() -> Result<()> {
        let values = [
            "--compare-q8-runtime",
            "packed-f32-128",
            "--q8-runtime-prefill",
            "dequant-sgemm",
            "--q8-runtime-quality-report",
            "quality.jsonl",
            "--profile-q8-runtime",
        ];
        let args = parse_args(&values)?;
        assert!(args.should_profile_runtime);
        assert!(!args.should_time_ollama);
        assert_eq!(
            args.primary_runtime_config(),
            args.candidate_runtime_config()
        );
        assert_eq!(
            args.primary_runtime_config()
                .unwrap()
                .ffn_config()?
                .threads(),
            Some(128)
        );
        assert!(parse_args(&["--profile-q8-runtime"]).is_err());
        for extra in [
            "--time-ollama",
            "--compare-stage-profile",
            "--compare-head128-attention",
        ] {
            assert!(parse_args(&[values.as_slice(), &[extra]].concat()).is_err());
        }
        Ok(())
    }
}
