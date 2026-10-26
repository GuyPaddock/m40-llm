use super::{super::q8_runtime::RuntimeConfig, sha256_bytes, sha256_file};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{collections::BTreeSet, path::Path};

pub(super) struct QualityEvidence {
    pub record: Value,
    pub passes_exact_gate: bool,
    pub expected_output: String,
    expected_token_ids: Vec<u32>,
}

impl QualityEvidence {
    pub fn verify_generation(&self, token_ids: &[u32], output: &str) -> Result<()> {
        ensure!(
            self.expected_output == output && self.expected_token_ids == token_ids,
            "actual runtime generation differs from completed quality evidence; rerun quality"
        );
        Ok(())
    }
}

fn only_record<'a>(rows: &'a [Value], kind: &str) -> Result<&'a Value> {
    let mut matching = rows.iter().filter(|row| row["kind"] == kind);
    let row = matching.next().with_context(|| format!("missing {kind}"))?;
    ensure!(matching.next().is_none(), "duplicate {kind}");
    Ok(row)
}

fn validate(
    rows: &[Value],
    config: RuntimeConfig,
    control_config: RuntimeConfig,
    model_sha: &str,
    corpus_bytes: &[u8],
    prompt_ids: &[u32],
) -> Result<bool> {
    config.validate_comparison(control_config)?;
    let manifest = only_record(rows, "q8_runtime_manifest")?;
    let summary = only_record(rows, "q8_runtime_summary")?;
    let browser = only_record(rows, "q8_runtime_browser_comparison")?;
    ensure!(
        manifest["reference"]["sha256"] == model_sha
            && manifest["candidate"]["sha256"] == model_sha,
        "quality report model differs from HTTP model"
    );
    ensure!(
        manifest["prefill_execution_contract"] == super::super::q8_runtime::PREFILL_EXECUTION_CONTRACT,
        "quality report predates the prefix completion contract; rerun quality with the corrected runtime"
    );
    ensure!(
        manifest["candidate_config"] == serde_json::to_value(config)?,
        "quality report runtime candidate differs"
    );
    ensure!(
        manifest["control_config"] == serde_json::to_value(control_config)?,
        "quality reference differs"
    );
    ensure!(
        manifest["browser_prompt_ids"] == serde_json::to_value(prompt_ids)?,
        "quality benchmark prompt differs"
    );
    ensure!(
        manifest["corpus_sha256"] == sha256_bytes(corpus_bytes)?,
        "quality corpus differs from frozen checkpoint"
    );
    ensure!(
        manifest["selected_splits"] == json!(["calibration", "held_out"]),
        "quality requires both frozen splits"
    );
    let corpus = m40_llm::evaluation::corpus::Corpus::parse(std::str::from_utf8(corpus_bytes)?)?;
    let expected: BTreeSet<_> = corpus
        .calibration
        .tasks
        .iter()
        .chain(&corpus.held_out.tasks)
        .map(|task| task.id.as_str())
        .collect();
    let tasks: Vec<_> = rows
        .iter()
        .filter(|row| row["kind"] == "model_quality_task")
        .collect();
    let observed: BTreeSet<_> = tasks
        .iter()
        .map(|row| row["id"].as_str().context("quality task id"))
        .collect::<Result<_>>()?;
    ensure!(
        expected == observed && tasks.len() == expected.len(),
        "missing or duplicate frozen quality tasks"
    );
    let mut valid = 0;
    let mut regressions = 0;
    for task in tasks {
        let reference = task["reference_score"]["passed"]
            .as_bool()
            .context("reference task score")?;
        let candidate = task["candidate_score"]["passed"]
            .as_bool()
            .context("candidate task score")?;
        valid += usize::from(reference);
        regressions += usize::from(reference && !candidate);
    }
    ensure!(
        summary["dense_valid_tasks"] == valid && summary["dense_valid_regressions"] == regressions,
        "inconsistent quality score summary"
    );
    let generations: Vec<_> = rows
        .iter()
        .filter(|row| row["kind"] == "q8_runtime_browser_generation")
        .collect();
    ensure!(
        generations.len() == 2,
        "missing browser generation evidence"
    );
    let control = generations
        .iter()
        .find(|row| row["label"] == "control")
        .context("control generation")?;
    let candidate = generations
        .iter()
        .find(|row| row["label"] == "candidate")
        .context("candidate generation")?;
    if !config.attention_projection.backend().is_native() {
        let layers = manifest["candidate"]["layers"]
            .as_u64()
            .context("projection quality layer count")?;
        let launches = layers
            .checked_mul(512)
            .context("projection launch count overflow")?;
        ensure!(
            layers > 0
                && manifest["candidate"]["active_layer_count"] == layers
                && control["actual_attention_projection_backend"]
                    == control_config.attention_projection.backend().as_str()
                && candidate["actual_attention_projection_backend"]
                    == config.attention_projection.backend().as_str(),
            "projection quality requires actual model backends and all source layers"
        );
        for (generation, choice) in [
            (control, control_config.attention_projection),
            (candidate, config.attention_projection),
        ] {
            if let Some(operations) = choice.backend().get_operations() {
                for op in operations {
                    ensure!(
                        generation["dispatch"]["operations"][op]["launches"] == launches,
                        "projection quality dispatch does not match all actual generated steps"
                    );
                }
            }
            for other in [
                m40_llm::infer::Q8AttentionProjectionBackend::Warp256,
                m40_llm::infer::Q8AttentionProjectionBackend::PackedWarp256,
            ] {
                if other != choice.backend() {
                    for op in other.get_operations().unwrap() {
                        ensure!(
                            generation["dispatch"]["operations"][op]["launches"]
                                .as_u64()
                                .unwrap_or(0)
                                == 0,
                            "projection quality used a different backend's dispatch"
                        );
                    }
                }
            }
        }
    }
    let fixed = generations.iter().all(|row| {
        row["generation"]["generated_token_ids"]
            .as_array()
            .is_some_and(|ids| ids.len() == 512)
            && row["generation"]["metrics"]["generated_tokens"] == 512
            && row["generation"]["metrics"]["stop_reason"] == "length"
            && row["prompt_token_ids"] == manifest["browser_prompt_ids"]
    });
    let identical = control["generation"]["generated_token_ids"]
        == candidate["generation"]["generated_token_ids"]
        && control["generation"]["output"] == candidate["generation"]["output"]
        && control["generation"]["metrics"]["stop_reason"]
            == candidate["generation"]["metrics"]["stop_reason"];
    ensure!(
        browser["has_fixed_workload"] == fixed && browser["has_identical_generation"] == identical,
        "inconsistent browser quality summary"
    );
    let passes = !config.is_numerical() && fixed && identical && valid > 0 && regressions == 0;
    ensure!(
        summary["passes_exact_quality_gate"] == passes,
        "inconsistent exact quality gate"
    );
    ensure!(
        summary["tracked_device_bytes_initial"] == summary["tracked_device_bytes_after"],
        "quality allocation cleanup failed"
    );
    Ok(passes)
}

pub(super) fn load(
    path: &Path,
    config: RuntimeConfig,
    control_config: RuntimeConfig,
    model_sha: &str,
    prompt_ids: &[u32],
) -> Result<QualityEvidence> {
    let text = std::fs::read_to_string(path)?;
    let rows = text
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<Vec<Value>, _>>()?;
    let corpus = std::fs::read("data/research_quality/v1.json")?;
    let passes_exact_gate = validate(
        &rows,
        config,
        control_config,
        model_sha,
        &corpus,
        prompt_ids,
    )?;
    let candidate = rows
        .iter()
        .find(|row| row["kind"] == "q8_runtime_browser_generation" && row["label"] == "candidate")
        .context("candidate quality generation")?;
    Ok(QualityEvidence {
        passes_exact_gate,
        expected_output: candidate["generation"]["output"]
            .as_str()
            .context("quality output")?
            .to_owned(),
        expected_token_ids: serde_json::from_value(
            candidate["generation"]["generated_token_ids"].clone(),
        )?,
        record: json!({"kind":"q8_runtime_quality_evidence",
        "path":std::fs::canonicalize(path)?,"sha256":sha256_file(path)?,"candidate_config":config,
        "control_config":control_config,
        "manifest":only_record(&rows,"q8_runtime_manifest")?,"summary":only_record(&rows,"q8_runtime_summary")?,
        "passes_exact_quality_gate":passes_exact_gate,"is_numerical_track":config.is_numerical()}),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A valid report cannot qualify a later runtime with changed actual IDs
    /// or output, even when its model/configuration names still match.
    #[test]
    fn test_completed_quality_requires_actual_generation_match() -> Result<()> {
        let evidence = QualityEvidence {
            record: json!({}),
            passes_exact_gate: true,
            expected_output: "ready".into(),
            expected_token_ids: vec![3, 4, 5],
        };
        evidence.verify_generation(&[3, 4, 5], "ready")?;
        assert!(evidence.verify_generation(&[3, 7, 5], "ready").is_err());
        assert!(evidence.verify_generation(&[3, 4], "ready").is_err());
        assert!(evidence.verify_generation(&[3, 4, 5], "changed").is_err());
        Ok(())
    }

    /// Missing or duplicate completion evidence must not qualify a timing run;
    /// a report from another model or another candidate must fail preflight.
    #[test]
    fn test_quality_preflight_rejects_incomplete_or_mismatched_evidence() -> Result<()> {
        let config = RuntimeConfig::default();
        let corpus = include_bytes!("../../data/research_quality/v1.json");
        assert!(validate(&[], config, config, "model", corpus, &[1, 2]).is_err());
        let rows = vec![
            json!({"kind":"q8_runtime_manifest","reference":{"sha256":"different"}}),
            json!({"kind":"q8_runtime_summary"}),
            json!({"kind":"q8_runtime_browser_comparison"}),
        ];
        assert!(validate(&rows, config, config, "model", corpus, &[1, 2])
            .unwrap_err()
            .to_string()
            .contains("model differs"));
        let mut duplicate = rows.clone();
        duplicate.push(rows[0].clone());
        assert!(
            validate(&duplicate, config, config, "model", corpus, &[1, 2])
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
        Ok(())
    }

    /// Timing promotion requires complete consistent evidence. A real changed
    /// continuation is a failed gate, not a malformed report or an exact win.
    /// Evidence from the unsafe prefix schedule must not qualify a repaired run.
    /// Projection candidates additionally require matching actual backends and
    /// full one-row dispatch, rather than a changed configuration label alone.
    #[test]
    fn test_quality_preflight_distinguishes_failed_gate_from_stale_summary() -> Result<()> {
        let config = RuntimeConfig::default();
        let corpus = include_bytes!("../../data/research_quality/v1.json");
        let parsed = m40_llm::evaluation::corpus::Corpus::parse(std::str::from_utf8(corpus)?)?;
        let mut rows = vec![
            json!({"kind":"q8_runtime_manifest","reference":{"sha256":"model"},"candidate":{"sha256":"model"},
                "prefill_execution_contract":super::super::super::q8_runtime::PREFILL_EXECUTION_CONTRACT,
                "candidate_config":config,"control_config":config,"corpus_sha256":sha256_bytes(corpus)?,
                "selected_splits":["calibration","held_out"],"browser_prompt_ids":[1,2]}),
            json!({"kind":"q8_runtime_summary","dense_valid_tasks":12,"dense_valid_regressions":0,
                "passes_exact_quality_gate":true,"tracked_device_bytes_initial":0,"tracked_device_bytes_after":0}),
            json!({"kind":"q8_runtime_browser_comparison","has_fixed_workload":true,"has_identical_generation":true}),
        ];
        for label in ["control", "candidate"] {
            rows.push(json!({"kind":"q8_runtime_browser_generation","label":label,"prompt_token_ids":[1,2],
                "generation":{"generated_token_ids":vec![3;512],"output":"same",
                    "metrics":{"generated_tokens":512,"stop_reason":"length"}}}));
        }
        rows.extend(
            parsed
                .calibration
                .tasks
                .iter()
                .chain(&parsed.held_out.tasks)
                .map(|task| {
                    json!({"kind":"model_quality_task","id":task.id,
                "reference_score":{"passed":true},"candidate_score":{"passed":true}})
                }),
        );
        assert!(validate(&rows, config, config, "model", corpus, &[1, 2])?);
        let qualified =
            super::super::super::q8_runtime::ControlChoice::QualifiedPacked128Sgemm.config();
        assert!(validate(&rows, config, qualified, "model", corpus, &[1, 2])
            .unwrap_err()
            .to_string()
            .contains("quality reference differs"));
        rows[0]["control_config"] = serde_json::to_value(qualified)?;
        assert!(validate(
            &rows,
            config,
            qualified,
            "model",
            corpus,
            &[1, 2]
        )?);
        rows[0]["control_config"] = serde_json::to_value(config)?;
        let projection_control =
            super::super::super::q8_runtime::ControlChoice::QualifiedSplitSgemm.config();
        let projection_candidate = RuntimeConfig {
            attention_projection:
                super::super::super::q8_runtime::AttentionProjectionChoice::Warp256,
            ..projection_control
        };
        let mut projection = rows.clone();
        projection[0]["control_config"] = serde_json::to_value(projection_control)?;
        projection[0]["candidate_config"] = serde_json::to_value(projection_candidate)?;
        projection[0]["candidate"]["layers"] = json!(36);
        projection[0]["candidate"]["active_layer_count"] = json!(36);
        projection[3]["actual_attention_projection_backend"] = json!("native");
        projection[4]["actual_attention_projection_backend"] = json!("warp-256");
        let operation = "gemm_f32xq8_0_gguf_coltile4_warp";
        projection[4]["dispatch"] = json!({"operations":{
            "qkv_f32xq8_0_gguf_warp":{"launches":18432},
            "gemm_f32xq8_0_gguf_coltile4_warp":{"launches":18432}}});
        assert!(validate(
            &projection,
            projection_candidate,
            projection_control,
            "model",
            corpus,
            &[1, 2]
        )?);
        projection[4]["dispatch"]["operations"][operation]["launches"] = json!(0);
        assert!(validate(
            &projection,
            projection_candidate,
            projection_control,
            "model",
            corpus,
            &[1, 2]
        )
        .is_err());
        projection[4]["dispatch"]["operations"][operation]["launches"] = json!(18432);
        projection[4]["actual_attention_projection_backend"] = Value::Null;
        assert!(validate(
            &projection,
            projection_candidate,
            projection_control,
            "model",
            corpus,
            &[1, 2]
        )
        .is_err());
        let packed_control =
            super::super::super::q8_runtime::ControlChoice::QualifiedWarpSgemm.config();
        let packed_config = RuntimeConfig {
            attention_projection:
                super::super::super::q8_runtime::AttentionProjectionChoice::PackedWarp256,
            ..packed_control
        };
        let mut packed = projection.clone();
        packed[0]["control_config"] = serde_json::to_value(packed_control)?;
        packed[0]["candidate_config"] = serde_json::to_value(packed_config)?;
        packed[3]["actual_attention_projection_backend"] = json!("warp-256");
        packed[3]["dispatch"] = json!({"operations":{
            "qkv_f32xq8_0_gguf_warp":{"launches":18432},
            "gemm_f32xq8_0_gguf_coltile4_warp":{"launches":18432}}});
        packed[4]["actual_attention_projection_backend"] = json!("packed-warp-256");
        packed[4]["dispatch"] = json!({"operations":{
            "qkv_f32xq8_0_gguf_packed_warp":{"launches":18432},
            "gemm_f32xq8_0_gguf_coltile4_packed_warp":{"launches":18432}}});
        assert!(validate(
            &packed,
            packed_config,
            packed_control,
            "model",
            corpus,
            &[1, 2]
        )?);
        for has_wrong_backend in [false, true] {
            let mut invalid = packed.clone();
            if has_wrong_backend {
                invalid[4]["actual_attention_projection_backend"] = json!("warp-256");
            } else {
                invalid[4]["dispatch"]["operations"]["qkv_f32xq8_0_gguf_packed_warp"]["launches"] =
                    json!(0);
            }
            assert!(validate(
                &invalid,
                packed_config,
                packed_control,
                "model",
                corpus,
                &[1, 2]
            )
            .is_err());
        }
        packed[4]["dispatch"]["operations"]["qkv_f32xq8_0_gguf_warp"] = json!({"launches":1});
        assert!(validate(
            &packed,
            packed_config,
            packed_control,
            "model",
            corpus,
            &[1, 2]
        )
        .is_err());
        for contract in [Value::Null, json!("prefix-unfenced-v0")] {
            let mut stale = rows.clone();
            stale[0]["prefill_execution_contract"] = contract;
            assert!(validate(&stale, config, config, "model", corpus, &[1, 2])
                .unwrap_err()
                .to_string()
                .contains("rerun quality"));
        }
        assert!(validate(&rows, config, config, "model", corpus, &[2, 1])
            .unwrap_err()
            .to_string()
            .contains("prompt differs"));
        rows[4]["generation"]["generated_token_ids"][20] = json!(4);
        assert!(validate(&rows, config, config, "model", corpus, &[1, 2]).is_err());
        rows[2]["has_identical_generation"] = json!(false);
        rows[1]["passes_exact_quality_gate"] = json!(false);
        assert!(!validate(&rows, config, config, "model", corpus, &[1, 2])?);
        rows.pop();
        assert!(validate(&rows, config, config, "model", corpus, &[1, 2])
            .unwrap_err()
            .to_string()
            .contains("quality tasks"));
        Ok(())
    }
}
