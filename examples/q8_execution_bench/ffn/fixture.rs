use super::super::Args;
use anyhow::{ensure, Context, Result};
use m40_llm::{
    evaluation::{
        corpus::Corpus,
        ffn_capture::{FfnCaptureArtifact, FfnTensorSource},
        identity::{file_sha256, sha256},
    },
    infer::{execution_weights::Q8SeparateLaunch, LoadedModel},
};
use serde_json::Value;
use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

fn read_bounded(path: &Path, limit: usize) -> Result<String> {
    let mut text = String::new();
    File::open(path)?
        .take((limit + 1) as u64)
        .read_to_string(&mut text)?;
    ensure!(
        text.len() <= limit,
        "FFN evidence file exceeds size limit: {}",
        path.display()
    );
    Ok(text)
}

pub(super) struct FfnInputs {
    pub artifact: FfnCaptureArtifact,
    pub artifact_sha256: String,
    pub report_sha256: String,
    pub manifest: Value,
}

impl FfnInputs {
    pub fn load(args: &Args) -> Result<Self> {
        let text = read_bounded(
            args.ffn_inputs.as_ref().context("provide --ffn-inputs")?,
            64 * 1024 * 1024,
        )?;
        let report = read_bounded(
            args.ffn_capture_report
                .as_ref()
                .context("provide --ffn-capture-report")?,
            4 * 1024 * 1024,
        )?;
        let corpus_text = read_bounded(&args.ffn_corpus, 4 * 1024 * 1024)?;
        let inputs = Self::parse(
            &text,
            &report,
            &Corpus::parse(&corpus_text)?,
            &sha256(corpus_text.as_bytes())?,
            &file_sha256(&args.model)?,
        )?;
        ensure!(
            inputs.artifact.plan.source_layer_count == 36
                && inputs.artifact.plan.selected_layer_ids == [0, 17, 35]
                && inputs.artifact.plan.d_model == 2048
                && inputs.artifact.plan.hidden_dim == 11008,
            "M02 baseline requires original Qwen2.5-3B layers 0/17/35"
        );
        Ok(inputs)
    }

    pub(super) fn parse(
        text: &str,
        report: &str,
        corpus: &Corpus,
        corpus_sha256: &str,
        model_sha256: &str,
    ) -> Result<Self> {
        let artifact = FfnCaptureArtifact::parse(text)?;
        artifact.plan.validate_corpus(corpus, corpus_sha256)?;
        let artifact_sha256 = sha256(text.as_bytes())?;
        let records = report
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ensure!(
            records.len() == artifact.tasks.len() + 2,
            "FFN report has missing/extra records"
        );
        let manifest = records.first().context("empty FFN report")?;
        let complete = records.last().context("empty FFN report")?;
        ensure!(
            manifest["kind"] == "ffn_capture_manifest"
                && manifest["schema_version"] == 1
                && manifest["build_profile"] == "release"
                && manifest["worktree"] == ""
                && complete["kind"] == "ffn_capture_complete"
                && complete["is_complete"] == true
                && complete["artifact_sha256"] == artifact_sha256,
            "FFN report requires clean release capture and matching completed artifact"
        );
        ensure!(
            manifest["model"]["sha256"] == model_sha256
                && artifact.plan.model_sha256 == model_sha256
                && manifest["corpus_sha256"] == corpus_sha256
                && manifest["execution_sha256"] == artifact.plan.execution_sha256
                && sha256(&serde_json::to_vec(&manifest["execution_configuration"])?)?
                    == artifact.plan.execution_sha256,
            "FFN model/corpus/execution identity mismatch"
        );
        for (record, task) in records[1..records.len() - 1].iter().zip(&artifact.tasks) {
            let prompt: Vec<u32> = serde_json::from_value(record["prompt_token_ids"].clone())?;
            let baseline: Vec<u32> =
                serde_json::from_value(record["baseline"]["generated_token_ids"].clone())?;
            let captured: Vec<u32> =
                serde_json::from_value(record["captured"]["generated_token_ids"].clone())?;
            ensure!(
                record["kind"] == "ffn_capture_task"
                    && record["id"] == task.id
                    && record["has_identical_generation"] == true
                    && prompt == task.prompt_token_ids
                    && baseline == task.generated_token_ids
                    && captured == baseline
                    && record["baseline"]["output"] == record["captured"]["output"]
                    && record["baseline"]["metrics"]["stop_reason"]
                        == record["captured"]["metrics"]["stop_reason"],
                "FFN capture task history mismatch"
            );
        }
        for layer in &artifact.layers {
            ensure!(
                layer.rows.len() >= 4,
                "FFN fixture needs at least four rows/layer"
            );
            for (values, dimension) in [
                (&layer.normalized_inputs, artifact.plan.d_model),
                (&layer.swiglu_inputs, artifact.plan.hidden_dim),
            ] {
                for a in 0..4 {
                    for b in a + 1..4 {
                        ensure!(
                            values[a * dimension..(a + 1) * dimension]
                                .iter()
                                .zip(&values[b * dimension..(b + 1) * dimension])
                                .any(|(x, y)| x.to_bits() != y.to_bits()),
                            "FFN four-row inputs must be distinct, not duplicated work"
                        );
                    }
                }
            }
        }
        Ok(Self {
            artifact,
            artifact_sha256,
            report_sha256: sha256(report.as_bytes())?,
            manifest: manifest.clone(),
        })
    }

    pub fn validate_environment(&self) -> Result<()> {
        let expected: BTreeMap<String, String> = serde_json::from_value(
            self.manifest["execution_configuration"]["environment"].clone(),
        )?;
        let actual: BTreeMap<_, _> = std::env::vars()
            .filter(|(key, _)| {
                key.starts_with("M40LLM_") || key.starts_with("CUDA_") || key == "OMP_NUM_THREADS"
            })
            .collect();
        validate_environment_values(&expected, &actual)
    }

    pub fn validate_model(&self, model: &LoadedModel) -> Result<()> {
        let props = model.cuda.current_device_props()?;
        ensure!(
            props.major == 5 && props.minor == 2,
            "FFN benchmark requires sm_52"
        );
        let plan = &self.artifact.plan;
        ensure!(
            model.model_config.block_count as usize == plan.source_layer_count
                && model.model_config.embedding_length as usize == plan.d_model
                && model.model_config.feed_forward_length as usize == plan.hidden_dim
                && model.kv_cache.is_none()
                && model.cuda.uploaded_weights_bytes()? == model.weights_len
                && !model.d_weights_base.is_null(),
            "FFN model shape/upload/layout mismatch"
        );
        ensure!(
            digest(&model.host_weights) == self.artifact.source_weights_sha256,
            "FFN uploaded source region differs from capture"
        );
        let prepared = model.q8_execution_stats();
        ensure!(
            prepared.has_prepared_vocabulary
                && prepared.single_row_down_split_launch == Some(Q8SeparateLaunch::new(4, 256)?)
                && prepared.resident_bytes
                    == self.manifest["prepared_weights"]["resident_bytes"]
                        .as_u64()
                        .context("missing prepared bytes")? as usize,
            "FFN baseline did not preserve the qualified prepared down/vocabulary cache"
        );
        for layer in &self.artifact.layers {
            let mapped = model.map_standard_layer(layer.source.layer_id)?;
            for (source, view) in [
                (&layer.source.gate, &mapped.w_gate),
                (&layer.source.up, &mapped.w_up),
                (&layer.source.down, &mapped.w_down),
            ] {
                let tensor = model
                    .gguf
                    .tensors
                    .iter()
                    .find(|tensor| tensor.name == source.name)
                    .context("missing FFN tensor")?;
                ensure!(
                    tensor.offset == source.byte_offset
                        && tensor.dtype == source.dtype
                        && tensor.shape == source.shape.map(|dim| dim as u64)
                        && view.byte_offset == source.byte_offset
                        && view.dtype == source.dtype
                        && view.shape == tensor.shape
                        && !view.dptr.is_null(),
                    "FFN source identity/shape differs from capture"
                );
                ensure!(
                    digest(source_bytes(model, source)?) == source.source_sha256,
                    "FFN tensor payload hash differs from capture"
                );
            }
        }
        Ok(())
    }
}

pub(super) fn validate_environment_values(
    expected: &BTreeMap<String, String>,
    actual: &BTreeMap<String, String>,
) -> Result<()> {
    ensure!(
        actual == expected,
        "FFN baseline environment must match the frozen capture configuration exactly"
    );
    for (key, value) in [
        ("M40LLM_ENABLE_NVCC", "1"),
        ("M40LLM_ENABLE_CUBLAS", "1"),
        ("M40LLM_PROJECTION_BACKEND", "large-model"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_Q8_EXECUTION_WEIGHTS", "off"),
        ("M40LLM_DECODE_GRAPH", "0"),
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_Q8_DECODE_COL_TILE", "4"),
        ("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4"),
        ("M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK", "4"),
        ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
    ] {
        ensure!(
            actual.get(key).map(String::as_str) == Some(value),
            "FFN baseline requires qualified control {key}={value}"
        );
    }
    Ok(())
}

fn digest(bytes: &[u8]) -> String {
    ring::digest::digest(&ring::digest::SHA256, bytes)
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(super) fn source_bytes<'a>(
    model: &'a LoadedModel,
    source: &FfnTensorSource,
) -> Result<&'a [u8]> {
    let begin = usize::try_from(source.byte_offset)?;
    let end = begin
        .checked_add(source.byte_len()?)
        .context("FFN tensor byte overflow")?;
    model
        .host_weights
        .get(begin..end)
        .context("FFN source outside model weight region")
}
