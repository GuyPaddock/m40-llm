use super::{
    records::{self, Report},
    run,
    tasks::TaskResult,
    Args,
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    evaluation::{corpus::Split, loss::TeacherForcedReport, teacher_forced::compare_loaded_models},
    gguf::{compact::CompactGgufPlan, load_gguf, GgufModel, GgufScalar, GgufValue},
    infer::{LayerMask, LoadedModel},
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Read,
    path::PathBuf,
    time::Instant,
};

pub(super) struct CompactComparison {
    report_path: PathBuf,
    report_sha256: String,
    record: Value,
    reference_skipped_layers: Vec<usize>,
    retained_original_layers: Vec<usize>,
    source_weight_bytes: usize,
    compact_weight_bytes: usize,
}

fn metadata_u64(model: &GgufModel, key: &str) -> Result<u64> {
    match model.metadata.get(key) {
        Some(GgufValue::Scalar(GgufScalar::U32(value))) => Ok(u64::from(*value)),
        Some(GgufValue::Scalar(GgufScalar::U64(value))) => Ok(*value),
        _ => anyhow::bail!("missing/invalid unsigned compact metadata {key}"),
    }
}

fn metadata_ids(model: &GgufModel, key: &str) -> Result<Vec<usize>> {
    let Some(GgufValue::Array(values)) = model.metadata.get(key) else {
        anyhow::bail!("missing compact layer mapping {key}");
    };
    values
        .iter()
        .map(|value| match value {
            GgufScalar::U32(id) => Ok(*id as usize),
            _ => anyhow::bail!("compact original IDs must be U32"),
        })
        .collect()
}

pub(super) fn require_identical_logits(comparison: &TeacherForcedReport) -> Result<()> {
    ensure!(
        comparison.input_tokens >= 2
            && comparison.tokens.len() == comparison.input_tokens - 1
            && comparison.reference.scored_tokens == comparison.tokens.len()
            && comparison.max_abs_logit_diff == 0.0
            && comparison.mean_abs_logit_diff == 0.0
            && comparison.reference == comparison.candidate
            && comparison
                .tokens
                .iter()
                .all(|token| token.max_abs_logit_diff == 0.0
                    && token.mean_abs_logit_diff == 0.0
                    && token.reference == token.candidate
                    && token.top10_overlap == token.reference.top_indices.len()),
        "masked-source/compact full logits differ; inspect retained diagnostic records"
    );
    Ok(())
}

pub(super) fn validate_diagnostic_capacity(prompt_tokens: usize, capacity: u32) -> Result<()> {
    ensure!(
        prompt_tokens > 0
            && prompt_tokens
                .checked_add(2)
                .is_some_and(|count| count <= capacity as usize),
        "compact prompt/decode logit diagnostics need two reserved tokens; no truncation"
    );
    Ok(())
}

impl CompactComparison {
    /// Check actual files and export lineage before allocating either GPU model.
    /// This is consistency checking, not a signed attestation or an inference pass.
    pub fn load(args: &Args, corpus_bytes: &[u8]) -> Result<Option<Self>> {
        let Some(path) = &args.compact_export_report else {
            return Ok(None);
        };
        let mut bytes = Vec::new();
        File::open(path)?
            .take(8 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() <= 8 * 1024 * 1024,
            "compact export record exceeds 8 MiB"
        );
        let record: Value = serde_json::from_slice(&bytes)?;
        let export = &record["export"];
        let evidence = &export["evidence"];
        ensure!(
            record["kind"] == "compact_export_complete"
                && record["is_complete"] == true
                && export["schema_version"] == 1
                && evidence["corpus_sha256"] == records::sha256(corpus_bytes)?
                && evidence["source_sha256"] == records::file_sha256(&args.reference_model)?
                && export["output_sha256"] == records::file_sha256(&args.model)?,
            "compact export source/output/corpus identity mismatch or incomplete record"
        );
        let source = load_gguf(&args.reference_model)?;
        let compact = load_gguf(&args.model)?;
        let original_count = u32::try_from(
            evidence["original_layer_count"]
                .as_u64()
                .context("original layer count")?,
        )?;
        let skipped: Vec<usize> =
            serde_json::from_value(evidence["removed_original_layers"].clone())?;
        let plan = CompactGgufPlan::build(
            &source,
            fs::metadata(&args.reference_model)?.len(),
            &LayerMask::new(original_count, &skipped)?,
        )?;
        ensure!(
            serde_json::to_value(&plan)? == export["plan"],
            "compact export copy plan differs from source"
        );
        let architecture = source.metadata["general.architecture"]
            .as_str()
            .context("source architecture")?;
        ensure!(
            compact.version == 3
                && compact
                    .metadata
                    .get("general.architecture")
                    .and_then(GgufValue::as_str)
                    == Some(architecture)
                && metadata_u64(&compact, &format!("{architecture}.block_count"))?
                    == plan.retained_original_layers().len() as u64
                && metadata_u64(&compact, "m40llm.pruning.original_layer_count")?
                    == u64::from(original_count)
                && metadata_u64(&compact, "m40llm.pruning.schema_version")? == 1
                && metadata_ids(&compact, "m40llm.pruning.removed_original_layers")? == skipped
                && metadata_ids(&compact, "m40llm.pruning.retained_original_layers")?
                    == plan.retained_original_layers()
                && export["output_file_bytes"].as_u64() == Some(fs::metadata(&args.model)?.len())
                && compact.tensors.len() == plan.tensors().len(),
            "compact GGUF metadata, layer mapping or file length differs from export"
        );
        for key in [
            "source_sha256",
            "ranking_sha256",
            "quality_report_sha256",
            "corpus_sha256",
        ] {
            ensure!(
                compact
                    .metadata
                    .get(&format!("m40llm.pruning.{key}"))
                    .and_then(GgufValue::as_str)
                    == evidence[key].as_str(),
                "compact GGUF evidence differs for {key}"
            );
        }
        let quality: Value = serde_json::from_str(
            compact
                .metadata
                .get("m40llm.pruning.quality_summary")
                .and_then(GgufValue::as_str)
                .context("compact quality metadata")?,
        )?;
        ensure!(
            quality == evidence["quality"],
            "compact quality costs differ from export evidence"
        );
        for (actual, expected) in compact.tensors.iter().zip(plan.tensors()) {
            ensure!(
                actual.name == expected.destination_name()
                    && actual.dtype == expected.dtype()
                    && actual.shape == expected.shape()
                    && actual.offset == expected.destination_offset(),
                "compact tensor layout differs from retained source"
            );
        }
        let block_key = format!("{architecture}.block_count");
        for (key, value) in &source.metadata {
            if [
                "general.name",
                "general.parameter_count",
                "general.size_label",
                &block_key,
            ]
            .contains(&key.as_str())
            {
                continue;
            }
            let copied = compact
                .metadata
                .get(key)
                .with_context(|| format!("compact metadata missing {key}"))?;
            ensure!(
                serde_json::to_value(value)? == serde_json::to_value(copied)?,
                "compact unchanged metadata differs for {key}"
            );
        }
        ensure!(
            compact
                .metadata
                .keys()
                .all(|key| source.metadata.contains_key(key)
                    || key.starts_with("m40llm.pruning.")
                    || matches!(key.as_str(), "general.name" | "general.parameter_count"))
                && !compact.metadata.contains_key("general.size_label"),
            "compact metadata adds unsupported overrides"
        );
        let tokenizer = |model: &GgufModel| -> Result<Value> {
            let values: BTreeMap<_, _> = model
                .metadata
                .iter()
                .filter(|(key, _)| key.starts_with("tokenizer."))
                .collect();
            Ok(serde_json::to_value(values)?)
        };
        ensure!(
            tokenizer(&source)? == tokenizer(&compact)?,
            "compact tokenizer metadata differs from source"
        );
        let source_weight_bytes = fs::metadata(&args.reference_model)?
            .len()
            .checked_sub(source.data_offset)
            .context("source data offset")?;
        let compact_weight_bytes = fs::metadata(&args.model)?
            .len()
            .checked_sub(compact.data_offset)
            .context("compact data offset")?;
        ensure!(
            export["source_tensor_data_bytes"].as_u64() == Some(source_weight_bytes)
                && export["output_tensor_data_bytes"].as_u64() == Some(compact_weight_bytes)
                && compact_weight_bytes == plan.destination_data_bytes(),
            "compact data storage differs from export"
        );
        Ok(Some(Self {
            report_path: path.clone(),
            report_sha256: records::sha256(&bytes)?,
            record,
            reference_skipped_layers: skipped,
            retained_original_layers: plan.retained_original_layers().to_vec(),
            source_weight_bytes: usize::try_from(source_weight_bytes)?,
            compact_weight_bytes: usize::try_from(compact_weight_bytes)?,
        }))
    }

    pub fn reference_skipped_layers(&self) -> &[usize] {
        &self.reference_skipped_layers
    }

    pub fn describe(&self) -> Value {
        json!({"report":self.report_path,"report_sha256":self.report_sha256,
            "source_sha256":self.record["export"]["evidence"]["source_sha256"],
            "compact_sha256":self.record["export"]["output_sha256"],
            "reference_role":"source with frozen mask; NOT unchanged dense model",
            "reference_skipped_original_layer_ids":self.reference_skipped_layers,
            "candidate_role":"compact artifact with every retained layer active",
            "retained_original_layer_ids":self.retained_original_layers,
            "source_tensor_data_bytes":self.source_weight_bytes,"compact_tensor_data_bytes":self.compact_weight_bytes,
            "prior_quality_costs":self.record["export"]["evidence"]["quality"],
            "historical_export_preparation_elapsed_ms":self.record["total_preparation_elapsed_ms"],
            "requires_live_equivalence_validation":true,"is_unchanged_model":false})
    }

    pub fn validate_identities(&self, reference: &Value, candidate: &Value) -> Result<()> {
        ensure!(
            reference["sha256"] == self.record["export"]["evidence"]["source_sha256"]
                && candidate["sha256"] == self.record["export"]["output_sha256"],
            "model file changed after compact preflight"
        );
        Ok(())
    }

    pub fn validate_models(&self, reference: &LoadedModel, candidate: &LoadedModel) -> Result<()> {
        ensure!(
            reference.layer_mask.skipped_layers() == self.reference_skipped_layers
                && !candidate.layer_mask.has_skipped_layers()
                && reference.active_layer_count()? == self.retained_original_layers.len()
                && candidate.model_config.block_count as usize
                    == self.retained_original_layers.len(),
            "compact comparison must mask only the original source, not the exported candidate"
        );
        let a = &reference.model_config;
        let b = &candidate.model_config;
        ensure!(
            (
                a.embedding_length,
                a.feed_forward_length,
                a.vocab_size,
                a.attention_head_count,
                a.attention_head_count_kv,
                a.attention_key_length,
                a.rope_layout_code()
            ) == (
                b.embedding_length,
                b.feed_forward_length,
                b.vocab_size,
                b.attention_head_count,
                b.attention_head_count_kv,
                b.attention_key_length,
                b.rope_layout_code()
            ),
            "compact loaded model dimensions/RoPE differ"
        );
        ensure!(
            reference.cuda.uploaded_weights_bytes()? == self.source_weight_bytes
                && candidate.cuda.uploaded_weights_bytes()? == self.compact_weight_bytes
                && !reference.d_weights_base.is_null()
                && !candidate.d_weights_base.is_null(),
            "compact comparison weight allocation differs from actual file storage"
        );
        let a_cache = reference.kv_cache.as_ref().context("source KV")?;
        let b_cache = candidate.kv_cache.as_ref().context("compact KV")?;
        ensure!(
            !a_cache.is_compressed()
                && !b_cache.is_compressed()
                && a_cache.num_heads() == a.attention_head_count_kv
                && b_cache.num_heads() == b.attention_head_count_kv
                && a_cache.head_dim() == a.attention_key_length
                && b_cache.head_dim() == b.attention_key_length
                && a_cache.max_batch_size() == a.block_count
                && b_cache.max_batch_size() == b.block_count
                && a_cache.max_seq_len() == b_cache.max_seq_len()
                && a_cache.actual_bytes() as u128 * u128::from(b.block_count)
                    == b_cache.actual_bytes() as u128 * u128::from(a.block_count)
                && a_cache.actual_bytes() > b_cache.actual_bytes(),
            "compact dense KV must reclaim physical layer slots with unchanged context"
        );
        Ok(())
    }

    pub fn record_task_boundaries(
        &self,
        report: &mut Report,
        split: Split,
        task_id: &str,
        models: (&mut LoadedModel, &mut LoadedModel),
        prompt_ids: &[u32],
        result: &TaskResult,
    ) -> Result<()> {
        ensure!(
            result.has_identical_generation,
            "compact/source generated tokens, output or stop reason differ for {task_id}"
        );
        ensure!(!prompt_ids.is_empty(), "compact prompt must be nonempty");
        let (reference, candidate) = models;
        let mut ids = prompt_ids.to_vec();
        if let Some(first) = result.first_reference_token_id {
            ids.push(first);
        }
        // The terminal target is never forwarded. It exposes full logits after
        // the first sampled token without treating this diagnostic as corpus loss.
        ids.push(0);
        let start = Instant::now();
        let comparison = compare_loaded_models(reference, candidate, &ids)?;
        let final_prompt = comparison
            .tokens
            .get(prompt_ids.len() - 1)
            .context("final prompt comparison")?;
        let first_decode = result
            .first_reference_token_id
            .and_then(|_| comparison.tokens.get(prompt_ids.len()));
        report.append(json!({"kind":"compact_task_logit_parity","split":split,"id":task_id,
            "prompt_tokens":prompt_ids.len(),"first_reference_token_id":result.first_reference_token_id,
            "compared_prefixes":comparison.tokens.len(),"max_abs_logit_diff":comparison.max_abs_logit_diff,
            "mean_abs_logit_diff":comparison.mean_abs_logit_diff,"final_prompt":final_prompt,"first_decode":first_decode,
            "terminal_dummy_target_id":0,"is_corpus_loss":false,
            "scope":"diagnostic full observed-prefix logits including prompt and continuation after first sampled token; even for immediate stop; NOT generation timing",
            "elapsed_ms":start.elapsed().as_secs_f64()*1000.0,
            "reference_memory":run::memory(reference)?,"candidate_memory":run::memory(candidate)?}))?;
        require_identical_logits(&comparison)
    }
}

#[cfg(test)]
#[path = "compact_tests.rs"]
mod tests;
