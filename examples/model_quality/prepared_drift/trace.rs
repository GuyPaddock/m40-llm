use super::super::records;
use anyhow::{ensure, Context, Result};
use m40_llm::infer::execution_weights::Q8ExecutionConfig;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

pub(super) fn candidate_config() -> Result<Q8ExecutionConfig> {
    Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))
}

pub(super) struct PinnedTrace {
    pub path: PathBuf,
    pub sha256: String,
    pub manifest: Value,
    pub prompt: String,
    pub prompt_ids: Vec<u32>,
    pub reference_ids: Vec<u32>,
    pub prepared_ids: Vec<u32>,
    pub first_difference: usize,
    pub sample_steps: BTreeSet<usize>,
}

fn unique<'a>(rows: &'a [Value], kind: &str) -> Result<&'a Value> {
    let matches: Vec<_> = rows.iter().filter(|row| row["kind"] == kind).collect();
    ensure!(matches.len() == 1, "need exactly one {kind} record");
    Ok(matches[0])
}

impl PinnedTrace {
    pub fn load(
        path: &Path,
        model_hash: &str,
        context: u32,
        environment: &BTreeMap<String, String>,
    ) -> Result<Self> {
        let mut bytes = Vec::new();
        File::open(path)?
            .take(8 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() <= 8 * 1024 * 1024,
            "prefix trace report exceeds 8 MiB"
        );
        Self::parse(path, &bytes, model_hash, context, environment)
    }

    pub(super) fn parse(
        path: &Path,
        bytes: &[u8],
        model_hash: &str,
        context: u32,
        environment: &BTreeMap<String, String>,
    ) -> Result<Self> {
        let text = std::str::from_utf8(bytes)?;
        ensure!(
            text.lines().count() <= 4096,
            "prefix trace has too many records"
        );
        let rows: Vec<Value> = text
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        let manifest = unique(&rows, "manifest")?;
        ensure!(
            manifest["model_sha256"] == model_hash
                && manifest["context_tokens"] == context
                && manifest["kv_mode"] == "off"
                && manifest["build_profile"] == "release"
                && manifest["worktree"] == ""
                && manifest["candidate_q8_config"] == serde_json::to_value(candidate_config()?)?
                && manifest["control_q8_execution"]["mode"] == "off",
            "prefix trace model/context/backend identity mismatch"
        );
        let mut current = environment.clone();
        if current
            .get("M40LLM_Q8_EXECUTION_WEIGHTS")
            .is_some_and(|value| value == "off")
        {
            current.remove("M40LLM_Q8_EXECUTION_WEIGHTS");
        }
        let mut previous: BTreeMap<String, String> =
            serde_json::from_value(manifest["environment"].clone())?;
        if previous
            .get("M40LLM_Q8_EXECUTION_WEIGHTS")
            .is_some_and(|value| value == "off")
        {
            previous.remove("M40LLM_Q8_EXECUTION_WEIGHTS");
        }
        ensure!(
            current == previous,
            "prefix trace base execution differs; no implicit kernel/stream change"
        );
        let setup = unique(&rows, "q8_execution_setup")?;
        ensure!(
            setup["candidate_execution"]["mode"] == "split"
                && setup["candidate_execution"]["families"] == json!(["down", "vocabulary"]),
            "trace is not the prepared down/vocabulary comparison"
        );
        ensure!(
            unique(&rows, "q8_promotion_decision")?["actual_generated_ids_equal"] == false
                && rows
                    .last()
                    .is_some_and(|row| row["kind"] == "q8_promotion_decision"),
            "need the completed, unresolved prepared comparison"
        );
        let traces: Vec<_> = rows
            .iter()
            .filter(|row| row["kind"] == "token_trace")
            .collect();
        ensure!(traces.len() == 2, "need exactly two actual token traces");
        let extract =
            |variant: &str| -> Result<(Vec<u32>, Vec<u32>)> {
                let matching: Vec<_> = traces
                    .iter()
                    .filter(|row| row["model_variant"] == variant)
                    .collect();
                ensure!(
                    matching.len() == 1,
                    "missing/duplicate {variant} token trace"
                );
                let row = matching[0];
                ensure!(row["runtime"] == "m40-llm" && row["matches_measured_output"] == true
                && row["is_timed"] == false && row["stage_profile_enabled"] == false,
                "trace must contain actual verified IDs, not timed profiler/re-tokenized output");
                Ok((
                    serde_json::from_value(row["prompt_token_ids"].clone())?,
                    serde_json::from_value(row["generated_token_ids"].clone())?,
                ))
            };
        let (prompt_ids, reference_ids) = extract("control")?;
        let (prepared_prompt, prepared_ids) = extract("prepared")?;
        let prompt = manifest["prompt"]
            .as_str()
            .context("missing recorded prompt")?
            .to_owned();
        ensure!(
            !prompt_ids.is_empty()
                && prompt_ids == prepared_prompt
                && manifest["prompt_token_ids"] == serde_json::to_value(&prompt_ids)?
                && manifest["prompt_sha256"] == records::sha256(prompt.as_bytes())?
                && manifest["prompt_token_ids_sha256"]
                    == records::sha256(&serde_json::to_vec(&prompt_ids)?)?
                && !reference_ids.is_empty()
                && reference_ids.len() == prepared_ids.len()
                && manifest["max_tokens"].as_u64() == Some(reference_ids.len() as u64)
                && prompt_ids
                    .len()
                    .checked_add(reference_ids.len())
                    .is_some_and(|count| count <= context as usize),
            "incomplete or inconsistent prompt/generated token trace"
        );
        let first_difference = reference_ids
            .iter()
            .zip(&prepared_ids)
            .position(|(a, b)| a != b)
            .context("recorded traces do not branch")?;
        let comparison = unique(&rows, "q8_token_comparison")?;
        ensure!(
            comparison["first_different_generated_step"].as_u64() == Some(first_difference as u64)
                && comparison["different_id_count"].as_u64()
                    == Some(
                        reference_ids
                            .iter()
                            .zip(&prepared_ids)
                            .filter(|(a, b)| a != b)
                            .count() as u64
                    )
                && comparison["control_generated_tokens"].as_u64()
                    == Some(reference_ids.len() as u64)
                && comparison["candidate_generated_tokens"].as_u64()
                    == Some(prepared_ids.len() as u64)
                && comparison["is_equal"] == false,
            "recorded token divergence metadata disagrees with actual IDs"
        );
        let last = first_difference
            .saturating_add(6)
            .min(reference_ids.len() - 1);
        let mut sample_steps: BTreeSet<_> = (first_difference.saturating_sub(2)..=last).collect();
        sample_steps.extend([0, 128, 256].into_iter().filter(|step| *step <= last));
        Ok(Self {
            path: path.to_owned(),
            sha256: records::sha256(bytes)?,
            manifest: manifest.clone(),
            prompt,
            prompt_ids,
            reference_ids,
            prepared_ids,
            first_difference,
            sample_steps,
        })
    }

    pub fn describe(&self) -> Value {
        json!({"path":self.path,"sha256":self.sha256,"historical_revision":self.manifest["revision"],
            "first_different_generated_step":self.first_difference,"sample_steps":self.sample_steps,
            "prompt_tokens":self.prompt_ids.len(),"recorded_generated_tokens":self.reference_ids.len(),
            "history":"recorded native IDs only; targets never enter their own prefix"})
    }
}
