use anyhow::{ensure, Context, Result};
use m40_llm::{
    evaluation::{
        corpus::{Corpus, Task},
        identity::{file_sha256, sha256},
    },
    gguf,
    tokenizer::Tokenizer,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

pub const MODEL_SHA256: &str = "4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d";
pub const CORPUS_SHA256: &str = "2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38";
pub const CAPACITY: usize = 3072;
pub const RECENT: usize = 1024;
pub const BLOCK: usize = 32;
const LONG_MIN: usize = 2048;
const LONG_LIMIT: usize = 2112;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextKind {
    Short,
    Long,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub id: String,
    pub context: ContextKind,
    pub task: Task,
    pub prompt: String,
    pub prompt_ids: Vec<u32>,
    pub task_end_token: usize,
    pub archive_notes: usize,
}

impl Case {
    pub fn maximum_consumed_length(&self) -> usize {
        self.prompt_ids.len() + self.task.max_tokens - 1
    }

    pub fn validate_bounds(&self) -> Result<()> {
        ensure!(
            !self.prompt_ids.is_empty() && (1..=160).contains(&self.task.max_tokens),
            "invalid case length/output bound"
        );
        ensure!(
            self.maximum_consumed_length() <= CAPACITY,
            "case exceeds full-coverage capacity"
        );
        ensure!(
            self.maximum_consumed_length()
                .saturating_sub(RECENT)
                .div_ceil(BLOCK)
                <= 64,
            "top64 would omit old blocks"
        );
        match self.context {
            ContextKind::Short => ensure!(
                self.maximum_consumed_length() <= RECENT && self.archive_notes == 0,
                "short case must have no old cache"
            ),
            ContextKind::Long => {
                ensure!(
                    (LONG_MIN..LONG_LIMIT).contains(&self.prompt_ids.len()),
                    "long prompt outside frozen range"
                );
                ensure!(
                    self.task_end_token <= self.prompt_ids.len() - RECENT,
                    "task must be entirely in old cache"
                );
                ensure!(self.archive_notes > 0, "long case missing archive");
                ensure!(
                    (self.prompt_ids.len() - RECENT).div_ceil(BLOCK) > 8,
                    "top8 must omit some old blocks"
                );
            }
        }
        self.task.scoring.validate()
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub model_sha256: String,
    pub corpus_sha256: String,
    pub tokenizer_sha256: String,
    pub preparation_sha256: String,
    pub cases: Vec<Case>,
}

pub fn render_prompt(user: &str) -> String {
    format!("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

fn render_long_task(task: &str, archive: &str) -> String {
    render_prompt(&format!("Complete the task below. Treat the archive as irrelevant background.\n<task>\n{task}\n</task>\n<archive>\n{archive}</archive>\nAnswer the task in <task> only, using its requested format."))
}

fn find_task_end(tokenizer: &Tokenizer, prompt: &str, ids: &[u32]) -> Result<usize> {
    ensure!(
        tokenizer.decode(ids)? == prompt,
        "prompt tokenizer does not round-trip"
    );
    let byte_end = prompt.find("</task>").context("missing task end")? + "</task>".len();
    let (mut low, mut high) = (0, ids.len());
    while low < high {
        let middle = (low + high) / 2;
        let prefix = tokenizer.decode(&ids[..middle])?;
        ensure!(
            prompt.starts_with(&prefix),
            "token prefix is not a prompt prefix"
        );
        if prefix.len() >= byte_end {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    Ok(low)
}

pub fn create_cases(corpus: &Corpus, tokenizer: &Tokenizer) -> Result<Vec<Case>> {
    ensure!(
        corpus.held_out.tasks.len() == 6,
        "expected six frozen held-out tasks"
    );
    let mut cases = Vec::new();
    for task in &corpus.held_out.tasks {
        let prompt = render_prompt(&task.prompt);
        let ids = tokenizer.encode_with_specials(&prompt, false, false)?;
        let short = Case {
            id: format!("{}-short", task.id),
            context: ContextKind::Short,
            task: task.clone(),
            task_end_token: ids.len(),
            prompt,
            prompt_ids: ids,
            archive_notes: 0,
        };
        short.validate_bounds()?;
        cases.push(short);

        let mut archive = String::new();
        let mut count = 0;
        let (prompt, ids) = loop {
            let prompt = render_long_task(&task.prompt, &archive);
            let ids = tokenizer.encode_with_specials(&prompt, false, false)?;
            if ids.len() >= LONG_MIN {
                break (prompt, ids);
            }
            ensure!(count < 256, "archive failed to reach target length");
            count += 1;
            archive.push_str(&format!(
                "Archive {count:03}: stone, wool, copper; label {:04}.\n",
                (count * 73 + 19) % 10000
            ));
        };
        let long = Case {
            id: format!("{}-long", task.id),
            context: ContextKind::Long,
            task: task.clone(),
            task_end_token: find_task_end(tokenizer, &prompt, &ids)?,
            prompt,
            prompt_ids: ids,
            archive_notes: count,
        };
        long.validate_bounds()?;
        cases.push(long);
    }
    Ok(cases)
}

pub fn tokenizer_identity(model: &gguf::GgufModel) -> Result<String> {
    let metadata: BTreeMap<_, _> = model
        .metadata
        .iter()
        .filter(|(key, _)| key.starts_with("tokenizer."))
        .collect();
    sha256(&serde_json::to_vec(&metadata)?)
}

impl Manifest {
    pub fn create(
        corpus: &Corpus,
        tokenizer: &Tokenizer,
        tokenizer_sha256: String,
    ) -> Result<Self> {
        Ok(Self {
            schema: "m40-kv-quality-inputs-v1".into(),
            model_sha256: MODEL_SHA256.into(),
            corpus_sha256: CORPUS_SHA256.into(),
            tokenizer_sha256,
            preparation_sha256: sha256(include_str!("fixture.rs").as_bytes())?,
            cases: create_cases(corpus, tokenizer)?,
        })
    }

    /// Rebuild only deterministic inputs, never using an inference result. This
    /// rejects changed prompts, token IDs, source rubrics and shortened bounds.
    pub fn validate(
        &self,
        corpus: &Corpus,
        tokenizer: &Tokenizer,
        tokenizer_sha256: String,
    ) -> Result<()> {
        let expected = Self::create(corpus, tokenizer, tokenizer_sha256)?;
        ensure!(
            serde_json::to_value(self)? == serde_json::to_value(expected)?,
            "manifest differs from frozen preparation"
        );
        Ok(())
    }
}

pub fn prepare(model_path: &Path, corpus_path: &Path, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "refuse to overwrite inputs");
    ensure!(
        file_sha256(model_path)? == MODEL_SHA256,
        "unexpected model source"
    );
    let bytes = fs::read(corpus_path)?;
    ensure!(sha256(&bytes)? == CORPUS_SHA256, "unexpected corpus source");
    let corpus = Corpus::parse(std::str::from_utf8(&bytes)?)?;
    let gguf = gguf::load_gguf(model_path)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)?;
    ensure!(tokenizer.is_qwen2(), "requires pinned Qwen tokenizer");
    let manifest = Manifest::create(&corpus, &tokenizer, tokenizer_identity(&gguf)?)?;
    manifest.validate(&corpus, &tokenizer, tokenizer_identity(&gguf)?)?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?;
    serde_json::to_writer_pretty(&mut file, &manifest)?;
    writeln!(file)?;
    for case in &manifest.cases {
        println!(
            "{} prompt_tokens={} task_end={} notes={} maximum_consumed={}",
            case.id,
            case.prompt_ids.len(),
            case.task_end_token,
            case.archive_notes,
            case.maximum_consumed_length()
        );
    }
    println!("manifest_sha256={}", file_sha256(output)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Result<Corpus> {
        Corpus::parse(include_str!("../../data/research_quality/v1.json"))
    }

    /// Every original rubric and output bound survives wrapping, short contexts
    /// stay uncompressed, and long tasks lie entirely before the recent window.
    #[test]
    fn test_preparation_preserves_tasks_and_separates_contexts() -> Result<()> {
        let corpus = corpus()?;
        let tokenizer = Tokenizer::byte_level();
        let cases = create_cases(&corpus, &tokenizer)?;
        assert_eq!(cases.len(), 12);
        for (pair, task) in cases.chunks_exact(2).zip(&corpus.held_out.tasks) {
            assert_eq!(pair[0].prompt, render_prompt(&task.prompt));
            assert_eq!(pair[0].context, ContextKind::Short);
            assert_eq!(pair[1].context, ContextKind::Long);
            for case in pair {
                assert_eq!(
                    serde_json::to_value(&case.task)?,
                    serde_json::to_value(task)?
                );
                assert_eq!(tokenizer.decode(&case.prompt_ids)?, case.prompt);
                case.validate_bounds()?;
            }
            let end = tokenizer.decode(&pair[1].prompt_ids[..pair[1].task_end_token])?;
            assert!(end.contains(&format!("{}\n</task>", task.prompt)));
            assert!(pair[1].task_end_token < pair[1].prompt_ids.len() - RECENT);
        }
        Ok(())
    }

    /// Altered IDs, rubrics or source identity cannot turn a different set of
    /// examples into an admitted frozen manifest.
    #[test]
    fn test_manifest_rejects_changed_evidence() -> Result<()> {
        let corpus = corpus()?;
        let tokenizer = Tokenizer::byte_level();
        let original = Manifest::create(&corpus, &tokenizer, "tokenizer".into())?;
        original.validate(&corpus, &tokenizer, "tokenizer".into())?;
        for field in ["token", "prompt", "bound", "model"] {
            let mut changed: Manifest = serde_json::from_value(serde_json::to_value(&original)?)?;
            match field {
                "token" => changed.cases[0].prompt_ids[0] += 1,
                "prompt" => changed.cases[0].task.prompt.push('!'),
                "bound" => changed.cases[0].task.max_tokens += 1,
                _ => changed.model_sha256 = "other".into(),
            }
            assert!(changed
                .validate(&corpus, &tokenizer, "tokenizer".into())
                .is_err());
        }
        Ok(())
    }

    /// Full-coverage and no-old conclusions fail when capacity or task position
    /// violates their mathematical preconditions, even with valid task scoring.
    #[test]
    fn test_bounds_reject_incomplete_coverage_and_recent_tasks() -> Result<()> {
        let mut cases = create_cases(&corpus()?, &Tokenizer::byte_level())?;
        cases[0].prompt_ids.resize(RECENT + 1, 1);
        assert!(cases[0].validate_bounds().is_err());
        cases[1].task_end_token = cases[1].prompt_ids.len();
        assert!(cases[1].validate_bounds().is_err());
        cases[1].prompt_ids.resize(CAPACITY + 1, 1);
        assert!(cases[1].validate_bounds().is_err());
        Ok(())
    }
}
