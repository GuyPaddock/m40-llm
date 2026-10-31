use super::scoring::ScoringRule;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const BROWSER_PROMPT: &str = "Explain everything that happens between the time a user types a web address into their browser and the time the page is fully loaded and displayed on their screen in as much detail as possible";
const BROWSER_ANSWER_PREFIX: &str = "When a user types a web address into their browser and the page is fully loaded and displayed on their screen, a series of complex and often automated processes occur.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskFamily {
    Chat,
    FactualRetrieval,
    MultiFact,
    CodeConfig,
    Arithmetic,
    LongForm,
}

impl TaskFamily {
    const ALL: [Self; 6] = [
        Self::Chat,
        Self::FactualRetrieval,
        Self::MultiFact,
        Self::CodeConfig,
        Self::Arithmetic,
        Self::LongForm,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Calibration,
    HeldOut,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub family: TaskFamily,
    pub prompt: String,
    pub max_tokens: usize,
    pub scoring: ScoringRule,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Passage {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationSplit {
    pub tasks: Vec<Task>,
    pub passages: Vec<Passage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Corpus {
    pub schema_version: u32,
    pub corpus_id: String,
    pub provenance: String,
    pub calibration: EvaluationSplit,
    pub held_out: EvaluationSplit,
}

fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn reject_benchmark_content(text: &str) -> Result<()> {
    let normalized = normalize(text);
    ensure!(
        !normalized.contains(&normalize(BROWSER_PROMPT))
            && !normalized.contains(&normalize(BROWSER_ANSWER_PREFIX)),
        "reserved browser benchmark content"
    );
    Ok(())
}

fn reject_benchmark_value(value: &serde_json::Value) -> Result<()> {
    match value {
        serde_json::Value::String(text) => reject_benchmark_content(text)?,
        serde_json::Value::Array(values) => {
            for value in values {
                reject_benchmark_value(value)?;
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values() {
                reject_benchmark_value(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

impl Corpus {
    pub fn parse(json: &str) -> Result<Self> {
        let corpus: Self = serde_json::from_str(json)?;
        corpus.validate()?;
        Ok(corpus)
    }

    pub fn split(&self, split: Split) -> &EvaluationSplit {
        match split {
            Split::Calibration => &self.calibration,
            Split::HeldOut => &self.held_out,
        }
    }

    /// Validate the data contract and reject exact/normalized leakage, including
    /// reserved browser prompt/answer material. Provenance still requires review;
    /// this is not a semantic detector for paraphrased contamination.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1 && !self.corpus_id.is_empty() && !self.provenance.is_empty(),
            "invalid corpus identity"
        );
        let mut ids = HashSet::new();
        let mut contents = HashSet::new();
        for split in [&self.calibration, &self.held_out] {
            ensure!(
                !split.passages.is_empty(),
                "each split needs fixed loss passages"
            );
            let families: HashSet<_> = split.tasks.iter().map(|task| task.family).collect();
            ensure!(
                TaskFamily::ALL
                    .iter()
                    .all(|family| families.contains(family)),
                "each split must cover all six task families"
            );
            for (id, text) in split
                .tasks
                .iter()
                .map(|task| (&task.id, &task.prompt))
                .chain(
                    split
                        .passages
                        .iter()
                        .map(|passage| (&passage.id, &passage.text)),
                )
            {
                ensure!(
                    !id.is_empty() && ids.insert(id),
                    "duplicate/empty corpus id: {id}"
                );
                let normalized = normalize(text);
                ensure!(
                    !normalized.is_empty() && contents.insert(normalized.clone()),
                    "duplicate/empty content across evaluation items: {id}"
                );
                reject_benchmark_content(text)?;
            }
            for task in &split.tasks {
                ensure!(
                    (1..=512).contains(&task.max_tokens),
                    "invalid output bound in {}",
                    task.id
                );
                task.scoring.validate()?;
                reject_benchmark_value(&serde_json::to_value(&task.scoring)?)?;
            }
        }
        Ok(())
    }
}
