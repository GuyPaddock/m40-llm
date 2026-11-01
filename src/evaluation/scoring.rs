use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id: String,
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScoringRule {
    Exact {
        answers: Vec<String>,
    },
    JsonObject {
        expected: Value,
    },
    Concepts {
        facts: Vec<Fact>,
        min_words: usize,
        min_paragraphs: usize,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TaskScore {
    pub passed: bool,
    pub score: f64,
    pub matched: Vec<String>,
    pub missing: Vec<String>,
    pub format_valid: bool,
}

fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn strip_fence(text: &str) -> &str {
    let text = text.trim();
    if let Some(body) = text
        .strip_prefix("```")
        .and_then(|v| v.split_once('\n'))
        .map(|(_, body)| body)
    {
        body.strip_suffix("```").map(str::trim).unwrap_or(text)
    } else {
        text
    }
}

fn equal_value(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::String(a), Value::String(b)) => normalize(a) == normalize(b),
        _ => a == b,
    }
}

fn contains_phrase(output: &str, phrase: &str) -> bool {
    let words = |value: &str| {
        value
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(str::to_lowercase)
            .collect::<Vec<_>>()
    };
    let output = words(output);
    let phrase = words(phrase);
    !phrase.is_empty() && output.windows(phrase.len()).any(|window| window == phrase)
}

impl ScoringRule {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Exact { answers } => ensure!(
                !answers.is_empty() && answers.iter().all(|answer| !answer.trim().is_empty()),
                "empty exact-answer rule"
            ),
            Self::JsonObject { expected } => ensure!(
                expected.as_object().is_some_and(|object| !object.is_empty()
                    && object
                        .values()
                        .all(|value| value.is_string() || value.is_number() || value.is_boolean())),
                "JSON rule requires nonempty scalar-valued object"
            ),
            Self::Concepts {
                facts,
                min_words,
                min_paragraphs,
            } => {
                ensure!(
                    !facts.is_empty() && *min_words > 0 && *min_paragraphs > 0,
                    "invalid concept rubric"
                );
                let unique: std::collections::HashSet<_> =
                    facts.iter().map(|fact| &fact.id).collect();
                ensure!(
                    unique.len() == facts.len()
                        && facts.iter().all(|fact| !fact.id.is_empty()
                            && !fact.aliases.is_empty()
                            && fact.aliases.iter().all(|v| !normalize(v).is_empty())),
                    "invalid/duplicate concept identifiers"
                );
            }
        }
        Ok(())
    }

    /// Deterministic surface rules, not a semantic judge. JSON accepts optional
    /// fences but rejects extra keys; concepts use word-bounded aliases and
    /// explicit length/paragraph checks rather than mere nonempty fluency.
    pub fn score(&self, output: &str) -> Result<TaskScore> {
        self.validate()?;
        let mut matched = Vec::new();
        let mut missing = Vec::new();
        let format_valid;
        let total;
        match self {
            Self::Exact { answers } => {
                let passed = answers
                    .iter()
                    .any(|answer| normalize(output) == normalize(answer));
                if passed {
                    matched.push("answer".into());
                } else {
                    missing.push("answer".into());
                }
                total = 1;
                format_valid = true;
            }
            Self::JsonObject { expected } => {
                let actual = serde_json::from_str::<Value>(strip_fence(output)).ok();
                let expected = expected.as_object().expect("validated rule");
                let actual = actual.as_ref().and_then(Value::as_object);
                format_valid = actual.is_some_and(|object| object.len() == expected.len());
                for (key, value) in expected {
                    if actual
                        .and_then(|object| object.get(key))
                        .is_some_and(|actual| equal_value(actual, value))
                    {
                        matched.push(key.clone());
                    } else {
                        missing.push(key.clone());
                    }
                }
                total = expected.len();
            }
            Self::Concepts {
                facts,
                min_words,
                min_paragraphs,
            } => {
                format_valid = output.split_whitespace().count() >= *min_words
                    && output
                        .replace("\r\n", "\n")
                        .trim()
                        .split("\n\n")
                        .filter(|p| !p.trim().is_empty())
                        .count()
                        >= *min_paragraphs;
                for fact in facts {
                    if fact
                        .aliases
                        .iter()
                        .any(|alias| contains_phrase(output, alias))
                    {
                        matched.push(fact.id.clone());
                    } else {
                        missing.push(fact.id.clone());
                    }
                }
                total = facts.len();
            }
        }
        Ok(TaskScore {
            passed: format_valid && missing.is_empty(),
            score: matched.len() as f64 / total.max(1) as f64,
            matched,
            missing,
            format_valid,
        })
    }
}
