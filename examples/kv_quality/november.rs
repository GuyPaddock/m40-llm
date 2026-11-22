use super::fixture::{render_cases, tokenizer_identity, Manifest, MODEL_SHA256};
use anyhow::{ensure, Result};
use m40_llm::{
    evaluation::{
        corpus::Task,
        identity::{file_sha256, sha256},
    },
    gguf,
    tokenizer::Tokenizer,
};
use std::{fs::OpenOptions, io::Write, path::Path};

const TASKS: &str = include_str!("../../data/research_quality/2026-11-21-kv-tasks.json");

pub fn create(tokenizer: &Tokenizer, identity: String) -> Result<Manifest> {
    let tasks: Vec<Task> = serde_json::from_str(TASKS)?;
    ensure!(tasks.len() == 24, "incomplete November task set");
    Ok(Manifest {
        schema: "m40-november-kv-quality-v1".into(),
        model_sha256: MODEL_SHA256.into(),
        corpus_sha256: sha256(TASKS.as_bytes())?,
        tokenizer_sha256: identity,
        preparation_sha256: sha256(
            concat!(include_str!("fixture.rs"), include_str!("november.rs")).as_bytes(),
        )?,
        cases: render_cases(&tasks, tokenizer)?,
    })
}

pub fn prepare(model: &Path, output: &Path) -> Result<()> {
    ensure!(file_sha256(model)? == MODEL_SHA256, "incorrect model bytes");
    let model = gguf::load_gguf(model)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&model.metadata)?;
    let manifest = create(&tokenizer, tokenizer_identity(&model)?)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    serde_json::to_writer(&mut file, &manifest)?;
    writeln!(file)?;
    file.flush()?;
    println!(
        "cases={} sha256={}",
        manifest.cases.len(),
        file_sha256(output)?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both contexts must tokenize back to the frozen text with the answer
    /// rubric intact and the entire long-context task outside recent history.
    #[test]
    fn test_november_case_coverage() -> Result<()> {
        let tokenizer = Tokenizer::byte_level();
        let manifest = create(&tokenizer, "test".into())?;
        assert_eq!(manifest.cases.len(), 48);
        for pair in manifest.cases.chunks_exact(2) {
            assert_eq!(pair[0].task.id, pair[1].task.id);
            assert_eq!(tokenizer.decode(&pair[0].prompt_ids)?, pair[0].prompt);
            assert_eq!(tokenizer.decode(&pair[1].prompt_ids)?, pair[1].prompt);
            assert!(
                pair[1].task_end_token < pair[1].prompt_ids.len() - super::super::fixture::RECENT
            );
            pair[0].validate_bounds()?;
            pair[1].validate_bounds()?;
        }
        Ok(())
    }
}
