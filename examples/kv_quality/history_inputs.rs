use super::fixture::{self, Case, Manifest};
use anyhow::{ensure, Context, Result};
use m40_llm::{evaluation::identity::file_sha256, gguf, tokenizer::Tokenizer};
use std::{fs, io::Write, path::Path};

pub fn cases(model: &Path, inputs: &Path) -> Result<Vec<Case>> {
    ensure!(file_sha256(model)? == fixture::MODEL_SHA256, "wrong model");
    ensure!(
        file_sha256(inputs)? == "18d0c6f69dbdfa3bf976780b5b20e14d06dfec5f32980418a9d4f308da9f1cc3",
        "wrong source grid"
    );
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf::load_gguf(model)?.metadata)?;
    let manifest: Manifest = serde_json::from_slice(&fs::read(inputs)?)?;
    let mut cases = Vec::new();
    for (id, target) in [
        ("nov-crate-long", 2048),
        ("nov-crate-long", 3200),
        ("nov-relay-long", 3200),
        ("nov-fallback-long", 3200),
    ] {
        let mut case = manifest
            .cases
            .iter()
            .find(|case| case.id == id)
            .context("source case missing")?
            .clone();
        while case.prompt_ids.len() < target {
            case.archive_notes += 1;
            ensure!(case.archive_notes < 256, "archive bound exceeded");
            let n = case.archive_notes;
            let offset = case.prompt.find("</archive>").context("archive missing")?;
            case.prompt.insert_str(
                offset,
                &format!(
                    "Archive {n:03}: stone, wool, copper; label {:04}.\n",
                    (n * 73 + 19) % 10000
                ),
            );
            case.prompt_ids = tokenizer.encode_with_specials(&case.prompt, false, false)?;
        }
        ensure!(
            (target..target + 64).contains(&case.prompt_ids.len()),
            "incorrect boundary"
        );
        ensure!(
            tokenizer.decode(&case.prompt_ids)? == case.prompt
                && case.maximum_consumed_length() <= 4096
                && case.task_end_token <= case.prompt_ids.len() - 1024,
            "invalid history fixture"
        );
        case.id = format!("{}-{target}", case.task.id);
        cases.push(case);
    }
    Ok(cases)
}

pub fn prepare(model: &Path, inputs: &Path, output: &Path) -> Result<()> {
    let cases = cases(model, inputs)?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    serde_json::to_writer(&mut file, &cases)?;
    writeln!(file)?;
    file.flush()?;
    println!("cases={} sha256={}", cases.len(), file_sha256(output)?);
    Ok(())
}
