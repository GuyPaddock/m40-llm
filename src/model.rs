// src/model.rs
use anyhow::{Result, Context};
use directories::ProjectDirs;
use std::{fs, path::PathBuf};
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LocalModel {
    pub name: String,
    pub path: PathBuf,      // path to GGUF
    pub size_bytes: u64,
}

pub fn models_root() -> Result<PathBuf> {
    let proj = ProjectDirs::from("dev", "guy", "m40-llm")
        .context("Could not determine project dirs")?;
    let dir = proj.data_dir().join("models");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn model_dir(name: &str) -> Result<PathBuf> {
    let root = models_root()?;
    Ok(root.join(name.replace(':', "_")))
}

pub async fn pull_model(name: &str, source: Option<String>) -> Result<LocalModel> {
    // For now, just assume `source` is a direct GGUF URL or HF raw URL
    // e.g., https://huggingface.co/.../resolve/main/model.gguf
    let url = source.unwrap_or_else(|| {
        // Placeholder: map `mistral:7b` → a concrete URL
        // In practice, you’d have a registry config.
        match name {
            "mistral:7b" | "mistral:7b-instruct" => {
                "https://example.com/mistral-7b-instruct.Q4_K_M.gguf".to_string()
            }
            _ => panic!("Unknown model alias: {name}. For now, specify --source URL"),
        }
    });

    let dir = model_dir(name)?;
    fs::create_dir_all(&dir)?;

    let target = dir.join("model.gguf");
    println!("Downloading {url} → {}", target.display());

    let resp = reqwest::get(&url).await?;
    if !resp.status().is_success() {
        anyhow::bail!("Download failed: {}", resp.status());
    }

    let mut file = tokio::fs::File::create(&target).await?;
    let mut stream = resp.bytes_stream();

    use tokio::io::AsyncWriteExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
    }
    file.flush().await?;

    let meta = fs::metadata(&target)?;
    Ok(LocalModel {
        name: name.to_string(),
        path: target,
        size_bytes: meta.len(),
    })
}

pub fn list_models() -> Result<Vec<LocalModel>> {
    let root = models_root()?;
    let mut out = Vec::new();
    if !root.exists() {
        return Ok(out);
    }
    for entry in fs::read_dir(&root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let gguf = entry.path().join("model.gguf");
        if gguf.exists() {
            let meta = fs::metadata(&gguf)?;
            out.push(LocalModel {
                name,
                path: gguf,
                size_bytes: meta.len(),
            });
        }
    }
    Ok(out)
}
