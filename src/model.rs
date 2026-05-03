// src/model.rs
use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tokio_stream::StreamExt;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LocalModel {
    pub name: String,
    pub path: PathBuf, // path to GGUF
    pub size_bytes: u64,
}

pub fn models_root() -> Result<PathBuf> {
    let proj =
        ProjectDirs::from("dev", "guy", "m40-llm").context("Could not determine project dirs")?;
    let dir = proj.data_dir().join("models");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn model_dir(name: &str) -> Result<PathBuf> {
    let root = models_root()?;
    Ok(root.join(name.replace(':', "_")))
}

/// Resolve a CLI model argument.
///
/// Direct GGUF file paths and directories containing `model.gguf` bypass the
/// app data model registry. Otherwise the argument is treated as a pulled model
/// name, preserving the existing `mistral:7b`-style workflow.
pub fn resolve_model_arg(model: &str) -> Result<LocalModel> {
    let path = Path::new(model);
    if path.is_file() {
        let path = path.canonicalize()?;
        let meta = fs::metadata(&path)?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(model)
            .to_string();
        return Ok(LocalModel {
            name,
            path,
            size_bytes: meta.len(),
        });
    }

    let dir_model = path.join("model.gguf");
    if dir_model.is_file() {
        let path = dir_model.canonicalize()?;
        let meta = fs::metadata(&path)?;
        let name = Path::new(model)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(model)
            .to_string();
        return Ok(LocalModel {
            name,
            path,
            size_bytes: meta.len(),
        });
    }

    list_models()?
        .into_iter()
        .find(|m| m.name == model.replace(':', "_"))
        .ok_or_else(|| anyhow::anyhow!("model not found locally or as a path: {model}"))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_temp_dir(test_name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("m40-llm-{test_name}-{}", std::process::id()))
    }

    #[test]
    fn resolve_model_arg_accepts_direct_file() {
        let dir = unique_temp_dir("direct-file");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("tiny.gguf");
        fs::write(&path, b"GGUF").expect("write gguf");

        let local = resolve_model_arg(path.to_str().expect("utf-8 path")).expect("resolve");
        assert_eq!(local.name, "tiny");
        assert_eq!(local.size_bytes, 4);
        assert!(local.path.ends_with("tiny.gguf"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_model_arg_accepts_model_dir() {
        let dir = unique_temp_dir("model-dir");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        fs::write(dir.join("model.gguf"), b"GGUF").expect("write gguf");

        let local = resolve_model_arg(dir.to_str().expect("utf-8 path")).expect("resolve");
        assert_eq!(local.name, dir.file_name().unwrap().to_string_lossy());
        assert_eq!(local.size_bytes, 4);
        assert!(local.path.ends_with("model.gguf"));

        let _ = fs::remove_dir_all(&dir);
    }
}
