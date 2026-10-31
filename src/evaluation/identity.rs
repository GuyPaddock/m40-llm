//! Research artifact identities using the project's existing coreutils dependency.

use anyhow::{ensure, Context, Result};
use std::{
    io::Write,
    path::Path,
    process::{Command, Output, Stdio},
};

fn parse_digest(output: Output) -> Result<String> {
    ensure!(
        output.status.success(),
        "sha256sum failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let digest = output.stdout.get(..64).context("missing SHA-256 digest")?;
    ensure!(
        digest.iter().all(u8::is_ascii_hexdigit) && output.stdout.get(64) == Some(&b' '),
        "invalid sha256sum output"
    );
    Ok(std::str::from_utf8(digest)?.to_ascii_lowercase())
}

/// Hash a file without loading its payload into host memory. GNU coreutils'
/// zero-delimited output avoids filename escaping, including newlines/backslashes.
pub fn file_sha256(path: &Path) -> Result<String> {
    parse_digest(
        Command::new("sha256sum")
            .args(["--zero", "--binary", "--"])
            .arg(path)
            .output()
            .context("research identity requires coreutils sha256sum")?,
    )
}

pub fn sha256(bytes: &[u8]) -> Result<String> {
    let mut process = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("research identity requires coreutils sha256sum")?;
    let written = process.stdin.take().context("hash stdin")?.write_all(bytes);
    let output = process.wait_with_output()?;
    written?;
    parse_digest(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    /// Stable published empty/ASCII vectors protect the digest itself rather
    /// than comparing two equally wrong artifact-identity implementations.
    #[test]
    fn test_content_sha256_matches_known_vectors() -> Result<()> {
        assert_eq!(
            sha256(b"")?,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256(b"abc")?,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        Ok(())
    }

    /// File and content identities agree for binary input even when the filename
    /// needs coreutils escaping; missing files must not produce a digest.
    #[test]
    fn test_file_sha256_handles_binary_and_escaped_paths() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "m40-hash-{}-{}-\\\n.bin",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        let bytes: Vec<u8> = (0..262_144).map(|index| index as u8).collect();
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(&bytes)?;
        drop(file);
        let result = file_sha256(&path);
        fs::remove_file(&path)?;
        assert_eq!(result?, sha256(&bytes)?);
        assert!(file_sha256(&path).is_err());
        Ok(())
    }
}
