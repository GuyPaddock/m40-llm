use anyhow::{ensure, Result};
use serde_json::Value;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
    process::Command,
};

pub struct Report {
    file: File,
}

impl Report {
    pub fn create(path: &Path) -> Result<Self> {
        Ok(Self {
            file: OpenOptions::new().create_new(true).write(true).open(path)?,
        })
    }

    pub fn append(&mut self, value: Value) -> Result<()> {
        serde_json::to_writer(&mut self.file, &value)?;
        writeln!(self.file)?;
        self.file.flush()?;
        Ok(())
    }
}

pub fn command_text(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    ensure!(
        output.status.success(),
        "{program} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    /// Completed rows survive a partial report and a second writer cannot
    /// overwrite that evidence. Completion is a separate explicit record.
    #[test]
    fn test_report_preserves_partial_rows_and_rejects_overwrite() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "m40-kv-report-{}-{}.jsonl",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        {
            let mut report = Report::create(&path)?;
            report.append(serde_json::json!({"kind":"generation","text":"first\nsecond"}))?;
            assert!(Report::create(&path).is_err());
        }
        let rows = fs::read_to_string(&path)?;
        let row: Value = serde_json::from_str(rows.trim())?;
        assert_eq!(row["text"], "first\nsecond");
        assert_eq!(rows.lines().count(), 1);
        fs::remove_file(path)?;
        Ok(())
    }
}
