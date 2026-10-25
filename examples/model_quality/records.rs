use serde::Serialize;

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityConclusion {
    Preserved,
    Regression,
    Inconclusive,
}

impl QualityConclusion {
    pub fn compare(reference_passed: bool, candidate_passed: bool) -> Self {
        if !reference_passed {
            Self::Inconclusive
        } else if candidate_passed {
            Self::Preserved
        } else {
            Self::Regression
        }
    }
}

#[cfg(all(feature = "cuda", nvcc))]
mod io {
    use anyhow::{ensure, Context, Result};
    use serde_json::Value;
    use std::{
        fs::{File, OpenOptions},
        io::Write,
        path::Path,
        process::{Command, Stdio},
    };

    pub fn command_text(program: &str, args: &[&str]) -> Result<String> {
        let output = Command::new(program).args(args).output()?;
        ensure!(
            output.status.success(),
            "{program}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(String::from_utf8(output.stdout)?.trim().to_owned())
    }

    pub fn file_sha256(path: &Path) -> Result<String> {
        Ok(command_text(
            "sha256sum",
            &[path.to_str().context("UTF-8 path required")?],
        )?
        .split_whitespace()
        .next()
        .context("missing file hash")?
        .to_owned())
    }

    pub fn sha256(bytes: &[u8]) -> Result<String> {
        let mut process = Command::new("sha256sum")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        process
            .stdin
            .take()
            .context("hash stdin")?
            .write_all(bytes)?;
        let output = process.wait_with_output()?;
        ensure!(output.status.success(), "sha256sum failed");
        Ok(String::from_utf8(output.stdout)?
            .split_whitespace()
            .next()
            .context("missing content hash")?
            .to_owned())
    }

    pub struct Report {
        file: File,
    }

    impl Report {
        pub fn create(path: &Path) -> Result<Self> {
            Ok(Self {
                file: OpenOptions::new().write(true).create_new(true).open(path)?,
            })
        }

        pub fn append(&mut self, record: Value) -> Result<()> {
            serde_json::to_writer(&mut self.file, &record)?;
            writeln!(self.file)?;
            self.file.flush()?;
            Ok(())
        }
    }
}

#[cfg(all(feature = "cuda", nvcc))]
pub use io::{command_text, file_sha256, sha256, Report};

#[cfg(test)]
mod tests {
    use super::QualityConclusion;

    /// A dense failure cannot establish a pruning failure or success, even when
    /// the candidate happens to pass. Dense-valid regressions remain visible.
    #[test]
    fn test_quality_conclusions_require_passing_reference() {
        assert_eq!(
            QualityConclusion::compare(false, false),
            QualityConclusion::Inconclusive
        );
        assert_eq!(
            QualityConclusion::compare(false, true),
            QualityConclusion::Inconclusive
        );
        assert_eq!(
            QualityConclusion::compare(true, false),
            QualityConclusion::Regression
        );
        assert_eq!(
            QualityConclusion::compare(true, true),
            QualityConclusion::Preserved
        );
    }

    #[cfg(all(feature = "cuda", nvcc))]
    /// Reports retain completed records and reject existing destinations instead
    /// of silently replacing or appending to another experiment's evidence.
    #[test]
    fn test_report_keeps_rows_and_refuses_overwrite() -> anyhow::Result<()> {
        let destination = crate::test_files::TestReportPath::new();
        let path = destination.path();
        let mut report = super::Report::create(path)?;
        report.append(serde_json::json!({"case":"first","output":"raw\nanswer"}))?;
        report.append(serde_json::json!({"case":"second","passed":false}))?;
        let records = std::fs::read_to_string(path)?
            .lines()
            .map(serde_json::from_str::<serde_json::Value>)
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["output"], "raw\nanswer");
        assert_eq!(records[1]["passed"], false);
        assert!(super::Report::create(path).is_err());
        assert_eq!(
            super::file_sha256(path)?,
            super::sha256(&std::fs::read(path)?)?
        );
        Ok(())
    }
}
