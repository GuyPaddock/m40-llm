use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

pub struct TestReportPath(PathBuf);

impl TestReportPath {
    pub fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "m40-quality-{}-{}-{}.jsonl",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos(),
            NEXT_PATH.fetch_add(1, Ordering::Relaxed)
        )))
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestReportPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
