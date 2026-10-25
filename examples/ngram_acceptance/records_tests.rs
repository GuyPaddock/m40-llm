use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_ARTIFACT: AtomicUsize = AtomicUsize::new(0);

struct Artifact(std::path::PathBuf);

impl Artifact {
    fn write(records: &[Value]) -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "m40-ngram-{}-{}.jsonl",
            std::process::id(),
            NEXT_ARTIFACT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        for record in records {
            writeln!(file, "{record}")?;
        }
        Ok(Self(path))
    }
}

impl Drop for Artifact {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn browser_records() -> Vec<Value> {
    let environment: serde_json::Map<_, _> = SELECTED_ENV
        .iter()
        .map(|(k, v)| ((*k).to_owned(), json!(v)))
        .collect();
    let mut records = vec![
        json!({"kind":"manifest","build_profile":"release","kv_mode":"off","context_tokens":1024,"environment":environment,
        "model_sha256":"ab".repeat(32),"prompt_token_ids":vec![1;65],"m40_stop_token_ids":[99]}),
        json!({"kind":"token_trace","runtime":"m40-llm","matches_measured_output":true,"prompt_token_ids":vec![1;65],"generated_token_ids":vec![7;512],"output":"stable"}),
    ];
    for index in 0..5 {
        records.push(json!({"kind":"trial","runtime":"m40-llm","phase":"screen","is_fixed_workload":true,
        "metrics":{"generated_tokens":512,"prompt_tokens":65,"stop_reason":"length","decode_elapsed_ms":60.0,"prefill_elapsed_ms":30.0},
        "client_elapsed_ms":100.0+index as f64,"output":"stable","cache":{"reused_prompt_tokens":0,"evaluated_prompt_tokens":65}}));
    }
    records
}

/// The reader must use actual IDs and validated stable trials, calculate the
/// timing median, and reject altered configuration, cache, output, or prompts.
#[test]
fn test_browser_reader_validates_actual_trace_and_measured_source() -> Result<()> {
    let records = browser_records();
    let artifact = Artifact::write(&records)?;
    let browser = read_browser(&artifact.0)?;
    assert_eq!(browser.case.trace.generated_token_ids, vec![7; 512]);
    assert_eq!(browser.case.baseline.unwrap().http_elapsed_ms, 102.0);
    for (index, key, value) in [
        (1, "matches_measured_output", json!(false)),
        (1, "prompt_token_ids", json!(vec![2; 65])),
        (2, "output", json!("changed")),
        (0, "kv_mode", json!("block-select-exact")),
        (
            2,
            "cache",
            json!({"reused_prompt_tokens":1,"evaluated_prompt_tokens":64}),
        ),
    ] {
        let mut changed = records.clone();
        changed[index][key] = value;
        assert!(
            read_browser(&Artifact::write(&changed)?.0).is_err(),
            "accepted changed {key}"
        );
    }
    let mut changed = records;
    changed[0]["environment"]["M40LLM_Q8_EXECUTION_WEIGHTS"] = json!("split");
    assert!(read_browser(&Artifact::write(&changed)?.0).is_err());
    changed[0]["environment"]["M40LLM_Q8_EXECUTION_WEIGHTS"] = json!("off");
    changed[0]["environment"]["M40LLM_PREFILL_CHUNK_SIZE"] = json!("64");
    assert!(read_browser(&Artifact::write(&changed)?.0).is_err());
    Ok(())
}

/// A formal regression checkpoint requires every frozen case and the same
/// model/configuration; partial captures and a different model cannot qualify.
#[test]
fn test_regression_reader_requires_complete_matching_checkpoint() -> Result<()> {
    let artifact = Artifact::write(&browser_records())?;
    let browser = read_browser(&artifact.0)?;
    let mut records = vec![
        json!({"kind":"ngram_regression_manifest","model_sha256":browser.model_sha256,"configuration":browser.configuration}),
    ];
    for &(label, prompt, _, expected) in REGRESSION_PROMPTS {
        records.push(json!({"kind":"ngram_regression_trace","case":label,"user_prompt":prompt,"prompt_token_ids":[1,2,3],
            "generated_token_ids":[7,99],"stop_ids":[99],"output":expected.unwrap_or("prose"),"is_stable_two_runs":true}));
    }
    assert_eq!(
        read_regressions(&Artifact::write(&records)?.0, &browser)?.len(),
        5
    );
    let mut incomplete = records.clone();
    incomplete.pop();
    assert!(read_regressions(&Artifact::write(&incomplete)?.0, &browser).is_err());
    records[0]["model_sha256"] = json!("different");
    assert!(read_regressions(&Artifact::write(&records)?.0, &browser).is_err());
    Ok(())
}
