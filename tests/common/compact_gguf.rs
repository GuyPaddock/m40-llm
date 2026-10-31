#![allow(dead_code)]
//! Synthetic wire-format and evaluation records for CPU export checks only.

use anyhow::Result;
use m40_llm::{
    evaluation::{
        corpus::{Corpus, Split},
        identity::sha256,
        layer_ranking::{
            LayerInfluence, LayerInfluenceValues, LayerRankingPassage, LayerRankingPlan,
            LayerRemovalRanking, LayerRemovalStage, LayerRemovalTrial,
        },
        loss::LossMetrics,
        scoring::ScoringRule,
    },
    gguf::{compact::CompactExportEvidence, GgmlDType, GgufScalar, GgufValue},
};
use serde_json::{json, Value};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

pub struct CompactFixture {
    pub directory: PathBuf,
    pub source: PathBuf,
    pub destination: PathBuf,
    pub ranking: PathBuf,
    pub report: PathBuf,
    pub corpus: PathBuf,
    pub source_bytes: Vec<u8>,
    pub source_metadata: Vec<(String, Vec<u8>)>,
    pub report_rows: Vec<Value>,
}

fn string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn scalar_entry(key: &str, value: GgufScalar) -> (String, Vec<u8>) {
    let mut out = Vec::new();
    string(&mut out, key);
    match value {
        GgufScalar::Str(value) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            string(&mut out, &value);
        }
        GgufScalar::U32(value) => {
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&value.to_le_bytes());
        }
        GgufScalar::U64(value) => {
            out.extend_from_slice(&10u32.to_le_bytes());
            out.extend_from_slice(&value.to_le_bytes());
        }
        _ => panic!("unsupported fixture scalar"),
    }
    (key.into(), out)
}

fn string_array_entry(key: &str, values: &[String]) -> (String, Vec<u8>) {
    let mut out = Vec::new();
    string(&mut out, key);
    out.extend_from_slice(&9u32.to_le_bytes());
    out.extend_from_slice(&8u32.to_le_bytes());
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        string(&mut out, value);
    }
    (key.into(), out)
}

fn source_bytes(
    architecture: &str,
    alignment: usize,
    has_u64_count: bool,
) -> (Vec<u8>, Vec<(String, Vec<u8>)>) {
    let mut entries = vec![
        scalar_entry("general.architecture", GgufScalar::Str(architecture.into())),
        scalar_entry("general.name", GgufScalar::Str("Fixture instruct".into())),
        scalar_entry(
            "general.author",
            GgufScalar::Str("Original fixture author".into()),
        ),
        scalar_entry("general.license", GgufScalar::Str("apache-2.0".into())),
        scalar_entry("general.size_label", GgufScalar::Str("3B".into())),
        scalar_entry("general.parameter_count", GgufScalar::U64(123)),
        scalar_entry("general.quantization_version", GgufScalar::U32(2)),
        scalar_entry(
            &format!("{architecture}.block_count"),
            if has_u64_count {
                GgufScalar::U64(5)
            } else {
                GgufScalar::U32(5)
            },
        ),
        scalar_entry(
            &format!("{architecture}.embedding_length"),
            GgufScalar::U32(32),
        ),
        scalar_entry("tokenizer.ggml.model", GgufScalar::Str("gpt2".into())),
        scalar_entry("tokenizer.ggml.pre", GgufScalar::Str("qwen2".into())),
        scalar_entry(
            "tokenizer.chat_template",
            GgufScalar::Str(
                "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}".into(),
            ),
        ),
        string_array_entry(
            "tokenizer.ggml.tokens",
            &(0..64).map(|id| format!("token-{id}")).collect::<Vec<_>>(),
        ),
        string_array_entry("tokenizer.ggml.merges", &["a b".into(), "\u{120} t".into()]),
    ];
    if alignment != 32 {
        entries.push(scalar_entry(
            "general.alignment",
            GgufScalar::U32(alignment as u32),
        ));
    }
    let mut empty = Vec::new();
    string(&mut empty, "fixture.empty_float64");
    empty.extend_from_slice(&9u32.to_le_bytes());
    empty.extend_from_slice(&12u32.to_le_bytes());
    empty.extend_from_slice(&0u64.to_le_bytes());
    entries.push(("fixture.empty_float64".into(), empty));
    let mut nan = Vec::new();
    string(&mut nan, "fixture.nan_payload");
    nan.extend_from_slice(&6u32.to_le_bytes());
    nan.extend_from_slice(&0x7fc0_1234u32.to_le_bytes());
    entries.push(("fixture.nan_payload".into(), nan));
    let mut tensors = vec![(
        "token_embd.weight".to_owned(),
        GgmlDType::F16,
        vec![32u64, 64u64],
    )];
    tensors.extend((0..5).map(|layer| {
        (
            format!("blk.{layer}.attn_q.weight"),
            GgmlDType::Q8_0,
            vec![32, 32],
        )
    }));
    tensors.push(("output_norm.weight".into(), GgmlDType::F32, vec![32]));
    tensors.push(("output.weight".into(), GgmlDType::Q8_0, vec![32, 64]));
    let mut payload = Vec::new();
    let mut descriptors = Vec::new();
    for (index, (name, dtype, shape)) in tensors.iter().enumerate() {
        payload.resize(payload.len().div_ceil(alignment) * alignment, 0);
        string(&mut descriptors, name);
        descriptors.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dim in shape {
            descriptors.extend_from_slice(&dim.to_le_bytes());
        }
        let elements: u64 = shape.iter().product();
        let (tag, bytes) = match dtype {
            GgmlDType::F32 => (0u32, elements * 4),
            GgmlDType::F16 => (1, elements * 2),
            GgmlDType::Q8_0 => (8, elements / 32 * 34),
            _ => unreachable!(),
        };
        descriptors.extend_from_slice(&tag.to_le_bytes());
        descriptors.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        payload.extend((0..bytes).map(|offset| offset.wrapping_add(index as u64 * 17) as u8));
    }
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (_, entry) in &entries {
        out.extend_from_slice(entry);
    }
    out.extend_from_slice(&descriptors);
    out.resize(out.len().div_ceil(alignment) * alignment, 0);
    out.extend_from_slice(&payload);
    (out, entries)
}

impl CompactFixture {
    pub fn new(architecture: &str, alignment: usize, has_u64_count: bool) -> Result<Self> {
        let directory = std::env::temp_dir().join(format!(
            "m40-compact-{}-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory)?;
        let mut fixture = Self {
            source: directory.join("source.gguf"),
            destination: directory.join("compact.gguf"),
            ranking: directory.join("ranking.json"),
            report: directory.join("quality.jsonl"),
            corpus: directory.join("corpus.json"),
            directory,
            source_bytes: Vec::new(),
            source_metadata: Vec::new(),
            report_rows: Vec::new(),
        };
        (fixture.source_bytes, fixture.source_metadata) =
            source_bytes(architecture, alignment, has_u64_count);
        fs::write(&fixture.source, &fixture.source_bytes)?;
        let corpus_bytes = include_bytes!("../../data/research_quality/v1.json");
        fs::write(&fixture.corpus, corpus_bytes)?;
        let corpus = Corpus::parse(std::str::from_utf8(corpus_bytes)?)?;
        let corpus_hash = sha256(corpus_bytes)?;
        let model_hash = sha256(&fixture.source_bytes)?;
        let plan = LayerRankingPlan::new(
            model_hash.clone(),
            corpus_hash.clone(),
            "a".repeat(64),
            5,
            32,
            &corpus,
            corpus
                .calibration
                .passages
                .iter()
                .map(|passage| LayerRankingPassage {
                    id: passage.id.clone(),
                    token_ids: vec![1, 2, 3],
                })
                .collect(),
        )?;
        let ranking = LayerRemovalRanking {
            schema_version: 1,
            influence: LayerInfluence {
                schema_version: 1,
                plan,
                layers: (0..5)
                    .map(|layer_id| LayerInfluenceValues {
                        layer_id,
                        block_influence: vec![0.1 + layer_id as f64 / 10.0; 4],
                    })
                    .collect(),
            },
            stages: vec![LayerRemovalStage {
                winner_layer_id: 1,
                trials: (1..4)
                    .map(|layer_id| {
                        Ok(LayerRemovalTrial {
                            layer_id,
                            calibration_loss: LossMetrics::from_sum(
                                8.0 + layer_id as f64 * 4.0,
                                4,
                            )?,
                        })
                    })
                    .collect::<Result<_>>()?,
            }],
        };
        ranking.validate()?;
        let ranking_bytes = serde_json::to_vec_pretty(&ranking)?;
        fs::write(&fixture.ranking, &ranking_bytes)?;
        let ranking_hash = sha256(&ranking_bytes)?;
        let configured = json!({"artifact_sha256":ranking_hash,"removed_layer_count":1,"skipped_original_layer_ids":[1],"fit_split":"calibration"});
        let mut rows = vec![
            json!({"kind":"model_quality_manifest","schema_version":1,"build_profile":"release","selected_splits":["calibration","held_out"],
            "corpus_sha256":corpus_hash,"has_identical_model_source":true,
            "reference":{"sha256":model_hash,"layers":5,"skipped_layer_ids":[]},
            "candidate":{"sha256":model_hash,"layers":5,"d_model":32,"skipped_layer_ids":[1],"active_layer_ids":[0,2,3,4]},
            "configuration":{"kv_mode":"off","candidate_layer_ranking":configured,"candidate_activation_calibration":null},
            "in_process_transforms":[{"kind":"diagnostic_layer_mask","skipped_layer_ids":[1],"calibration_ranking":configured}]}),
        ];
        for split in [Split::Calibration, Split::HeldOut] {
            let items = corpus.split(split);
            let mut reference_passes = 0;
            for task in &items.tasks {
                let reference_output = match &task.scoring {
                    ScoringRule::Exact { answers } => &answers[0],
                    _ => "",
                };
                let reference = task.scoring.score(reference_output)?;
                let candidate = task.scoring.score("")?;
                reference_passes += usize::from(reference.passed);
                rows.push(json!({"kind":"model_quality_task","id":task.id,"split":split,"family":task.family,"max_tokens":task.max_tokens,
                    "user_prompt":task.prompt,"scoring_rule":task.scoring,"reference":{"output":reference_output},"candidate":{"output":""},
                    "reference_score":reference,"candidate_score":candidate,"quality_conclusion":if reference.passed { "regression" } else { "inconclusive" }}));
            }
            let (reference_nll, candidate_nll) = if split == Split::Calibration {
                (4.0, 6.0)
            } else {
                (5.0, 7.0)
            };
            for passage in &items.passages {
                rows.push(json!({"kind":"model_quality_passage","id":passage.id,"split":split,"text":passage.text,"token_ids":[1,2,3],
                    "comparison":{"input_tokens":3,"reference":LossMetrics::from_sum(reference_nll,2)?,"candidate":LossMetrics::from_sum(candidate_nll,2)?}}));
            }
            rows.push(json!({"kind":"model_quality_summary","split":split,"tasks":items.tasks.len(),"dense_reference_passes":reference_passes,
                "candidate_passes":0,"dense_valid_regressions":reference_passes,"inconclusive_tasks":items.tasks.len()-reference_passes,
                "reference_loss":LossMetrics::from_sum(reference_nll * 2.0,4)?,"candidate_loss":LossMetrics::from_sum(candidate_nll * 2.0,4)?}));
        }
        rows.push(json!({"kind":"model_quality_complete","is_complete":true,"device_bytes_before":0,"device_bytes_after":0}));
        fixture.report_rows = rows;
        fixture.write_report(&fixture.report_rows)?;
        Ok(fixture)
    }

    pub fn write_report(&self, rows: &[Value]) -> Result<()> {
        let text = rows
            .iter()
            .map(serde_json::to_string)
            .collect::<Result<Vec<_>, _>>()?
            .join("\n")
            + "\n";
        fs::write(&self.report, text)?;
        Ok(())
    }

    pub fn evidence(&self) -> Result<CompactExportEvidence> {
        CompactExportEvidence::load(&self.ranking, &self.report, &self.corpus, 1)
    }

    pub fn has_partial_files(&self) -> Result<bool> {
        for entry in fs::read_dir(&self.directory)? {
            if entry?.file_name().to_string_lossy().ends_with(".partial") {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl Drop for CompactFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

pub fn get_u64(model: &m40_llm::gguf::GgufModel, key: &str) -> u64 {
    match &model.metadata[key] {
        GgufValue::Scalar(GgufScalar::U32(value)) => u64::from(*value),
        GgufValue::Scalar(GgufScalar::U64(value)) => *value,
        _ => panic!("expected unsigned metadata"),
    }
}
