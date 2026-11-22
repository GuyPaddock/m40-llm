use super::{
    config::{self, CacheArm},
    evidence::{command_text, Report},
    execution::{self, CacheGeometry},
    fixture::Case,
    history_inputs,
    metrics::{compare_logits, measure_repetition},
    prefix::{collect_with_geometry, PrefixObservations},
};
use anyhow::{ensure, Context, Result};
use m40_llm::{cuda::CudaContext, evaluation::identity::file_sha256, infer::LoadedModel};
use serde_json::json;
use std::{collections::BTreeMap, fs, io::Write, path::Path};

const CAPACITY: usize = 4096;

#[derive(Clone, Copy)]
struct Arm {
    label: &'static str,
    consumer: CacheArm,
    block: usize,
}

const DENSE: Arm = Arm {
    label: "dense",
    consumer: CacheArm::Dense,
    block: 32,
};
const FULL: Arm = Arm {
    label: "full_mixed",
    consumer: CacheArm::QuantizedAll,
    block: 64,
};
const TOP64: Arm = Arm {
    label: "top64",
    consumer: CacheArm::QuantizedAll,
    block: 32,
};
const TOP8: Arm = Arm {
    label: "top8",
    consumer: CacheArm::QuantizedTop8,
    block: 32,
};

impl Arm {
    fn geometry(self) -> CacheGeometry {
        CacheGeometry {
            capacity: CAPACITY,
            block_size: self.block,
            requires_full_block_coverage: self.label == FULL.label,
        }
    }
}

fn generate(
    model: &mut LoadedModel,
    case: &Case,
    arm: Arm,
    report: &mut Report,
) -> Result<Vec<u32>> {
    eprintln!("boundary generation {} {}", case.id, arm.label);
    let (generation, dispatch, cache) =
        execution::generate_with_geometry(model, case, arm.consumer, arm.geometry(), CAPACITY)?;
    let ids = generation.token_ids[generation.prompt_token_len..].to_vec();
    let score = case.task.scoring.score(&generation.output)?;
    report.append(json!({"kind":"generation","case":case.id,"arm":arm.label,"ids":ids,"output":generation.output,
        "score":score,"metrics":generation.metrics,"repetition":measure_repetition(&ids),"cache":cache,"dispatch":dispatch}))?;
    Ok(ids)
}

fn observe(
    model: &mut LoadedModel,
    case: &Case,
    arm: Arm,
    ids: &[u32],
    directory: &Path,
    report: &mut Report,
) -> Result<PrefixObservations> {
    eprintln!("boundary prefixes {} {}", case.id, arm.label);
    let observation =
        collect_with_geometry(model, arm.consumer, &case.prompt_ids, ids, arm.geometry())?;
    for (step, values) in observation.steps.iter().zip(&observation.logits) {
        let path = directory.join(format!("{}-{}-{step}.f32le", case.id, arm.label));
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        file.write_all(&bytes)?;
    }
    report.append(json!({"kind":"prefix","case":case.id,"arm":arm.label,"steps":observation.steps,
        "targets":observation.steps.iter().map(|&i|ids[i]).collect::<Vec<_>>(),"dispatch":observation.dispatch,"cache":observation.state}))?;
    Ok(observation)
}

fn compare(
    case: &Case,
    arms: [Arm; 2],
    observations: [&PrefixObservations; 2],
    ids: &[u32],
    report: &mut Report,
) -> Result<bool> {
    let [a, b] = observations;
    ensure!(a.steps == b.steps, "different prediction steps");
    let mut exact = true;
    for ((&step, left), right) in a.steps.iter().zip(&a.logits).zip(&b.logits) {
        let same = left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(x, y)| x.to_bits() == y.to_bits());
        exact &= same;
        report.append(json!({"kind":"comparison","case":case.id,"reference":arms[0].label,"candidate":arms[1].label,
            "step":step,"exact":same,"logits":compare_logits(left,right,ids[step])?}))?;
    }
    Ok(exact)
}

pub fn run(model: &Path, november: &Path, inputs: &Path, output: &Path) -> Result<()> {
    let mut variables: BTreeMap<String, String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("M40LLM_") || k.starts_with("CUDA_"))
        .collect();
    ensure!(
        variables
            .remove("M40LLM_KV_SELECTED_BLOCK_ORDER")
            .as_deref()
            == Some("chronological"),
        "require chronological order"
    );
    config::validate_environment(&variables)?;
    ensure!(
        command_text("git", &["status", "--porcelain"])?.is_empty(),
        "dirty source"
    );
    let cases = history_inputs::cases(model, november)?;
    let frozen: Vec<Case> = serde_json::from_slice(&fs::read(inputs)?)?;
    ensure!(
        serde_json::to_value(&cases)? == serde_json::to_value(&frozen)?,
        "inputs differ from frozen cases"
    );
    let directory = output.with_extension("logits");
    let mut report = Report::create(output)?;
    fs::create_dir(&directory)?;
    let baseline = CudaContext::total_device_bytes();
    {
        let mut loaded = execution::load_weights(model)?;
        report.append(json!({"kind":"manifest","source":command_text("git",&["rev-parse","HEAD"])?,
            "model":file_sha256(model)?,"inputs":file_sha256(inputs)?,"binary":file_sha256(&std::env::current_exe()?)?,
            "order":"chronological","flags":variables,"gpu":format!("{:?}",loaded.cuda.current_device_props()?),"capacity":CAPACITY}))?;
        let references = cases
            .iter()
            .map(|case| generate(&mut loaded, case, DENSE, &mut report))
            .collect::<Result<Vec<_>>>()?;
        for (index, (case, ids)) in cases.iter().zip(references).enumerate() {
            let reference = observe(&mut loaded, case, DENSE, &ids, &directory, &mut report)?;
            let full_ids = generate(&mut loaded, case, FULL, &mut report)?;
            let full = observe(&mut loaded, case, FULL, &ids, &directory, &mut report)?;
            compare(case, [DENSE, FULL], [&reference, &full], &ids, &mut report)?;
            for arm in [TOP64, TOP8] {
                let actual = generate(&mut loaded, case, arm, &mut report)?;
                let partial = observe(&mut loaded, case, arm, &ids, &directory, &mut report)?;
                let blocks = partial.state["old_blocks"]
                    .as_u64()
                    .context("old block count missing")?;
                ensure!((index == 0) == (blocks <= 64), "wrong history regime");
                compare(
                    case,
                    [DENSE, arm],
                    [&reference, &partial],
                    &ids,
                    &mut report,
                )?;
                let same = compare(case, [FULL, arm], [&full, &partial], &ids, &mut report)?;
                if index == 0 && arm.label == TOP64.label {
                    ensure!(
                        full_ids == actual && same,
                        "full-coverage equivalence control failed"
                    );
                }
            }
        }
    }
    ensure!(
        baseline == CudaContext::total_device_bytes(),
        "cache or model leak"
    );
    report.append(json!({"kind":"complete","cases":cases.len(),"generations":cases.len()*4,"initial_bytes":baseline,"final_bytes":CudaContext::total_device_bytes()}))
}
