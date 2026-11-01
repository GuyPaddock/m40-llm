use super::{
    activations::ActivationSetup,
    records::{self, Report},
    research, run, tasks, Args, SplitChoice,
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream},
    evaluation::{
        corpus::{Corpus, Split},
        layer_ranking::{
            LayerRankingPassage, LayerRankingPlan, LayerRemovalRanker, LayerRemovalRanking,
            LayerRemovalTrial,
        },
        loss::LossMetrics,
        teacher_forced::score_loaded_model,
    },
    gguf::GgmlDType,
    infer::{LayerMask, LoadedModel},
    tokenizer::Tokenizer,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub(super) struct RankedVariant {
    ranking: LayerRemovalRanking,
    path: PathBuf,
    sha256: String,
    count: usize,
}

impl RankedVariant {
    pub fn skipped_layers(&self) -> Result<Vec<usize>> {
        self.ranking.selected_layers(self.count)
    }

    pub fn describe(&self) -> Result<Value> {
        Ok(json!({"artifact":self.path,"artifact_sha256":self.sha256,
            "removed_layer_count":self.count,"skipped_original_layer_ids":self.skipped_layers()?,
            "fit_split":"calibration","selection":"iterative calibration mean NLL; BI then original ID for ties",
            "is_model_semantics_change":true}))
    }

    pub fn validate_model(
        &self,
        model: &LoadedModel,
        tokenizer: &Tokenizer,
        corpus: &Corpus,
    ) -> Result<()> {
        let plan = &self.ranking.influence.plan;
        ensure!(
            plan.layer_count == model.model_config.block_count
                && plan.hidden_dim == model.model_config.embedding_length as usize,
            "layer ranking model dimensions differ"
        );
        for (expected, passage) in plan.passages.iter().zip(&corpus.calibration.passages) {
            ensure!(
                expected.token_ids == tasks::encode_passage(tokenizer, &passage.text)?,
                "layer ranking calibration tokenization differs"
            );
        }
        Ok(())
    }
}

pub(super) fn load_variant(
    args: &Args,
    environment: &BTreeMap<String, String>,
    corpus: &Corpus,
    corpus_bytes: &[u8],
) -> Result<Option<RankedVariant>> {
    let Some(path) = &args.layer_ranking else {
        return Ok(None);
    };
    research::validate_execution(environment)?;
    let text = fs::read_to_string(path)?;
    let ranking = LayerRemovalRanking::parse(&text)?;
    let plan = &ranking.influence.plan;
    plan.validate_corpus(corpus)?;
    ensure!(
        plan.model_sha256 == records::file_sha256(&args.model)?
            && plan.model_sha256 == records::file_sha256(&args.reference_model)?
            && plan.corpus_sha256 == records::sha256(corpus_bytes)?
            && plan.execution_sha256 == research::execution_hash(environment, args.context_tokens)?,
        "frozen layer ranking model/corpus/base-execution identity mismatch"
    );
    let count = args
        .remove_layer_count
        .context("provide --remove-layer-count")? as usize;
    ranking.selected_layers(count)?;
    Ok(Some(RankedVariant {
        ranking,
        path: path.clone(),
        sha256: records::sha256(text.as_bytes())?,
        count,
    }))
}

/// This runner exclusively owns the model. Every scoring call has already
/// dropped its session; drain both streams before releasing KV/workspace owners.
fn clear_forward_state(model: &mut LoadedModel) -> Result<()> {
    model.cuda.synchronize_stream(CudaStream::Decode)?;
    model.cuda.synchronize_stream(CudaStream::Prefill)?;
    drop(model.kv_cache.take());
    let workspace = model
        .forward_workspace
        .lock()
        .map_err(|_| anyhow::anyhow!("poisoned ranking workspace"))?
        .take();
    drop(workspace);
    Ok(())
}

fn score_mask(
    model: &mut LoadedModel,
    context: u32,
    mask: &[usize],
    plan: &LayerRankingPlan,
) -> Result<(Vec<LossMetrics>, f64, f64)> {
    LayerMask::new(model.model_config.block_count, mask)?;
    let setup = Instant::now();
    clear_forward_state(model)?;
    model.set_diagnostic_layer_mask(mask)?;
    model.allocate_kv_cache_for_layers(context)?;
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let losses = plan
        .passages
        .iter()
        .map(|passage| score_loaded_model(model, &passage.token_ids))
        .collect::<Result<Vec<_>>>()?;
    Ok((losses, setup_ms, started.elapsed().as_secs_f64() * 1000.0))
}

pub(super) fn rank(args: &Args) -> Result<()> {
    ensure!(
        matches!(args.split, SplitChoice::Calibration)
            && args.step_timing_trials == 0
            && args.skip_layers.is_empty()
            && args.activation_calibration.is_none()
            && args.capture_activations.is_none()
            && args.layer_ranking.is_none(),
        "layer ranking requires --split calibration without model variants or timing trials"
    );
    let environment = run::environment()?;
    research::validate_execution(&environment)?;
    let corpus_bytes = fs::read(&args.corpus)?;
    let corpus = Corpus::parse(std::str::from_utf8(&corpus_bytes)?)?;
    let model_hash = records::file_sha256(&args.model)?;
    ensure!(
        model_hash == records::file_sha256(&args.reference_model)?,
        "rank only an unchanged reference model"
    );
    let destination = args
        .rank_layers
        .as_ref()
        .context("ranking artifact destination")?;
    ensure!(
        !destination.exists(),
        "refuse to overwrite frozen layer ranking"
    );
    ensure!(
        destination != &args.output,
        "ranking artifact and report need different paths"
    );
    let mut report = Report::create(&args.output)?;
    let initial = CudaContext::total_device_bytes();
    let load_start = Instant::now();
    let (mut model, tokenizer) = run::load_model(
        &args.model,
        args.context_tokens,
        &[],
        ActivationSetup::Disabled,
    )?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    run::validate_inputs(
        &corpus,
        &[Split::Calibration],
        &tokenizer,
        &tokenizer,
        args.context_tokens,
        model.model_config.vocab_size,
    )?;
    for layer in 0..model.model_config.block_count as usize {
        let weights = model.map_standard_layer(layer)?;
        ensure!(
            [
                weights.wq.dtype,
                weights.wk.dtype,
                weights.wv.dtype,
                weights.wo.dtype,
                weights.w_gate.dtype,
                weights.w_up.dtype,
                weights.w_down.dtype
            ]
            .iter()
            .all(|&dtype| dtype == GgmlDType::Q8_0),
            "initial layer-ranking runner requires native Q8_0 projections"
        );
    }
    let passages = corpus
        .calibration
        .passages
        .iter()
        .map(|passage| {
            Ok(LayerRankingPassage {
                id: passage.id.clone(),
                token_ids: tasks::encode_passage(&tokenizer, &passage.text)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let plan = LayerRankingPlan::new(
        model_hash,
        records::sha256(&corpus_bytes)?,
        research::execution_hash(&environment, args.context_tokens)?,
        model.model_config.block_count,
        model.model_config.embedding_length as usize,
        &corpus,
        passages,
    )?;
    ensure!(
        args.layer_ranking_removals as usize <= plan.layer_count as usize - 2,
        "too few interior layers"
    );
    let host_budget = args
        .layer_ranking_budget_mb
        .checked_mul(1024 * 1024)
        .context("ranking budget overflow")?;
    clear_forward_state(&mut model)?;
    model.begin_layer_influence_capture(plan.clone(), host_budget)?;
    model.allocate_kv_cache_for_layers(args.context_tokens)?;
    report.append(json!({"kind":"layer_ranking_manifest","schema_version":1,
        "created_unix_s":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),"plan":plan,"variant_id":args.variant_id,
        "model":run::model_identity(&args.model,&model,&tokenizer)?,"environment":environment,
        "revision":records::command_text("git",&["rev-parse","HEAD"])?,"worktree":records::command_text("git",&["status","--short"])?,
        "executable_sha256":records::file_sha256(&std::env::current_exe()?)?,
        "rustc":records::command_text("rustc",&["--version"])?,"nvcc":records::command_text("nvcc",&["--version"])?,
        "host_compiler":records::command_text("g++",&["--version"])?,
        "cuda_flags":research::CUDA_FLAGS,"device":format!("{:?}",model.cuda.current_device_props()?),
        "build_profile":if cfg!(debug_assertions){"dev-test"}else{"release"},
        "cold_load_ms":load_ms,"requested_removals":args.layer_ranking_removals,
        "load_scope":"shared GGUF/tokenizer/file/upload/dense KV/validation; excludes later hashing/capture reconfiguration; not cold-disk I/O",
        "host_capture_budget_bytes":host_budget,"fit_split":"calibration",
        "timing_scope":"diagnostic capture/loss/setup wall time, not generated or HTTP throughput"}))?;
    tasks::warmup(&model, &tokenizer)?;
    let mut baseline = Vec::new();
    for passage in &plan.passages {
        model.begin_layer_influence_passage(&passage.id)?;
        let started = Instant::now();
        let loss = score_loaded_model(&mut model, &passage.token_ids)?;
        model.finish_layer_influence_passage()?;
        report.append(json!({"kind":"layer_influence_passage","id":passage.id,"token_ids":passage.token_ids,
            "loss":loss,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"is_timed_benchmark":false}))?;
        baseline.push(loss);
    }
    let influence = model.finish_layer_influence_capture()?;
    report.append(json!({"kind":"layer_influence_complete","influence":influence,
        "ranked_interior_layer_ids":influence.ranked_layers()?,"reference_loss":LossMetrics::combine(&baseline)?}))?;
    let mut ranker = LayerRemovalRanker::new(influence, args.layer_ranking_removals as usize)?;
    for stage in 0..args.layer_ranking_removals {
        let selected = ranker.selected_layers();
        let mut trials = Vec::new();
        for layer_id in ranker.candidate_layers() {
            let mut mask = selected.clone();
            mask.push(layer_id);
            mask.sort_unstable();
            let (passage_losses, setup_ms, loss_ms) =
                score_mask(&mut model, args.context_tokens, &mask, &plan)?;
            let calibration_loss = LossMetrics::combine(&passage_losses)?;
            report.append(
                json!({"kind":"layer_removal_trial","stage":stage,"candidate_layer_id":layer_id,
                "prior_selected_layer_ids":selected,"skipped_original_layer_ids":mask,
                "passage_losses":passage_losses,"calibration_loss":calibration_loss,
                "setup_elapsed_ms":setup_ms,"loss_elapsed_ms":loss_ms,"memory":run::memory(&model)?,
                "is_timed_benchmark":false,"fit_split":"calibration"}),
            )?;
            eprintln!(
                "calibration removal stage {} layer {layer_id}: PPL={:?}",
                stage + 1,
                calibration_loss.perplexity
            );
            trials.push(LayerRemovalTrial {
                layer_id,
                calibration_loss,
            });
        }
        let winner = ranker.record_stage(trials)?;
        report.append(
            json!({"kind":"layer_removal_stage_complete","stage":stage,"winner_layer_id":winner,
            "skipped_original_layer_ids":ranker.selected_layers(),"fit_split":"calibration"}),
        )?;
    }
    let ranking = ranker.finish()?;
    clear_forward_state(&mut model)?;
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "layer-ranking device allocation leak"
    );
    let mut artifact = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    serde_json::to_writer_pretty(&mut artifact, &ranking)?;
    drop(artifact);
    report.append(json!({"kind":"layer_ranking_complete","artifact":destination,
        "artifact_sha256":records::file_sha256(destination)?,"masks":(1..=ranking.stages.len())
            .map(|count|ranking.selected_layers(count)).collect::<Result<Vec<_>>>()?,
        "device_bytes_before":initial,"device_bytes_after":CudaContext::total_device_bytes(),"is_complete":true}))?;
    Ok(())
}

#[cfg(test)]
#[path = "layer_tests.rs"]
mod tests;
