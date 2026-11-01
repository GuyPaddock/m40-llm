use super::*;
use clap::Parser;
use m40_llm::evaluation::layer_ranking::{LayerInfluence, LayerInfluenceValues};

fn plan(corpus: &Corpus, model_hash: String, execution_hash: String) -> Result<LayerRankingPlan> {
    let passages = corpus
        .calibration
        .passages
        .iter()
        .map(|passage| LayerRankingPassage {
            id: passage.id.clone(),
            token_ids: vec![2, 3, 4],
        })
        .collect();
    LayerRankingPlan::new(
        model_hash,
        records::sha256(include_bytes!("../../data/research_quality/v1.json"))?,
        execution_hash,
        6,
        64,
        corpus,
        passages,
    )
}

fn ranking(plan: LayerRankingPlan) -> Result<LayerRemovalRanking> {
    let rows = plan.observed_rows()?;
    let layers = (0..6)
        .map(|layer_id| LayerInfluenceValues {
            layer_id,
            block_influence: vec![layer_id as f64 / 10.0; rows],
        })
        .collect();
    let mut ranker = LayerRemovalRanker::new(
        LayerInfluence {
            schema_version: 1,
            plan,
            layers,
        },
        3,
    )?;
    for _ in 0..3 {
        ranker.record_stage(
            ranker
                .candidate_layers()
                .into_iter()
                .map(|layer_id| {
                    Ok(LayerRemovalTrial {
                        layer_id,
                        calibration_loss: LossMetrics::from_sum(
                            layer_id as f64 * rows as f64,
                            rows,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        )?;
    }
    ranker.finish()
}

/// Application binds the frozen masks to original file/corpus/execution hashes,
/// retains artifact lineage and rejects changed context/configuration before CUDA.
#[test]
fn test_ranked_variant_identity_and_frozen_mask_preflight() -> Result<()> {
    let model = crate::test_files::TestReportPath::new();
    fs::write(model.path(), b"fixture-model")?;
    let artifact = crate::test_files::TestReportPath::new();
    let environment = BTreeMap::from([("M40LLM_Q8_DECODE_STREAM".into(), "decode".into())]);
    let corpus = Corpus::parse(include_str!("../../data/research_quality/v1.json"))?;
    let mut ranking = ranking(plan(
        &corpus,
        records::file_sha256(model.path())?,
        research::execution_hash(&environment, 1024)?,
    )?)?;
    serde_json::to_writer(fs::File::create(artifact.path())?, &ranking)?;
    let mut args = Args::try_parse_from([
        "model_quality",
        "--reference-model",
        model.path().to_str().unwrap(),
        "--model",
        model.path().to_str().unwrap(),
        "--variant-id",
        "test",
        "--output",
        "new.jsonl",
        "--layer-ranking",
        artifact.path().to_str().unwrap(),
        "--remove-layer-count",
        "2",
    ])?;
    let corpus_bytes = include_bytes!("../../data/research_quality/v1.json");
    let variant = load_variant(&args, &environment, &corpus, corpus_bytes)?.unwrap();
    assert_eq!(variant.skipped_layers()?, [1, 2]);
    assert_eq!(
        variant.describe()?["artifact_sha256"],
        records::file_sha256(artifact.path())?
    );
    args.context_tokens = 2048;
    assert!(load_variant(&args, &environment, &corpus, corpus_bytes).is_err());
    args.context_tokens = 1024;
    ranking.influence.plan.corpus_sha256 = "9".repeat(64);
    serde_json::to_writer(fs::File::create(artifact.path())?, &ranking)?;
    assert!(load_variant(&args, &environment, &corpus, corpus_bytes).is_err());
    Ok(())
}

/// Sequential mask trials fully release old forward state only after sessions
/// finish. Returning to the empty mask recovers exact loss and original KV IDs,
/// without accumulating device allocations or accepting protected-layer masks.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_ranking_mask_trials_reset_state_and_preserve_source_allocation() -> Result<()> {
    let _lock = crate::cuda_test_scope::lock_cuda();
    let _dense = m40_llm::kv_compression::ScopedRuntimeConfig::new(
        m40_llm::kv_compression::KvCompressionConfig::dense_reference(),
    );
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let (gguf, weights) =
            crate::fixture::make_tied_identity_tiny_gguf(crate::fixture::TinyGgufConfig {
                vocab: 96,
                d_model: 64,
                hidden: 128,
                head_count: 1,
                block_count: 6,
                context_length: 16,
            });
        let mut model = LoadedModel::from_gguf(gguf, weights, -1)?;
        let source_bytes = model.weights_len;
        let corpus = Corpus::parse(include_str!("../../data/research_quality/v1.json"))?;
        let plan = plan(&corpus, "1".repeat(64), "2".repeat(64))?;
        let original = score_mask(&mut model, 16, &[], &plan)?.0;
        for mask in [&[1][..], &[1, 2], &[1, 2, 3]] {
            let (losses, setup_ms, loss_ms) = score_mask(&mut model, 16, mask, &plan)?;
            assert_eq!(model.layer_mask.skipped_layers(), mask);
            assert_eq!(model.active_layer_count()?, 6 - mask.len());
            assert_eq!(model.weights_len, source_bytes);
            assert_eq!(model.kv_cache.as_ref().unwrap().max_batch_size(), 6);
            assert!(setup_ms >= 0.0 && loss_ms > 0.0);
            assert!(losses.iter().all(|loss| loss.scored_tokens == 2));
        }
        assert_eq!(score_mask(&mut model, 16, &[], &plan)?.0, original);
        let before = m40_llm::profile::snapshot();
        assert!(score_mask(&mut model, 16, &[0], &plan).is_err());
        assert_eq!(
            m40_llm::profile::counter_delta(&before, &m40_llm::profile::snapshot()),
            m40_llm::profile::OpCounters::default()
        );
        assert_eq!(score_mask(&mut model, 16, &[], &plan)?.0, original);
        clear_forward_state(&mut model)?;
        drop(model);
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}
