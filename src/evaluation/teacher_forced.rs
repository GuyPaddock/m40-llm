//! Full-logit evaluation on existing sessions; no sampling or new CUDA kernels.

use super::loss::{compare_teacher_forced, TeacherForcedReport};
use crate::{
    decode_session::DecodeSession, infer::LoadedModel, kv_compression::KvCompressionConfig,
};
use anyhow::{ensure, Context, Result};

fn create_session(model: &mut LoadedModel, ids: &[u32]) -> Result<DecodeSession> {
    let cache = model
        .kv_cache
        .as_ref()
        .context("teacher forcing requires allocated dense KV")?;
    ensure!(
        !cache.is_compressed(),
        "teacher forcing requires explicit dense reference KV"
    );
    ensure!(
        ids.len() <= cache.max_seq_len() as usize
            && ids.len() <= model.model_config.context_length as usize,
        "teacher-forced passage exceeds context; no silent truncation"
    );
    let (_, _, _, vocabulary, _) = model.map_lm_head()?;
    ensure!(
        ids.iter().all(|&id| (id as usize) < vocabulary),
        "teacher-forced input token outside vocabulary"
    );
    let (d_model, _) = model.validate_full_layer_decode()?;
    model.reset_kv_cache()?;
    DecodeSession::new_for_sequence_with_kv_config(
        model,
        0,
        KvCompressionConfig::dense_reference(),
        d_model,
        true,
        "teacher_forced",
        "teacher_forced:input",
        "teacher_forced:output",
    )
}

/// Exclusive model borrows keep the session's raw model pointer live and prevent
/// concurrent high-level workspace/KV use. Separate model instances isolate KV;
/// reset once per passage, reuse session scratch/prefix state for every token.
pub fn compare_loaded_models(
    reference: &mut LoadedModel,
    candidate: &mut LoadedModel,
    ids: &[u32],
) -> Result<TeacherForcedReport> {
    ensure!(
        !reference
            .kv_cache
            .as_ref()
            .zip(candidate.kv_cache.as_ref())
            .is_some_and(|(a, b)| a.has_same_storage(b)),
        "reference and candidate must not share mutable KV storage"
    );
    let mut reference_session = create_session(reference, ids)?;
    let mut candidate_session = create_session(candidate, ids)?;
    compare_teacher_forced(
        ids,
        |prefix| reference_session.logits_for_ids(prefix, |_| {}),
        |prefix| candidate_session.logits_for_ids(prefix, |_| {}),
    )
}
