use super::{
    q8_runtime::{self, PrefillChoice, RuntimeConfig},
    qwen_workload,
    records::{self, Report},
    run, Args,
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{ffi_debug_read_kv_token, CudaContext, CudaStream, DeviceBuffer},
    decode_session::DecodeSession,
    evaluation::loss::score_logits,
    infer::{BatchMetadata, BatchSequence, LoadedModel, VarlenPrefillPlan},
    kv_compression::{KvCompressionConfig, ScopedRuntimeConfig},
};
use serde::Serialize;
use serde_json::json;
use std::ffi::c_void;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "kebab-case")]
enum ProbeMode {
    SessionSequential,
    ManualSequential,
    DrainedSequential,
    GreedyEachPrefix,
    PackedPrefix,
}

struct KvSample {
    layer: u32,
    physical_slot: u32,
    position: u32,
    length: u32,
    key: Vec<f32>,
    value: Vec<f32>,
}

struct Snapshot {
    actual_token: u32,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    kv: Vec<KvSample>,
    attention: AttentionComparison,
}

#[derive(Debug, Serialize)]
struct AttentionComparison {
    cached_decode_vs_cpu: Difference,
    packed_replay_vs_cpu: Difference,
    packed_replay_vs_cached_decode: Difference,
}

#[derive(Debug, Serialize)]
struct Difference {
    max_abs_diff: f64,
    mean_abs_diff: f64,
    different_bits: usize,
}

fn compare(a: &[f32], b: &[f32]) -> Result<Difference> {
    ensure!(
        !a.is_empty() && a.len() == b.len() && a.iter().chain(b).all(|x| x.is_finite()),
        "invalid/nonfinite diagnostic vectors"
    );
    let mut max_abs_diff = 0.0_f64;
    let mut sum = 0.0;
    let mut different_bits = 0;
    for (&a, &b) in a.iter().zip(b) {
        let delta = (f64::from(a) - f64::from(b)).abs();
        max_abs_diff = max_abs_diff.max(delta);
        sum += delta;
        different_bits += usize::from(a.to_bits() != b.to_bits());
    }
    Ok(Difference {
        max_abs_diff,
        mean_abs_diff: sum / a.len() as f64,
        different_bits,
    })
}

fn drain(model: &LoadedModel) -> Result<()> {
    model.cuda.synchronize_stream(CudaStream::Decode)?;
    model.cuda.synchronize_stream(CudaStream::Prefill)
}

fn read_f32(model: &LoadedModel, pointer: *const c_void, elements: usize) -> Result<Vec<f32>> {
    let mut values = vec![0.0_f32; elements];
    unsafe {
        model
            .cuda
            .memcpy_d2h(values.as_mut_ptr().cast(), pointer, elements * 4)?;
    }
    Ok(values)
}

/// Replay one real last-layer query against its stored FP16 K/V, widened to
/// FP32 for packed attention. This separates kernel math from cache precision;
/// it does not reconstruct the earlier unrounded packed-prefix inputs.
fn compare_cached_attention(model: &LoadedModel, length: usize) -> Result<AttentionComparison> {
    let cache = model.kv_cache.as_ref().context("KV")?;
    let pointers = model
        .forward_workspace
        .lock()
        .map_err(|_| anyhow::anyhow!("poisoned workspace"))?
        .as_ref()
        .context("forward workspace")?
        .ptrs();
    ensure!(
        pointers.rows == 1,
        "attention replay requires final one-token workspace"
    );
    let q_heads = model.model_config.attention_head_count;
    let kv_heads = cache.num_heads();
    let head_dim = cache.head_dim();
    let dimension = (q_heads * head_dim) as usize;
    let kv_width = (kv_heads * head_dim) as usize;
    let query = read_f32(model, pointers.dq.cast_const(), dimension)?;
    let actual = read_f32(model, pointers.datt.cast_const(), dimension)?;
    let slot =
        model.kv_cache_physical_slot_for_layer_sequence(model.model_config.block_count - 1, 0)?;
    let mut keys = Vec::with_capacity(length * kv_width);
    let mut values = Vec::with_capacity(length * kv_width);
    for position in 0..length {
        let mut key = vec![0u8; kv_width * 2];
        let mut value = key.clone();
        let rc = unsafe {
            ffi_debug_read_kv_token(
                &model.cuda,
                cache,
                slot,
                position as u32,
                key.as_mut_ptr(),
                value.as_mut_ptr(),
            )
        };
        ensure!(rc == 0, "attention replay KV read failed: {rc}");
        for (output, bytes) in [(&mut keys, &key), (&mut values, &value)] {
            output.extend(
                bytes.chunks_exact(2).map(|pair| {
                    half::f16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32()
                }),
            );
        }
    }
    let mut cpu = vec![0.0_f32; dimension];
    let group = (q_heads / kv_heads) as usize;
    for head in 0..q_heads as usize {
        let mut scores = Vec::with_capacity(length);
        for token in 0..length {
            let key_base = token * kv_width + (head / group) * head_dim as usize;
            let dot: f64 = (0..head_dim as usize)
                .map(|d| {
                    f64::from(query[head * head_dim as usize + d]) * f64::from(keys[key_base + d])
                })
                .sum();
            scores.push(dot / f64::from(head_dim).sqrt());
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
        let denominator: f64 = weights.iter().sum();
        for d in 0..head_dim as usize {
            let sum: f64 = weights
                .iter()
                .enumerate()
                .map(|(token, &weight)| {
                    weight
                        * f64::from(
                            values[token * kv_width + (head / group) * head_dim as usize + d],
                        )
                })
                .sum();
            cpu[head * head_dim as usize + d] = (sum / denominator) as f32;
        }
    }
    let q = DeviceBuffer::new_tagged(&model.cuda, query.len() * 4, "prefill_probe:q")?;
    let k = DeviceBuffer::new_tagged(&model.cuda, keys.len() * 4, "prefill_probe:k")?;
    let v = DeviceBuffer::new_tagged(&model.cuda, values.len() * 4, "prefill_probe:v")?;
    let out = DeviceBuffer::new_tagged(&model.cuda, dimension * 4, "prefill_probe:attention")?;
    unsafe {
        model
            .cuda
            .memcpy_h2d(q.as_mut_ptr(), query.as_ptr().cast(), query.len() * 4)?;
        model
            .cuda
            .memcpy_h2d(k.as_mut_ptr(), keys.as_ptr().cast(), keys.len() * 4)?;
        model
            .cuda
            .memcpy_h2d(v.as_mut_ptr(), values.as_ptr().cast(), values.len() * 4)?;
    }
    let plan = VarlenPrefillPlan::new(
        &model.cuda,
        BatchMetadata::new(vec![BatchSequence {
            seq_len: length as u32,
            kv_len: length as u32,
            query_len: 1,
        }])?,
        head_dim,
    )?;
    unsafe {
        plan.dispatch(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            q_heads,
            kv_heads,
            out.as_mut_ptr(),
        )?;
    }
    let replay = read_f32(model, out.as_ptr(), dimension)?;
    Ok(AttentionComparison {
        cached_decode_vs_cpu: compare(&actual, &cpu)?,
        packed_replay_vs_cpu: compare(&replay, &cpu)?,
        packed_replay_vs_cached_decode: compare(&replay, &actual)?,
    })
}

fn read_snapshot(
    model: &LoadedModel,
    hidden: *const c_void,
    actual_token: u32,
    length: usize,
) -> Result<Snapshot> {
    drain(model)?;
    let dimension = model.model_config.embedding_length as usize;
    let mut host_hidden = vec![0.0_f32; dimension];
    unsafe {
        model
            .cuda
            .memcpy_d2h(host_hidden.as_mut_ptr().cast(), hidden, dimension * 4)?;
    }
    let logits = unsafe { model.logits_from_hidden(hidden)? };
    score_logits(&logits, actual_token)?;
    let cache = model.kv_cache.as_ref().context("dense cache")?;
    ensure!(
        !cache.is_compressed(),
        "prefill diagnostic requires dense cache"
    );
    let width = model.model_config.attention_head_count_kv as usize
        * model.model_config.attention_key_length as usize;
    let mut positions = vec![0, 1, 32, 63, length - 1];
    positions.retain(|&pos| pos < length);
    positions.sort_unstable();
    positions.dedup();
    let mut kv = Vec::new();
    for layer in 0..model.model_config.block_count {
        let physical_slot = model.kv_cache_physical_slot_for_layer_sequence(layer, 0)?;
        let stored_length = cache.debug_sequence_length(&model.cuda, physical_slot)?;
        ensure!(
            stored_length as usize == length,
            "wrong device KV length at layer {layer}"
        );
        for &position in &positions {
            let mut key = vec![0u8; width * 2];
            let mut value = key.clone();
            let rc = unsafe {
                ffi_debug_read_kv_token(
                    &model.cuda,
                    cache,
                    physical_slot,
                    position as u32,
                    key.as_mut_ptr(),
                    value.as_mut_ptr(),
                )
            };
            ensure!(
                rc == 0,
                "read physical KV layer {layer} position {position}: {rc}"
            );
            let decode = |bytes: &[u8]| {
                bytes
                    .chunks_exact(2)
                    .map(|pair| {
                        half::f16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32()
                    })
                    .collect()
            };
            kv.push(KvSample {
                layer,
                physical_slot,
                position: position as u32,
                length: stored_length,
                key: decode(&key),
                value: decode(&value),
            });
        }
    }
    let attention = compare_cached_attention(model, length)?;
    Ok(Snapshot {
        actual_token,
        hidden: host_hidden,
        logits,
        kv,
        attention,
    })
}

/// The manual variant changes only the stream-drain boundary; it does not
/// replace projection, attention, RoPE, KV append or greedy-head arithmetic.
fn run_manual_prompt(model: &LoadedModel, ids: &[u32], should_drain: bool) -> Result<Snapshot> {
    let dimension = model.model_config.embedding_length as usize;
    let (_, _, _, vocabulary, _) = model.map_lm_head()?;
    let input = DeviceBuffer::new_tagged(&model.cuda, dimension * 4, "prefill_probe:input")?;
    let output = DeviceBuffer::new_tagged(&model.cuda, dimension * 4, "prefill_probe:output")?;
    let logits = DeviceBuffer::new_tagged(&model.cuda, vocabulary * 4, "prefill_probe:logits")?;
    let norm = model
        .map_output_norm()?
        .map(|_| DeviceBuffer::new_tagged(&model.cuda, dimension * 4, "prefill_probe:norm"))
        .transpose()?;
    let token = DeviceBuffer::new_tagged(&model.cuda, 4, "prefill_probe:token")?;
    for (position, &id) in ids.iter().enumerate() {
        if should_drain {
            drain(model)?;
        }
        unsafe {
            model.load_token_embedding_to_f32(id as u64, input.as_mut_ptr())?;
            model.forward_one_token_all_layers_for_sequence(
                input.as_ptr(),
                0,
                (position + 1) as u32,
                output.as_mut_ptr(),
            )?;
        }
    }
    let actual = unsafe {
        model.greedy_token_from_hidden_into(
            output.as_ptr(),
            logits.as_mut_ptr(),
            norm.as_ref().map(DeviceBuffer::as_mut_ptr),
            token.as_mut_ptr(),
        )?
    };
    read_snapshot(model, output.as_ptr(), actual, ids.len())
}

fn probe(model: &LoadedModel, ids: &[u32], mode: ProbeMode) -> Result<Snapshot> {
    ensure!(
        !ids.is_empty()
            && ids.len() <= model.kv_cache.as_ref().context("KV")?.max_seq_len() as usize,
        "invalid diagnostic prefix length"
    );
    let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    drain(model)?;
    model.reset_kv_cache()?;
    match mode {
        ProbeMode::ManualSequential => return run_manual_prompt(model, ids, false),
        ProbeMode::DrainedSequential => return run_manual_prompt(model, ids, true),
        _ => {}
    }
    let dimension = model.validate_full_layer_decode()?.0;
    let mut session = DecodeSession::new_for_sequence_with_kv_config(
        model,
        0,
        KvCompressionConfig::dense_reference(),
        dimension,
        true,
        "prefill_drift",
        "prefill_drift:x",
        "prefill_drift:out",
    )?;
    if matches!(mode, ProbeMode::PackedPrefix) {
        session.prefill_packed_prefix_ids(ids)?;
    }
    let actual = if matches!(mode, ProbeMode::GreedyEachPrefix) {
        let mut last = 0;
        for length in 1..=ids.len() {
            last = session.greedy_token_for_ids(&ids[..length])?;
        }
        last
    } else {
        session.greedy_token_for_ids(ids)?
    };
    read_snapshot(model, session.d_out_ptr()?.cast_const(), actual, ids.len())
}

fn record(
    report: &mut Report,
    label: &str,
    mode: ProbeMode,
    reference: &Snapshot,
    candidate: &Snapshot,
) -> Result<()> {
    ensure!(
        reference.kv.len() == candidate.kv.len(),
        "KV sample coverage differs"
    );
    let reference_prediction = score_logits(&reference.logits, reference.actual_token)?;
    let candidate_prediction = score_logits(&candidate.logits, reference.actual_token)?;
    let overlap = reference_prediction
        .top_indices
        .iter()
        .filter(|id| candidate_prediction.top_indices.contains(id))
        .count();
    report.append(json!({"kind":"q8_prefill_drift_output","label":label,"mode":mode,
        "reference_actual_greedy_token_id":reference.actual_token,"candidate_actual_greedy_token_id":candidate.actual_token,
        "hidden_difference":compare(&reference.hidden,&candidate.hidden)?,
        "full_logit_difference":compare(&reference.logits,&candidate.logits)?,"top10_overlap":overlap,
        "reference_generic_prediction":reference_prediction,"candidate_generic_prediction":candidate_prediction,
        "last_layer_cached_attention":candidate.attention,
        "scope":"untimed; actual greedy token plus generic full-vocabulary head; no sampling-policy change"}))?;
    for (a, b) in reference.kv.iter().zip(&candidate.kv) {
        ensure!(
            (a.layer, a.physical_slot, a.position, a.length)
                == (b.layer, b.physical_slot, b.position, b.length),
            "KV layer/position/length mapping differs"
        );
        let mut dimensions = vec![0, 1, 31, 63, 64, 95, 127, 128, 191, 255, a.key.len() - 1];
        dimensions.retain(|&d| d < a.key.len());
        dimensions.sort_unstable();
        dimensions.dedup();
        let sampled_values: Vec<_> = dimensions
            .iter()
            .map(|&d| {
                json!({"dimension":d,
            "reference_key":a.key[d],"reference_value":a.value[d],
            "candidate_key":b.key[d],"candidate_value":b.value[d]})
            })
            .collect();
        report.append(json!({"kind":"q8_prefill_drift_kv","label":label,"mode":mode,
            "layer":a.layer,"physical_slot":a.physical_slot,"absolute_position":a.position,"sequence_length":a.length,
            "key_difference":compare(&a.key,&b.key)?,"value_difference":compare(&a.value,&b.value)?,
            "compared_elements_per_vector":a.key.len(),"sampled_values":sampled_values}))?;
    }
    Ok(())
}

pub(super) fn diagnose(args: Args) -> Result<()> {
    ensure!(
        cfg!(have_cublas),
        "prefill drift SGEMM comparison requires a cuBLAS-enabled build"
    );
    ensure!(
        args.context_tokens == qwen_workload::CONTEXT_TOKENS,
        "prefill drift requires context 1024"
    );
    let environment = run::environment()?;
    q8_runtime::validate_environment(&environment)?;
    ensure!(
        records::file_sha256(&args.model)? == records::file_sha256(&args.reference_model)?,
        "use identical source GGUF"
    );
    let mut report = Report::create(&args.output)?;
    let initial = CudaContext::total_device_bytes();
    let (model, tokenizer) =
        RuntimeConfig::default().load_model(&args.model, args.context_tokens)?;
    let prompt = qwen_workload::render_user_prompt(qwen_workload::USER_PROMPT);
    let ids = tokenizer.encode_with_specials(&prompt, false, false)?;
    ensure!(
        ids.len() == 65,
        "canonical browser prompt must contain 65 IDs"
    );
    report.append(json!({"kind":"q8_prefill_drift_manifest","variant_id":args.variant_id,
        "model":run::model_identity(&args.model,&model,&tokenizer)?,"environment":environment,
        "prepared_control":model.q8_execution_stats(),"prompt_token_ids":ids,
        "revision":records::command_text("git",&["rev-parse","HEAD"])?,"worktree":records::command_text("git",&["status","--short"])?,
        "executable_sha256":records::file_sha256(&std::env::current_exe()?)?,
        "build_profile":"release","cuda_flags":super::research::CUDA_FLAGS,
        "scope":"untimed correctness attribution, explicit dense KV; no performance or promotion claim"}))?;
    let reference = probe(&model, &ids, ProbeMode::SessionSequential)?;
    record(
        &mut report,
        "sequential",
        ProbeMode::SessionSequential,
        &reference,
        &reference,
    )?;
    for (label, mode) in [
        ("manual-sequential", ProbeMode::ManualSequential),
        ("drained-sequential", ProbeMode::DrainedSequential),
        ("greedy-each-prefix", ProbeMode::GreedyEachPrefix),
        ("native-packed", ProbeMode::PackedPrefix),
        ("sequential-repeat", ProbeMode::SessionSequential),
    ] {
        let candidate = probe(&model, &ids, mode)?;
        record(&mut report, label, mode, &reference, &candidate)?;
    }
    let config = RuntimeConfig {
        prefill: PrefillChoice::DequantSgemm,
        ..Default::default()
    };
    let (candidate, _) = config.load_model(&args.model, args.context_tokens)?;
    let snapshot = probe(&candidate, &ids, ProbeMode::PackedPrefix)?;
    record(
        &mut report,
        "dequant-sgemm",
        ProbeMode::PackedPrefix,
        &reference,
        &snapshot,
    )?;
    drop(candidate);
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "diagnostic leaked tracked allocation"
    );
    report.append(json!({"kind":"q8_prefill_drift_complete","tracked_device_bytes_initial":initial,
        "tracked_device_bytes_after":CudaContext::total_device_bytes(),"changes_runtime_defaults":false}))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Differences must include all values and reject missing/nonfinite samples;
    /// exact bits remain distinguishable from merely close floating-point values.
    #[test]
    fn test_differences_cover_values_and_reject_invalid_vectors() -> Result<()> {
        let diff = compare(&[1.0, 2.0], &[1.0, 2.5])?;
        assert_eq!(diff.max_abs_diff, 0.5);
        assert_eq!(diff.mean_abs_diff, 0.25);
        assert_eq!(diff.different_bits, 1);
        assert!(compare(&[], &[]).is_err());
        assert!(compare(&[1.0], &[1.0, 2.0]).is_err());
        assert!(compare(&[f32::NAN], &[1.0]).is_err());
        Ok(())
    }

    /// Real fixture forwards expose physical KV for every sampled layer and
    /// absolute token. Drained and per-prefix-head execution must match the
    /// corrected session; manual unfenced execution is a diagnostic negative
    /// control and may race input reuse. Snapshot storage owns no allocation.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_prefill_snapshots_cover_physical_kv_and_release_scratch() -> Result<()> {
        use m40_llm::infer::{
            execution_weights::Q8ExecutionConfig, q8_prefill::Q8PrefillBackend, Q8FfnConfig,
        };
        let _scope = crate::cuda_test_scope::lock_cuda();
        let _env: Vec<_> = [
            ("M40LLM_ENABLE_NVCC", "1"),
            ("M40LLM_PROJECTION_BACKEND", "large-model"),
            ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
            ("M40LLM_Q8_DECODE_STREAM", "decode"),
            ("M40LLM_FUSED_Q8_QKV", "1"),
            ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
            ("M40LLM_DECODE_GRAPH", "0"),
        ]
        .into_iter()
        .map(|(k, v)| crate::env_restore::EnvRestore::set(k, v))
        .collect();
        let _hardware = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        {
            let (gguf, bytes) = crate::q8_fixture::build_prefill_fixture(true);
            let mut model = LoadedModel::from_gguf_with_q8_backends(
                gguf,
                bytes,
                -1,
                Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
                Q8FfnConfig::native(),
                Q8PrefillBackend::Native,
            )?;
            model.allocate_kv_cache_for_layers(16)?;
            let ids = [65, 66, 67];
            let reference = probe(&model, &ids, ProbeMode::SessionSequential)?;
            let destination = crate::test_files::TestReportPath::new();
            let mut report = Report::create(destination.path())?;
            for mode in [
                ProbeMode::ManualSequential,
                ProbeMode::DrainedSequential,
                ProbeMode::GreedyEachPrefix,
                ProbeMode::PackedPrefix,
            ] {
                let snapshot = probe(&model, &ids, mode)?;
                assert_eq!(snapshot.kv.len(), 6);
                assert!(snapshot.kv.iter().all(|s| s.length == 3
                    && s.position < 3
                    && s.key.len() == 128
                    && s.value.len() == 128));
                assert!(snapshot.attention.cached_decode_vs_cpu.max_abs_diff <= 2e-3);
                assert!(snapshot.attention.packed_replay_vs_cpu.max_abs_diff <= 2e-3);
                if matches!(
                    mode,
                    ProbeMode::DrainedSequential | ProbeMode::GreedyEachPrefix
                ) {
                    assert!(compare(&reference.hidden, &snapshot.hidden)?.max_abs_diff <= 1e-5);
                    assert_eq!(reference.actual_token, snapshot.actual_token);
                }
                record(&mut report, "fixture", mode, &reference, &snapshot)?;
            }
            let rows = std::fs::read_to_string(destination.path())?
                .lines()
                .map(serde_json::from_str::<serde_json::Value>)
                .collect::<Result<Vec<_>, _>>()?;
            assert_eq!(
                rows.iter()
                    .filter(|row| row["kind"] == "q8_prefill_drift_output")
                    .count(),
                4
            );
            assert_eq!(
                rows.iter()
                    .filter(|row| row["kind"] == "q8_prefill_drift_kv")
                    .count(),
                24
            );
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }
}
