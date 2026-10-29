//! Actual output-normalized vocabulary inputs, outside every timing boundary.
use super::{
    guarded_words::GuardedWords,
    packet::PreparedWarpPacket,
    run::{self, Args, Report},
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream},
    decode_session::DecodeSession,
    evaluation::identity::sha256,
    infer::{execution_weights::Q8SeparateLaunch, LoadedModel, Q8FfnOwnership},
    kv_compression::KvCompressionConfig,
    profile,
};
use serde_json::{json, Value};
use std::fs;

#[cfg(test)]
#[path = "../../tests/common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[cfg(test)]
#[path = "../../tests/common/env_restore.rs"]
mod env_restore;
#[cfg(test)]
#[path = "../../tests/common/q8_model.rs"]
#[allow(dead_code)]
mod model_fixture;

struct LogitProbe<'m> {
    model: &'m LoadedModel,
    packet: PreparedWarpPacket,
    normalized: GuardedWords,
    control: GuardedWords,
    candidate: GuardedWords,
    token: GuardedWords,
    partials: GuardedWords,
    vocab: usize,
    scratch_bytes: usize,
}

impl<'m> LogitProbe<'m> {
    fn new(model: &'m LoadedModel) -> Result<Self> {
        let (_, head, width, vocab, _) = model.map_lm_head()?;
        ensure!(model.map_output_norm()?.is_some(), "requires output norm");
        let scratch_bytes = CudaContext::argmax_f32_partitioned_scratch_bytes(vocab as u32)?;
        Ok(Self {
            model,
            packet: PreparedWarpPacket::new(model, head.dptr, 512 * 1024 * 1024)?,
            normalized: GuardedWords::new(&model.cuda, width)?,
            control: GuardedWords::new(&model.cuda, vocab)?,
            candidate: GuardedWords::new(&model.cuda, vocab)?,
            token: GuardedWords::new(&model.cuda, 1)?,
            partials: GuardedWords::new(&model.cuda, scratch_bytes / 4)?,
            vocab,
            scratch_bytes,
        })
    }

    /// The session owns a completed full-forward hidden vector at this prefix.
    fn verify(&self, session: &DecodeSession, expected_token: u32) -> Result<Value> {
        let ctx = &self.model.cuda;
        let resident = CudaContext::total_device_bytes();
        let (_, stream) = unsafe {
            self.model.enqueue_logits_from_hidden_into(
                session.d_out_ptr()?,
                self.control.pointer(),
                Some(self.normalized.pointer()),
            )?
        };
        ctx.synchronize_stream(stream)?;
        let original_input = self.normalized.read(ctx)?;
        let expected = self.control.read(ctx)?;
        ensure!(
            original_input
                .iter()
                .chain(&expected)
                .all(|&v| f32::from_bits(v).is_finite()),
            "nonfinite captured output norm/logits"
        );
        let before = profile::snapshot();
        unsafe {
            self.packet.device.enqueue_async(
                self.normalized.pointer(),
                self.candidate.pointer(),
                1,
                Q8SeparateLaunch::new(4, 128)?,
                CudaStream::Decode,
            )?;
            ctx.argmax_f32_async(
                self.control.pointer(),
                self.vocab as u32,
                self.token.pointer(),
                CudaStream::Decode,
            )?;
        }
        let counters = profile::counter_delta(&before, &profile::snapshot());
        ensure!(
            counters.device_allocs == 0
                && counters.device_frees == 0
                && counters.h2d_copies == 0
                && counters.d2h_copies == 0
                && counters.stream_syncs == 0
                && counters.cublas_calls == 0
                && counters.launches == 2,
            "packet/control argmax enqueue changed its resource contract"
        );
        ctx.synchronize_stream(CudaStream::Decode)?;
        ensure!(
            self.token.read(ctx)? == [expected_token],
            "control argmax differs"
        );
        ensure!(
            self.candidate.read(ctx)? == expected,
            "actual vocabulary bits differ"
        );
        // Check the new argmax independently on both real control and packet logits.
        for input in [&self.control, &self.candidate] {
            let before = profile::snapshot();
            unsafe {
                ctx.argmax_f32_partitioned_async(
                    input.pointer(),
                    self.vocab as u32,
                    self.partials.pointer(),
                    self.scratch_bytes,
                    self.token.pointer(),
                    CudaStream::Decode,
                )?;
            }
            let counters = profile::counter_delta(&before, &profile::snapshot());
            ensure!(
                counters.device_allocs == 0
                    && counters.device_frees == 0
                    && counters.h2d_copies == 0
                    && counters.d2h_copies == 0
                    && counters.stream_syncs == 0
                    && counters.cublas_calls == 0
                    && counters.launches == 2,
                "partitioned argmax enqueue changed its resource contract"
            );
            ctx.synchronize_stream(CudaStream::Decode)?;
            ensure!(
                self.token.read(ctx)? == [expected_token],
                "partitioned argmax differs"
            );
            self.partials.read(ctx)?;
        }
        ensure!(
            self.normalized.read(ctx)? == original_input,
            "normalized input mutated"
        );
        ensure!(
            self.control.read(ctx)? == expected,
            "control logits mutated"
        );
        ensure!(
            CudaContext::total_device_bytes() == resident,
            "probe residency changed"
        );
        let hash = |words: &[u32]| {
            sha256(
                &words
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        };
        Ok(
            json!({"vocabulary":self.vocab,"expected_token":expected_token,
            "output_norm_sha256":hash(&original_input)?,"logits_sha256":hash(&expected)?,
            "full_vocabulary_bits_equal":true,"control_and_partitioned_argmax_equal":true,
            "scratch_bytes":self.scratch_bytes,"tracked_device_bytes":resident}),
        )
    }
}

pub fn execute(args: &Args, report: &mut Report) -> Result<()> {
    let path = "docs/experiments/2026-10-27-maxwell-compound-runtime-quality.jsonl";
    let text = fs::read_to_string(path)?;
    let records = text
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let frozen = records
        .iter()
        .find(|r| r["kind"] == "q8_runtime_browser_generation" && r["label"] == "candidate")
        .context("frozen compound continuation")?;
    let generation = &frozen["generation"];
    let mut ids: Vec<u32> = serde_json::from_value(frozen["prompt_token_ids"].clone())?;
    let expected: Vec<u32> = serde_json::from_value(generation["generated_token_ids"].clone())?;
    ensure!(
        ids.len() == 65 && expected.len() == 512,
        "wrong frozen workload"
    );
    let initial = CudaContext::total_device_bytes();
    {
        let model = run::load(&args.model, Q8FfnOwnership::JointOnly)?;
        let probe = LogitProbe::new(&model)?;
        let (_, head, _, _, _) = model.map_lm_head()?;
        let weight = model
            .q8_execution
            .get_separate_weight(head.dptr)
            .context("prepared head")?;
        let planes = unsafe { weight.read_planes()? };
        report.record(json!({"kind":"portfolio_actual_logits_setup","source_report":path,
            "source_report_sha256":sha256(text.as_bytes())?,"model":run::identity(&model),
            "packet_preparation":probe.packet.report,"scope":"correctness only; not operator or runtime timing"}))?;
        let mut session = DecodeSession::new_for_sequence_with_kv_config(
            &model,
            0,
            KvCompressionConfig::dense_reference(),
            2048,
            true,
            "portfolio:logits",
            "portfolio:x",
            "portfolio:out",
        )?;
        session.prefill_packed_prefix_ids(&ids)?;
        for (step, &token) in expected.iter().enumerate() {
            ensure!(
                session.greedy_token_for_ids(&ids)? == token,
                "frozen greedy differs at {step}"
            );
            if [0, 1, 8, 32, 128, 256, 511].contains(&step) {
                let result = probe.verify(&session, token)?;
                report.record(json!({"kind":"portfolio_actual_logits_validation","step":step,
                    "prefix_tokens":ids.len(),"prefix_ids_sha256":sha256(&ids.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?,
                    "result":result}))?;
            }
            ids.push(token);
        }
        probe.packet.verify()?;
        ensure!(
            unsafe { weight.read_planes()? } == planes,
            "qualified head mutated"
        );
    }
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "probe leaked device allocations"
    );
    report.record(
        json!({"kind":"portfolio_actual_logits_complete","is_gate_passed":true,
        "actual_generated_ids":expected,"sampled_full_vocabularies":7,"device_bytes_after":initial,
        "runtime_integration_allowed":false}),
    )
}

/// Capture real output-normalized states from a full tiny Qwen forward,
/// preserving vocabulary bits, greedy choice, guards and all device owners.
#[cfg(test)]
#[test]
#[ignore = "requires M40 hardware"]
fn test_actual_logits_preserve_packet_and_argmax_bits() -> Result<()> {
    use m40_llm::infer::execution_weights::Q8ExecutionConfig;
    let _lock = cuda_test_scope::lock_cuda();
    let _env: Vec<_> = [
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_Q8_DECODE_COL_TILE", "4"),
        ("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4"),
        ("M40LLM_PROJECTION_BACKEND", "large-model"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_CUDA_GREEDY_ARGMAX", "1"),
        ("M40LLM_DECODE_GRAPH", "0"),
    ]
    .into_iter()
    .map(|(k, v)| env_restore::EnvRestore::set(k, v))
    .collect();
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let (mut gguf, mut bytes) = model_fixture::build_prefill_fixture(true);
        gguf.tensors.push(m40_llm::gguf::GgufTensor {
            name: "output_norm.weight".into(),
            shape: vec![256],
            dtype: m40_llm::gguf::GgmlDType::F32,
            offset: bytes.len() as u64,
        });
        for index in 0..256 {
            bytes.extend_from_slice(&(0.5 + index as f32 / 512.0).to_le_bytes());
        }
        let mut model = LoadedModel::from_gguf_with_q8_execution(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
        )?;
        model.allocate_kv_cache_for_layers(32)?;
        let probe = LogitProbe::new(&model)?;
        let mut session = DecodeSession::new_for_sequence_with_kv_config(
            &model,
            0,
            KvCompressionConfig::dense_reference(),
            256,
            true,
            "probe-test",
            "probe-test:x",
            "probe-test:out",
        )?;
        let mut ids = vec![1, 2, 3];
        for _ in 0..4 {
            let token = session.greedy_token_for_ids(&ids)?;
            let result = probe.verify(&session, token)?;
            assert_eq!(result["full_vocabulary_bits_equal"], true);
            assert_eq!(result["control_and_partitioned_argmax_equal"], true);
            assert!(probe.verify(&session, (token + 1) % 320).is_err());
            ids.push(token);
        }
        probe.packet.verify()?;
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}
