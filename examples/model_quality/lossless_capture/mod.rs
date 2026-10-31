mod snapshot;
#[cfg(test)]
mod tests;

use super::{
    records::{self, Report},
    run, Args, SplitChoice,
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream},
    decode_session::DecodeSession,
    infer::execution_weights::Q8ExecutionConfig,
    kv_compression::{KvCompressionConfig, ScopedRuntimeConfig},
    lossless_kv::LosslessKvBlocks,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Seek, Write},
    path::Path,
    process::Command,
    time::Instant,
};

const SOURCE: &str = "0eee578:docs/perf_baselines.md";
const SOURCE_SHA: &str = "a918722b18973bb2621a0e2f0fd9ccde0d01ff610456904773c1a975a9b10ade";
const MODEL_SHA: &str = "4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d";
const TOKENS: usize = 8192;
const SEQUENCES: usize = 8;
const CONTEXTS: [usize; 4] = [1024, 2048, 4096, 8192];

fn validate_request(args: &Args) -> Result<()> {
    ensure!(
        args.context_tokens == TOKENS as u32
            && matches!(args.split, SplitChoice::Calibration)
            && args.skip_layers.is_empty()
            && args.step_timing_trials == 0,
        "lossless capture requires context 8192 and calibration split without transforms/timing"
    );
    let destination = args
        .capture_lossless_kv
        .as_ref()
        .context("missing lossless artifact")?;
    let canonical = |path: &Path| -> Result<_> {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        Ok(fs::canonicalize(parent)?.join(path.file_name().context("expected filename")?))
    };
    ensure!(
        !destination.exists()
            && !args.output.exists()
            && canonical(destination)? != canonical(&args.output)?,
        "lossless capture needs distinct new artifact and report paths"
    );
    Ok(())
}

fn inspect_gpu() -> Result<Value> {
    let processes = records::command_text(
        "nvidia-smi",
        &[
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
    )?;
    for line in processes.lines() {
        let pid: u32 = line
            .split(',')
            .next()
            .context("missing PID")?
            .trim()
            .parse()?;
        ensure!(
            pid == std::process::id(),
            "other GPU process present: {line}"
        );
    }
    Ok(
        json!({"processes":processes,"device":records::command_text("nvidia-smi",&["--query-gpu=uuid,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used","--format=csv,noheader"])?}),
    )
}

fn source_text() -> Result<Vec<u8>> {
    let result = Command::new("git").args(["show", SOURCE]).output()?;
    ensure!(
        result.status.success() && result.stdout.len() <= 1024 * 1024,
        "cannot read bounded pinned capture source"
    );
    ensure!(
        records::sha256(&result.stdout)? == SOURCE_SHA,
        "capture source hash mismatch"
    );
    Ok(result.stdout)
}

fn select_spans(tokens: &[u32]) -> Result<Vec<Vec<u32>>> {
    ensure!(
        tokens.len() >= TOKENS * SEQUENCES,
        "pinned source lacks eight disjoint 8192-token spans"
    );
    let spans: Vec<_> = tokens[..TOKENS * SEQUENCES]
        .chunks_exact(TOKENS)
        .map(<[u32]>::to_vec)
        .collect();
    for index in 0..spans.len() {
        ensure!(
            spans[..index].iter().all(|prior| prior != &spans[index]),
            "duplicate capture token span"
        );
    }
    Ok(spans)
}

fn verify_environment() -> Result<BTreeMap<String, String>> {
    let actual = run::environment()?;
    let bytes = fs::read("docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl")?;
    ensure!(
        records::sha256(&bytes)?
            == "d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801",
        "capture environment evidence changed"
    );
    let manifest = std::str::from_utf8(&bytes)?
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .find(|v| v["kind"] == "ffn_capture_manifest")
        .context("capture environment manifest missing")?;
    let expected: BTreeMap<String, String> =
        serde_json::from_value(manifest["execution_configuration"]["environment"].clone())?;
    ensure!(
        actual == expected,
        "lossless capture requires the frozen FFN control environment"
    );
    Ok(actual)
}

fn store_stream(
    file: &mut File,
    report: &mut Report,
    case: Value,
    words: &[u16],
    head_dim: usize,
) -> Result<()> {
    ensure!(
        head_dim > 0 && words.len() == TOKENS * head_dim,
        "unexpected capture stream length"
    );
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let offset = file.stream_position()?;
    file.write_all(&bytes)?;
    file.flush()?;
    report.append(json!({"kind":"lossless_kv_stream","case":case,"raw_offset":offset,"raw_bytes":bytes.len(),"raw_sha256":records::sha256(&bytes)?,"head_dim":head_dim,"tokens":TOKENS}))?;
    for context in CONTEXTS {
        let prefix = &words[..context * head_dim];
        let began = Instant::now();
        let encoded = LosslessKvBlocks::encode(prefix)?;
        let encode_ms = began.elapsed().as_secs_f64() * 1000.0;
        let began = Instant::now();
        let decoded = encoded.decode()?;
        let decode_ms = began.elapsed().as_secs_f64() * 1000.0;
        ensure!(decoded == prefix, "real FP16 KV reconstruction differs");
        let stats = encoded.get_stats();
        let exceptional = prefix
            .iter()
            .filter(|&&w| matches!((w >> 10) & 31, 0 | 31))
            .count();
        report.append(json!({"kind":"lossless_kv_compression","case":case,"context_tokens":context,
            "source_prefix_sha256":records::sha256(&bytes[..context*head_dim*2])?,"encoded_sha256":records::sha256(encoded.get_bytes())?,
            "stats":stats,"net_saved_bytes":stats.original_bytes as i64-stats.encoded_bytes as i64,
            "encode_validate_cpu_ms":encode_ms,"decode_cpu_ms":decode_ms,"exceptional_words":exceptional,
            "has_bit_exact_reconstruction":true,"scope":"CPU storage analysis; no GPU traffic or attention speed claim"}))?;
    }
    Ok(())
}

pub(super) fn capture(args: &Args) -> Result<()> {
    validate_request(args)?;
    ensure!(!cfg!(debug_assertions), "lossless capture requires release");
    ensure!(
        records::command_text("git", &["status", "--short"])?.is_empty(),
        "capture requires clean source"
    );
    let environment = verify_environment()?;
    ensure!(
        records::file_sha256(&args.model)? == MODEL_SHA
            && records::file_sha256(&args.reference_model)? == MODEL_SHA,
        "capture requires the pinned original Qwen Q8 model"
    );
    let text = source_text()?;
    let mut report = Report::create(&args.output)?;
    let artifact = args.capture_lossless_kv.as_ref().unwrap();
    let mut raw = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(artifact)?;
    let gpu = inspect_gpu()?;
    let initial = CudaContext::total_device_bytes();
    let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let began = Instant::now();
    let (model, tokenizer) = run::load_model_with_setup(
        &args.model,
        TOKENS as u32,
        Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
        |model| {
            ensure!(
                model.model_config.block_count == 36
                    && model.model_config.embedding_length == 2048
                    && model.model_config.attention_head_count_kv == 2
                    && model.model_config.attention_key_length == 128
                    && model.q8_prefill.is_native(),
                "unexpected capture model/backend"
            );
            Ok(())
        },
    )?;
    let setup_ms = began.elapsed().as_secs_f64() * 1000.0;
    let spans = select_spans(&tokenizer.encode(std::str::from_utf8(&text)?)?)?;
    let mut inputs = Vec::new();
    for (index, ids) in spans.iter().enumerate() {
        inputs.push(json!({"sequence":index,"source_token_start":index*TOKENS,"tokens":ids,
            "token_sha256":records::sha256(&ids.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?}));
    }
    report.append(json!({"kind":"lossless_kv_capture_manifest","schema_version":1,"variant_id":args.variant_id,"revision":records::command_text("git",&["rev-parse","HEAD"])?,
        "executable_sha256":records::file_sha256(&std::env::current_exe()?)?,"model":run::model_identity(&args.model,&model,&tokenizer)?,
        "environment":environment,"source":SOURCE,"source_sha256":SOURCE_SHA,"inputs":inputs,"artifact":artifact,
        "gpu_before":gpu,"setup_ms":setup_ms,"layers":[0,17,35],"contexts":CONTEXTS,"sequences":SEQUENCES,
        "layout":"sequence, layer, head, role K then V; each stream token-major/channel-minor u16 little endian",
        "prefill":"native packed prefix plus one final ordinary decode input","kv_mode":"explicit dense FP16 off","graphs":false,
        "input_scope":"disjoint technical-document token spans; compression fixture, not task or representativeness evidence",
        "rustc":records::command_text("rustc",&["--version"])?,"nvcc":records::command_text("nvcc",&["--version"])?}))?;
    for (sequence, ids) in spans.iter().enumerate() {
        inspect_gpu()?;
        model.reset_kv_cache()?;
        eprintln!(
            "[lossless_kv] sequence={sequence} prefill tokens={}",
            ids.len()
        );
        let began = Instant::now();
        let mut session = DecodeSession::new_for_sequence_with_kv_config(
            &model,
            0,
            KvCompressionConfig::dense_reference(),
            2048,
            true,
            "lossless_kv",
            "lossless_kv:x",
            "lossless_kv:out",
        )?;
        session.prefill_packed_prefix_ids(ids)?;
        let prefix_memory = run::memory(&model)?;
        let next = session.greedy_token_for_ids(ids)?;
        model.cuda.synchronize_stream(CudaStream::Prefill)?;
        model.cuda.synchronize_stream(CudaStream::Decode)?;
        ensure!(
            session.processed_len() == TOKENS,
            "capture did not process every input token"
        );
        let input_processing_ms = began.elapsed().as_secs_f64() * 1000.0;
        for layer in [0, 17, 35] {
            let began = Instant::now();
            let heads = snapshot::read_layer(&model, layer, TOKENS)?;
            let readback_ms = began.elapsed().as_secs_f64() * 1000.0;
            for (head, words) in heads.iter().enumerate() {
                for (role, stream) in [("key", &words.key), ("value", &words.value)] {
                    store_stream(
                        &mut raw,
                        &mut report,
                        json!({"sequence":sequence,"layer":layer,"head":head,"role":role}),
                        stream,
                        128,
                    )?;
                }
            }
            report.append(json!({"kind":"lossless_kv_readback","sequence":sequence,"layer":layer,"cpu_wall_ms":readback_ms}))?;
        }
        report.append(json!({"kind":"lossless_kv_sequence","sequence":sequence,"processed_input_tokens":ids.len(),"generated_token_count":0,
            "unused_next_token":next,"stop_reason":"fixed input consumed; no free generation",
            "input_processing_ms":input_processing_ms,"prefill_memory_snapshot":prefix_memory,"memory_after":run::memory(&model)?,"gpu_after":inspect_gpu()?}))?;
        drop(session);
        eprintln!("[lossless_kv] sequence={sequence} captured");
    }
    raw.flush()?;
    let artifact_bytes = raw.stream_position()?;
    ensure!(
        artifact_bytes == (SEQUENCES * 3 * 2 * 2 * TOKENS * 128 * 2) as u64,
        "incomplete raw capture"
    );
    drop(raw);
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "capture leaked tracked buffers"
    );
    report.append(json!({"kind":"lossless_kv_capture_complete","is_complete":true,"artifact":artifact,
        "artifact_bytes":artifact_bytes,"artifact_sha256":records::file_sha256(artifact)?,"tracked_device_bytes_before":initial,
        "tracked_device_bytes_after":CudaContext::total_device_bytes(),"gpu_after":inspect_gpu()?}))
}
