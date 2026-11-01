use super::{
    records::{self, Trace},
    runtime,
    source::{DraftSource, SourceKind},
};
use anyhow::{ensure, Context, Result};
use m40_llm::{cuda::CudaContext, infer::LoadedModel, profile, verification::VerificationSession};
use serde_json::{json, Value};
use std::{
    fs::{File, OpenOptions},
    path::Path,
    time::Instant,
};
#[path = "../compare_ollama/protocol.rs"]
#[allow(dead_code)]
mod protocol;

struct ResultRow {
    data: Value,
    logits: Vec<Vec<f32>>,
    kv: Vec<u16>,
}

fn snapshot(s: &VerificationSession<'_>) -> Result<Vec<u16>> {
    let length = s.committed_len();
    ensure!(
        s.debug_lengths()? == vec![length; 36],
        "live layer lengths differ"
    );
    let mut bits = Vec::with_capacity(36 * length as usize * 512);
    for layer in 0..36 {
        for pos in 0..length {
            let (k, v) = s.debug_kv_token(layer, pos)?;
            bits.extend(k);
            bits.extend(v);
        }
    }
    ensure!(
        bits.iter().all(|v| half::f16::from_bits(*v).is_finite()),
        "nonfinite live KV"
    );
    Ok(bits)
}
fn compare_logits(
    actual: &[f32],
    expected: &[f32],
    max: &mut f64,
    has_exact: &mut bool,
) -> Result<()> {
    ensure!(
        actual.len() == 151936 && actual.len() == expected.len(),
        "full vocabulary missing"
    );
    for (&a, &b) in actual.iter().zip(expected) {
        let diff = (f64::from(a) - f64::from(b)).abs();
        ensure!(
            a.is_finite() && b.is_finite() && diff <= 1e-3 + 2e-5 * f64::from(b).abs(),
            "live logit parity failed"
        );
        *max = max.max(diff);
        *has_exact &= a.to_bits() == b.to_bits();
    }
    Ok(())
}

/// Only prompt, budget, stop IDs and actual target predictions enter drafting.
/// Frozen future vectors, when provided, are read solely after GPU evaluation.
fn execute(
    model: &mut LoadedModel,
    prompt: &[u32],
    budget: usize,
    stops: &[u32],
    is_direct: bool,
    should_capture: bool,
    oracle: Option<&[Vec<f32>]>,
) -> Result<ResultRow> {
    let began = Instant::now();
    let setup = Instant::now();
    let mut source = is_direct.then(|| DraftSource::new(SourceKind::PromptCopy, prompt, stops));
    let mut s = if is_direct {
        VerificationSession::new_direct_four_rows(model, stops)?
    } else {
        VerificationSession::new(model, stops)?
    };
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.;
    let prefill = Instant::now();
    let first = s.prefill(prompt)?;
    let prefill_ms = prefill.elapsed().as_secs_f64() * 1000.;
    let before = profile::snapshot();
    let decode = Instant::now();
    let mut ids = vec![first];
    let mut row = 0;
    let mut vectors = Vec::new();
    let (mut max_diff, mut has_exact) = (0f64, true);
    let mut compared = 0;
    let mut cleared_words = 0;
    let mut attempts = 0;
    let mut rejections = 0;
    while ids.len() < budget && !stops.contains(ids.last().unwrap()) {
        let seed = *ids.last().unwrap();
        let proposal = if budget - ids.len() >= 4 {
            source.as_mut().and_then(|src| src.propose(seed))
        } else {
            None
        };
        if let Some(p) = proposal {
            attempts += 1;
            let start = s.committed_len();
            let a = s.begin(
                &[seed, p.guesses[0], p.guesses[1], p.guesses[2]],
                should_capture,
            )?;
            let accepted = a.accepted_len();
            ensure!(accepted >= 1, "known target seed rejected");
            if let Some(reference) = oracle {
                for (offset, actual) in a
                    .captured_logits()
                    .context("missing live logits")?
                    .iter()
                    .take(accepted)
                    .enumerate()
                {
                    compare_logits(
                        actual,
                        &reference[row + offset],
                        &mut max_diff,
                        &mut has_exact,
                    )?;
                    compared += 1;
                }
            }
            let result = a.commit()?;
            if accepted < 4 {
                rejections += 1;
                if should_capture {
                    for layer in 0..36 {
                        for pos in start + accepted as u32..start + 4 {
                            let (k, v) = s.debug_kv_token(layer, pos)?;
                            ensure!(
                                k.iter().chain(&v).all(|x| *x == 0),
                                "live rejected suffix not cleared"
                            );
                            cleared_words += k.len() + v.len();
                        }
                    }
                }
            }
            source.as_mut().unwrap().commit(&result.accepted_tokens);
            row += accepted;
            ids.extend_from_slice(&result.accepted_tokens[1..]);
            ids.push(result.next_token);
            if accepted < 4 && ids.len() < budget && !stops.contains(&result.next_token) {
                let (next, values) = if should_capture {
                    s.serial_step_with_logits(result.next_token)?
                } else {
                    (s.serial_step(result.next_token)?, Vec::new())
                };
                if let Some(reference) = oracle {
                    compare_logits(&values, &reference[row], &mut max_diff, &mut has_exact)?;
                    compared += 1;
                }
                source.as_mut().unwrap().commit(&[result.next_token]);
                row += 1;
                ids.push(next);
            }
        } else {
            let (next, values) = if should_capture {
                s.serial_step_with_logits(seed)?
            } else {
                (s.serial_step(seed)?, Vec::new())
            };
            if let Some(reference) = oracle {
                compare_logits(&values, &reference[row], &mut max_diff, &mut has_exact)?;
                compared += 1;
            }
            if should_capture && !is_direct {
                vectors.push(values);
            }
            if let Some(src) = source.as_mut() {
                src.commit(&[seed]);
            }
            row += 1;
            ids.push(next);
        }
    }
    let decode_ms = decode.elapsed().as_secs_f64() * 1000.;
    let counters = profile::counter_delta(&before, &profile::snapshot());
    ensure!(
        counters.cublas_calls == 0 && counters.device_allocs == 0 && counters.device_frees == 0,
        "unexpected target resources"
    );
    ensure!(
        row + 1 == ids.len() && s.committed_len() as usize == prompt.len() + row,
        "live output/KV accounting"
    );
    let kv = if should_capture {
        snapshot(&s)?
    } else {
        Vec::new()
    };
    let teardown = Instant::now();
    drop(s);
    drop(source);
    let teardown_ms = teardown.elapsed().as_secs_f64() * 1000.;
    let total_ms = began.elapsed().as_secs_f64() * 1000.;
    let reason = if stops.contains(ids.last().unwrap()) {
        "stop"
    } else {
        "length"
    };
    Ok(ResultRow {
        data: json!({"is_direct":is_direct,"generated_token_ids":ids,"stop_reason":reason,
        "setup_ms":setup_ms,"prefill_ms":prefill_ms,"decode_ms":decode_ms,"teardown_ms":teardown_ms,
        "total_ms":total_ms,"resident_tps":1000.*ids.len() as f64/total_ms,
        "attempts":attempts,"rejections":rejections,"compared_full_logit_rows":compared,
        "max_abs_logit_diff":max_diff,"has_bitwise_logits":has_exact,"cleared_suffix_words":cleared_words,
        "has_full_logit_tolerance_passed":oracle.is_some(),
        "target_counters":counters,"includes_debug_readbacks":should_capture}),
        logits: vectors,
        kv,
    })
}
fn check_result(result: &ResultRow, trace: &Trace) -> Result<()> {
    ensure!(
        result.data["generated_token_ids"] == json!(trace.generated_token_ids)
            && result.data["stop_reason"] == trace.stop_reason,
        "live actual IDs/count/stop differ: {}",
        trace.id
    );
    Ok(())
}
fn gpu() -> Result<Value> {
    let processes = records::command(
        "nvidia-smi",
        &[
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
    )?;
    ensure!(
        processes.lines().count() == 1
            && processes.starts_with(&format!("{},", std::process::id())),
        "GPU not isolated"
    );
    let device = records::command(
        "nvidia-smi",
        &[
            "--query-gpu=uuid,name,pstate,clocks.sm,clocks.mem",
            "--format=csv,noheader",
        ],
    )?;
    let parts: Vec<_> = device.split(',').map(str::trim).collect();
    ensure!(
        parts
            == [
                "GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852",
                "Tesla M40 24GB",
                "P0",
                "1113 MHz",
                "3004 MHz"
            ],
        "GPU clocks changed"
    );
    Ok(json!({"device":device,"processes":processes}))
}
fn collect(
    model: &mut LoadedModel,
    trace: &Trace,
    out: &mut File,
    phase: &str,
    pairs: &mut Vec<protocol::Pair>,
    count: usize,
) -> Result<()> {
    for i in pairs.len()..count {
        let order = if i % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        let mut tps = [0.; 2];
        for direct in order {
            let before = gpu()?;
            let result = execute(
                model,
                &trace.prompt_token_ids,
                trace.max_tokens,
                &trace.stop_ids,
                direct,
                false,
                None,
            )?;
            check_result(&result, trace)?;
            tps[usize::from(direct)] = result.data["resident_tps"].as_f64().unwrap();
            records::write(
                out,
                json!({"kind":"live_copy_trial","id":trace.id,"trial":i,"phase":phase,
                "gpu_before":before,"gpu_after":gpu()?,"result":result.data}),
            )?;
        }
        pairs.push(protocol::Pair {
            control_tps: tps[0],
            candidate_tps: tps[1],
        });
    }
    Ok(())
}
fn summarize(
    out: &mut File,
    id: &str,
    phase: &str,
    pairs: &[protocol::Pair],
) -> Result<protocol::Comparison> {
    let c = protocol::compare_pairs(pairs)?;
    let mut summary = serde_json::to_value(&c)?;
    summary
        .as_object_mut()
        .unwrap()
        .remove("reaches_original_target");
    records::write(
        out,
        json!({"kind":"live_copy_summary","id":id,"phase":phase,"comparison":summary,
        "scope":"specialized resident copy workload, not fixed512/HTTP/Ollama"}),
    )?;
    Ok(c)
}
pub fn run(
    model_path: &Path,
    heldout: &Path,
    output: &Path,
    correctness: Option<&Path>,
) -> Result<()> {
    let traces = records::read_traces(Some(heldout))?;
    let executable = records::hash_file(&std::env::current_exe()?)?;
    if let Some(path) = correctness {
        let rows = records::rows(path)?;
        ensure!(
            rows[0]["executable_sha256"] == executable
                && rows[0]["is_correctness_only"] == true
                && rows[0]["heldout_sha256"] == records::hash_file(heldout)?
                && rows.last().is_some_and(
                    |r| r["is_correctness_passed"] == true && r["device_bytes_after"] == 0
                )
                && rows
                    .iter()
                    .filter(|r| r["kind"] == "live_copy_correctness"
                        && r["has_exact_ids_and_kv"] == true)
                    .count()
                    == 21,
            "complete executable-bound live correctness required before timing"
        );
        ensure!(rows.len() == 23, "incomplete live correctness rows");
        for trace in &traces {
            let found: Vec<_> = rows
                .iter()
                .filter(|r| r["kind"] == "live_copy_correctness" && r["id"] == trace.id)
                .collect();
            ensure!(
                found.len() == 1,
                "missing or duplicate live correctness case"
            );
            let r = found[0];
            ensure!(
                r["serial"]["generated_token_ids"] == json!(trace.generated_token_ids)
                    && r["candidate"]["generated_token_ids"] == json!(trace.generated_token_ids)
                    && r["candidate"]["compared_full_logit_rows"]
                        == trace.generated_token_ids.len() - 1
                    && r["candidate"]["has_full_logit_tolerance_passed"] == true
                    && r["candidate"]["stop_reason"] == trace.stop_reason,
                "live correctness behavior differs"
            );
        }
    }
    let initial = CudaContext::total_device_bytes();
    let runtime::Runtime {
        mut model,
        reference,
        environment,
        ..
    } = runtime::load(model_path)?;
    let prepared = serde_json::to_value(model.q8_execution_stats())?;
    let joint = serde_json::to_value(model.q8_ffn_pair_stats())?;
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    records::write(
        &mut out,
        json!({"kind":"live_copy_manifest","revision":records::command("git",&["rev-parse","HEAD"])?,
        "executable_sha256":executable,"heldout_sha256":records::hash_file(heldout)?,
        "is_correctness_only":correctness.is_none(),"source":"unchanged prompt-copy4..16; explicit research only",
        "model_sha256":reference["model"]["sha256"],"environment":environment,
        "prepared":model.q8_execution_stats(),"joint":model.q8_ffn_pair_stats(),"runtime_defaults_changed":false}),
    )?;
    if correctness.is_none() {
        for trace in &traces {
            let serial = execute(
                &mut model,
                &trace.prompt_token_ids,
                trace.max_tokens,
                &trace.stop_ids,
                false,
                true,
                None,
            )?;
            check_result(&serial, trace)?;
            let live = execute(
                &mut model,
                &trace.prompt_token_ids,
                trace.max_tokens,
                &trace.stop_ids,
                true,
                true,
                Some(&serial.logits),
            )?;
            check_result(&live, trace)?;
            ensure!(
                serial.kv == live.kv
                    && live.data["compared_full_logit_rows"] == trace.generated_token_ids.len() - 1,
                "live valid KV/logit coverage differs"
            );
            records::write(
                &mut out,
                json!({"kind":"live_copy_correctness","id":trace.id,"has_exact_ids_and_kv":true,
                "valid_kv_words":live.kv.len(),"serial":serial.data,"candidate":live.data}),
            )?;
            eprintln!("live correctness {}", trace.id);
        }
    } else {
        for trace in traces
            .iter()
            .filter(|t| t.split == "heldout" && t.class == "copy")
        {
            for direct in [false, true] {
                let result = execute(
                    &mut model,
                    &trace.prompt_token_ids,
                    trace.max_tokens,
                    &trace.stop_ids,
                    direct,
                    false,
                    None,
                )?;
                check_result(&result, trace)?;
                records::write(
                    &mut out,
                    json!({"kind":"live_copy_warmup","id":trace.id,"result":result.data}),
                )?;
            }
            let mut pairs = Vec::new();
            collect(&mut model, trace, &mut out, "screen", &mut pairs, 5)?;
            let mut c = summarize(&mut out, &trace.id, "screen", &pairs)?;
            if c.needs_variation_extension {
                collect(&mut model, trace, &mut out, "screen", &mut pairs, 10)?;
                c = summarize(&mut out, &trace.id, "screen", &pairs)?;
                ensure!(
                    !c.needs_variation_extension,
                    "unstable ten-pair screen; stop"
                );
            }
            if c.needs_fresh_confirmation {
                let mut fresh = Vec::new();
                collect(&mut model, trace, &mut out, "confirmation", &mut fresh, 20)?;
                c = summarize(&mut out, &trace.id, "confirmation", &fresh)?;
                ensure!(
                    !c.needs_variation_extension,
                    "unstable fresh confirmation; stop"
                );
            }
            records::write(
                &mut out,
                json!({"kind":"live_copy_gate","id":trace.id,"passes_resident_gate":c.passes_promotion_gate}),
            )?;
            ensure!(
                c.passes_promotion_gate,
                "live copy performance gate failed; stop without rescue"
            );
        }
    }
    ensure!(
        serde_json::to_value(model.q8_execution_stats())? == prepared
            && serde_json::to_value(model.q8_ffn_pair_stats())? == joint,
        "live prepared ownership changed"
    );
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "live drafter leaked"
    );
    records::write(
        &mut out,
        json!({"kind":"live_copy_complete","is_complete":true,
        "is_correctness_passed":correctness.is_none(),"device_bytes_after":CudaContext::total_device_bytes(),
        "is_http_claim":false,"defaults_changed":false}),
    )
}
