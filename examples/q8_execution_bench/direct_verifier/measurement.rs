use super::*;
use m40_llm::{cuda::CudaStream, profile};
use std::{
    fs::File,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn millis(began: Instant) -> f64 {
    began.elapsed().as_secs_f64() * 1000.
}

fn run_arm(model: &mut LoadedModel, frozen: &Frozen, is_direct: bool) -> Result<Value> {
    let trace = &frozen.browser;
    let ctx = model.cuda.clone();
    let start = ctx.create_event()?;
    let decode_start = ctx.create_event()?;
    let stop = ctx.create_event()?;
    let before = profile::snapshot();
    let tracked_before = CudaContext::total_device_bytes();
    let started_unix_us = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
    start.record(CudaStream::Decode)?;
    let total = Instant::now();
    let setup = Instant::now();
    let mut session = new_session(model, &frozen.stop_ids, is_direct)?;
    let session_setup_ms = millis(setup);
    let session_scratch_bytes = session.scratch_bytes();
    let tracked_after_setup = CudaContext::total_device_bytes();
    let prefill = Instant::now();
    let first = session.prefill(&trace.prompt)?;
    let prefill_ms = millis(prefill);
    ensure!(first == trace.tokens[0], "timed prompt changed");
    let tracked_after_prefill = CudaContext::total_device_bytes();
    let before_target = profile::snapshot();
    decode_start.record(CudaStream::Decode)?;
    let decode = Instant::now();
    let mut actual = vec![first];
    let mut row = 0;
    let mut batches_ms = Vec::new();
    let mut serial_steps_ms = Vec::new();
    while row < 511 {
        if is_direct && row + 4 <= 511 {
            let batch = Instant::now();
            let attempt = session.begin(&trace.tokens[row..row + 4], false)?;
            ensure!(
                attempt.accepted_len() == 4,
                "timed perfect draft rejected at {row}"
            );
            let predictions = attempt.predictions().to_vec();
            let result = attempt.commit()?;
            batches_ms.push(millis(batch));
            ensure!(
                !result.has_stopped && predictions == trace.tokens[row + 1..row + 5],
                "timed actual predictions changed"
            );
            actual.extend(predictions);
            row += 4;
        } else {
            let step = Instant::now();
            let next = session.serial_step(*actual.last().unwrap())?;
            serial_steps_ms.push(millis(step));
            actual.push(next);
            row += 1;
        }
    }
    let decode_ms = millis(decode);
    let after_target = profile::snapshot();
    let target_counters = profile::counter_delta(&before_target, &after_target);
    ensure!(
        target_counters.cublas_calls == 0,
        "timed target entered SGEMM"
    );
    ensure!(
        actual == trace.tokens && !frozen.stop_ids.contains(actual.last().unwrap()),
        "timed IDs/count/stop changed"
    );
    let committed_len = session.committed_len();
    let tracked_after_decode = CudaContext::total_device_bytes();
    let teardown = Instant::now();
    drop(session);
    let session_teardown_ms = millis(teardown);
    let total_ms = millis(total);
    stop.record(CudaStream::Decode)?;
    let after = profile::snapshot();
    let gpu_total_interval_ms = start.elapsed_sync(&stop, "direct_verifier_total")?;
    let gpu_decode_through_teardown_ms =
        decode_start.elapsed_sync(&stop, "direct_verifier_decode_teardown")?;
    let counters = profile::counter_delta(&before, &after);
    let lengths = (0..36)
        .map(|slot| {
            model
                .kv_cache
                .as_ref()
                .unwrap()
                .debug_sequence_length(&ctx, slot)
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        committed_len == 576 && lengths == vec![576; 36],
        "timed final dense lengths differ"
    );
    let dispatch = target_dispatch(&before_target, &after_target);
    require_target_dispatch(&dispatch, is_direct)?;
    let batch_sum: f64 = batches_ms.iter().sum();
    let serial_sum: f64 = serial_steps_ms.iter().sum();
    Ok(
        json!({"is_direct":is_direct,"generated_token_ids":actual,"generated_tokens":512,"prompt_tokens":65,
        "stop_reason":"length","target_input_rows":511,"committed_len":committed_len,"layer_lengths":lengths,
        "session_setup_ms":session_setup_ms,"prefill_ms":prefill_ms,"decode_ms":decode_ms,
        "session_teardown_ms":session_teardown_ms,"total_ms":total_ms,"resident_total_tps":512000./total_ms,
        "batch_ms":batches_ms,"serial_step_ms":serial_steps_ms,"decode_other_ms":decode_ms-batch_sum-serial_sum,
        "gpu_total_interval_ms":gpu_total_interval_ms,"gpu_decode_through_teardown_ms":gpu_decode_through_teardown_ms,
        "target_counters":target_counters,"total_counters":counters,"target_dispatch":dispatch,
        "session_scratch_bytes":session_scratch_bytes,"tracked_before":tracked_before,
        "tracked_after_setup":tracked_after_setup,"tracked_after_prefill":tracked_after_prefill,
        "tracked_after_decode":tracked_after_decode,"tracked_after_teardown":CudaContext::total_device_bytes(),
        "started_unix_us":started_unix_us,"completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros(),
        "scope":"resident generation includes session construction, prompt preparation, target/commit, serial tails and teardown; debug cache reads and event completion excluded"}),
    )
}

fn rejection(model: &mut LoadedModel, frozen: &Frozen, accepted: usize) -> Result<Value> {
    let mut s = new_session(model, &frozen.stop_ids, true)?;
    s.prefill(&frozen.browser.prompt)?;
    let mut draft = frozen.browser.tokens[..4].to_vec();
    draft[accepted] = (draft[accepted] + 1) % 151936;
    let before = profile::snapshot();
    let began = Instant::now();
    let target = Instant::now();
    let a = s.begin(&draft, false)?;
    let target_ms = millis(target);
    ensure!(a.accepted_len() == accepted, "rejection prefix differs");
    let commit = Instant::now();
    let outcome = a.commit()?;
    let rollback_ms = millis(commit);
    ensure!(
        outcome.next_token == frozen.browser.tokens[accepted],
        "rejection correction token differs"
    );
    let correction = Instant::now();
    let next = s.serial_step(outcome.next_token)?;
    let correction_ms = millis(correction);
    let elapsed_ms = millis(began);
    let counters = profile::counter_delta(&before, &profile::snapshot());
    ensure!(
        next == frozen.browser.tokens[accepted + 1] && counters.cublas_calls == 0,
        "measured correction differs"
    );
    ensure!(
        s.debug_lengths()? == vec![66 + accepted as u32; 36],
        "rejection final lengths differ"
    );
    Ok(
        json!({"accepted_prefix":accepted,"useful_progress":accepted+1,"target_ms":target_ms,
        "rollback_ms":rollback_ms,"correction_ms":correction_ms,"elapsed_ms":elapsed_ms,"counters":counters,
        "next_token":next,"has_exact_correction":true,
        "scope":"target+rollback+ordinary correction; diagnostic cache reads, session setup/prefill/teardown excluded and separately charged by generation measurements"}),
    )
}

fn collect_pairs(
    model: &mut LoadedModel,
    frozen: &Frozen,
    out: &mut File,
    phase: &str,
    pairs: &mut Vec<protocol::Pair>,
    count: usize,
) -> Result<()> {
    for trial in pairs.len()..count {
        let order = if trial % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        let mut tps = [0.; 2];
        for is_direct in order {
            let before = check_gpu()?;
            let result = run_arm(model, frozen, is_direct)?;
            tps[usize::from(is_direct)] = result["resident_total_tps"].as_f64().unwrap();
            write_record(
                out,
                json!({"kind":"direct_verifier_trial","phase":phase,"trial":trial,"is_direct":is_direct,
                "gpu_before":before,"gpu_after":check_gpu()?,"result":result}),
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
    phase: &str,
    pairs: &[protocol::Pair],
) -> Result<protocol::Comparison> {
    let comparison = protocol::compare_pairs(pairs)?;
    let mut values = serde_json::to_value(&comparison)?;
    values
        .as_object_mut()
        .unwrap()
        .remove("reaches_original_target");
    write_record(
        out,
        json!({"kind":"direct_verifier_summary","phase":phase,"comparison":values,
        "pairs":pairs.iter().map(|p|json!({"control_tps":p.control_tps,"candidate_tps":p.candidate_tps})).collect::<Vec<_>>(),
        "is_http_claim":false,"has_real_drafter":false}),
    )?;
    Ok(comparison)
}

pub(super) fn run(model: &mut LoadedModel, frozen: &Frozen, out: &mut File) -> Result<()> {
    for is_direct in [false, true] {
        let result = run_arm(model, frozen, is_direct)?;
        write_record(
            out,
            json!({"kind":"direct_verifier_warmup","result":result,"gpu_after":check_gpu()?}),
        )?;
    }
    let mut pairs = Vec::new();
    collect_pairs(model, frozen, out, "screen", &mut pairs, 5)?;
    let mut comparison = summarize(out, "screen", &pairs)?;
    if comparison.needs_variation_extension {
        collect_pairs(model, frozen, out, "screen", &mut pairs, 10)?;
        comparison = summarize(out, "screen", &pairs)?;
        ensure!(
            !comparison.needs_variation_extension,
            "ten-pair screen remains unstable; stop"
        );
    }
    if comparison.needs_fresh_confirmation {
        let mut confirmation = Vec::new();
        collect_pairs(model, frozen, out, "confirmation", &mut confirmation, 20)?;
        comparison = summarize(out, "confirmation", &confirmation)?;
        ensure!(
            !comparison.needs_variation_extension,
            "confirmation remains unstable; stop"
        );
    }
    write_record(
        out,
        json!({"kind":"direct_verifier_gate","passes_resident_gate":comparison.passes_promotion_gate,
        "is_http_claim":false,"has_real_drafter":false}),
    )?;
    for accepted in 0..4 {
        for trial in 0..5 {
            let before = check_gpu()?;
            let result = rejection(model, frozen, accepted)?;
            write_record(
                out,
                json!({"kind":"direct_verifier_rejection","trial":trial,
                "gpu_before":before,"gpu_after":check_gpu()?,"result":result}),
            )?;
        }
    }
    Ok(())
}
