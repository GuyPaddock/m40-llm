use super::*;
use m40_llm::profile;
use std::fs::File;

fn snapshot(session: &VerificationSession<'_>, length: usize) -> Result<Vec<u16>> {
    ensure!(
        session.debug_lengths()? == vec![length as u32; 36],
        "layer lengths differ"
    );
    let mut out = Vec::with_capacity(36 * length * 512);
    for layer in 0..36 {
        for position in 0..length {
            let (k, v) = session.debug_kv_token(layer, position as u32)?;
            out.extend(k);
            out.extend(v);
        }
    }
    ensure!(
        out.iter().all(|&v| half::f16::from_bits(v).is_finite()),
        "nonfinite valid KV"
    );
    Ok(out)
}

fn snapshot_hash(snapshot: &[u16]) -> Result<String> {
    sha256(
        &snapshot
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn compare_state(
    session: &VerificationSession<'_>,
    length: usize,
    reference: &[u16],
    reference_length: usize,
) -> Result<()> {
    let actual = snapshot(session, length)?;
    for layer in 0..36 {
        let a = &actual[layer * length * 512..(layer + 1) * length * 512];
        let b = &reference
            [layer * reference_length * 512..layer * reference_length * 512 + length * 512];
        ensure!(
            a == b,
            "valid dense KV bits differ at layer {layer}, length {length}"
        );
    }
    Ok(())
}

fn compare_logits(actual: &[f32], expected: &[f32]) -> Result<Value> {
    ensure!(
        actual.len() == 151936 && actual.len() == expected.len(),
        "full vocabulary missing"
    );
    let (mut max, mut sum, mut equal) = (0f64, 0f64, true);
    for (&a, &b) in actual.iter().zip(expected) {
        let d = (f64::from(a) - f64::from(b)).abs();
        ensure!(
            a.is_finite() && b.is_finite() && d <= 1e-3 + 2e-5 * f64::from(b).abs(),
            "full-logit tolerance failed: {a} vs {b}"
        );
        max = max.max(d);
        sum += d;
        equal &= a.to_bits() == b.to_bits();
    }
    Ok(
        json!({"values":actual.len(),"max_abs_diff":max,"mean_abs_diff":sum/actual.len() as f64,"is_bitwise_equal":equal,"has_tolerance_passed":true}),
    )
}

fn browser(model: &mut LoadedModel, frozen: &Frozen, out: &mut File) -> Result<Vec<u16>> {
    let trace = &frozen.browser;
    let (reference_logits, reference_kv) = {
        let mut s = new_session(model, &frozen.stop_ids, false)?;
        ensure!(
            s.prefill(&trace.prompt)? == trace.tokens[0],
            "serial prompt differs"
        );
        let mut logits = Vec::with_capacity(511);
        for index in 0..511 {
            let (next, values) = s.serial_step_with_logits(trace.tokens[index])?;
            ensure!(
                next == trace.tokens[index + 1],
                "serial frozen token {index} differs"
            );
            logits.push(values);
        }
        (logits, snapshot(&s, 576)?)
    };
    let mut s = new_session(model, &frozen.stop_ids, true)?;
    ensure!(
        s.prefill(&trace.prompt)? == trace.tokens[0],
        "direct prompt differs"
    );
    let before = profile::snapshot();
    let mut index = 0;
    let mut actual = vec![trace.tokens[0]];
    let mut differences = Vec::new();
    while index < 511 {
        if index + 4 <= 511 {
            let attempt = s.begin(&trace.tokens[index..index + 4], true)?;
            ensure!(
                attempt.accepted_len() == 4,
                "direct frozen acceptance failed at {index}"
            );
            let predictions = attempt.predictions().to_vec();
            ensure!(
                predictions == trace.tokens[index + 1..index + 5],
                "direct actual IDs differ at {index}"
            );
            for (offset, values) in attempt
                .captured_logits()
                .context("missing captured logits")?
                .iter()
                .enumerate()
            {
                differences.push(json!({"input_row":index+offset,"actual_next":predictions[offset],
                    "expected_next":trace.tokens[index+offset+1],"comparison":compare_logits(values,&reference_logits[index+offset])?}));
            }
            let result = attempt.commit()?;
            ensure!(!result.has_stopped, "unexpected frozen stop");
            actual.extend(predictions);
            index += 4;
        } else {
            let (next, values) = s.serial_step_with_logits(trace.tokens[index])?;
            ensure!(next == trace.tokens[index + 1], "serial tail differs");
            differences.push(
                json!({"input_row":index,"actual_next":next,"expected_next":trace.tokens[index+1],
                "comparison":compare_logits(&values,&reference_logits[index])?}),
            );
            actual.push(next);
            index += 1;
        }
    }
    let after = profile::snapshot();
    let counters = profile::counter_delta(&before, &after);
    let dispatch = target_dispatch(&before, &after);
    require_target_dispatch(&dispatch, true)?;
    ensure!(
        counters.cublas_calls == 0,
        "target verification entered cuBLAS"
    );
    ensure!(
        actual == trace.tokens && !frozen.stop_ids.contains(actual.last().unwrap()),
        "browser count/stop differs"
    );
    let actual_kv = snapshot(&s, 576)?;
    ensure!(
        actual_kv == reference_kv,
        "full committed dense KV bits changed"
    );
    write_record(
        out,
        json!({"kind":"direct_verifier_browser_correctness","actual_token_ids":actual,
        "generated_tokens":512,"stop_reason":"length","full_logit_comparisons":differences,
        "has_exact_valid_kv":true,"valid_kv_words":actual_kv.len(),"valid_kv_sha256":snapshot_hash(&actual_kv)?,
        "target_counters":counters,"target_dispatch":dispatch,"session_scratch_bytes":s.scratch_bytes()}),
    )?;
    Ok(reference_kv)
}

fn check_cleared(s: &VerificationSession<'_>, start: usize, end: usize) -> Result<()> {
    for layer in 0..36 {
        for position in start..end {
            let (k, v) = s.debug_kv_token(layer, position as u32)?;
            ensure!(
                k.iter().chain(&v).all(|&v| v == 0),
                "rejected suffix not cleared at {layer}/{position}"
            );
        }
    }
    Ok(())
}

fn transactions(
    model: &mut LoadedModel,
    frozen: &Frozen,
    reference_kv: &[u16],
    out: &mut File,
) -> Result<()> {
    let trace = &frozen.browser;
    {
        let alias = model.kv_cache.as_ref().unwrap().clone();
        ensure!(
            new_session(model, &frozen.stop_ids, true).is_err(),
            "aliased cache accepted"
        );
        drop(alias);
    }
    let mut s = new_session(model, &frozen.stop_ids, true)?;
    for forced_partial in [false, true] {
        for accepted in 0..4 {
            s.prefill(&trace.prompt)?;
            let mut draft = trace.tokens[..4].to_vec();
            if !forced_partial {
                draft[accepted] = (draft[accepted] + 1) % 151936;
            }
            let a = s.begin(&draft, false)?;
            ensure!(
                a.accepted_len() == if forced_partial { 4 } else { accepted },
                "wrong rejected-prefix length"
            );
            let result = if forced_partial {
                a.commit_prefix(accepted)?
            } else {
                a.commit()?
            };
            ensure!(
                result.next_token == trace.tokens[accepted]
                    && result.accepted_tokens == trace.tokens[..accepted],
                "wrong committed prefix/correction"
            );
            compare_state(&s, 65 + accepted, reference_kv, 576)?;
            check_cleared(&s, 65 + accepted, 69)?;
            ensure!(
                s.serial_step(result.next_token)? == trace.tokens[accepted + 1],
                "corrected target differs"
            );
            compare_state(&s, 66 + accepted, reference_kv, 576)?;
            write_record(
                out,
                json!({"kind":"direct_verifier_transaction","accepted":accepted,
                "is_forced_partial":forced_partial,"has_exact_retained_kv":true,
                "has_cleared_suffix_before_correction":true,"has_exact_correction":true}),
            )?;
        }
    }
    s.prefill(&trace.prompt)?;
    drop(s.begin(&trace.tokens[..4], false)?);
    compare_state(&s, 65, reference_kv, 576)?;
    check_cleared(&s, 65, 69)?;
    let mut wrong = trace.tokens[..4].to_vec();
    wrong[1] = (wrong[1] + 1) % 151936;
    ensure!(
        s.begin(&wrong, false)?.commit_prefix(2).is_err(),
        "unverified commit succeeded"
    );
    compare_state(&s, 65, reference_kv, 576)?;
    check_cleared(&s, 65, 69)?;
    ensure!(
        s.begin(&trace.tokens[..2], false).is_err() && s.begin(&[151936; 4], false).is_err(),
        "invalid candidates accepted"
    );
    drop(s);
    {
        let mut stopped = new_session(model, &[trace.tokens[0]], true)?;
        stopped.prefill(&trace.prompt)?;
        let result = stopped.begin(&trace.tokens[..4], false)?.commit()?;
        ensure!(
            result.has_stopped && result.accepted_tokens == trace.tokens[..1],
            "stop acceptance escaped"
        );
        compare_state(&stopped, 66, reference_kv, 576)?;
        check_cleared(&stopped, 66, 69)?;
    }
    model.allocate_kv_cache_for_layers(68)?;
    {
        let mut small = new_session(model, &frozen.stop_ids, true)?;
        small.prefill(&trace.prompt)?;
        ensure!(
            small.begin(&trace.tokens[..4], false).is_err(),
            "capacity overflow accepted"
        );
        compare_state(&small, 65, reference_kv, 576)?;
    }
    model.allocate_kv_cache_for_layers(1024)?;
    write_record(
        out,
        json!({"kind":"direct_verifier_invalid_cases","has_rejected_alias":true,
        "has_drop_rollback":true,"has_rejected_unverified_commit":true,"has_rejected_invalid_width_and_id":true,
        "has_stop_cap":true,"has_rejected_capacity":true}),
    )
}

fn quality(
    model: &mut LoadedModel,
    frozen: &Frozen,
    tokenizer: &Tokenizer,
    out: &mut File,
) -> Result<()> {
    for trace in &frozen.tasks {
        for is_direct in [false, true] {
            let mut s = new_session(model, &frozen.stop_ids, is_direct)?;
            let first = s.prefill(&trace.prompt)?;
            let mut actual = vec![first];
            let mut index = 0;
            while actual.len() < trace.tokens.len()
                && !frozen.stop_ids.contains(actual.last().unwrap())
            {
                if is_direct && index + 4 < trace.tokens.len() {
                    let a = s.begin(&trace.tokens[index..index + 4], false)?;
                    ensure!(
                        a.accepted_len() == 4,
                        "frozen task draft rejected: {}",
                        trace.id
                    );
                    let predictions = a.predictions().to_vec();
                    a.commit()?;
                    actual.extend(predictions);
                    index += 4;
                } else {
                    let next = s.serial_step(*actual.last().unwrap())?;
                    actual.push(next);
                    index += 1;
                }
                ensure!(
                    actual == trace.tokens[..actual.len()],
                    "task target IDs changed: {}",
                    trace.id
                );
            }
            let stop = if frozen.stop_ids.contains(actual.last().unwrap()) {
                "stop"
            } else {
                "length"
            };
            let output = tokenizer.decode_ignoring_specials(&actual)?;
            ensure!(
                actual == trace.tokens && stop == trace.stop_reason && output == trace.output,
                "task output/count/stop changed: {}",
                trace.id
            );
            write_record(
                out,
                json!({"kind":"direct_verifier_quality","id":trace.id,"is_direct":is_direct,
                "actual_token_ids":actual,"output":output,"stop_reason":stop,"has_identical_generation":true}),
            )?;
        }
    }
    Ok(())
}

pub(super) fn run(
    model: &mut LoadedModel,
    frozen: &Frozen,
    tokenizer: &Tokenizer,
    out: &mut File,
) -> Result<()> {
    let reference_kv = browser(model, frozen, out)?;
    transactions(model, frozen, &reference_kv, out)?;
    quality(model, frozen, tokenizer, out)
}
