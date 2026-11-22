def check($ok; $reason): if $ok then . else error($reason) end;
def variation($c): [$c.control.mad_over_median, $c.candidate.mad_over_median] | max;
def settled_gate($c): $c.gain >= $c.promotion_threshold and $c.bootstrap_gain_ci95[0] > 0 and variation($c) <= 0.05;

["control","down","vocabulary","argmax","all"] as $arms
| ($quality | map(select(.kind == "portfolio_generation"))) as $generation
| ($generation[] | select(.arm == "control")) as $control
| check(all(["revision","executable_sha256","model_sha256","frozen_environment"][]; . as $key
        | $quality[0][$key] != null and $quality[0][$key] == $timing[0][$key])
    and ([$timing[] | select(.kind == "portfolio_http_binding") | .quality_sha256] == [$digest]);
    "quality/timing provenance differs")
| check(($generation | map(.arm)) == $arms and all($generation[];
        .prompt_ids == $control.prompt_ids and .generated_ids == $control.generated_ids
        and .output == $control.output and (.prompt_ids | length) == 65 and (.generated_ids | length) == 512);
    "portfolio workload IDs differ")
| ($quality | map(select(.kind == "portfolio_task"))) as $tasks
| ($tasks | map(select(.arm == "control"))) as $control_tasks
| ($quality | map(select(.kind == "portfolio_kv"))) as $kv
| ($kv[] | select(.arm == "control") | .sha256) as $kv_sha
| check(($kv | map(.arm)) == $arms
    and all($kv[]; .is_bitwise_equal and .sha256 == $kv_sha and .length == 576 and .valid_words == 10616832);
    "invalid KV digest comparison")
| ($quality | map(select(.kind == "portfolio_logits"))) as $logits
| check(($logits | length) == 35 and all($logits | group_by(.step)[];
        (map(.arm) | sort) == ($arms | sort)
        and (map(.sha256) | unique | length) == 1
        and all(.[]; .vocabulary == 151936 and .is_bitwise_equal and .original_sha256 == .sha256));
    "invalid full logits comparison")
| check(($tasks | length) == 60 and ($control_tasks | map(select(.score.passed)) | length) == 7
    and all($tasks | group_by(.id)[];
        (map(.arm) | sort) == ($arms | sort) and (map([.ids,.score.passed]) | unique | length) == 1);
    "unmatched quality tasks")
| check(all($arms[]; . as $arm
        | ([$quality[] | select(.kind == "portfolio_quality_arm_complete" and .arm == $arm
            and .is_gate_passed and .device_bytes_after == 0 and .dense_valid == 7 and .inconclusive == 5)] | length) == 1)
    and $quality[-1].kind == "portfolio_quality_complete" and $quality[-1].is_gate_passed;
    "incomplete quality gate")
| ([$timing[] | select(.kind == "portfolio_http_summary")]) as $summaries
| ([$timing[] | select(.kind == "portfolio_http_trial")]) as $trials
| check(all($trials[]; . as $trial
    | ($generation[] | select(.arm == (if $trial.arm == 0 then "control" else $trial.candidate end))) as $expected
    | (.arm == 0 or .arm == 1) and .row.output == $expected.output and .row.dispatch == $expected.dispatch
      and .row.metrics.prompt_tokens == 65 and .row.metrics.generated_tokens == 512 and .row.metrics.stop_reason == "length"
      and .row.http_ms > 0 and ((.row.http_tok_per_sec - 512000 / .row.http_ms) | fabs) < 1e-10);
    "HTTP event differs from its qualified workload")
| check(all($summaries[]; .comparison as $c
    | $c.promotion_threshold == ([0.03, 2 * variation($c)] | max)
      and $c.passes_promotion_gate == settled_gate($c)
      and .reaches_historical_http_target == ($c.candidate.median >= 56.153149));
    "wrong recorded decision")
| [$arms[1:][] as $arm
    | ($summaries | map(select(.candidate == $arm))) as $series
    | $series[-1] as $final
    | check($final.comparison.pairs > 0 and $final.next_pairs == null
        and ([$timing[] | select(.kind == "portfolio_http_arm_complete" and .candidate == $arm and .device_bytes_after == 0)] | length) == 1;
        "incomplete HTTP qualification")
    | ($trials | map(select(.candidate == $arm))) as $rows
    | check(all($rows | group_by([.phase,.pair])[]; (map(.arm) | sort) == [0,1]);
        "incomplete HTTP pair")
    | {candidate:$arm, final_phase:$final.phase, final_comparison:$final.comparison,
        historical_target_reached:$final.reaches_historical_http_target, timed_requests:($rows | length)}
] as $reports
| check($timing[-1].kind == "portfolio_http_complete" and $timing[-1].device_bytes_after == 0
    and $timing[-1].is_gate_passed == $reports[-1].final_comparison.passes_promotion_gate;
    "missing final gate")
| {revision:$quality[0].revision, hash:$digest, prompt_tokens:65, generated_tokens:512, passing_tasks:7,
    dense_failures:5, valid_kv_words:10616832, exact_logit_rows_per_arm:7, summaries:$summaries, results:$reports,
    loads:[$quality[] | select(.kind == "portfolio_load")
        | {arm, original_load_ms, tracked_device_delta, packet_setup_device_high_water,
            packet_bytes:.packets.resident_bytes, prepared_bytes:.packets.combined_prepared_bytes,
            model_setup_ms:.packets.preparation_ms, temporary_host_bytes:.packets.peak_temporary_host_bytes}]}
