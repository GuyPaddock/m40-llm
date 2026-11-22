def check($condition; $message):
    if $condition then . else error($message) end;

def counters:
    .target_counters | .cublas_calls == 0 and .device_allocs == 0 and .device_frees == 0;

def work:
    [.generated_token_ids,.stop_reason,.attempts,.rejections,.lookup_count,
        .lookup_misses,.accepted_lengths_1_to_4,.cleared_suffix_words];

def validate_result:
    counters and (.total_ms > 0)
    and (((.resident_tps - (1000 * (.generated_token_ids | length) / .total_ms)) | fabs) < 1e-10)
    and (.attempts == (.accepted_lengths_1_to_4 | add))
    and (.rejections == (.accepted_lengths_1_to_4[:3] | add))
    and (.lookup_count == .lookup_misses + .attempts);

def variation($comparison):
    [$comparison.control.mad_over_median, $comparison.candidate.mad_over_median] | max;

def comparison_checks:
    .shared_vs_ordinary as $ordinary | .shared_vs_multi_row as $incremental
    | (2 * variation($incremental)) as $threshold
    | ([0.03, 2 * variation($ordinary)] | max) as $ordinary_threshold
    | check(
        .incremental_threshold == $threshold
        and $ordinary.promotion_threshold == $ordinary_threshold
        and .is_unstable == (variation($ordinary) > 0.05 or variation($incremental) > 0.05)
        and .passes_incremental == ($incremental.gain > $threshold and $incremental.bootstrap_gain_ci95[0] > 0)
        and $ordinary.passes_promotion_gate == (
            $ordinary.gain >= $ordinary_threshold and $ordinary.bootstrap_gain_ci95[0] > 0
            and variation($ordinary) <= 0.05
        )
        and .needs_confirmation == (
            (($ordinary.gain - $ordinary_threshold) | fabs) <= 0.01
            or ($ordinary.gain >= $ordinary_threshold and $ordinary.bootstrap_gain_ci95[0] <= 0)
            or (($incremental.gain - $threshold) | fabs) <= 0.01
            or ($incremental.gain > $threshold and $incremental.bootstrap_gain_ci95[0] <= 0)
        ); "invalid promotion checks");

$correctness[0] as $pre
| $timing[0] as $timed
| ["revision","executable_sha256","model_sha256","heldout_sha256","environment","arms"] as $identity
| check(all($identity[]; . as $key | $pre[$key] != null and $pre[$key] == $timed[$key]);
    "mismatched build, model or inputs")
| check($pre.is_correctness_only and ($timed.is_correctness_only | not)
    and $pre.arms == [null, "multi_row", "shared_weights"]
    and $correctness[-1].is_correctness_passed
    and all([$correctness[-1], $timing[-1]][]; .does_release_all_device_bytes and .device_bytes_after == 0);
    "failed correctness or incomplete device cleanup")
| $correctness[1:-1] as $cases
| check(($cases | length) == 21 and ($cases | map(.id) | unique | length) == 21
    and all($cases[]; . as $row
        | .has_exact_valid_kv and .valid_kv_words > 0 and (.valid_kv_words % (36 * 512)) == 0
        and .kind == "copy_schedule_correctness"
        and (.results | map(.down_projection)) == $pre.arms
        and all(.results[]; .includes_debug_readbacks and validate_result)
        and (.results[1]|work) == (.results[2]|work)
        and all(.results[1:][]; .generated_token_ids == $row.results[0].generated_token_ids
            and .stop_reason == $row.results[0].stop_reason and .has_full_logit_tolerance_passed
            and .compared_full_logit_rows == ((.generated_token_ids | length) - 1)));
    "incomplete exact output/KV or logit coverage")
| check(any($cases[]; .results[2].cleared_suffix_words > 0); "natural rollback missing")
| [$timing[] | select(.kind == "copy_schedule_comparison") | comparison_checks] as $comparisons
| [$timing[] | select(.kind == "copy_schedule_gate")] as $gates
| check(($gates | map(.id) | sort) == ["held-copy-long","held-copy-short"]; "missing copy gates")
| [$gates[] | . as $gate
    | ($cases[] | select(.id == $gate.id)) as $case
    | ([$comparisons[] | select(.id == $gate.id)]) as $series
    | ($series[-1]) as $final
    | ($series | map(.phase)) as $phases
    | check(any([["screen"], ["screen","screen"], ["screen","confirmation"],
        ["screen","screen","confirmation"]][]; . == $phases); "unexpected series")
    | check($gate.disposition == (
        if $final.is_unstable then "INCONCLUSIVE"
        elif $final.shared_vs_ordinary.passes_promotion_gate and $final.passes_incremental then "PASS"
        else "FAIL" end); "mismatched final gate")
    | ([$timing[] | select(.kind == "copy_schedule_trial" and .id == $gate.id)]) as $trials
    | check(all($trials[]; .arm >= 0 and .arm <= 2
        and .result.down_projection == $pre.arms[.arm]
        and (.result.includes_debug_readbacks | not) and (.result | validate_result)
        and (.result | work[:7]) == ($case.results[.arm] | work[:7]));
        "timed work, IDs or counters differ")
    | check(all(($trials | group_by([.phase, .round]))[];
        (map(.arm) | sort) == [0,1,2]); "unmatched timed round")
    | ([$trials[] | select(.phase == "warmup")] | length) as $warmup
    | ($series | map(select(.phase == "screen")) | last | .shared_vs_ordinary.pairs) as $screen_count
    | ($series | map(select(.phase == "confirmation")) | last | .shared_vs_ordinary.pairs // 0) as $confirmation_count
    | check($warmup == 3 and ($trials | length) == 3 * (1 + $screen_count + $confirmation_count);
        "unexpected count of warmed and timed arms")
    | {id, disposition, prompt_tokens: ($case.valid_kv_words / (36 * 512) - ($case.results[0].generated_token_ids|length) + 1),
        generated_tokens: ($case.results[0].generated_token_ids|length), rounds: ($series | map([.phase, .shared_vs_ordinary.pairs]))}
] as $workloads
| {correctness_cases: ($cases|length), workloads: $workloads, timings_and_key_checks_replayed: true, comparisons: $comparisons}
