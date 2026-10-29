def require($test; $message): if $test then . else error($message) end;
def median: sort | length as $n | if $n % 2 == 1 then .[$n/2|floor] else (.[($n/2)-1] + .[$n/2])/2 end;
def rows($kind): [.[] | select(.kind == $kind)];
. as $all
| require(.[0].revision == $quality[0].revision and .[0].executable_sha256 == $quality[0].executable_sha256 and .[0].model_sha256 == $quality[0].model_sha256 and .[0].frozen_environment == $quality[0].frozen_environment; "quality identity differs")
| require(.[-1].kind == "portfolio_http_complete" and .[-1].is_gate_passed and .[-1].device_bytes_after == 0; "HTTP incomplete")
| require((rows("portfolio_http_arm_complete") | map(.candidate) | sort) == ["all","argmax","down","vocabulary"]; "comparison coverage")
| require((rows("portfolio_http_arm_complete") | all(.device_bytes_after == 0)); "teardown leak")
| require((rows("portfolio_http_trial") | length) == 120; "timing coverage")
| require((rows("portfolio_http_trial") | all(. as $r | ($quality[] | select(.kind == "portfolio_generation" and .arm == (if $r.arm == 0 then "control" else $r.candidate end))) as $q |
    .row.output == $q.output and .row.dispatch == $q.dispatch
    and .row.metrics.generated_tokens == 512 and .row.metrics.prompt_tokens == 65 and .row.metrics.stop_reason == "length"
    and (.row.http_tok_per_sec - 512000/.row.http_ms | fabs) < 1e-10)); "timed work differs")
| require((rows("portfolio_http_trial") | all([.gpu_before,.gpu_after] | all(.clocks | endswith(", P0, 1113, 3004")))); "clocks differ")
| require((rows("portfolio_http_trial") | all([.gpu_before,.gpu_after] | all((.processes | split("\n") | map(select(length > 0)) | length) == 1))); "GPU not isolated")
| require((rows("portfolio_http_trial") | map([.gpu_before,.gpu_after][] | .processes | split(",")[1]) | unique | length) == 1; "GPU process changes")
| require((rows("portfolio_http_trial") | group_by([.candidate,.phase,.pair]) | all(length == 2 and (map(.arm)|sort) == [0,1])); "unpaired or duplicate observations")
| (rows("portfolio_http_summary")) as $summaries
| require(($summaries | length) == 6; "summary coverage")
| require(($summaries | all(. as $summary | [$all[] | select(.kind == "portfolio_http_trial" and .candidate == $summary.candidate and .phase == $summary.phase)] as $trials |
    ($trials | length) == 2*.comparison.pairs
    and .comparison.pairs == (if .phase == "confirmation" then 20 else 5 end)
    and ([$trials[]|select(.arm == 0)|.row.http_tok_per_sec]|median) == .comparison.control.median
    and ([$trials[]|select(.arm == 1)|.row.http_tok_per_sec]|median) == .comparison.candidate.median
    and (.comparison.gain - (.comparison.candidate.median/.comparison.control.median-1)|fabs) < 1e-12
    and .comparison.bootstrap_resamples == 10000 and .comparison.bootstrap_seed == 0
    and .comparison.bootstrap_gain_ci95[0] > 0
    and .comparison.bootstrap_gain_ci95[0] <= .comparison.bootstrap_gain_ci95[1]
    and .comparison.needs_variation_extension == false)); "summary inconsistent with raw pairs")
| require(($summaries | map(select(.candidate == "all" and .phase == "confirmation")) | length) == 1; "combined confirmation missing")
| require(($summaries | map(select(.candidate == "all" and .phase == "confirmation")) | all(.comparison.passes_promotion_gate and .comparison.gain >= .comparison.promotion_threshold and .comparison.promotion_threshold >= 0.03)); "combined performance gate failed")
| {is_passed:true, revision:.[0].revision, executable_sha256:.[0].executable_sha256,
   timed_requests:120, tokens_per_request:512, phases_not_pooled:true,
   results:[$summaries[] | {candidate,phase,pairs:.comparison.pairs,control_http_tps:.comparison.control.median,
     candidate_http_tps:.comparison.candidate.median,gain_percent:(100*.comparison.gain),
     paired_ci95_percent:(.comparison.bootstrap_gain_ci95|map(100*.)),
     control_variation_percent:(100*.comparison.control.mad_over_median),
     candidate_variation_percent:(100*.comparison.candidate.mad_over_median),
     requires_confirmation:.comparison.needs_fresh_confirmation,next_pairs} ]}
