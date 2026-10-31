def decision: if .confirmation == null then .screen else .confirmation end;
def range_of(f): [map(f) | min, max];

if .[-1].kind != "q4_channel_complete" or .[-1].is_complete != true or .[-1].cases != 216 then
  error("require the completed full Q4 grid")
else . end
| . as $records
| [.[] | select(.kind == "q4_channel_correctness")] as $checks
| if ($checks | length) != 810 or ($checks | all(.passed)) != true then
    error("missing or failed Q4 numerical checks")
  else . end
| [.[] | select(.kind == "q4_channel_comparison")] as $comparisons
| if ($comparisons | length) != 756 then error("missing Q4 comparisons") else . end
| ($comparisons
    | group_by([.case.layer, .case.rows, .case.selection, .case.retained_percent, .arms[1]])
    | map({rows: .[0].case.rows, retention: .[0].case.retained_percent,
        candidate: .[0].arms[1], passes_all_controls: all(.passes_operator_screen),
        strongest_control_speedup: (min_by(decision.native_median_ms) | decision.throughput_gain + 1)})) as $cases
| {
    revision: .[0].revision, executable_sha256: .[0].executable_sha256,
    model_sha256: .[0].model_sha256, fixture_sha256: .[0].fixture_sha256,
    cases: .[-1].cases, numerical_checks: ($checks | length),
    maximum_absolute_numerical_error: ($checks | map(.error.max_abs_diff) | max),
    comparison_count: ($comparisons | length),
    pair_count: ([$records[] | select(.kind == "q4_channel_pair")] | length),
    extended_screens: ($comparisons | map(select((.screen.pairs | length) == 10)) | length),
    confirmations: ($comparisons | map(select(.confirmation != null)) | length),
    unstable: ($comparisons | map(select(.screen.is_stable == false) |
        {case, arms, variation: .screen.throughput_mad_over_median})),
    batch_one: ($comparisons | map(select(.case.rows == 1))
        | group_by([.arms[1], .case.retained_percent])
        | map({candidate: .[0].arms[1], retention: .[0].case.retained_percent,
            comparisons: length, passes: (map(select(.passes_operator_screen)) | length),
            control_ms: range_of(decision.native_median_ms),
            candidate_ms: range_of(decision.candidate_median_ms),
            speedup: range_of(decision.throughput_gain + 1),
            minimum_gain_lower_bound: (map(decision.bootstrap_gain_ci95[0]) | min)})),
    by_batch: ($cases | group_by([.candidate, .rows])
        | map({candidate: .[0].candidate, rows: .[0].rows, cases: length,
            passes: (map(select(.passes_all_controls)) | length),
            retention: (group_by(.retention) | map({retained_percent: .[0].retention,
                passes: (map(select(.passes_all_controls)) | length),
                speedup_against_strongest_control: range_of(.strongest_control_speedup)}))}))
  }
