# gzip -dc 2026-10-31-rotated-preparation-timing.jsonl.gz | jq -s -f this-file
def records($kind): map(select(.kind == $kind));
def bounds: [min, max];
records("rotated_preparation_comparison") as $comparisons |
records("rotated_preparation_correctness") as $before |
records("rotated_preparation_case_complete") as $after |
($before + $after) as $checks |
{
  manifest: (records("rotated_preparation_manifest")[0] |
    {revision, executable_sha256, model_sha256, fixture_sha256,
     capture_report_sha256, model_setup_ms, build_profile, timing_boundary}),
  completion: records("rotated_preparation_complete"),
  counts: (group_by(.kind) | map({kind: .[0].kind, count: length})),
  all_numerical_checks_passed: ($checks | length == 48 and all(.checks.passed)),
  all_canonical_bits_equal: ($checks | all(.checks.canonical_bits_equal)),
  all_encoded_bytes_equal: ($checks | all(.checks.encoded_bytes_equal)),
  all_consumer_bits_equal: ($checks | all(.checks.consumer_bits_equal)),
  all_case_ownership_passed: ($after | length == 24 and all(.ownership_passed)),
  all_width_ownership_passed: (records("rotated_preparation_width_complete") |
    length == 6 and all(.ownership_passed)),
  maximum_rotation_error: ([$checks[].checks.all_arms[].max_rotation_error] | max),
  maximum_output_error: ([$checks[].checks.all_arms[].max_output_error] | max),
  maximum_output_bound_fraction: ([$checks[].checks.all_arms[].max_output_bound_fraction] | max),
  maximum_throughput_variation: ([$comparisons[] |
    (.screen.throughput_mad_over_median[], .confirmation.throughput_mad_over_median[]?)] | max),
  extended_screens: ([$comparisons[] | select((.screen.pairs | length) > 5)] | length),
  confirmations: ([$comparisons[] | select(.confirmation != null)] | length),
  all_comparisons_stable: ($comparisons | all((.confirmation // .screen).is_stable)),
  all_comparisons_passed: ($comparisons | length == 48 and all(.passes_operator_screen)),
  scratch_bytes: ([$before[].scratch_bytes] | bounds),
  peak_tracked_device_bytes: ([$before[].tracked_device_bytes] | max),
  weight_preparation: records("rotated_preparation_weights"),
  comparisons: [$comparisons[] | {case, passed: .passes_operator_screen,
    screen_pairs: (.screen.pairs | length), confirmation_pairs: (.confirmation.pairs // [] | length),
    summary: ((.confirmation // .screen) |
      {native_median_ms, candidate_median_ms, throughput_gain,
       bootstrap_gain_ci95, required_gain, is_stable})}],
  by_shape: ($comparisons | group_by([.case.stage,.case.columns,.case.rows]) | map({
    stage: .[0].case.stage, columns: .[0].case.columns, rows: .[0].case.rows,
    control_ms: (map((.confirmation // .screen).native_median_ms) | bounds),
    candidate_ms: (map((.confirmation // .screen).candidate_median_ms) | bounds),
    gain: (map((.confirmation // .screen).throughput_gain) | bounds),
    saved_ms: (map((.confirmation // .screen) | .native_median_ms - .candidate_median_ms) | bounds),
    all_passed: all(.passes_operator_screen)
  }))
}
