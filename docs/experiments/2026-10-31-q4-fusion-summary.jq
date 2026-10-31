# Reproduce with: gzip -dc 2026-10-31-q4-fusion-timing.jsonl.gz | jq -s -f this-file
def records($kind): map(select(.kind == $kind));
def bounds: [min, max];
records("ffn_fusion_comparison") as $comparisons |
records("ffn_fusion_correctness") as $checks |
{
  manifest: (records("ffn_fusion_manifest")[0] |
    {revision, executable_sha256, model_sha256, fixture_sha256,
     capture_report_sha256, build_profile, timing_boundary}),
  completion: records("ffn_fusion_complete"),
  counts: (group_by(.kind) | map({kind: .[0].kind, count: length})),
  all_numerical_checks_passed: ($checks | length == 12 and all(.checks.passed)),
  all_hidden_bits_equal: ($checks | all(.checks.serial_fused_hidden_bits_equal)),
  all_same_down_output_bits_equal: ($checks | all(.checks.same_down_complete_bits_equal)),
  all_case_ownership_passed: (records("ffn_fusion_case_complete") |
    length == 12 and all(.ownership_passed)),
  all_layer_ownership_passed: (records("ffn_fusion_layer_complete") |
    length == 3 and all(.ownership_passed)),
  maximum_fp64_absolute_error: ([$checks[].checks.all_arms[] |
    (.hidden.max_abs_diff, .output.max_abs_diff)] | max),
  maximum_throughput_variation: ([$comparisons[].screen.throughput_mad_over_median[]] | max),
  extended_screens: ([$comparisons[] | select((.screen.pairs | length) > 5)] | length),
  confirmations: ([$comparisons[] | select(.confirmation != null)] | length),
  all_comparisons_stable: ($comparisons | all((.confirmation // .screen).is_stable)),
  scratch_bytes: ([$checks[].scratch_bytes] | bounds),
  peak_tracked_device_bytes: ([$checks[].tracked_device_bytes] | max),
  prepare_upload_ms: ([records("ffn_fusion_layer")[].prepare_upload_ms] | bounds),
  complete_cases: ([$comparisons[] | select(.case.stage == "complete")] |
    group_by([.case.layer, .case.rows]) | map({
      layer: .[0].case.layer, rows: .[0].case.rows,
      clears_all_controls: (length == 3 and all(.passes_operator_screen)),
      comparisons: map({control: .arms[0], passed: .passes_operator_screen,
        summary: ((.confirmation // .screen) |
          {native_median_ms, candidate_median_ms, throughput_gain,
           bootstrap_gain_ci95, required_gain, is_stable})})
    })),
  diagnostic_gate_up_gain_over_serial: ([$comparisons[] |
    select(.case.stage == "gate_up" and .arms[0] == "serial_down") |
    (.confirmation // .screen).throughput_gain] | bounds)
}
