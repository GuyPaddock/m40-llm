def rows($kind): [.[] | select(.kind == $kind)];
def comparison_key: [.case.fixture.layer,.case.fixture.length,.case.batch,.comparison];
def distribution: {min:min,max:max};
(rows("lossless_attention_manifest")) as $manifest |
(rows("lossless_attention_validation")) as $validations |
(rows("lossless_append_validation")) as $appends |
(rows("lossless_attention_case_complete")) as $cases |
(rows("lossless_attention_complete")) as $complete |
(rows("lossless_pair")) as $pairs |
(rows("lossless_comparison") | group_by(comparison_key) | map(last)) as $final |
([ [0,17,35][] as $layer | [1024,2048,4096,8192][] | [$layer,.] ] | sort) as $grid |
["dense_vs_staged","staged_vs_fused","dense_vs_fused","native_serial_vs_fused_batched","fused_serial_vs_batched","native_append_attention_vs_cpu_replacement_attention"] as $names |
if ($manifest|length) != 1 or $manifest[0].is_correctness_only != false
  or ($validations|length) != 12 or ($appends|length) != 48
  or ($cases|length) != 12 or ($complete|length) != 1
  or $complete[0].is_complete != true or $complete[0].cases != 12
  or $complete[0].generated_token_count != 0 or ($final|length) != 288
  or ([$cases[].case | [.layer,.length]] | sort) != $grid
  or ([$final[] | comparison_key] | sort)
      != ([$grid[] as $case | [1,2,4,8][] as $batch | $names[] | $case + [$batch,.]] | sort)
  or ($validations | any(.result.has_bit_exact_reconstruction != true
      or .result.has_matched_prototype_output_bits != true
      or (.result.errors|length) != 20
      or (.result.errors|any(.max_absolute_fp64_error > 0.001))))
  or ($appends | any(.is_valid != true))
  or ($cases | any(.tracked_device_bytes_before != .tracked_device_bytes_after))
then error("incomplete or invalid lossless timing grid")
else {
  revision:$manifest[0].revision,
  executable_sha256:$manifest[0].executable_sha256,
  artifact_sha256:$manifest[0].kv_artifact_sha256,
  generated_token_count:0,
  cases:($cases|length),arm_batch_checks:([$validations[].result.errors[]]|length),
  append_checks:($appends|length),pairs:($pairs|length),comparison_groups:($final|length),
  confirmation_groups:([$final[]|select(.is_confirmation)]|length),
  timing_errors:[$final[]|select(.timing_error != null)|{key:comparison_key,error:.timing_error}],
  maximum_absolute_fp64_error:([$validations[].result.errors[].max_absolute_fp64_error]|max),
  by_comparison:($final | group_by(.comparison) | map({
    comparison:.[0].comparison,groups:length,
    passes:([.[]|select(.wall_comparison.passes_operator_speed_gate and .timing_error==null)]|length),
    gain_range:([.[].wall_comparison.gain]|distribution),
    control_ms_range:([.[]|1000/.wall_comparison.control.median]|distribution),
    candidate_ms_range:([.[]|1000/.wall_comparison.candidate.median]|distribution)
  })),
  rows:[$final[]|{
    layer:.case.fixture.layer,length:.case.fixture.length,batch:.case.batch,
    comparison,control_ms:(1000/.wall_comparison.control.median),
    candidate_ms:(1000/.wall_comparison.candidate.median),
    gain:.wall_comparison.gain,gain_ci95:.wall_comparison.bootstrap_gain_ci95,
    passes:.wall_comparison.passes_operator_speed_gate,timing_error,
    pairs:.wall_comparison.pairs,is_confirmation,
    control_variation:.wall_comparison.control.mad_over_median,
    candidate_variation:.wall_comparison.candidate.mad_over_median,
    cuda_control_ms:(1000/.cuda_timeline_control.median),
    cuda_candidate_ms:(1000/.cuda_timeline_candidate.median)
  }],
  complete:$complete[0],is_model_speedup_measured:false
} end
