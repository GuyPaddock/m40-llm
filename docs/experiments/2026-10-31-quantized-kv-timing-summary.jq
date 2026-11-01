def rows($kind): [.[] | select(.kind == $kind)];
def fixture_key: [.layer,.shape.length,.shape.head_dim,.backing];
def comparison_key: (.case.fixture|fixture_key) + [.case.batch,.comparison];
([ [0,17,35][] as $layer | [1024,2048,4096,8192][] as $length |
   [[64,"q8"],[64,"mixed"],[128,"mixed"]][] | [$layer,$length] + . ] | sort) as $grid |
(rows("kv_attention_benchmark_manifest")) as $manifest |
(rows("kv_attention_validation")) as $validation |
(rows("kv_attention_case_complete")) as $cases |
(rows("kv_attention_benchmark_complete")) as $complete |
(rows("kv_attention_pair")) as $pairs |
(rows("kv_attention_comparison")|group_by(comparison_key)|map(last)) as $final |
if ($manifest|length) != 1 or $manifest[0].is_correctness_only != false
  or ($validation|length) != 36 or ($cases|length) != 36 or ($complete|length) != 1
  or $complete[0].is_complete != true or $complete[0].cases != 36
  or $complete[0].generated_token_count != 0 or ($final|length) != 384
  or ([$cases[].case|fixture_key]|sort) != $grid
  or ([$validation[].case|fixture_key]|sort) != $grid
  or ([$final[]|comparison_key]|sort)
    != ([$grid[] as $fixture | [1,2,4,8][] as $batch |
         (if $fixture[3]=="mixed" then ["staged_vs_direct","serial_vs_batched_direct","dense_context_only"]
          else ["staged_vs_direct","dense_context_only"] end)[] | $fixture+[$batch,.]]|sort)
  or ($validation|any(.result.is_valid != true
    or .result.staged_values_bit_exact_after_identity_append != true
    or .result.query_and_output_guards_intact != true
    or .result.inactive_batch_rows_unchanged != true
    or .result.repeated_serial_outputs_bit_exact != true
    or ([.result.staged_vs_fp64,.result.direct_vs_fp64,.result.dense_vs_fp64,.result.staged_vs_direct,.result.batched_vs_serial_direct]
        |any(. != null and (.elements<=0 or .max_absolute_error>=0.001)))))
  or ($cases|any(.tracked_device_bytes_before != .tracked_device_bytes_after))
then error("incomplete or invalid quantized KV timing grid")
else {
  revision:$manifest[0].revision,executable_sha256:$manifest[0].executable_sha256,
  artifact_sha256:$manifest[0].kv_artifact_sha256,cases:36,
  pairs:($pairs|length),comparison_groups:($final|length),
  confirmation_groups:([$final[]|select(.is_confirmation)]|length),
  extended_groups:([$final[]|select(.wall_comparison.pairs==10)]|length),
  timing_errors:[$final[]|select(.timing_error!=null)|{key:comparison_key,error:.timing_error}],
  maximum_staged_fp64_error:([$validation[].result.staged_vs_fp64.max_absolute_error]|max),
  maximum_direct_fp64_error:([$validation[].result.direct_vs_fp64.max_absolute_error]|max),
  maximum_dense_fp64_error:([$validation[].result.dense_vs_fp64.max_absolute_error]|max),
  maximum_staged_direct_error:([$validation[].result.staged_vs_direct.max_absolute_error]|max),
  maximum_batched_serial_error:([$validation[].result.batched_vs_serial_direct|select(.!=null)|.max_absolute_error]|max),
  maximum_selection_approximation:([$validation[].result.selection_only_vs_full_fp16.max_absolute_error]|max),
  maximum_quantization_approximation:([$validation[].result.quantization_only_at_fixed_positions.max_absolute_error]|max),
  by_format_comparison:($final|group_by([.case.fixture.backing,.case.fixture.shape.head_dim,.comparison])|map({
    backing:.[0].case.fixture.backing,head_dim:.[0].case.fixture.shape.head_dim,comparison:.[0].comparison,
    groups:length,coverage_matches:.[0].coverage_matches,
    passes:([.[]|select(.wall_comparison.passes_operator_speed_gate==true and .timing_error==null)]|length),
    gain_range:([.[].wall_comparison.gain]|{min:min,max:max}),
    latency_ratio_range:([.[]|.wall_comparison.control.median/.wall_comparison.candidate.median]|{min:min,max:max})
  })),
  rows:[$final[]|{
    layer:.case.fixture.layer,length:.case.fixture.shape.length,head_dim:.case.fixture.shape.head_dim,
    backing:.case.fixture.backing,batch:.case.batch,comparison,coverage_matches,
    control_ms:(1000/.wall_comparison.control.median),candidate_ms:(1000/.wall_comparison.candidate.median),
    gain:.wall_comparison.gain,gain_ci95:.wall_comparison.bootstrap_gain_ci95,
    passes:.wall_comparison.passes_operator_speed_gate,timing_error,
    pairs:.wall_comparison.pairs,is_confirmation,
    control_variation:.wall_comparison.control.mad_over_median,
    candidate_variation:.wall_comparison.candidate.mad_over_median,
    gpu_control_ms:(1000/.gpu_control.median),gpu_candidate_ms:(1000/.gpu_candidate.median)
  }],complete:$complete[0],is_model_speedup_measured:false,is_model_quality_measured:false
} end
