def rows($kind): [.[] | select(.kind == $kind)];
def case_key: [.layer,.shape.length,.shape.head_dim,.backing];
([ [0,17,35][] as $layer | [1024,2048,4096,8192][] as $length |
   [[64,"q8"],[64,"mixed"],[128,"mixed"]][] | [$layer,$length] + . ] | sort) as $grid |
(rows("kv_attention_benchmark_manifest")) as $manifest |
(rows("kv_attention_validation")) as $validation |
(rows("kv_attention_case_complete")) as $cases |
(rows("kv_attention_benchmark_complete")) as $complete |
if length != 110 or ($manifest|length) != 1 or $manifest[0].is_correctness_only != true
  or ($validation|length) != 36 or ($cases|length) != 36 or ($complete|length) != 1
  or $complete[0].is_complete != true or $complete[0].cases != 36
  or $complete[0].generated_token_count != 0
  or ([$cases[].case|case_key]|sort) != $grid
  or ([$validation[].case|case_key]|sort) != $grid
  or ($validation|any(.result.is_valid != true
    or .result.staged_values_bit_exact_after_identity_append != true
    or .result.query_and_output_guards_intact != true
    or .result.inactive_batch_rows_unchanged != true
    or .result.repeated_serial_outputs_bit_exact != true
    or ([.result.staged_vs_fp64,.result.direct_vs_fp64,.result.dense_vs_fp64,.result.staged_vs_direct,.result.batched_vs_serial_direct]
      |any(. != null and (.elements<=0 or .max_absolute_error>=0.001)))))
  or ($cases|any(.tracked_device_bytes_before != .tracked_device_bytes_after))
then error("incomplete or failed quantized KV release correctness grid")
else {
  revision:$manifest[0].revision,executable_sha256:$manifest[0].executable_sha256,
  artifact_sha256:$manifest[0].kv_artifact_sha256,cases:36,
  maximum_staged_fp64_error:([$validation[].result.staged_vs_fp64.max_absolute_error]|max),
  maximum_direct_fp64_error:([$validation[].result.direct_vs_fp64.max_absolute_error]|max),
  maximum_dense_fp64_error:([$validation[].result.dense_vs_fp64.max_absolute_error]|max),
  maximum_staged_direct_error:([$validation[].result.staged_vs_direct.max_absolute_error]|max),
  maximum_batched_serial_error:([$validation[].result.batched_vs_serial_direct|select(.!=null)|.max_absolute_error]|max),
  maximum_selection_approximation:([$validation[].result.selection_only_vs_full_fp16.max_absolute_error]|max),
  maximum_quantization_approximation:([$validation[].result.quantization_only_at_fixed_positions.max_absolute_error]|max),
  complete:$complete[0],is_timing_measured:false,is_model_quality_measured:false
} end
