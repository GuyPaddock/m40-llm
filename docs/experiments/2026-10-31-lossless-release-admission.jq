def rows($kind): [.[] | select(.kind == $kind)];
def case_key: [.layer,.length];
([ [0,17,35][] as $layer | [1024,2048,4096,8192][] | [$layer,.] ] | sort) as $grid |
([ [1,2,4,8][] as $batch | ["staged","dense","fused","serial_fused","native"][] | [$batch,.] ] | sort) as $arms |
(rows("lossless_attention_manifest")) as $manifest |
(rows("lossless_attention_setup")) as $setups |
(rows("lossless_attention_validation")) as $validations |
(rows("lossless_append_validation")) as $appends |
(rows("lossless_attention_case_complete")) as $cases |
(rows("lossless_attention_complete")) as $complete |
if length != 86 or ($manifest|length) != 1 or ($setups|length) != 12
  or ($validations|length) != 12 or ($appends|length) != 48
  or ($cases|length) != 12 or ($complete|length) != 1
  or $manifest[0].is_correctness_only != true
  or $complete[0].is_complete != true or $complete[0].cases != 12
  or $complete[0].generated_token_count != 0
  or ($validations | any(.result.has_bit_exact_reconstruction != true
      or .result.has_matched_prototype_output_bits != true
      or ([.result.errors[] | [.batch,.arm]] | sort) != $arms
      or (.result.errors | any(.max_absolute_fp64_error > 0.001))))
  or ([$validations[].case | case_key] | sort) != $grid
  or ([$cases[].case | case_key] | sort) != $grid
  or ([$setups[].case | case_key] | sort) != $grid
  or ([$appends[] | [.case.fixture.layer,.case.fixture.length,.case.batch]] | sort)
     != ([$grid[] as $case | [1,2,4,8][] | $case + [.]] | sort)
  or ($appends | any(.is_valid != true))
  or ($cases | any(.tracked_device_bytes_before != .tracked_device_bytes_after))
then error("incomplete or failed lossless release correctness grid")
else {
  revision:$manifest[0].revision,
  executable_sha256:$manifest[0].executable_sha256,
  artifact_sha256:$manifest[0].kv_artifact_sha256,
  cases:($cases|length),
  arm_batch_checks:([$validations[].result.errors[]]|length),
  append_checks:($appends|length),
  maximum_absolute_fp64_error:([$validations[].result.errors[].max_absolute_fp64_error]|max),
  complete:$complete[0],
  has_exact_reconstruction:true,
  has_matched_output_bits:true,
  has_tracked_cleanup:true,
  is_timing_measured:false
} end
