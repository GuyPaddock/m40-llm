def records($kind): [.[]|select(.kind==$kind)];
def key: [.case.name,.comparison];
records("cooperative_transfer_manifest") as $manifest |
records("cooperative_transfer_complete") as $complete |
records("cooperative_transfer_validation") as $validation |
records("cooperative_transfer_case_complete") as $cases |
records("cooperative_transfer_warmup") as $warmup |
(records("cooperative_transfer_costs")|group_by(key)|map(last)) as $final |
if ($manifest|length)!=1 or $manifest[0].is_validation_only!=false
  or $manifest[0].iterations!=100 or $manifest[0].warmup_minimum_wall_ms_per_arm!=1000
  or $manifest[0].warmup_minimum_calls_per_arm!=3 or ($warmup|length)!=54
  or ($warmup|any(.calls<3 or .wall_ms<1000 or .iterations!=(if .case.name=="idle_synchronization" then 1000 else 100 end)))
  or ($complete|length)!=1 or $complete[0].is_complete!=true or $complete[0].transfer_cases!=13
  or $complete[0].generated_token_count!=0 or ($validation|length)!=13
  or ($validation|any(.is_bit_exact!=true or .guards_unchanged!=true))
  or ($cases|length)!=13 or ($cases|any(.tracked_device_bytes_before!=.tracked_device_bytes_after))
  or ($final|length)!=27
then error("incomplete transfer timing grid")
else {revision:$manifest[0].revision,executable_sha256:$manifest[0].executable_sha256,
  placement:($manifest[0].placement|{cpu,numa_node,threads}),cases:13,groups:27,
  pairs:(records("cooperative_transfer_pair")|length),
  warmup_records:($warmup|length),minimum_warmup_ms:([$warmup[].wall_ms]|min),
  extensions:([$final[]|select(.paired_operation_rate_comparison.pairs==10)]|length),
  errors:[$final[]|select(.timing_error!=null)|{case,comparison,timing_error}],
  rows:[$final[]|{case,comparison,arms,wall_ns,cuda_timeline_ns,
    pairs:.paired_operation_rate_comparison.pairs,timing_error}],
  tracked_final:$complete[0].tracked_device_bytes_after,
  is_complete_decode_measured:false,generated_token_count:0}
end
