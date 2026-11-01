def records($kind): [.[]|select(.kind==$kind)];
def key: [.case.layer,.case.family,.case.batch];
records("cpu_projection_manifest") as $manifest |
records("cpu_projection_complete") as $complete |
records("cpu_projection_validation") as $validation |
(records("cpu_projection_comparison")|group_by(key)|map(last)) as $final |
([ [0,17,35][] as $layer | ["gate","up","down"][] as $family | [1,2,4,8][] | [$layer,$family,.]]|sort) as $grid |
if ($manifest|length)!=1 or $manifest[0].is_validation_only!=false
  or ($complete|length)!=1 or $complete[0].is_complete!=true or $complete[0].cases!=36
  or $complete[0].generated_token_count!=0 or ($validation|map(key)|sort)!=$grid
  or ($final|map(key)|sort)!=$grid
  or ($validation|any(.scalar.max_tolerance_ratio>1 or .avx2.max_tolerance_ratio>1
    or .avx2.elements<=0 or .scalar.elements!=.avx2.elements))
then error("incomplete CPU projection timing grid")
else {revision:$manifest[0].revision,executable_sha256:$manifest[0].executable_sha256,
  placement:($manifest[0].placement|{cpu,numa_node,threads}),cases:36,
  pairs:(records("cpu_projection_pair")|length),
  extensions:([$final[]|select(.comparison.pairs==10)]|length),
  confirmations:([$final[]|select(.is_confirmation)]|length),
  errors:[$final[]|select(.timing_error!=null)|{case,timing_error}],
  scalar_max_error:([$validation[].scalar.max_absolute_error]|max),
  avx2_max_error:([$validation[].avx2.max_absolute_error]|max),
  rows:[$final[]|{layer:.case.layer,family:.case.family,batch:.case.batch,
    scalar_ms:(1000/.comparison.control.median),avx2_ms:(1000/.comparison.candidate.median),
    gain:.comparison.gain,ci95:.comparison.bootstrap_gain_ci95,
    scalar_variation:.comparison.control.mad_over_median,avx2_variation:.comparison.candidate.mad_over_median,
    passes:.comparison.passes_operator_speed_gate,pairs:.comparison.pairs,is_confirmation,timing_error}],
  is_complete_decode_measured:false,generated_token_count:0}
end
