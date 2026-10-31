def decided: if .confirmation == null then .screen else .confirmation end;
def span: {min:min,max:max};
def stats:
  {cases:length,passed:map(select(.passes_operator_screen))|length,
   unstable:map(select((decided.is_stable)|not))|length,
   extended:map(select(.screen.pairs|length>5))|length,
   confirmed:map(select(.confirmation!=null))|length,
   gain:map(decided.throughput_gain)|span,
   control_ms:map(decided.native_median_ms)|span,
   candidate_ms:map(decided.candidate_median_ms)|span,
   variation:map(decided.throughput_mad_over_median[])|max};
if (map(select(.kind=="shiftadd_complete" and .is_complete))|length)!=1 then error("incomplete run") else . end |
{
 manifest:map(select(.kind=="shiftadd_manifest"))[0],
 counts:(group_by(.kind)|map({key:.[0].kind,value:length})|from_entries),
 comparisons:(map(select(.kind=="shiftadd_comparison"))|
   {all:stats,by_control:(group_by(.case.control)|map({control:.[0].case.control,all:stats,batch_one:(map(select(.case.rows==1))|stats)})),
    cases:map({case,pass:.passes_operator_screen,decision:decided})}),
 weights:map(select(.kind=="shiftadd_weights")|{layer_id,role,report}),
 projection_errors:map(select(.kind=="shiftadd_validation" and .phase=="before")|
   {case,error:.preparation.projection_fit_error,preparation:.preparation}),
 correctness:{
   validations:map(select(.kind=="shiftadd_validation"))|length,
   output_values:map(select(.kind=="shiftadd_validation")|.checks.checks[].arithmetic.values)|add,
   max_bound_fraction:map(select(.kind=="shiftadd_validation")|.checks.checks[].arithmetic.maximum_bound_fraction)|max,
   max_error:map(select(.kind=="shiftadd_validation")|.checks.checks[].arithmetic.maximum_abs_error)|max},
 peak_tracked_device_bytes:map(select(.kind=="shiftadd_validation" and .phase=="before")|.peak_tracked_device_bytes)|max,
 cleanup:map(select(.kind=="shiftadd_cleanup")),
 complete:map(select(.kind=="shiftadd_complete"))[0]
}
