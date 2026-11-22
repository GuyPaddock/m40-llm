def median: sort | length as $n | if $n%2==1 then .[($n/2|floor)] else (.[($n/2)-1]+.[$n/2])/2 end;
map(select(.kind=="trial" and .phase=="screen")) as $rows |
map(select(.kind=="q8_execution_summary"))[0].comparison as $summary |
def arm($name): [$rows[] | select(.model_variant==$name)];
def stats($name): arm($name) | {
  pairs:length,
  client_ms:(map(.client_elapsed_ms)|median),
  http_tps:(map(1000*.metrics.generated_tokens/.client_elapsed_ms)|median),
  prefill_ms:(map(.metrics.prefill_elapsed_ms)|median),
  decode_tps:(map(1000*(.metrics.generated_tokens-1)/.metrics.decode_elapsed_ms)|median),
  preparation_bytes:(map(.q8_prepared_bytes)|unique),
  complete_prefill_tps:(map(1000*.metrics.prompt_tokens/.metrics.prefill_elapsed_ms)|median)
};
stats("control") as $control | stats("prepared") as $prepared |
if ($rows|length)!=10 or ($rows|all(.is_fixed_workload and .metrics.generated_tokens==512 and .metrics.prompt_tokens==65)|not)
  or ([$rows|group_by(.pair)[]| (map(.model_variant)|sort)==["control","prepared"] and (map(.output_sha256)|unique|length)==1]|all|not)
  or (($control.http_tps-$summary.control.median)|fabs)>0.000001
  or (($prepared.http_tps-$summary.candidate.median)|fabs)>0.000001
  or (map(select(.kind=="q8_token_comparison" and .different_id_count==0 and .candidate_generated_tokens==512))|length)!=1
then error("screen does not match measurement contract")
else {control:$control,prepared:$prepared,recomputed_gain:($prepared.http_tps/$control.http_tps-1),reported_bootstrap_interval:$summary.bootstrap_gain_ci95} end
