def median: sort as $a | length as $n |
  if $n%2==1 then $a[$n/2|floor] else ($a[$n/2-1]+$a[$n/2])/2 end;
def validate($condition;$message): if $condition then . else error($message) end;
map(select(.kind=="direct_verifier_manifest")) as $m |
map(select(.kind=="direct_verifier_trial")) as $trials |
[$trials[]|select(.is_direct)|.result] as $direct |
validate(($m|length)==1 and $m[0].model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" and $m[0].rows==4;"wrong four-row model") |
validate(($trials|length)==10 and ($direct|length)==5;"missing paired resident trials") |
validate(all($direct[]; (.batch_ms|length)==127 and all(.batch_ms[];.>0) and
  .generated_tokens==512 and .committed_len==576 and (.layer_lengths|length)==36 and
  all(.layer_lengths[];.==576));"unmatched committed target work") |
validate(([$trials[].result.generated_token_ids]|unique|length)==1;"outputs differ") |
($direct|map(.batch_ms|add/length)|median) as $batch |
{source:$m[0].revision,model_sha256:$m[0].model_sha256,layers:36,
 four_row_batches:([$direct[].batch_ms|length]|add),
 median_of_request_mean_batch_ms:$batch,
 five_percent_target_latency_saving_ms:($batch*0.05),
 equivalent_saving_per_layer_us:($batch*0.05*1000/36),
 five_percent_target_throughput_saving_ms:($batch*(1-1/1.05)),
 median_resident_request_ms:($direct|map(.total_ms)|median),
 resident_request_latency_saving_ms:($direct|map(.total_ms)|median*0.05),
 scope:"historical measured four-row begin + verified commit, perfect drafts; operator extrapolation screens feasibility only"}
