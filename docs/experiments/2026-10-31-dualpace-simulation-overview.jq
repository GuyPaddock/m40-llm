def range_values: {min:min,max:max};
if length!=108 or any(.is_simulation!=true or .is_serving_measurement!=false or .generated_token_count!=0 or (.modes|length)!=3)
then error("incomplete simulation set") else
{scenarios:length,modes:([.[].modes[]]|length),
  admitted:([.[].modes[]|select(.is_admitted)]|length),
  rejected:[.[]|.description as $description|.modes[]|select(.is_admitted!=true)|{description:$description,mode,error}],
  by_mode:([.[].modes[]|select(.is_admitted)]|group_by(.mode)|map({mode:.[0].mode,cases:length,
    ratio_to_gpu_only:(map(.throughput_ratio_to_gpu_only)|range_values),
    above_gpu_only:(map(select(.throughput_ratio_to_gpu_only>1))|length),
    simulated_tokens_per_second:(map(.simulated_tokens_per_second)|range_values),
    first_decode_p50_ms:(map(.first_decode_latency.p50_ns/1e6)|range_values),
    inter_token_p50_ms:(map(.inter_token_latency.p50_ns/1e6)|range_values),
    inter_token_p95_ms:(map(.inter_token_latency.p95_ns/1e6)|range_values),
    inter_token_p99_ms:(map(.inter_token_latency.p99_ns/1e6)|range_values),
    host_reserved_bytes:(map(.host_reserved_bytes)|range_values),gpu_reserved_bytes:(map(.gpu_reserved_bytes)|range_values)})),
  local_assistance_saving_ms:([.[].local_assistance[].saving_before_queueing_ns/1e6]|range_values),
  assisted_vs_independent_ratio:([.[]|(.modes|map(select(.mode=="independent" and .is_admitted))|.[0].elapsed_ns) as $control |
    .modes[]|select(.mode=="assisted" and .is_admitted)|$control/.elapsed_ns]|range_values),
  winning_assisted:[.[]|. as $row|.modes[]|select(.mode=="assisted" and .is_admitted and .throughput_ratio_to_gpu_only>1)|{node:$row.cpu_numa_node,description:$row.description,ratio:.throughput_ratio_to_gpu_only}],
  is_measured_serving_result:false}
end
