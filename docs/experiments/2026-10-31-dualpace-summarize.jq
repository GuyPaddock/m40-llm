# Summarize one complete simulator report. Preserve declared request ownership
# across modes so the GPU-only control remains comparable to CPU-owned requests.
def require($condition; $message): if $condition then . else error($message) end;
def get_quantiles:
  sort | length as $count | if $count == 0 then null else
  {samples:$count,p50_ns:.[($count*0.50|ceil)-1],
   p95_ns:.[($count*0.95|ceil)-1],p99_ns:.[($count*0.99|ceil)-1]} end;
def get_intervals: .completion_ns as $times | [range(1; $times|length) | $times[.] - $times[.-1]];
def get_queues:
  [.[] | {resource,queue_wait_ns}] | group_by(.resource) |
  map({resource:.[0].resource,intervals:length,total_wait_ns:(map(.queue_wait_ns)|add),
    wait_distribution:(map(.queue_wait_ns)|get_quantiles)});

require(length == 5 and .[0].kind == "cooperative_simulation_manifest"
  and .[-1].kind == "cooperative_simulation_complete" and .[-1].is_complete == true;
  "incomplete simulator report") |
.[0] as $manifest | .[-1] as $complete | .[1:4] as $rows |
require($manifest.schema_version == 1 and $manifest.is_simulation == true
  and $manifest.is_serving_measurement == false and $manifest.generated_token_count == 0
  and ($rows|all(.kind == "cooperative_simulation" and (.is_admitted|type) == "boolean"))
  and ($rows|map(if .is_admitted then .result.mode else .mode end)|sort) == ["assisted","gpu_only","independent"]
  and $complete.admitted_scenarios == ($rows|map(select(.is_admitted))|length);
  "report modes or simulation provenance differ") |
($manifest.trace.requests|map({key:.id,value:.})|from_entries) as $requests |
($manifest.trace.requests|map(.token_profiles|length)|add) as $tokens |
require(($requests|length) == ($manifest.trace.requests|length); "duplicate input request") |
require(($rows|all(if .is_admitted then
  .result.elapsed_ns > 0 and .result.simulated_tokens == $tokens
  and (.result.requests|map(.id)|sort) == ($requests|keys)
  and (.result.requests|all(. as $request |
    (.completion_ns|length) == ($requests[$request.id].token_profiles|length)
    and (.completion_ns|length) > 0
    and (.completion_ns|all(type == "number" and floor == .))
    and .completion_ns[0] >= $requests[$request.id].ready_ns
    and (get_intervals|all(. > 0))))
  else (.error|type == "string" and length > 0) end)); "invalid request completion or rejected mode") |
($rows|map(select(.is_admitted and .result.mode == "gpu_only"))|.[0].result.elapsed_ns) as $baseline |
{schema_version:1,is_simulation:true,is_serving_measurement:false,generated_token_count:0,
 trace_sha256:$manifest.trace_sha256,executable_sha256:$manifest.executable_sha256,
 description:$manifest.trace.description,cpu_numa_node:$manifest.trace.cpu_numa_node,
 timing_scope:$manifest.timing_scope,evidence:$manifest.trace.evidence,
 local_assistance:[ $manifest.trace.profiles|to_entries[] | .key as $profile | .value[] |
   select(.assist != null) | {profile:$profile,stage:.name,cpu_ns:.cpu.duration_ns,gpu_ns:.gpu.duration_ns,
     h2d_ns:.assist.h2d.duration_ns,d2h_ns:.assist.d2h.duration_ns,
     synchronization_ns:.assist.synchronization.duration_ns,
     saving_before_queueing_ns:(.cpu.duration_ns - .gpu.duration_ns - .assist.h2d.duration_ns
       - .assist.d2h.duration_ns - .assist.synchronization.duration_ns)} ],
 modes:[$rows[] | if .is_admitted == false then {mode,is_admitted,error} else
   .result as $result | ($result|del(.intervals,.requests)) + {is_admitted:true,
     throughput_ratio_to_gpu_only:(if $baseline == null then null else $baseline/$result.elapsed_ns end),
     queueing:($result.intervals|get_queues),
     by_declared_owner:[$result.requests|group_by($requests[.id].is_cpu_resident)[] |
       (map(.id)) as $ids |
       {owner:(if $requests[.[0].id].is_cpu_resident then "cpu" else "gpu" end),
        requests:length,simulated_tokens:(map(.completion_ns|length)|add),
        first_decode_latency:([.[]|.completion_ns[0]-$requests[.id].ready_ns]|get_quantiles),
        inter_token_latency:([.[]|get_intervals[]]|get_quantiles),
        queueing:([$result.intervals[]|select(.request_id as $id|$ids|index($id))]|get_queues)}]}
   end]}
