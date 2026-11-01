def require($condition; $message):
  if $condition then . else error($message) end;
def percentile($p): sort | (($p*(length-1)) as $i
  | .[$i|floor] + ( .[$i|ceil] - .[$i|floor] ) * ($i-($i|floor)));
def close($a;$b): (($a-$b)|fabs) < 1e-9;
. as $rows
| map(select(.kind=="compact_cost_setup")) as $setups
| map(select(.kind=="compact_cost_trial")) as $trials
| map(select(.kind=="compact_cost_decision")) as $decisions
| require(($setups|length)==1 and ($decisions|length)==1; "setup/decision coverage")
| $setups[0] as $s | $decisions[0] as $d
| require(($rows[0].revision|startswith("c9c1853"));"source revision")
| require($s.parameter_bytes==161876992 and $s.additional_device_bytes==163728908;"allocation accounting")
| require($s.validation.is_passed and $s.validation.hidden.values==768 and $s.validation.logits.values==455808;"numeric coverage")
| require(($trials|map(.trial)|sort)==[0,1,2,3,4];"trial coverage")
| require(all($trials[]; (.all_in_proposal_ms|length)==200
    and all(.all_in_proposal_ms[]; .>0)
    and close(.median_ms; (.all_in_proposal_ms|percentile(0.5)))
    and close(.p95_ms; (.all_in_proposal_ms|percentile(0.95)))
    and .counters=={launches:1000,cublas_calls:400,stream_syncs:200,stream_waits:0,
      device_allocs:0,device_frees:0,h2d_copies:0,d2h_copies:200,h2d_bytes:0,d2h_bytes:2400}
    and .started_unix_us<.completed_unix_us
    and .gpu_before.clocks=="GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, P0, 1113, 3004"
    and .gpu_after.clocks==.gpu_before.clocks
    and (.gpu_before.processes|split("\n")|length)==1
    and .gpu_before.processes==.gpu_after.processes);"trial work/isolation/timing")
| require(all(range(1;5); $trials[.].started_unix_us > $trials[.-1].completed_unix_us);"overlapping trials")
| [$trials[].all_in_proposal_ms[]] as $times
| require(close($d.median_ms;($times|percentile(0.5)))
    and close($d.p95_ms;($times|percentile(0.95)))
    and close($d.p99_ms;($times|percentile(0.99)));"complete timing summary")
| require($d.admission_median_ms==3.884 and $d.passes_cost_admission==($d.median_ms<=3.884)
    and $d.tracked_device_bytes_after==0 and $d.is_complete
    and ($d.is_http_measured|not) and ($d.is_acceptance_measured|not)
    and ($d.is_training_authorized|not);"decision or claim scope")
| {is_passed:true,trials:5,complete_decisions:1000,median_ms:$d.median_ms,p95_ms:$d.p95_ms,
    no_timed_device_allocation:true,cleanup_bytes:0,scope:"synthetic proposal cost, not learned acceptance or measured HTTP"}
