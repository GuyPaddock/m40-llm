# jq -s --slurpfile trace QUALITY --slurpfile http QUALIFIED_HTTP -f THIS RAW
def require($ok;$why): if $ok then . else error($why) end;
def median: sort|.[length/2|floor];
. as $rows
| map(select(.kind=="greedy_adapter_manifest"))|require(length==1;"manifest multiplicity")|.[0] as $m
| $rows|map(select(.kind=="greedy_adapter_pair")) as $pairs
| $rows|map(select(.kind=="greedy_adapter_complete")) as $done
| $trace|map(select(.kind=="q8_runtime_browser_generation" and .label=="candidate"))
| require(length==1;"qualified input generation")|.[0].generation.generated_token_ids as $ids
| $rows
| require(length==7 and ($pairs|map(.trial))==[0,1,2,3,4]
    and ($done|length)==1 and $done[0].is_complete and $done[0].tracked_device_bytes==0;"bounded coverage/device tracking")
| require($m.schema_version==1 and $m.worktree=="" and $m.profile=="release"
    and $m.revision=="a2b8458c02c5d9c13b6e934b9252b9753f9b2897"
    and $m.executable_sha256=="546c86757ce194fbe1d44f1da6d85970aa8fd222d6b485f706ddb447f83d946a"
    and $m.trace_sha256=="8b499787179f484aeb32e37caf837208e0994da4995918e18e34ba922d3d949e"
    and $m.tokens==512 and $m.iterations==256 and $m.pairs==5
    and $m.cuda_visible_devices=="-1" and ($m.cpu_affinity|test("^Cpus_allowed_list:\\s+[0-9]+$"))
    and $m.generated_ids==$ids and ($ids|length)==512
    and $m.synthetic_vector_requested_bytes_per_trace==([$ids[]|4*(.+1)]|add);"clean release/frozen trace/hidden GPU")
| require(all($pairs[];. as $p|(.measurements|length)==2
    and [.measurements[].is_synthetic]==(if .trial%2==0 then [true,false] else [false,true] end)
    and all(.measurements[];.elapsed_ms>0 and .per_trace_ms==.elapsed_ms/256
      and .checksum==($ids|add)*256
      and .completed_unix_us>=.started_unix_us
      and (((.completed_unix_us-.started_unix_us)/1000-.elapsed_ms)|fabs)<0.1));"raw arithmetic/checksums/alternation")
| ([$pairs[].measurements[]]|sort_by(.started_unix_us)) as $intervals
| require(all(range(1;10);$intervals[.].started_unix_us>=$intervals[.-1].completed_unix_us);"nonoverlapping CPU intervals")
| [$intervals[]|select(.is_synthetic)|.per_trace_ms] as $synthetic
| [$intervals[]|select(.is_synthetic|not)|.per_trace_ms] as $direct
| [$http[]|select(.kind=="trial" and .runtime=="m40-llm" and .phase=="screen")|.client_elapsed_ms] as $http_ms
| require(($http_ms|length)==5;"qualified HTTP context")
| {passes_audit:true,source:$m.revision,executable_sha256:$m.executable_sha256,
    actual_qualified_ids:512,pairs:5,serial_intervals:10,iterations_per_arm:256,
    synthetic_median_ms_per_trace:($synthetic|median),synthetic_min_ms_per_trace:($synthetic|min),
    synthetic_max_ms_per_trace:($synthetic|max),direct_median_ms_per_trace:($direct|median),
    synthetic_median_us_per_token:(($synthetic|median)*1000/512),
    requested_vector_bytes_per_trace:$m.synthetic_vector_requested_bytes_per_trace,
    largest_requested_vector_bytes:4*(($ids|max)+1),
    qualified_http_median_ms:($http_ms|median),
    observed_adapter_to_http_fraction:(($synthetic|median)/($http_ms|median)),
    observed_max_adapter_to_http_fraction:(($synthetic|max)/($http_ms|median)),
    observed_cost_below_three_percent:(($synthetic|max)/($http_ms|median)<0.03),
    scope:"isolated CPU cost comparison, not an inference bound or GPU/HTTP speedup",
    runtime_integration_allowed:false,changes_runtime_defaults:false,tracked_device_bytes:0}
