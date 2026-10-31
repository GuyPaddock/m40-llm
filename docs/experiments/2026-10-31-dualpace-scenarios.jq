# Consume measured CPU/transfer records, the audited GPU cost library and
# recorded capacity inventory. Emit the frozen 54 scenarios for one NUMA node.
def require($condition; $message): if $condition then . else error($message) end;
def get_rows($input; $kind): $input | map(select(.kind == $kind));
def get_one($values; $label):
  $values | if length == 1 then .[0] else error("expected one " + $label) end;
def get_median: sort | if length % 2 == 1 then .[length/2|floor]
  else (.[length/2-1] + .[length/2]) / 2 end;
def is_positive: if type == "number" then isfinite and . > 0 else false end;
def is_capacity: is_positive and floor == . and . < 9007199254740992;
def get_case_key: [.layer,.family,.batch];
def get_cost($values; $factor):
  $values | require(length > 0 and all(.[]; is_positive); "invalid cost observations") |
  get_median * $factor | ceil |
  require(. > 0 and . <= 10000000000; "component service outside bounded study");
def get_shape($family): if $family == "down" then [11008,2048] else [2048,11008] end;
def get_transfer_cost($records; $name; $bytes; $index; $iterations):
  ($records | map(select(.kind == "cooperative_transfer_costs" and
    .case.name == $name and .comparison == "pageable_h2d_vs_d2h"))) as $summaries |
  require(($summaries | length) > 0; "missing transfer summary") |
  $summaries[-1] as $summary |
  ($records | map(select(.kind == "cooperative_transfer_pair" and
    .case.name == $name and .comparison == "pageable_h2d_vs_d2h")) | sort_by(.pair)) as $pairs |
  require($summary.timing_error == null and $summary.case.bytes == $bytes
    and ([5,10] | index($pairs | length)) != null
    and $summary.paired_operation_rate_comparison.pairs == ($pairs | length)
    and ($pairs | map(.pair)) == [range(0; $pairs | length)]; "incomplete or unstable transfer series") |
  require(($pairs | all(.case.bytes == $bytes and
    .order == (if .pair % 2 == 0 then [0,1] else [1,0] end)
    and (.observations | length) == 2
    and .observations[$index].arm == (if $index == 0 then "h2d" else "d2h" end)
    and .observations[$index].iterations == $iterations
    and .observations[$index].logical_transfer_bytes_per_operation == $bytes));
    "transfer byte count, iteration count or ordering differs") |
  get_cost($pairs | map(.observations[$index].wall_ns_per_operation); 1);

. as $input |
get_one(get_rows(.cpu; "cpu_projection_manifest"); "CPU manifest") as $cpu_manifest |
get_one(get_rows(.transfers; "cooperative_transfer_manifest"); "transfer manifest") as $transfer_manifest |
get_one(get_rows(.cpu; "cpu_projection_complete"); "CPU completion") as $cpu_complete |
get_one(get_rows(.transfers; "cooperative_transfer_complete"); "transfer completion") as $transfer_complete |
require($cpu_manifest.schema_version == 1 and $transfer_manifest.schema_version == 1
  and .cpu[0].kind == "cpu_projection_manifest" and .cpu[-1].kind == "cpu_projection_complete"
  and .transfers[0].kind == "cooperative_transfer_manifest" and .transfers[-1].kind == "cooperative_transfer_complete"
  and $cpu_manifest.is_validation_only == false and $transfer_manifest.is_validation_only == false
  and $cpu_complete.is_complete == true and $cpu_complete.cases == 36
  and $transfer_complete.is_complete == true and $transfer_complete.transfer_cases == 13
  and $cpu_complete.generated_token_count == 0 and $transfer_complete.generated_token_count == 0
  and (get_rows(.transfers; "cooperative_transfer_case_complete") | length) == 13;
  "cost reports are incomplete or validation-only") |
get_rows(.transfers; "cooperative_transfer_validation") as $transfer_checks |
require(($transfer_checks | length) == 13 and ($transfer_checks | map(.case.name) | unique | length) == 13
  and ($transfer_checks | all(.is_bit_exact == true and .guards_unchanged == true))
  and (get_rows(.transfers; "cooperative_transfer_case_complete") |
    all(.tracked_device_bytes_before == .tracked_device_bytes_after and .tracked_device_bytes_before != null));
  "transfer validation or cleanup is incomplete") |
$cpu_manifest.placement as $placement |
require($placement.threads == 1 and $transfer_manifest.placement.threads == 1
  and $placement.cpu == $transfer_manifest.placement.cpu
  and $placement.numa_node == $transfer_manifest.placement.numa_node
  and ([$placement.cpu,$placement.numa_node] == [15,0] or [$placement.cpu,$placement.numa_node] == [31,1]);
  "CPU and transfer placement differ from the frozen study") |
require(.gpu.schema_version == 1 and .gpu.is_full_decode_profile == false
  and .gpu.source_manifest.model_sha256 == "4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
  and $cpu_manifest.model_sha256 == .gpu.source_manifest.model_sha256
  and $cpu_manifest.input_sha256 == .gpu.source_manifest.fixture_sha256
  and $cpu_manifest.capture_report_sha256 == .gpu.source_manifest.capture_report_sha256
  and (.gpu.costs | length) == 72; "GPU/CPU model or capture identity differs") |
require(.capacity.schema_version == 1
  and .capacity.gpu.uuid == "GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852"
  and (.capacity.gpu.capacity_bytes | is_capacity)
  and ($transfer_manifest.gpu_state | startswith("GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852"))
  and $transfer_manifest.has_overlap_measurement == false; "GPU inventory or transfer scope differs") |
get_one(.capacity.nodes | map(select(.numa_node == $placement.numa_node)); "NUMA capacity") as $node |
require(($node.capacity_bytes | is_capacity); "invalid NUMA capacity") |
require((["cpu_costs","gpu_costs","transfers","capacity"] | all(. as $key |
  $input.evidence[$key].kind == "artifact"
  and ($input.evidence[$key].path | type == "string" and length > 0)
  and ($input.evidence[$key].sha256 | test("^[a-f0-9]{64}$")))); "missing evidence artifact") |
get_rows(.cpu; "cpu_projection_validation") as $validation |
get_rows(.cpu; "cpu_projection_setup") as $weights |
[([0,17,35][]) as $layer | (["gate","up","down"][]) as $family |
 ([1,2,4,8][]) as $batch | [$layer,$family,$batch]] | sort as $expected |
require(($validation | length) == 36 and ($weights | length) == 9
  and ($validation | map(.case | get_case_key) | sort) == $expected; "CPU validation grid differs") |
[
  $expected[] as $key |
  get_one($validation | map(select((.case | get_case_key) == $key)); "CPU validation") as $check |
  get_one($weights | map(select([.layer,.family] == $key[:2])); "CPU weight identity") as $weight |
  ($input.gpu.costs | map(select([.layer,.family,.batch] == $key))) as $gpu |
  require(($gpu | map(.arm) | sort) == ["prepared_q8","q8"]
    and ($gpu | all(.source == $weight.source and .q8_payload_bytes == $weight.q8_payload_bytes
      and .source.shape == get_shape($key[1]) and .input_sha256 == $check.case.input_sha256
      and .row_positions == $check.case.row_positions and .iterations_per_trial == 100))
    and $check.case.columns == get_shape($key[1])[0] and $check.case.rows == get_shape($key[1])[1]
    and ($check.case.row_positions | length) == $key[2]
    and $check.avx2.elements == ($key[2] * $check.case.rows)
    and $check.scalar.elements == $check.avx2.elements
    and $check.avx2.max_tolerance_ratio <= 1 and $check.avx2.max_tolerance_ratio >= 0
    and $check.scalar.max_tolerance_ratio <= 1 and $check.scalar.max_tolerance_ratio >= 0;
    "CPU/GPU input, weight or numerical validation differs") |
  ($input.cpu | map(select(.kind == "cpu_projection_comparison" and (.case | get_case_key) == $key))) as $summaries |
  require(($summaries | length) > 0; "missing CPU cost summary") |
  $summaries[-1] as $summary |
  ($input.cpu | map(select(.kind == "cpu_projection_pair" and (.case | get_case_key) == $key
    and .is_confirmation == $summary.is_confirmation)) | sort_by(.pair)) as $pairs |
  require($summary.timing_error == null and $summary.comparison.pairs == ($pairs | length)
    and (if $summary.is_confirmation then ($pairs | length) == 20
      else ([5,10] | index($pairs | length)) != null end)
    and ($pairs | map(.pair)) == [range(0; $pairs | length)]
    and ($pairs | all(.iterations == $cpu_manifest.iterations
      and .avx2_first == (.pair % 2 == 1))); "incomplete or unstable CPU series") |
  {layer:$key[0],family:$key[1],batch:$key[2],
   cpu_ns:get_cost($pairs | map(.avx2_ms); 1000000),
   gpu:[$gpu[] | {arm, duration_ns:get_cost(.raw_observations | map(.wall_ms); 1000000)}]}
] as $projections |
get_transfer_cost($input.transfers; "activation_11008_batch_1"; 44032; 0; $transfer_manifest.iterations) as $h2d |
get_transfer_cost($input.transfers; "activation_2048_batch_1"; 8192; 1; $transfer_manifest.iterations) as $d2h |
($projections | map(select(.batch == 1))) as $single |
($single | map(.cpu_ns) | add * 12) as $cpu_total |
[
  (["q8","prepared_q8"][]) as $arm |
  ($single | map(.gpu[] | select(.arm == $arm) | .duration_ns) | add * 12) as $gpu_total |
  ([0,1,4][]) as $cpu_missing |
  ([0,1,4][]) as $gpu_missing |
  ($cpu_total * $cpu_missing) as $cpu_extra |
  ($gpu_total * $gpu_missing) as $gpu_extra |
  [range(0;36) as $layer |
    (if $layer < 12 then 0 elif $layer < 24 then 17 else 35 end) as $sampled |
    ("layer_" + ($layer | tostring)) as $name |
    ([if $cpu_extra == 0 and $gpu_extra == 0 then empty else
      {name:($name + "_unmeasured"),
       cpu:{duration_ns:(($cpu_extra/36|floor) + (if $layer == 35 then $cpu_extra%36 else 0 end)),evidence:"missing_work"},
       gpu:{duration_ns:(($gpu_extra/36|floor) + (if $layer == 35 then $gpu_extra%36 else 0 end)),evidence:"missing_work"},
       assist:null} end] +
    [(["gate","up","down"][]) as $family |
      get_one($single | map(select(.layer == $sampled and .family == $family)); "sampled projection") as $projection |
      get_one($projection.gpu | map(select(.arm == $arm)); "GPU path") as $device |
      {name:($name + "_" + $family),cpu:{duration_ns:$projection.cpu_ns,evidence:"cpu_costs"},
       gpu:{duration_ns:$device.duration_ns,evidence:"gpu_costs"},
       assist:(if $family == "down" then
         {h2d:{duration_ns:$h2d,evidence:"transfers"},d2h:{duration_ns:$d2h,evidence:"transfers"},
          synchronization:{duration_ns:0,evidence:"transfers"},h2d_bytes:44032,d2h_bytes:8192}
         else null end)}])[]
  ] as $stages |
  (["balanced","gpu_loaded","staggered"][]) as $workload |
  (if $workload == "balanced" then [false,true]
    elif $workload == "gpu_loaded" then [false,false,false,true] else [false,true,false,true] end) as $owners |
  {name:("node" + ($placement.numa_node|tostring) + "-" + $arm + "-cpu" + ($cpu_missing|tostring) + "-gpu" + ($gpu_missing|tostring) + "-" + $workload),
   derived:{cpu_projection_ns:$cpu_total,gpu_projection_ns:$gpu_total,cpu_missing_ns:$cpu_extra,gpu_missing_ns:$gpu_extra,
     down_h2d_ns:$h2d,down_d2h_ns:$d2h},
   trace:{schema_version:3,has_copy_compute_overlap:false,cpu_threads:1,cpu_numa_node:$placement.numa_node,
     description:("Projected cooperative decode; " + $arm + "; " + $workload + "; missing CPU/GPU cost multipliers " +
       ($cpu_missing|tostring) + "/" + ($gpu_missing|tostring) + "; zero is a projection-only lower bound, not complete decoding"),
     memory:{host_capacity_bytes:$node.capacity_bytes,gpu_capacity_bytes:$input.capacity.gpu.capacity_bytes,
       host_fixed_bytes:(3279519744+1073741824),
       gpu_fixed_bytes:(3279519744+1073741824+(if $arm == "prepared_q8" then 108*23953408 else 0 end))},
     evidence:($input.evidence + {
       missing_work:{kind:"assumption",description:("Unmeasured CPU/GPU budgets " + ($cpu_extra|tostring) + "/" + ($gpu_extra|tostring) +
         " ns, distributed over 36 layer groups. Zero omits work; not a measured complete CPU decoder.")},
       extrapolation:{kind:"assumption",description:"Sampled layers 0/17/35 each represent twelve layers. Fixed eight-token costs, synthetic post-prefill arrivals, 1GiB workspace per processor, dense FP16 final KV. Isolated 100-operation GPU costs amortize completion; no concurrent contention or host-dispatch qualification."}}),
     profiles:{token:$stages},
     requests:[$owners | to_entries[] | {id:("request_"+(.key|tostring)),is_cpu_resident:.value,
       ready_ns:(if $workload == "staggered" and .key >= 2 then 4*($gpu_total+$gpu_extra) else 0 end),
       host_peak_kv_bytes:38043648,gpu_peak_kv_bytes:38043648,token_profiles:[range(0;8)|"token"]}]
   }}
] |
{schema_version:1,is_simulation:true,is_complete_decode_measured:false,cpu_placement:$placement,scenarios:.}
