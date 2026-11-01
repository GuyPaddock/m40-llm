# Audit unchanged numerical work, serial frequency perturbation and restoration.
def require($condition;$why): if $condition then . else error($why) end;
def median: sort|.[length/2|floor];
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity: \($kind)")|.[0];
def fields: split(",")|map(gsub("^ +| +$";""));
def shape: [.layer_id,.rows];
def verify_run:
  . as $run | .records as $records
  | ($records|one("ffn_read_manifest")) as $m
  | ($records|one("ffn_read_complete")) as $done
  | ($records|map(select(.kind=="ffn_read_pair"))) as $pairs
  | ($records|map(select(.kind=="ffn_read_validation"))) as $checks
  | ($records|map(select(.kind=="ffn_read_case_complete"))) as $finished
  | ($records|map(select(.kind=="ffn_read_warmup"))) as $warmups
  | [[0,1],[0,4],[17,1],[17,4],[35,1],[35,4]] as $shapes
  | require($m.revision=="c2db5fed1e1407951e676aa799845640f24a5d25"
      and $m.executable_sha256=="aa218ae8b24422d7ace65c1563fabc3e36ec93b61ae661eb5013d74fc5add579"
      and $m.worktree=="" and $m.build_profile=="release"
      and $m.model_sha256==$capture[0].model.sha256
      and $m.fixture_sha256=="eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687"
      and $m.capture_report_sha256=="d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801"
      and $m.execution_configuration==$capture[0].execution_configuration
      and ($m.prepared_weights|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))==
          ($capture[0].prepared_weights|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))
      and ($m.prepared_weights.tensors|length)==37
      and $m.memory=={original_weights:3279519744,prepared_weights:1192935424,resident_kv:0,tracked_device_bytes:4472455168}
      and $m.is_weight_read_diagnostic and ($m.is_q81_numerical_track|not) and ($m.is_fp32_ablation|not)
      and ($m.weight_read_contract.has_measured_dram_traffic|not)
      and ($m.weight_read_contract.runtime_integration_allowed|not)
      and $m.control=="qualified split-f32x4-128 with original prepared down/vocabulary"
      and $m.rows==[1,4] and $m.trials==5 and $m.iterations==10
      and $m.cuda_flags==["-O3","sm_52","compute_52","no fast math"];
      "frozen source, release configuration and original prepared identities")
  | require(($records|length)==56 and ($pairs|length)==30
      and all([$checks,$finished,$warmups][];length==6 and (map(shape)|sort)==$shapes)
      and $done.is_complete and $done.cases==6 and $done.device_bytes_before==0 and $done.device_bytes_after==0;
      "six shapes and complete cleanup")
  | require(all($checks[];. as $check|
      .checks.has_matching_cpu_checksums and .checks.has_intact_output_guards
      and .checks.checksum_count==.rows*11008*2
      and .checks.checksum_storage_bytes==.rows*11008*8+64
      and .preparation.additional_prepared_bytes==47906816 and .preparation.layout=="SeparateScalesValues"
      and .real_scratch_bytes==(if .rows==1 then 201152 else 803264 end)
      and .real_ffn_checks.guards_and_immutable_inputs and .real_ffn_checks.intermediate_capture_matches_normal_kernel
      and .real_ffn_checks.variant=="split-f32x4-128" and .real_ffn_checks.quantization==null
      and all([.real_ffn_checks.gate_vs_cpu,.real_ffn_checks.up_vs_cpu,.real_ffn_checks.swiglu_vs_cpu,
        .real_ffn_checks.complete_down_vs_cpu_on_actual_swiglu,.real_ffn_checks.isolated_down_vs_cpu_on_frozen_swiglu][];
        .max_abs_diff>=0 and .max_abs_diff<0.001 and .mean_abs_diff>=0 and .mean_abs_diff<=.max_abs_diff)
      and (([$finished[]|select(shape==($check|shape))]|.[0]) as $end|
        $end.checks==$check.checks and $end.has_intact_planes_and_inputs and ($end.runtime_integration_allowed|not)));
      "source-derived checksums, original CPU bounds, guards and immutable planes")
  | require(all($warmups[];.real_ffn.minimum_gpu_ms==1000 and .real_ffn.elapsed_gpu_ms>=1000
      and .real_ffn.iterations_per_batch==32 and .real_ffn.maximum_batches==256 and .real_ffn.batches<=256
      and .read_checksum.minimum_gpu_ms==100 and .read_checksum.gpu_ms>=100
      and .read_checksum.iterations_per_batch==256 and .read_checksum.maximum_batches==256 and .read_checksum.batches<=256);
      "bounded warmup excluded from primary samples")
  | require(([$pairs[]|shape]|unique|sort)==$shapes
      and all($pairs|group_by(shape)[];map(.trial)==[0,1,2,3,4])
      and all($pairs[];. as $pair|
        .tracked_device_bytes==(if .rows==1 then 4520651264 else 4521517568 end)
        and [.measurements[].stage]==(if .trial%2==0 then ["weight_read_checksum","gate_up_swiglu","isolated_down","complete"]
          else ["complete","isolated_down","gate_up_swiglu","weight_read_checksum"] end)
        and all(.measurements[];.enqueue_counters as $c|
          .iterations==10 and .gpu_ms>0 and .completed_unix_us>.started_unix_us
          and all(["device_allocs","device_frees","h2d_copies","h2d_bytes","d2h_copies","d2h_bytes","cublas_calls","stream_syncs"][];$c[.]==0)
          and $c.launches==(if .stage=="complete" then 20 else 10 end)
          and $c.stream_waits==(if .stage=="weight_read_checksum" or $pair.rows==1 then 0 elif .stage=="complete" then 20 else 10 end)
          and (if .stage=="weight_read_checksum" then .operation=="ffn_q8_weight_checksum_128"
            and .blocks_per_row==11008 and .threads==128 and .shared_bytes==1024
            and .logical_unique_weight_bytes_per_call==47906816*$pair.rows
            and (.has_inference_output|not) and (.has_measured_dram_traffic|not)
          else .control_dispatch=={operation:"ffn_q8_split_f32x4_128",launches:(if .stage=="isolated_down" then 0 else 10 end)} end)));
      "identical stage order, complete work and unchanged enqueue counters")
  | require($run.meta.exit_code==0 and $run.meta.requested_sm_mhz==(if $run.meta.state=="high" then 1114 else 823 end)
      and all($pairs[];all([.gpu_before,.gpu_after][];
        (.device|fields) as $d|
        $d[0]=="GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852" and $d[3]=="P0"
        and ($d[6]|split(" ")[0]|tonumber)==3004
        and ((($d[5]|split(" ")[0]|tonumber)-$run.meta.requested_sm_mhz)|fabs)<=2
        and (.processes|split("\n")|map(split(",")[0]|tonumber))==[$run.meta.pid]));
      "single owned process, P0 and observed supported clocks");

# Enumerate every ordered paired resample; outer processes are the unit.
def bootstrap_ratio($pairs;$depth;$indices):
  if $depth==0 then
    ([$indices[] as $i|$pairs[$i].low_ms]|median)/([$indices[] as $i|$pairs[$i].high_ms]|median)-1
  else range(0;5) as $i|bootstrap_ratio($pairs;$depth-1;$indices+[$i]) end;

. as $input
| require(.completion.exit_code==0 and .completion.has_mutated and .completion.is_original_policy_restored
    and .policy.before=="GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852,Disabled,3004,947,3004,947"
    and .policy.after==.policy.before and .policy.independent_after==.policy.before
    and .policy.boost_before=="On,On" and .policy.boost_after==.policy.boost_before
    and .policy.before_processes=="" and .policy.after_processes=="" and .policy.independent_after_processes=="";
    "original application clocks, boost, persistence and empty GPU restored")
| require((.runs|length)==10 and (.runs|map(.meta.pid)|unique|length)==10
    and (.runs|map(.meta|{pair,state}))==[range(0;5) as $p|
      if $p%2==0 then {pair:$p,state:"high"},{pair:$p,state:"low"}
      else {pair:$p,state:"low"},{pair:$p,state:"high"} end];"five independent alternating process pairs")
| .runs|map(verify_run) as $runs
| require(([$runs[].records[]|select(.kind=="ffn_read_validation")]|group_by(shape)
    |all(.[];map(.checks.checksum_sha256)|unique|length==1));"same source checksums across all clock states")
| ([$runs[].records[]|select(.kind=="ffn_read_pair")|.measurements[]]|sort_by(.started_unix_us)) as $intervals
| require(($intervals|length)==1200 and all(range(1;1200);$intervals[.].started_unix_us>=$intervals[.-1].completed_unix_us);
    "1200 serial primary GPU intervals")
| [$runs[] as $run|$run.records|map(select(.kind=="ffn_read_pair"))|group_by(shape)[] as $group|
    ([$group[]|.gpu_before,.gpu_after|.device|fields|.[5]|split(" ")[0]|tonumber]|median) as $sm|
    $group|map(.measurements[])|group_by(.stage)[]|
    {pair:$run.meta.pair,state:$run.meta.state,layer_id:$group[0].layer_id,rows:$group[0].rows,
     stage:.[0].stage,ms:(map(.gpu_ms)|median),sm_mhz:$sm}] as $medians
| {passes_audit:true,source:"c2db5fed1e1407951e676aa799845640f24a5d25",
    executable_sha256:"aa218ae8b24422d7ace65c1563fabc3e36ec93b61ae661eb5013d74fc5add579",
    outer_process_pairs:5,processes:10,inner_trials_per_shape:5,serial_primary_intervals:1200,gpu_boundaries:600,
    observed_sm_mhz:($medians|group_by(.state)|map({key:.[0].state,value:(map(.sm_mhz)|median)})|from_entries),
    memory_mhz:3004,original_policy_restored:true,tracked_device_bytes_after:0,
    cases:($medians|group_by([.layer_id,.rows,.stage])|map(. as $case|
      [group_by(.pair)[]|(map(select(.state=="high"))[0]) as $h|(map(select(.state=="low"))[0]) as $l|
        {pair:$h.pair,high_ms:$h.ms,low_ms:$l.ms,high_sm_mhz:$h.sm_mhz,low_sm_mhz:$l.sm_mhz}] as $pairs|
      ($pairs|map(.high_ms)|median) as $h|($pairs|map(.low_ms)|median) as $l|
      ([bootstrap_ratio($pairs;5;[])]|sort) as $bootstrap|
      {layer_id:$case[0].layer_id,rows:$case[0].rows,stage:$case[0].stage,process_pairs:$pairs,
       high_median_ms:$h,low_median_ms:$l,latency_increase:($l/$h-1),
       interval_95:[$bootstrap[78],$bootstrap[3046]],
       frequency_elasticity:(($l/$h|log)/(($pairs|map(.high_sm_mhz)|median)/($pairs|map(.low_sm_mhz)|median)|log)),
       is_material_response:($l/$h-1>=0.05 and $bootstrap[78]>0)})),
    bootstrap:{method:"exhaustive ordered paired percentile",resamples:3125,percentile_indices:[78,3046],unit:"outer process-pair medians"},
    is_optimization_claim:false,is_hardware_ceiling_claim:false,has_measured_dram_traffic:false,changes_defaults:false}
