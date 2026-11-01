# jq -s --argjson trials 5 --arg pid 1334267 -f THIS SCREEN.jsonl
# Confirmation uses trials=20 and pid=1336467; both pin the same release.
def require($condition;$message): if $condition then . else error($message) end;
def median: sort | if length%2==1 then .[length/2|floor] else (.[length/2-1]+.[length/2])/2 end;
. as $rows
| map(select(.kind=="ffn_ablation_manifest")) as $manifests
| $rows | map(select(.kind=="ffn_ablation_pair")) as $pairs
| $rows | map(select(.kind=="ffn_ablation_validation")) as $checks
| $rows | map(select(.kind=="ffn_ablation_screen")) as $screens
| $rows | map(select(.kind=="ffn_ablation_preparation")) as $control_setup
| $rows | map(select(.kind=="ffn_joint_preparation")) as $joint_setup
| $rows | map(select(.kind=="ffn_ablation_complete")) as $complete
| "split-f32x4-128" as $control
| "joint-f32x4-128" as $candidate
| require(($manifests|length)==1 and ($trials==5 or $trials==20);"manifest multiplicity/trial scope")
| $manifests[0] as $manifest
| require($manifest.revision=="8881763d1599d51f8c8228d4a36f3a4333379649"
    and $manifest.worktree=="" and $manifest.build_profile=="release"
    and $manifest.executable_sha256=="071a85f99cf318abbff4b13e7fd8b39416118e1ea5bbf130726a88db45a86c59"
    and $manifest.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $manifest.trials==$trials and $manifest.iterations==10 and $manifest.fp32_variants==[$candidate]
    and ($manifest.is_q81_numerical_track|not)
    and $manifest.memory=={original_weights:3279519744,prepared_weights:1192935424,resident_kv:0,tracked_device_bytes:4472455168}
    and $manifest.prepared_weights.families==["down","vocabulary"]
    and $manifest.prepared_weights.single_row_down_split_launch=={columns:4,threads:256};"source/configuration/control memory")
| require(($pairs|length)==6*$trials and ($checks|length)==18 and ($screens|length)==6
    and [$screens[]|[.layer_id,.rows]]==[[0,1],[0,4],[17,1],[17,4],[35,1],[35,4]]
    and ($complete|length)==1 and $complete[0].is_complete
    and $complete[0].device_bytes_before==0 and $complete[0].device_bytes_after==0;"coverage/cleanup")
| require(all($checks|group_by(.layer_id,.rows)[]; (map(.variant)|sort)==[null,$candidate,$control])
    and all($checks[];.checks.guards_and_immutable_inputs
      and (if .variant==$candidate then .checks.intermediate_capture_matches_normal_kernel
        and .timing_control_hidden_bits_preserved and .timing_control_output_bits_preserved else true end));"numerical/control validation")
| require(($control_setup|length)==6 and ($joint_setup|length)==6
    and [$control_setup[]|[.layer_id,.rows]]==[$screens[]|[.layer_id,.rows]]
    and [$joint_setup[]|[.layer_id,.rows]]==[$screens[]|[.layer_id,.rows]];"preparation coverage")
| require(all($control_setup[];. as $record|.preparation as $p|
    .scratch_capacity_bytes==(if .rows==1 then 201152 else 803264 end)
    and $p.layout=="SeparateScalesValues" and $p.additional_prepared_bytes==47906816
    and $p.retained_host_prepared_bytes==47906816 and $p.estimated_all_layers_extra_bytes==1724645376
    and $p.tracked_bytes_before==4472455168+$record.scratch_capacity_bytes
    and $p.tracked_bytes_after==$p.tracked_bytes_before+47906816
    and [$p.keys[].source.tensor_name]==["blk.\(.layer_id).ffn_gate.weight","blk.\(.layer_id).ffn_up.weight"]
    and all($p.keys[];.layout=="SeparateScalesValues" and .source.dtype=="Q8_0"
      and .source.shape=={input_dim:2048,output_dim:11008}
      and .source.model_sha256==$manifest.prepared_weights.tensors[0].key.source.model_sha256));"control preparation accounting")
| require(all($joint_setup[];. as $record|.preparation as $p|
    ($control_setup[]|select(.layer_id==$record.layer_id and .rows==$record.rows)) as $s|
    $p.layout=="JointGateUp4" and $p.key.gate==$s.preparation.keys[0].source
    and $p.key.up==$s.preparation.keys[1].source
    and $p.paired_scale_bytes==2818048 and $p.paired_payload_bytes==45088768
    and $p.additional_prepared_bytes==47906816 and $p.retained_host_prepared_bytes==47906816
    and $p.control_prepared_bytes==47906816 and $p.combined_prepared_bytes==95813632
    and $p.combined_prepared_bytes<=128*1024*1024
    and $p.tracked_bytes_before==$s.preparation.tracked_bytes_after
    and $p.tracked_bytes_after==$p.tracked_bytes_before+47906816
    and $p.packing_ms>=0 and $p.allocation_upload_ms>=0);"joint identity/planes/accounting")
| require(all($pairs|group_by(.layer_id,.rows)[];map(.trial)==[range($trials)])
    and all($pairs[];. as $pair|
      ($joint_setup[]|select(.layer_id==$pair.layer_id and .rows==$pair.rows)) as $s|
      .control_variant==$control and .variant==$candidate
      and .activation_quantization_ms==0 and .quantization_scratch_capacity_bytes==0
      and .threads==128 and .additional_prepared_bytes==95813632 and .joint_prepared_bytes==47906816
      and .tracked_device_bytes==$s.preparation.tracked_bytes_after
      and .scratch_capacity_bytes==(if .rows==1 then 201152 else 803264 end)
      and [.measurements[].variant]==(if .trial%2==0 then [$control,$candidate] else [$candidate,$control] end));"control/residency/alternating pairs")
| require(all($pairs[];. as $pair|all(.measurements[];
    [.stages[].stage]==(if $pair.trial%2==0 then ["gate_up_swiglu","isolated_down","complete"] else ["complete","isolated_down","gate_up_swiglu"] end)
    and all(.stages[];. as $stage|.enqueue_counters as $counts|
      .iterations==10 and .gpu_ms>0 and .completed_unix_us>.started_unix_us
      and all(["device_allocs","device_frees","h2d_copies","h2d_bytes","d2h_copies","d2h_bytes","cublas_calls","stream_syncs"][];$counts[.]==0)
      and $counts.launches==(if $stage.stage=="complete" then 20 else 10 end)
      and $counts.stream_waits==(if $pair.rows==1 then 0 elif $stage.stage=="complete" then 20 else 10 end))));"timed work/side effects")
| require(all($screens[];. as $screen|
    [$pairs[]|select(.layer_id==$screen.layer_id and .rows==$screen.rows)|
      [.measurements[]|select(.variant==$control)|.stages[]|select(.stage=="complete")|.gpu_ms]
      + [.measurements[]|select(.variant==$candidate)|.stages[]|select(.stage=="complete")|.gpu_ms]] as $actual|
    .control_variant==$control and .result.pairs==$actual
      and .result.control_median_ms==($actual|map(.[0])|median)
      and .result.candidate_median_ms==($actual|map(.[1])|median)
      and .result.throughput_gain==(.result.control_median_ms/.result.candidate_median_ms-1)
      and .result.passes_micro_screen==(.result.throughput_gain>=0.05 and .result.bootstrap_gain_ci95[0]>0)
      and (.result.passes_micro_screen|not) and (.runtime_integration_allowed|not)
      and (if .rows==1 then .result.throughput_gain>0 and .result.throughput_gain<0.05 else true end));"screen versus actual pairs/gate")
| [$pairs[]|.gpu_before,.gpu_after] as $gpu
| require(all($gpu[];(.device|contains(", P0,")) and (.device|contains("1113 MHz, 3004 MHz"))
    and (.processes|test("^"+$pid+", target/release/examples/q8_execution_bench, [0-9]+ MiB$")));"GPU clocks/process isolation")
| [$pairs[].measurements[].stages[]]|sort_by(.started_unix_us) as $intervals
| require(all(range(1;$intervals|length);$intervals[.].started_unix_us >= $intervals[.-1].completed_unix_us);"overlapping primary intervals")
| {is_complete:true,source:$manifest.revision,executable:$manifest.executable_sha256,
    trials:$trials,control:$control,candidate:$candidate,
    pairs:($pairs|length),primary_intervals:($intervals|length),gpu_boundary_samples:($gpu|length),validation_records:18,
    bits_equal_to_current_split_control:true,tracked_device_bytes_after:0,runtime_integration_allowed:false,
    screens:($screens|map({layer_id,rows,result})),
    stages:($pairs|group_by(.layer_id,.rows)|map(. as $shape|{layer_id:.[0].layer_id,rows:.[0].rows,
      arms:([$shape[].measurements[]]|group_by(.variant)|map({variant:.[0].variant,
        stages:([.[].stages[]]|group_by(.stage)|map({stage:.[0].stage,median_gpu_ms:map(.gpu_ms)|median}))}))})),
    memory:{control_pair_bytes:47906816,joint_pair_bytes:47906816,combined_pair_bytes:95813632,
      combined_retained_host_bytes:95813632,paired_scale_bytes:2818048,paired_payload_bytes:45088768,
      tracked_by_rows:($pairs|group_by(.rows)|map({rows:.[0].rows,bytes:map(.tracked_device_bytes)|unique})),
      joint_median_packing_ms:($joint_setup|map(.preparation.packing_ms)|median),
      joint_median_upload_ms:($joint_setup|map(.preparation.allocation_upload_ms)|median)},
    candidate_errors:($checks|map(select(.variant==$candidate))|{
      gate_max_abs:map(.checks.gate_vs_cpu.max_abs_diff)|max,
      up_max_abs:map(.checks.up_vs_cpu.max_abs_diff)|max,
      swiglu_max_abs:map(.checks.swiglu_vs_cpu.max_abs_diff)|max,
      down_max_abs:map(.checks.complete_down_vs_cpu_on_actual_swiglu.max_abs_diff)|max})}
