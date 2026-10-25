def require($condition;$message): if $condition then . else error($message) end;
def median: sort | .[length/2|floor];
. as $rows
| map(select(.kind=="ffn_ablation_manifest")) as $manifests
| $rows | map(select(.kind=="ffn_ablation_pair")) as $pairs
| $rows | map(select(.kind=="ffn_ablation_validation")) as $checks
| $rows | map(select(.kind=="ffn_ablation_screen")) as $screens
| $rows | map(select(.kind=="ffn_ablation_complete")) as $complete
| "split-f32x4-128" as $control
| "split-f32x4-cols2-128" as $candidate
| require(($manifests|length)==1;"manifest multiplicity")
| $manifests[0] as $manifest
| require($manifest.revision=="414f2a521662b7ea2840eb47df98d6ba90f34a78"
    and $manifest.worktree=="" and $manifest.build_profile=="release"
    and $manifest.executable_sha256=="a8cfe71ed2653d05c1adec506db26b7a7fbd510f03697067d9bfdcdb82e605a6"
    and $manifest.trials==5 and $manifest.iterations==10 and $manifest.fp32_variants==[$candidate]
    and ($manifest.is_q81_numerical_track|not)
    and $manifest.memory=={original_weights:3279519744,prepared_weights:1192935424,resident_kv:0,tracked_device_bytes:4472455168};"source/configuration/memory")
| require(($pairs|length)==30 and ($checks|length)==18 and ($screens|length)==6
    and [$screens[]|[.layer_id,.rows]]==[[0,1],[0,4],[17,1],[17,4],[35,1],[35,4]]
    and ($complete|length)==1 and $complete[0].is_complete
    and $complete[0].device_bytes_before==0 and $complete[0].device_bytes_after==0;"complete coverage/cleanup")
| require(all($checks|group_by(.layer_id,.rows)[]; (map(.variant)|sort)==[null,$control,$candidate])
    and all($checks[];.checks.guards_and_immutable_inputs
      and (if .variant==$candidate then .checks.intermediate_capture_matches_normal_kernel
        and .timing_control_hidden_bits_preserved and .timing_control_output_bits_preserved else true end));"numerical/control validation")
| require(all($pairs|group_by(.layer_id,.rows)[];map(.trial)==[0,1,2,3,4])
    and all($pairs[];.control_variant==$control and .variant==$candidate
      and .activation_quantization_ms==0 and .quantization_scratch_capacity_bytes==0
      and .threads==128 and .additional_prepared_bytes==47906816
      and [.measurements[].variant]==(if .trial%2==0 then [$control,$candidate] else [$candidate,$control] end));"control and alternating pairs")
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
      and (.result.passes_micro_screen|not) and .result.bootstrap_gain_ci95[1]<0);"observed negative screening result")
| [$pairs[]|.gpu_before,.gpu_after] as $gpu
| require(all($gpu[]; (.device|contains(", P0,")) and (.device|contains("1113 MHz, 3004 MHz"))
    and (.processes|test("^1300420, target/release/examples/q8_execution_bench, [0-9]+ MiB$")));"GPU clocks/process isolation")
| [$pairs[].measurements[].stages[]]|sort_by(.started_unix_us) as $intervals
| require(all(range(1;$intervals|length);$intervals[.].started_unix_us >= $intervals[.-1].completed_unix_us);"overlapping timed intervals")
| {is_complete:true,source:$manifest.revision,control:$control,candidate:$candidate,
    pairs:30,primary_intervals:($intervals|length),gpu_boundary_samples:60,validation_records:18,
    bits_equal_to_current_split_control:true,tracked_device_bytes_after:0,runtime_integration_allowed:false,
    screens:($screens|map({layer_id,rows,result})),
    stages:($pairs|group_by(.layer_id,.rows)|map(. as $shape|{layer_id:.[0].layer_id,rows:.[0].rows,
      arms:([$shape[].measurements[]]|group_by(.variant)|map({variant:.[0].variant,
        stages:([.[].stages[]]|group_by(.stage)|map({stage:.[0].stage,median_gpu_ms:map(.gpu_ms)|median}))}))})),
    candidate_errors:($checks|map(select(.variant==$candidate))|{
      gate_max_abs:map(.checks.gate_vs_cpu.max_abs_diff)|max,
      up_max_abs:map(.checks.up_vs_cpu.max_abs_diff)|max,
      swiglu_max_abs:map(.checks.swiglu_vs_cpu.max_abs_diff)|max,
      down_max_abs:map(.checks.complete_down_vs_cpu_on_actual_swiglu.max_abs_diff)|max})}
