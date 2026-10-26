# jq -s --slurpfile capture ...ffn-capture.jsonl --slurpfile previous ...unfused-ffn-screen.jsonl -f THIS ...cache-ffn-screen.jsonl
def require($condition;$message): if $condition then . else error($message) end;
def median: sort | .[length/2|floor];
. as $rows
| map(select(.kind=="ffn_ablation_manifest")) as $manifests
| $rows | map(select(.kind=="ffn_ablation_pair")) as $pairs
| $rows | map(select(.kind=="ffn_ablation_validation")) as $checks
| $rows | map(select(.kind=="ffn_ablation_screen")) as $screens
| $rows | map(select(.kind=="ffn_ablation_complete")) as $complete
| "split-f32x4-128" as $control
| "split-f32x4-weight-cg-128" as $candidate
| require(($manifests|length)==1;"manifest multiplicity")
| $manifests[0] as $manifest
| require($manifest.revision=="0cff8abc9878edbb124394738b907ca47ce8eae3"
    and $manifest.worktree=="" and $manifest.build_profile=="release"
    and $manifest.executable_sha256=="72d3e2317fc4a2f0dbe1ce58b18388898700d0cbf91ec00f68d9e224ff0c375b"
    and $manifest.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $manifest.fixture_sha256=="eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687"
    and $manifest.capture_report_sha256=="d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801"
    and $manifest.execution_configuration==$capture[0].execution_configuration
    and $manifest.trials==5 and $manifest.iterations==10 and $manifest.fp32_variants==[$candidate]
    and $manifest.cache_policy=={activations:"nc",candidate_weights:"cg",control_weights:"nc",scope:"fixed fused geometry, planes, FP32 association and prepared down"}
    and ($manifest.is_q81_numerical_track|not)
    and $manifest.memory=={original_weights:3279519744,prepared_weights:1192935424,resident_kv:0,tracked_device_bytes:4472455168}
    and $manifest.prepared_weights.families==["down","vocabulary"]
    and ($manifest.prepared_weights.tensors|map(.key))==($previous[0].prepared_weights.tensors|map(.key))
    and $manifest.prepared_weights.single_row_down_split_launch=={columns:4,threads:256};"source/configuration/control memory")
| require(($pairs|length)==30 and ($checks|length)==18 and ($screens|length)==6
    and [$screens[]|[.layer_id,.rows]]==[[0,1],[0,4],[17,1],[17,4],[35,1],[35,4]]
    and ($complete|length)==1 and $complete[0].is_complete and $complete[0].cases==6
    and $complete[0].device_bytes_before==0 and $complete[0].device_bytes_after==0;"complete coverage/cleanup")
| require(all($checks|group_by(.layer_id,.rows)[]; (map(.variant)|sort)==[null,$control,$candidate])
    and all($checks[];.checks.guards_and_immutable_inputs
      and (if .variant==$candidate then .checks.intermediate_capture_matches_normal_kernel
        and .timing_control_hidden_bits_preserved and .timing_control_output_bits_preserved
        and all([.checks.gate_vs_cpu,.checks.up_vs_cpu,.checks.swiglu_vs_cpu,.checks.complete_down_vs_cpu_on_actual_swiglu][];.max_abs_diff<0.001)
        else true end));"numerical/control validation")
| require(all($pairs|group_by(.layer_id,.rows)[];map(.trial)==[0,1,2,3,4])
    and all($pairs[];.control_variant==$control and .variant==$candidate
      and .activation_quantization_ms==0 and .quantization_scratch_capacity_bytes==0
      and .threads==128 and .additional_prepared_bytes==47906816 and .joint_prepared_bytes==0
      and .cache_policy=={control_weights:"nc",candidate_weights:"cg",activations:"nc",geometry:"one-column/four-value/128",prepared_down:"unchanged"}
      and [.measurements[].variant]==(if .trial%2==0 then [$control,$candidate] else [$candidate,$control] end));"control/alternating pairs")
| require(all($pairs[];. as $pair|all(.measurements[];. as $arm|
    [.stages[].stage]==(if $pair.trial%2==0 then ["gate_up_swiglu","isolated_down","complete"] else ["complete","isolated_down","gate_up_swiglu"] end)
    and all(.stages[];. as $stage|.enqueue_counters as $counts|
      .iterations==10 and .gpu_ms>0 and .completed_unix_us>.started_unix_us
      and all(["device_allocs","device_frees","h2d_copies","h2d_bytes","d2h_copies","d2h_bytes","cublas_calls","stream_syncs"][];$counts[.]==0)
      and $counts.launches==(if $stage.stage=="complete" then 20 else 10 end)
      and $stage.unfused_dispatch==null
      and (if $arm.variant==$candidate then
        $stage.weight_cg_dispatch=={fused_launches:(if $stage.stage=="isolated_down" then 0 else 10 end),weight_cache_policy:"cg",activation_cache_policy:"nc"}
        else $stage.weight_cg_dispatch==null end)
      and $counts.stream_waits==(if $pair.rows==1 then 0 elif $stage.stage=="complete" then 20 else 10 end))));"timed work/side effects")
| require(all($screens[];. as $screen|
    [$pairs[]|select(.layer_id==$screen.layer_id and .rows==$screen.rows)|
      [.measurements[]|select(.variant==$control)|.stages[]|select(.stage=="complete")|.gpu_ms]
      + [.measurements[]|select(.variant==$candidate)|.stages[]|select(.stage=="complete")|.gpu_ms]] as $actual|
    .control_variant==$control and .result.pairs==$actual
      and .result.control_median_ms==($actual|map(.[0])|median)
      and .result.candidate_median_ms==($actual|map(.[1])|median)
      and .result.throughput_gain==(.result.control_median_ms/.result.candidate_median_ms-1)
      and .result.bootstrap_resamples==10000 and .result.bootstrap_seed==0
      and (.result.bootstrap_gain_ci95|length)==2
      and all(.result.bootstrap_gain_ci95[];type=="number")
      and .result.bootstrap_gain_ci95[0]<=.result.bootstrap_gain_ci95[1]
      and .result.passes_micro_screen==(.result.throughput_gain>=0.05 and .result.bootstrap_gain_ci95[0]>0)
      and (.result.passes_micro_screen|not) and (.runtime_integration_allowed|not)
      and (if .rows==1 then .result.bootstrap_gain_ci95[1]<0.05 else true end));"observed below-gate screen")
| [$pairs[]|.gpu_before,.gpu_after] as $gpu
| require(all($gpu[];(.device|contains(", P0,")) and (.device|contains("1113 MHz, 3004 MHz"))
    and (.processes|test("^1627047, target/release/examples/q8_execution_bench, [0-9]+ MiB$")));"GPU clocks/process isolation")
| [$pairs[].measurements[].stages[]]|sort_by(.started_unix_us) as $intervals
| require(all(range(1;$intervals|length);$intervals[.].started_unix_us >= $intervals[.-1].completed_unix_us);"overlapping primary intervals")
| {is_complete:true,source:$manifest.revision,executable:$manifest.executable_sha256,control:$control,candidate:$candidate,
    pairs:30,primary_intervals:($intervals|length),gpu_boundary_samples:60,validation_records:18,
    bits_equal_to_current_split_control:true,prepared_cache_identities_unchanged:true,
    tracked_device_bytes_after:0,runtime_integration_allowed:false,
    screens:($screens|map({layer_id,rows,result})),
    stages:($pairs|group_by(.layer_id,.rows)|map(. as $shape|{layer_id:.[0].layer_id,rows:.[0].rows,
      arms:([$shape[].measurements[]]|group_by(.variant)|map({variant:.[0].variant,
        stages:([.[].stages[]]|group_by(.stage)|map({stage:.[0].stage,median_gpu_ms:map(.gpu_ms)|median}))}))})),
    candidate_errors:($checks|map(select(.variant==$candidate))|{
      gate_max_abs:map(.checks.gate_vs_cpu.max_abs_diff)|max,
      up_max_abs:map(.checks.up_vs_cpu.max_abs_diff)|max,
      swiglu_max_abs:map(.checks.swiglu_vs_cpu.max_abs_diff)|max,
      down_max_abs:map(.checks.complete_down_vs_cpu_on_actual_swiglu.max_abs_diff)|max})}
