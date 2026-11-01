# Independently check raw paired timings, dispatch, coverage and residency.
def require($condition;$message): if $condition then . else error($message) end;
def median: sort|.[length/2|floor];
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity: \($kind)")|.[0];
def key: [.layer_id,.rows];
. as $rows
| one("ffn_ablation_manifest") as $manifest
| $rows|one("ffn_ablation_complete") as $complete
| $rows|map(select(.kind=="ffn_ablation_pair")) as $pairs
| $rows|map(select(.kind=="ffn_ablation_validation")) as $checks
| $rows|map(select(.kind=="ffn_ablation_screen")) as $screens
| $rows|map(select(.kind=="ffn_ablation_warmup")) as $warmup
| "split-f32x4-128" as $control
| "split-f32x4-virtual-warp-128" as $candidate
| require($manifest.revision=="21ca3491412f1782736e6f0a3aba2d32be3e4700"
    and $manifest.worktree=="" and $manifest.build_profile=="release"
    and $manifest.executable_sha256=="59ab040d07ca1af396bea37acc0b13f9da0a8e1053cd6e0adc0e097fcb26975f"
    and $manifest.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $manifest.fixture_sha256=="eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687"
    and $manifest.capture_report_sha256=="d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801"
    and $manifest.execution_configuration==$capture[0].execution_configuration
    and $manifest.rows==[1,4] and $manifest.trials==5 and $manifest.iterations==10
    and $manifest.fp32_variants==[$candidate] and ($manifest.is_q81_numerical_track|not)
    and $manifest.memory=={original_weights:3279519744,prepared_weights:1192935424,resident_kv:0,tracked_device_bytes:4472455168}
    and $manifest.prepared_weights.families==["down","vocabulary"]
    and ($manifest.prepared_weights.tensors|map(.key))==($previous[0].prepared_weights.tensors|map(.key))
    and ($manifest.prepared_weights.tensors|length)==37
    and $manifest.prepared_weights.single_row_down_split_launch=={columns:4,threads:256};"source, frozen inputs and qualified cache")
| require(($pairs|length)==30 and ($checks|length)==18 and ($screens|length)==6
    and ([$screens[]|key]|sort)==[[0,1],[0,4],[17,1],[17,4],[35,1],[35,4]]
    and $complete.is_complete and $complete.cases==6
    and $complete.device_bytes_before==0 and $complete.device_bytes_after==0;"coverage and cleanup")
| require(($warmup|length)==6 and ([$warmup[]|key]|sort)==([$screens[]|key]|sort)
    and all($warmup[];.variant==$candidate and .control_variant==$control
      and .control.minimum_gpu_ms==1000 and .candidate.minimum_gpu_ms==100
      and all([.control,.candidate][];.elapsed_gpu_ms>=.minimum_gpu_ms
        and .iterations_per_batch==32 and .maximum_batches==256 and .batches<=256));"bounded warmup before measured pairs")
| require(all($checks|group_by(key)[];(map(.variant)|sort)==[null,$control,$candidate])
    and all($checks[];.checks.guards_and_immutable_inputs
      and .scratch_capacity_bytes==(if .rows==1 then 201152 else 803264 end)
      and (if .variant==$candidate then
        .timing_control_hidden_bits_preserved and .timing_control_output_bits_preserved
        and .checks.intermediate_capture_matches_normal_kernel
        and all([.checks.gate_vs_cpu,.checks.up_vs_cpu,.checks.swiglu_vs_cpu,
          .checks.complete_down_vs_cpu_on_actual_swiglu][];.max_abs_diff<0.001)
        else true end));"numerical control bits, CPU references and guards")
| require(all($pairs|group_by(key)[];map(.trial)==[0,1,2,3,4]
    and (map(.tracked_device_bytes)|unique|length)==1)
    and all($pairs[];. as $pair|
      .control_variant==$control and .variant==$candidate and .threads==128
      and .additional_prepared_bytes==47906816 and .joint_prepared_bytes==0
      and .scratch_capacity_bytes==(if .rows==1 then 201152 else 803264 end)
      and .activation_quantization_ms==0 and .quantization_payload_bytes==0
      and .quantization_scratch_capacity_bytes==0 and (.is_numerical_track|not)
      and [.measurements[].variant]==(if .trial%2==0 then [$control,$candidate] else [$candidate,$control] end)
      and all(.measurements[];. as $arm|
        [.stages[].stage]==(if $pair.trial%2==0 then ["gate_up_swiglu","isolated_down","complete"] else ["complete","isolated_down","gate_up_swiglu"] end)
        and all(.stages[];. as $stage|.enqueue_counters as $counts|
          .iterations==10 and .gpu_ms>0 and .completed_unix_us>.started_unix_us
          and all(["device_allocs","device_frees","h2d_copies","h2d_bytes","d2h_copies","d2h_bytes","cublas_calls","stream_syncs"][];$counts[.]==0)
          and $counts.launches==(if $stage.stage=="complete" then 20 else 10 end)
          and $counts.stream_waits==(if $pair.rows==1 then 0 elif $stage.stage=="complete" then 20 else 10 end)
          and .unfused_dispatch==null and .weight_cg_dispatch==null and .byte_permute_dispatch==null
          and (if $arm.variant==$candidate then
            .virtual_warp_dispatch=={operation:"ffn_q8_split_f32x4_virtual_warp_128",
              fused_launches:(if $stage.stage=="isolated_down" then 0 else 10 end),
              output_warps:4,virtual_threads_per_output:128,shared_bytes:0}
            else .virtual_warp_dispatch==null end))));"actual paired work and allocation-free dispatch")
| require(all($screens[];. as $screen|
    [$pairs[]|select(key==($screen|key))|
      [.measurements[]|select(.variant==$control)|.stages[]|select(.stage=="complete")|.gpu_ms]
      + [.measurements[]|select(.variant==$candidate)|.stages[]|select(.stage=="complete")|.gpu_ms]] as $actual|
    .control_variant==$control and .result.pairs==$actual
    and .result.control_median_ms==($actual|map(.[0])|median)
    and .result.candidate_median_ms==($actual|map(.[1])|median)
    and .result.throughput_gain==(.result.control_median_ms/.result.candidate_median_ms-1)
    and .result.bootstrap_resamples==10000 and .result.bootstrap_seed==0
    and (.result.bootstrap_gain_ci95|length)==2
    and .result.bootstrap_gain_ci95[0]<=.result.bootstrap_gain_ci95[1]
    and .result.passes_micro_screen==(.result.throughput_gain>=0.05 and .result.bootstrap_gain_ci95[0]>0)
    and (.runtime_integration_allowed|not));"raw pair medians and unchanged promotion gate")
| ([$pairs[]|.gpu_before,.gpu_after]) as $gpu
| require(all($gpu[];(.device|contains(", P0,")) and (.device|contains("1113 MHz, 3004 MHz"))
    and (.processes|test("^2456852, target/release/examples/q8_execution_bench, [0-9]+ MiB$")));"GPU clocks and process isolation")
| ([$pairs[].measurements[].stages[]]|sort_by(.started_unix_us)) as $intervals
| require(($intervals|length)==180 and all(range(1;$intervals|length);
    $intervals[.].started_unix_us >= $intervals[.-1].completed_unix_us);"serial primary intervals")
| {passes_audit:true,source:$manifest.revision,executable_sha256:$manifest.executable_sha256,
    pairs:30,gpu_boundaries:60,serial_primary_intervals:180,validation_records:18,
    bits_equal_to_control:true,qualified_prepared_cache_unchanged:true,tracked_device_bytes_after:0,
    screens:($screens|map({layer_id,rows,result})),
    stages:($pairs|group_by(key)|map(. as $shape|{layer_id:.[0].layer_id,rows:.[0].rows,
      arms:([$shape[].measurements[]]|group_by(.variant)|map({variant:.[0].variant,
        stages:([.[].stages[]]|group_by(.stage)|map({stage:.[0].stage,median_gpu_ms:map(.gpu_ms)|median}))}))})),
    candidate_errors:($checks|map(select(.variant==$candidate))|{
      gate_max_abs:map(.checks.gate_vs_cpu.max_abs_diff)|max,
      up_max_abs:map(.checks.up_vs_cpu.max_abs_diff)|max,
      swiglu_max_abs:map(.checks.swiglu_vs_cpu.max_abs_diff)|max,
      down_max_abs:map(.checks.complete_down_vs_cpu_on_actual_swiglu.max_abs_diff)|max}),
    any_micro_gate_passes:any($screens[];.result.passes_micro_screen),
    runtime_integration_allowed:false,is_e2e_claim:false,changes_runtime_defaults:false}
