# jq -s --arg source REV --arg elf SHA --arg pid PID --slurpfile capture CAPTURE -f THIS SCREEN
def require($ok;$why): if $ok then . else error($why) end;
def median: sort|.[length/2|floor];
def key: [.layer_id,.rows];
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity")|.[0];
. as $records
| one("ffn_ablation_manifest") as $m
| $records|one("ffn_ablation_complete") as $done
| $records|map(select(.kind=="ffn_ablation_pair")) as $pairs
| $records|map(select(.kind=="ffn_ablation_validation")) as $checks
| $records|map(select(.kind=="ffn_ablation_screen")) as $screens
| $records|map(select(.kind=="ffn_ablation_warmup")) as $warmups
| $records|map(select(.kind=="ffn_ablation_preparation")) as $preparation
| "split-f32x4-128" as $control
| "split-f32x4-role-parallel-256" as $candidate
| [[0,1],[0,4],[17,1],[17,4],[35,1],[35,4]] as $shapes
| require($m.schema_version==1 and $m.revision==$source
    and $m.worktree=="" and $m.build_profile=="release"
    and $m.executable_sha256==$elf
    and $m.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $m.fixture_sha256=="eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687"
    and $m.capture_report_sha256=="d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801"
    and $m.execution_configuration==$capture[0].execution_configuration
    and ($m.prepared_weights|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))
      ==($capture[0].prepared_weights|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))
    and ($m.prepared_weights.tensors|length)==37
    and $m.memory=={original_weights:3279519744,prepared_weights:1192935424,resident_kv:0,tracked_device_bytes:4472455168}
    and $m.rows==[1,4] and $m.trials==5 and $m.iterations==10
    and $m.fp32_variants==[$candidate] and $m.is_fp32_ablation and ($m.is_q81_numerical_track|not)
    and $m.role_parallel_contract=={roles:2,partials_per_role:128,threads:256,outputs_per_cta:1,
      has_changed_summation_ownership:false,has_activation_quantization:false,shared_bytes:1024}
    and $m.eight_value_contract==null
    and $m.cuda_flags==["-O3","sm_52","compute_52","no fast math"];"clean frozen source/control")
| require(($records|length)==68 and ($pairs|length)==30 and ($checks|length)==18 and ($screens|length)==6
    and ([$screens[]|key]|sort)==$shapes and $done.is_complete and $done.cases==6
    and $done.device_bytes_before==0 and $done.device_bytes_after==0;"coverage/cleanup")
| require(($warmups|length)==6 and ([$warmups[]|key]|sort)==$shapes
    and all($warmups[];.variant==$candidate and .control_variant==$control
      and .control.minimum_gpu_ms==1000 and .candidate.minimum_gpu_ms==100
      and all([.control,.candidate][];.elapsed_gpu_ms>=.minimum_gpu_ms and .iterations_per_batch==32
        and .maximum_batches==256 and .batches>0 and .batches<=256));"bounded untimed warmup")
| require(($preparation|length)==6 and ([$preparation[]|key]|sort)==$shapes
    and all($preparation[];.preparation.layout=="SeparateScalesValues"
      and .preparation.additional_prepared_bytes==47906816
      and (.preparation.keys|map(.source.shape))==[{input_dim:2048,output_dim:11008},{input_dim:2048,output_dim:11008}]
      and .scratch_capacity_bytes==(if .rows==1 then 201152 else 803264 end)
      and .preparation.tracked_bytes_after==(if .rows==1 then 4520563136 else 4521165248 end)
      and .preparation.tracked_bytes_after-.preparation.tracked_bytes_before==47906816);"shared preparation/scratch")
| require(([$checks[]|key]|unique|sort)==$shapes
    and all($checks|group_by(key)[];(map(.variant)|sort)==[null,$control,$candidate])
    and all($checks[];. as $check|.checks.guards_and_immutable_inputs
      and .scratch_capacity_bytes==(if .rows==1 then 201152 else 803264 end)
      and (if .variant==$candidate then .checks.intermediate_capture_matches_normal_kernel
        and all([.checks.gate_vs_cpu,.checks.up_vs_cpu,.checks.swiglu_vs_cpu,.checks.complete_down_vs_cpu_on_actual_swiglu][];
          .max_abs_diff>=0 and .max_abs_diff<0.001)
        and all(["gate","up","swiglu","complete_down"][];
          $check.drift_vs_timing_control[.] as $drift|
          $drift.max_abs_diff==0 and $drift.bitwise_equal
          and $drift.mean_abs_diff>=0 and $drift.mean_abs_diff<=$drift.max_abs_diff
          and $drift.compared_values==$check.rows*(if .=="complete_down" then 2048 else 11008 end))
        and .timing_control_hidden_bits_preserved==.drift_vs_timing_control.swiglu.bitwise_equal
        and .timing_control_output_bits_preserved==.drift_vs_timing_control.complete_down.bitwise_equal
        else true end));"CPU bounds/capture parity/reported control drift")
| require(([$pairs[]|key]|unique|sort)==$shapes and all($pairs|group_by(key)[];map(.trial)==[0,1,2,3,4])
    and all($pairs[];. as $pair|.control_variant==$control and .variant==$candidate and .threads==256
      and .tracked_device_bytes==(if .rows==1 then 4520563136 else 4521165248 end)
      and .additional_prepared_bytes==47906816 and .joint_prepared_bytes==0 and .task_pair_prepared_bytes==0
      and .scratch_capacity_bytes==(if .rows==1 then 201152 else 803264 end)
      and .activation_quantization_ms==0 and .quantization_payload_bytes==0 and .quantization_scratch_capacity_bytes==0
      and (.is_numerical_track|not)
      and [.measurements[].variant]==(if .trial%2==0 then [$control,$candidate] else [$candidate,$control] end)
      and all(.measurements[];. as $arm|
        [.stages[].stage]==(if $pair.trial%2==0 then ["gate_up_swiglu","isolated_down","complete"] else ["complete","isolated_down","gate_up_swiglu"] end)
        and all(.stages[];. as $stage|.enqueue_counters as $c|
          .iterations==10 and .gpu_ms>0 and .completed_unix_us>.started_unix_us
          and all(["device_allocs","device_frees","h2d_copies","h2d_bytes","d2h_copies","d2h_bytes","cublas_calls","stream_syncs"][];$c[.]==0)
          and $c.launches==(if .stage=="complete" then 20 else 10 end)
          and $c.stream_waits==(if $pair.rows==1 then 0 elif .stage=="complete" then 20 else 10 end)
          and .unfused_dispatch==null and .weight_cg_dispatch==null and .byte_permute_dispatch==null
          and .virtual_warp_dispatch==null and .persistent_tiles_dispatch==null and .cohorts_dispatch==null
          and .eight_value_dispatch==null and .task_pair_dispatch==null and .native_warp_dispatch==null
          and (if $arm.variant==$candidate then .role_parallel_dispatch=={operation:"ffn_q8_split_f32x4_role_parallel_256",
            fused_launches:(if .stage=="isolated_down" then 0 else 10 end),values_per_task:4,roles:2,partials_per_role:128,
            threads:256,blocks_per_row:11008,shared_bytes:1024,has_changed_summation_ownership:false,has_activation_quantization:false}
          else .role_parallel_dispatch==null end))));"actual complete work/dispatch/side effects")
| require(all($screens[];. as $screen|
    [$pairs[]|select(key==($screen|key))|
      [.measurements[]|select(.variant==$control)|.stages[]|select(.stage=="complete")|.gpu_ms]
      + [.measurements[]|select(.variant==$candidate)|.stages[]|select(.stage=="complete")|.gpu_ms]] as $raw|
    .variant==$candidate and .control_variant==$control and .result.pairs==$raw
    and .result.control_median_ms==($raw|map(.[0])|median) and .result.candidate_median_ms==($raw|map(.[1])|median)
    and .result.throughput_gain==(.result.control_median_ms/.result.candidate_median_ms-1)
    and .result.bootstrap_resamples==10000 and .result.bootstrap_seed==0
    and (.result.bootstrap_gain_ci95|length)==2 and .result.bootstrap_gain_ci95[0]<=.result.bootstrap_gain_ci95[1]
    and .result.bootstrap_gain_ci95[0]>=((($raw|map(.[0])|min)/($raw|map(.[1])|max))-1)
    and .result.bootstrap_gain_ci95[1]<=((($raw|map(.[0])|max)/($raw|map(.[1])|min))-1)
    and .is_decode_screen==(.rows==1)
    and .result.passes_micro_screen==(.result.throughput_gain>=0.05 and .result.bootstrap_gain_ci95[0]>0)
    and (.runtime_integration_allowed|not));"raw medians/paired bounds/unchanged gate")
| [$pairs[]|.gpu_before,.gpu_after] as $gpu
| require(all($gpu[];(.device|contains("GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, Tesla M40 24GB"))
    and (.device|contains(", P0,")) and (.device|contains("1113 MHz, 3004 MHz"))
    and (.processes|test("^"+$pid+", target/release/examples/q8_execution_bench, [0-9]+ MiB$")));"GPU isolation/clocks")
| ([$pairs[].measurements[].stages[]]|sort_by(.started_unix_us)) as $intervals
| require(($intervals|length)==180 and all(range(1;180);$intervals[.].started_unix_us>=$intervals[.-1].completed_unix_us);"serial intervals")
| {passes_audit:true,source:$m.revision,executable_sha256:$m.executable_sha256,pairs:30,gpu_boundaries:60,
    serial_primary_intervals:180,validation_records:18,qualified_prepared_cache_unchanged:true,tracked_device_bytes_after:0,
    screens:($screens|map({layer_id,rows,result})),
    stages:($pairs|group_by(key)|map(. as $shape|{layer_id:.[0].layer_id,rows:.[0].rows,
      arms:([$shape[].measurements[]]|group_by(.variant)|map({variant:.[0].variant,
        stages:([.[].stages[]]|group_by(.stage)|map({stage:.[0].stage,median_gpu_ms:map(.gpu_ms)|median}))}))})),
    candidate_checks:[$checks[]|select(.variant==$candidate)|{layer_id,rows,checks,drift_vs_timing_control}],
    any_micro_gate_passes:any($screens[];.result.passes_micro_screen),runtime_integration_allowed:false,is_e2e_claim:false,
    bootstrap_scope:"benchmark records seeded intervals; raw medians/gains, gates and resample bounds independently checked",
    changes_runtime_defaults:false}
