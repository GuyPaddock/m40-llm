# jq -s --arg source REV --arg elf SHA --arg pid PID --argjson trials 5
#   --slurpfile capture docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl -f THIS RAW
def require($ok;$why): if $ok then . else error($why) end;
def median: sort | if length%2==1 then .[length/2|floor] else (.[length/2-1]+.[length/2])/2 end;
def key: [.layer_id,.rows];
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity")|.[0];
def variation: map(1/.) as $v | ($v|median) as $m | $v|map((.-$m)|fabs)|median/$m;
. as $records
| one("ffn_compound_manifest") as $m
| $records|one("ffn_ablation_complete") as $done
| $records|map(select(.kind=="ffn_compound_trial")) as $pairs
| $records|map(select(.kind=="ffn_compound_validation")) as $checks
| $records|map(select(.kind=="ffn_compound_summary")) as $screens
| $records|map(select(.kind=="ffn_compound_warmup")) as $warmups
| $records|map(select(.kind=="ffn_ablation_preparation")) as $setup
| $records|map(select(.kind=="ffn_joint_preparation")) as $joint
| ["split-f32x4-128","joint-f32x4-128","split-f32x4-shared-scales-128","joint-f32x4-shared-scales-128"] as $arms
| [[0,1],[0,4],[17,1],[17,4],[35,1],[35,4]] as $shapes
| require($m.revision==$source and $m.worktree=="" and $m.executable_sha256==$elf and $m.build_profile=="release"
    and $m.schema_version==1 and ($trials==5 or $trials==10 or $trials==20)
    and $m.trials==$trials and $m.iterations==10 and $m.rows==[1,4] and $m.fp32_variants==$arms
    and $m.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $m.fixture_sha256=="eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687"
    and $m.capture_report_sha256=="d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801"
    and $m.execution_configuration==$capture[0].execution_configuration
    and ($m.prepared_weights|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))
      ==($capture[0].prepared_weights|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))
    and ($m.prepared_weights.tensors|length)==37
    and $m.memory=={original_weights:3279519744,prepared_weights:1192935424,resident_kv:0,tracked_device_bytes:4472455168}
    and $m.compound_contract=={threads:128,outputs_per_cta:1,max_k:4096,shared_bytes_formula:"1024+8*(K/32)",
      block_barriers:2,has_changed_summation_ownership:false,has_activation_quantization:false,
      complete_ffn_gate:0.05,comparisons:["compound-vs-split","compound-vs-joint"]}
    and $m.cuda_flags==["-O3","sm_52","compute_52","no fast math"] and ($m.is_q81_numerical_track|not);"frozen clean control/contract")
| require(($records|length)==68+6*$trials and ($pairs|length)==6*$trials and ($checks|length)==24
    and ($screens|length)==6 and ([$screens[]|key]|sort)==$shapes
    and $done.is_complete and $done.cases==6 and $done.device_bytes_before==0 and $done.device_bytes_after==0;"coverage/cleanup")
| require(($setup|length)==6 and ($joint|length)==6
    and ([$setup[]|key]|sort)==$shapes and ([$joint[]|key]|sort)==$shapes
    and all($setup[];.preparation.additional_prepared_bytes==47906816
      and .scratch_capacity_bytes==(if .rows==1 then 201152 else 803264 end))
    and all($joint[];. as $r|($setup[]|select(key==($r|key))) as $s|
      .preparation.key.gate==$s.preparation.keys[0].source and .preparation.key.up==$s.preparation.keys[1].source
      and .preparation.paired_scale_bytes==2818048 and .preparation.paired_payload_bytes==45088768
      and .preparation.additional_prepared_bytes==47906816 and .preparation.combined_prepared_bytes==95813632
      and .preparation.tracked_bytes_after-.preparation.tracked_bytes_before==47906816);"shared owners and source identities")
| require(all($checks|group_by(key)[];(map(.variant)|sort)==($arms|sort))
    and all($checks[];.has_control_bits and .checks.guards_and_immutable_inputs
      and .checks.intermediate_capture_matches_normal_kernel
      and all([.checks.gate_vs_cpu,.checks.up_vs_cpu,.checks.swiglu_vs_cpu,.checks.complete_down_vs_cpu_on_actual_swiglu][];
        .max_abs_diff>=0 and .max_abs_diff<0.001));"numerics/guards/capture/bit parity")
| require(($warmups|length)==24 and all($warmups|group_by(key)[];(map(.variant)|sort)==($arms|sort))
    and all($warmups[];.warmup.minimum_gpu_ms==1000 and .warmup.elapsed_gpu_ms>=1000
      and .warmup.iterations_per_batch==32 and .warmup.batches>0 and .warmup.batches<=256);"all-arm warmup")
| require(([$pairs[]|key]|unique|sort)==$shapes and all($pairs|group_by(key)[];map(.trial)==[range($trials)])
    and all($pairs[];. as $p|
      ([0,1,2,3] as $order|($p.trial%4) as $n|($order[$n:]+$order[:$n])|if $p.trial%2==1 then reverse else . end) as $order|
      .order==$order and [.measurements[].variant]==[$order[]|$arms[.]]
      and .split_prepared_bytes==47906816 and .joint_prepared_bytes==47906816
      and .scratch_capacity_bytes==(if .rows==1 then 201152 else 803264 end)
      and .tracked_device_bytes==4472455168+95813632+.scratch_capacity_bytes
      and all(.measurements[];. as $a|
        [.stages[].stage]==(if $p.trial%2==0 then ["gate_up_swiglu","isolated_down","complete"] else ["complete","isolated_down","gate_up_swiglu"] end)
        and all(.stages[];.enqueue_counters as $c|
          .iterations==10 and .gpu_ms>0 and .completed_unix_us>.started_unix_us
          and all(["device_allocs","device_frees","h2d_copies","h2d_bytes","d2h_copies","d2h_bytes","cublas_calls","stream_syncs"][];$c[.]==0)
          and $c.launches==(if .stage=="complete" then 20 else 10 end)
          and $c.stream_waits==(if $p.rows==1 then 0 elif .stage=="complete" then 20 else 10 end)
          and (if $a.variant==$arms[2] or $a.variant==$arms[3] then
            .shared_scales_dispatch.operation==(if $a.variant==$arms[2] then "ffn_q8_split_f32x4_shared_scales_128" else "ffn_q8_joint_f32x4_shared_scales_128" end)
            and .shared_scales_dispatch.shared_bytes==1536 and .shared_scales_dispatch.threads==128
            and .shared_scales_dispatch.block_barriers==2 and .shared_scales_dispatch.blocks_per_row==11008
            and .shared_scales_dispatch.fused_launches==(if .stage=="isolated_down" then 0 else 10 end)
          else .shared_scales_dispatch==null end))));"complete work/order/dispatch/side effects")
| require(all($screens[];. as $s|all([
    ["compound_vs_split",0,3],["compound_vs_joint",1,3],["joint_vs_split",0,1],["shared_vs_split",0,2]
    ][];. as $comparison|
      [$pairs[]|select(key==($s|key))|
        [.measurements[]|select(.variant==$arms[$comparison[1]])|.stages[]|select(.stage=="complete")|.gpu_ms]
        + [.measurements[]|select(.variant==$arms[$comparison[2]])|.stages[]|select(.stage=="complete")|.gpu_ms]] as $raw|
      $s[$comparison[0]] as $r|
      $r.pairs==$raw and $r.native_median_ms==($raw|map(.[0])|median)
      and $r.candidate_median_ms==($raw|map(.[1])|median)
      and $r.throughput_gain==($r.native_median_ms/$r.candidate_median_ms-1)
      and $r.bootstrap_seed==0 and $r.bootstrap_resamples==10000
      and ($r.bootstrap_gain_ci95|length)==2 and $r.bootstrap_gain_ci95[0]<=$r.bootstrap_gain_ci95[1]
      and $r.bootstrap_gain_ci95[0]>=((($raw|map(.[0])|min)/($raw|map(.[1])|max))-1)
      and $r.bootstrap_gain_ci95[1]<=((($raw|map(.[0])|max)/($raw|map(.[1])|min))-1)
      and $r.passes_micro_screen==($r.throughput_gain>=0.05 and $r.bootstrap_gain_ci95[0]>0)));"raw paired summaries/gates")
| [$pairs[]|.gpu_before,.gpu_after] as $gpu
| require(all($gpu[];(.device|contains("GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, Tesla M40 24GB"))
    and (.device|contains(", P0,")) and (.device|contains("1113 MHz, 3004 MHz"))
    and (.processes|test("^"+$pid+", target/release/examples/q8_execution_bench, [0-9]+ MiB$")));"GPU isolation/clocks")
| ([$pairs[].measurements[].stages[]]|sort_by(.started_unix_us)) as $intervals
| require(($intervals|length)==72*$trials and all(range(1;$intervals|length);
    $intervals[.].started_unix_us>=$intervals[.-1].completed_unix_us);"serial primary intervals")
| {passes_audit:true,source:$source,executable_sha256:$elf,trials:$trials,
    gpu_boundaries:($gpu|length),primary_intervals:($intervals|length),validation_records:24,
    tracked_device_bytes_after:0,screens:$screens,
    timing_variation:($pairs|group_by(key)|map(. as $shape|{layer_id:.[0].layer_id,rows:.[0].rows,
      arms:([$shape[].measurements[]]|group_by(.variant)|map({variant:.[0].variant,
        complete_mad_fraction:([.[].stages[]|select(.stage=="complete")|.gpu_ms]|variation)}))})),
    stages:($pairs|group_by(key)|map(. as $shape|{layer_id:.[0].layer_id,rows:.[0].rows,
      arms:([$shape[].measurements[]]|group_by(.variant)|map({variant:.[0].variant,
        stages:([.[].stages[]]|group_by(.stage)|map({stage:.[0].stage,median_gpu_ms:map(.gpu_ms)|median}))}))})),
    passes_all_one_row_gates:all($screens[]|select(.rows==1);.compound_vs_split.passes_micro_screen),
    is_e2e_claim:false,changes_runtime_defaults:false}
