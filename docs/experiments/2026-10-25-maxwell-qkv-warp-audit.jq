# jq -s -f THIS docs/experiments/2026-10-25-maxwell-qkv-warp-screen.jsonl
def require($condition;$message): if $condition then . else error($message) end;
def median: sort | .[length/2|floor];
. as $rows
| map(select(.kind=="qkv_warp_manifest")) as $manifests
| $rows | map(select(.kind=="qkv_warp_validation")) as $checks
| $rows | map(select(.kind=="qkv_warp_pair")) as $pairs
| $rows | map(select(.kind=="qkv_warp_screen")) as $screens
| $rows | map(select(.kind=="qkv_warp_complete")) as $complete
| require(($manifests|length)==1;"manifest multiplicity")
| $manifests[0] as $manifest
| require($manifest.revision=="499d2dea3988ec4418856c6d779cec75d991cd85" and $manifest.worktree==""
    and $manifest.executable_sha256=="04aab98908b357e522b4aaefecf96a0969459b37eef2919ee796af5b0dc4a04b"
    and $manifest.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $manifest.build_profile=="release" and $manifest.trials==5 and $manifest.iterations==10
    and $manifest.memory=={original_weights:3279519744,prepared_weights:1192935424,tracked_device_bytes:4472455168,resident_kv:0,additional_prepared_bytes:0}
    and $manifest.prepared_weights.families==["down","vocabulary"]
    and $manifest.prepared_weights.single_row_down_split_launch=={columns:4,threads:256};"source/executable/control/configuration")
| require(($checks|length)==6 and ($pairs|length)==30 and ($screens|length)==6
    and [$checks[]|[.layer_id,.rows]]==[[0,1],[0,4],[17,1],[17,4],[35,1],[35,4]]
    and [$screens[]|[.layer_id,.rows]]==[$checks[]|[.layer_id,.rows]]
    and ($complete|length)==1 and $complete[0].is_complete
    and $complete[0].device_bytes_before==0 and $complete[0].device_bytes_after==0;"coverage/cleanup")
| require(all($checks[];. as $check|
    .checks.control_bits_equal and .checks.guards_and_immutable_read_spans
    and .scratch_bytes==(if .rows==1 then 18688 else 73984 end)
    and .tracked_device_bytes==4472455168+.scratch_bytes
    and [.sources[].name]==["blk.\(.layer_id).attn_q.weight","blk.\(.layer_id).attn_k.weight","blk.\(.layer_id).attn_v.weight"]
    and [.sources[].shape]==[[2048,2048],[2048,256],[2048,256]]
    and all(.sources[];.dtype=="Q8_0" and (.source_sha256|test("^[0-9a-f]{64}$")))
    and .sources==[.checks.outputs[].source]
    and [.checks.biases[].bytes]==[8192,1024,1024]
    and [.checks.outputs[].cpu.compared_values]==[2048*.rows,256*.rows,256*.rows]);"numerics/source/bias/memory evidence")
| require(all($pairs|group_by(.layer_id,.rows)[];map(.trial)==[0,1,2,3,4])
    and all($pairs[];. as $pair|($checks[]|select(.layer_id==$pair.layer_id and .rows==$pair.rows)) as $check|
      .tracked_device_bytes==$check.tracked_device_bytes and .scratch_bytes==$check.scratch_bytes
      and [.measurements[].is_warp]==(if .trial%2==0 then [false,true] else [true,false] end)
      and all(.measurements[];. as $arm|.enqueue_counters as $counts|
        .iterations==10 and .gpu_ms>0 and .gpu_ms_per_request==(.gpu_ms/$pair.rows)
        and .completed_unix_us>.started_unix_us
        and $counts.launches==10*$pair.rows and .operation_launches==$counts.launches
        and .operation==(if .is_warp then "qkv_f32xq8_0_gguf_warp" else "qkv_f32xq8_0_gguf_decode" end)
        and all(["device_allocs","device_frees","h2d_copies","h2d_bytes","d2h_copies","d2h_bytes","cublas_calls","stream_syncs","stream_waits"][];$counts[.]==0)));
    "arm order/actual dispatch/timed effects/residency")
| require(all($screens[];. as $screen|
    [$pairs[]|select(.layer_id==$screen.layer_id and .rows==$screen.rows)|
      [.measurements[]|select(.is_warp==false)|.gpu_ms]+[.measurements[]|select(.is_warp)|.gpu_ms]] as $actual|
    .result.pairs==$actual and .result.control_median_ms==($actual|map(.[0])|median)
    and .result.candidate_median_ms==($actual|map(.[1])|median)
    and .result.throughput_gain==(.result.control_median_ms/.result.candidate_median_ms-1)
    and .result.passes_micro_screen==(.result.throughput_gain>=0.05 and .result.bootstrap_gain_ci95[0]>0)
    and .result.passes_micro_screen
    and .optimistic_profile_weighted_e2e_gain==(1/(1-0.1172+0.1172/(1+.result.throughput_gain))-1)
    and .optimistic_profile_weighted_e2e_gain<0.03 and (.runtime_integration_allowed|not));"actual-pair positive stage result and below-gate ceiling")
| [$pairs[]|.gpu_before,.gpu_after] as $gpu
| require(all($gpu[];(.device|contains(", P0,")) and (.device|contains("1113 MHz, 3004 MHz"))
    and (.processes|test("^1405396, target/release/examples/q8_execution_bench, [0-9]+ MiB$")));"GPU clocks/isolation")
| [$pairs[].measurements[]]|sort_by(.started_unix_us) as $intervals
| require(all(range(1;$intervals|length);$intervals[.].started_unix_us >= $intervals[.-1].completed_unix_us);"overlapping primary intervals")
| {is_complete:true,source:$manifest.revision,executable:$manifest.executable_sha256,
    pairs:30,primary_intervals:($intervals|length),gpu_boundary_samples:60,validation_records:6,
    all_qkv_control_bits_equal:true,no_timed_allocation_copy_sync:true,tracked_device_bytes_after:0,
    stage_micro_gate_passed:true,standalone_runtime_integration_allowed:false,
    screens:$screens,
    memory:($checks|group_by(.rows)|map({rows:.[0].rows,scratch_bytes:map(.scratch_bytes)|unique,
      tracked_device_bytes:map(.tracked_device_bytes)|unique})),
    max_cpu_errors:([range(3)]|map(. as $i|{projection:["q","k","v"][$i],
      max_abs:[$checks[].checks.outputs[$i].cpu.max_abs_diff]|max,
      max_mean_abs:[$checks[].checks.outputs[$i].cpu.mean_abs_diff]|max}))}
