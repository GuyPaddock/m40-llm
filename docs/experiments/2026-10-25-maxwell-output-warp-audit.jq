# jq -s --slurpfile peer docs/experiments/2026-10-25-maxwell-qkv-warp-screen.jsonl -f THIS RAW
def require($condition;$message): if $condition then . else error($message) end;
def median: sort | .[length/2|floor];
. as $records
| map(select(.kind=="output_warp_manifest")) as $manifests
| $records | map(select(.kind=="output_warp_validation")) as $checks
| $records | map(select(.kind=="output_warp_pair")) as $pairs
| $records | map(select(.kind=="output_warp_screen")) as $screens
| $records | map(select(.kind=="output_warp_complete")) as $complete
| require(($manifests|length)==1;"manifest multiplicity")
| $manifests[0] as $manifest
| require($manifest.revision=="12191f55c1aedd917e8a83048b989bbc89802384" and $manifest.worktree==""
    and $manifest.executable_sha256=="8d0995e317beaa17e37dcf4d1632abf61c9b867f2dc46746951a3cfeef82241d"
    and $manifest.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $manifest.fixture_sha256==$peer[0].fixture_sha256
    and $manifest.capture_report_sha256==$peer[0].capture_report_sha256
    and $manifest.execution_configuration==$peer[0].execution_configuration
    and $manifest.build_profile=="release" and $manifest.trials==5 and $manifest.iterations==10
    and $manifest.memory=={original_weights:3279519744,prepared_weights:1192935424,tracked_device_bytes:4472455168,resident_kv:0,additional_prepared_bytes:0}
    and $manifest.prepared_weights.families==["down","vocabulary"]
    and $manifest.prepared_weights.single_row_down_split_launch=={columns:4,threads:256}
    and $manifest.qkv_peer.sha256=="9c1468a93abc8b4828e04d5880f1d1fbb70c36fe4790c73c27087a0098217b73"
    and $manifest.qkv_peer.revision==$peer[0].revision
    and $manifest.qkv_peer.executable_sha256==$peer[0].executable_sha256;"source/executable/control/peer/configuration")
| require(($checks|length)==6 and ($pairs|length)==30 and ($screens|length)==6
    and [$checks[]|[.layer_id,.rows]]==[[0,1],[0,4],[17,1],[17,4],[35,1],[35,4]]
    and [$screens[]|[.layer_id,.rows]]==[$checks[]|[.layer_id,.rows]]
    and ($complete|length)==1 and $complete[0].is_complete
    and $complete[0].device_bytes_before==0 and $complete[0].device_bytes_after==0;"coverage/cleanup")
| require(all($checks[];
    .checks.control_bits_equal and .checks.guards_and_immutable_read_spans
    and .scratch_bytes==(if .rows==1 then 16512 else 65664 end)
    and .tracked_device_bytes==4472455168+.scratch_bytes
    and [.sources[].name]==["blk.\(.layer_id).attn_output.weight"]
    and [.sources[].shape]==[[2048,2048]]
    and all(.sources[];.dtype=="Q8_0" and (.source_sha256|test("^[0-9a-f]{64}$")))
    and .sources==[.checks.outputs[].source] and .checks.biases==[null]
    and [.checks.outputs[].cpu.compared_values]==[2048*.rows]);"numerics/source/memory evidence")
| require(all($pairs|group_by(.layer_id,.rows)[];map(.trial)==[0,1,2,3,4])
    and all($pairs[];. as $pair|($checks[]|select(.layer_id==$pair.layer_id and .rows==$pair.rows)) as $check|
      .tracked_device_bytes==$check.tracked_device_bytes and .scratch_bytes==$check.scratch_bytes
      and [.measurements[].is_warp]==(if .trial%2==0 then [false,true] else [true,false] end)
      and all(.measurements[];.enqueue_counters as $counts|
        .iterations==10 and .gpu_ms>0 and .gpu_ms_per_request==(.gpu_ms/$pair.rows)
        and .completed_unix_us>.started_unix_us
        and $counts.launches==10*$pair.rows and .operation_launches==$counts.launches
        and .operation==(if .is_warp then "gemm_f32xq8_0_gguf_coltile4_warp" else "gemm_f32xq8_0_gguf_f32_decode" end)
        and all(["device_allocs","device_frees","h2d_copies","h2d_bytes","d2h_copies","d2h_bytes","cublas_calls","stream_syncs","stream_waits"][];$counts[.]==0)));
    "arm order/actual dispatch/timed effects/residency")
| require(all($screens[];. as $screen|
    ($peer[]|select(.kind=="qkv_warp_screen" and .layer_id==$screen.layer_id and .rows==$screen.rows)) as $qkv|
    [$pairs[]|select(.layer_id==$screen.layer_id and .rows==$screen.rows)|
      [.measurements[]|select(.is_warp==false)|.gpu_ms]+[.measurements[]|select(.is_warp)|.gpu_ms]] as $actual|
    .result.pairs==$actual and .result.control_median_ms==($actual|map(.[0])|median)
    and .result.candidate_median_ms==($actual|map(.[1])|median)
    and .result.throughput_gain==(.result.control_median_ms/.result.candidate_median_ms-1)
    and .result.passes_micro_screen==(.result.throughput_gain>=0.05 and .result.bootstrap_gain_ci95[0]>0)
    and .result.passes_micro_screen and $qkv.result.passes_micro_screen
    and .optimistic_profile_weighted_e2e_gain==(1/(1-0.0825+0.0825/(1+.result.throughput_gain))-1)
    and .optimistic_combined_e2e_gain==(1/(1-0.1172-0.0825+0.1172/(1+$qkv.result.throughput_gain)+0.0825/(1+.result.throughput_gain))-1)
    and .optimistic_combined_e2e_gain>=0.03
    and .runtime_integration_allowed==(.rows==1));"actual pairs/stage gates/prespecified combined ceiling")
| [$pairs[]|.gpu_before,.gpu_after] as $gpu
| require(all($gpu[];(.device|contains(", P0,")) and (.device|contains("1113 MHz, 3004 MHz"))
    and (.processes|test("^1444020, target/release/examples/q8_execution_bench, [0-9]+ MiB$")));"GPU clocks/isolation")
| [$pairs[].measurements[]]|sort_by(.started_unix_us) as $intervals
| require(all(range(1;$intervals|length);$intervals[.].started_unix_us >= $intervals[.-1].completed_unix_us);"overlapping primary intervals")
| {is_complete:true,source:$manifest.revision,executable:$manifest.executable_sha256,
    qkv_peer:$manifest.qkv_peer,pairs:30,primary_intervals:($intervals|length),gpu_boundary_samples:60,
    validation_records:6,all_output_control_bits_equal:true,no_timed_allocation_copy_sync:true,
    tracked_device_bytes_after:0,stage_micro_gate_passed:true,combined_feasibility_gate_passed:true,
    runtime_quality_and_http_qualification_complete:false,screens:$screens,
    memory:($checks|group_by(.rows)|map({rows:.[0].rows,scratch_bytes:map(.scratch_bytes)|unique,
      tracked_device_bytes:map(.tracked_device_bytes)|unique})),
    max_cpu_abs_error:([$checks[].checks.outputs[].cpu.max_abs_diff]|max),
    max_cpu_mean_abs_error:([$checks[].checks.outputs[].cpu.mean_abs_diff]|max)}
