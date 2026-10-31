# Verify actual paired measurements, dispatch and eligibility, not summary labels.
def require($condition;$message): if $condition then . else error($message) end;
def median: sort|.[length/2|floor];
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity: \($kind)")|.[0];
def key: [.family,.layer_id,.rows];
def check_result($pairs):
    . as $result
    | require($result.pairs==$pairs and ($pairs|length)==5;"actual contemporaneous pairs")
    | ($pairs|map(.[0])|median) as $control
    | ($pairs|map(.[1])|median) as $candidate
    | require($result.native_median_ms==$control and $result.candidate_median_ms==$candidate
        and (($result.throughput_gain-($control/$candidate-1))|fabs)<1e-12
        and $result.passes_micro_screen==($result.throughput_gain>=0.05 and $result.bootstrap_gain_ci95[0]>0)
        and $result.bootstrap_resamples==10000 and $result.bootstrap_seed==0;"paired medians and original micro gate");
. as $rows
| one("prepared_warp_manifest") as $manifest
| $rows|one("prepared_warp_complete") as $complete
| $rows|map(select(.kind=="prepared_warp_screen")) as $screens
| $rows|map(select(.kind=="prepared_warp_pair")) as $pairs
| $rows|map(select(.kind=="prepared_warp_validation")) as $validation
| $capture[0] as $capture
| $rows
| require($manifest.revision=="5218f41805467e595b5a5e4bdfa78ae4ece8cdc2" and $manifest.worktree==""
    and $manifest.executable_sha256=="6aab5b02fcf77ba3efb259589526d23fe3ced73a53480a75eb7a7fbdb4ab9f75"
    and $manifest.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $manifest.fixture_sha256=="eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687"
    and $manifest.capture_report_sha256=="d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801"
    and $manifest.execution_configuration==$capture.execution_configuration
    and $manifest.prepared_weights.resident_bytes==1192935424
    and ($manifest.prepared_weights.tensors|length)==37
    and $manifest.rows==[1,4] and $manifest.trials==5 and $manifest.iterations==10
    and $manifest.build_profile=="release" and ($manifest.runtime_defaults_changed|not);"source, executable, frozen inputs and retained cache")
| require(($screens|length)==8 and ($validation|length)==8 and ($pairs|length)==40
    and ([$screens[]|key]|sort)==[["down",0,1],["down",0,4],["down",17,1],["down",17,4],["down",35,1],["down",35,4],["vocabulary",null,1],["vocabulary",null,4]];"case coverage")
| require(all($validation[];if .family=="down" then
    .control_hidden_bits_equal and .control_complete_output_bits_equal and .control_down_bits_equal
    and .control_checks.guards_and_immutable_inputs and .candidate_checks.guards_and_immutable_inputs
    else .control_bits_equal and .cpu_sample_max_abs_diff>=0 end);"control bits and numerical validation")
| require(all($pairs[];. as $pair|
    (.measurements|length)==2 and ([.measurements[].is_warp]|sort)==[false,true]
    and [.measurements[].is_warp]==(if .trial%2==0 then [false,true] else [true,false] end)
    and all(.measurements[];. as $arm|
      (if $pair.family=="down" then .timings[] else . end) as $timing|
      $timing.iterations==10 and $timing.projection_launches==10 and $timing.gpu_ms>0
      and $timing.projection_operation==(if $arm.is_warp then "gemm_f32xq8_separate_warp" else "gemm_f32xq8_separate_f32" end)
      and $timing.enqueue_counters.device_allocs==0 and $timing.enqueue_counters.device_frees==0
      and $timing.enqueue_counters.stream_syncs==0 and $timing.enqueue_counters.cublas_calls==0
      and $timing.enqueue_counters.h2d_copies==0 and $timing.enqueue_counters.d2h_copies==0
      and $timing.enqueue_counters.launches==(if $timing.stage=="complete" then 20 else 10 end)));"timed dispatch and allocation-free enqueue")
| require(all($pairs|group_by(key)[];length==5 and (map(.trial)|sort)==[0,1,2,3,4]
    and (map(.resident_bytes)|unique|length)==1);"paired coverage and stable residency")
| ([$pairs[]|(.gpu_before,.gpu_after)]|map(.device|split(",")|map(gsub("^ +| +$";""))|[.[3],.[5],.[6]])|unique) as $clocks
| require($clocks==[["P0","1113 MHz","3004 MHz"]]
    and all($pairs[];all([.gpu_before,.gpu_after][];
      (.processes|split("\n")|length)==1 and (.processes|startswith("1568441, target/release/examples/q8_execution_bench,"))));"clock and process isolation")
| [$screens[]|. as $screen|($pairs|map(select(key==($screen|key)))|sort_by(.trial)) as $case_pairs
    | if .family=="down" then
        .complete_ffn|check_result([$case_pairs[]|[.measurements[]|select(.is_warp==false)|.timings[]|select(.stage=="complete")|.gpu_ms] + [.measurements[]|select(.is_warp==true)|.timings[]|select(.stage=="complete")|.gpu_ms]])
        | . as $complete_result
        | $screen.isolated_projection|check_result([$case_pairs[]|[.measurements[]|select(.is_warp==false)|.timings[]|select(.stage=="isolated_down")|.gpu_ms] + [.measurements[]|select(.is_warp==true)|.timings[]|select(.stage=="isolated_down")|.gpu_ms]])
        | {family:$screen.family,layer_id:$screen.layer_id,rows:$screen.rows,
            complete_ffn:$complete_result,isolated_projection:.}
      else .result|check_result([$case_pairs[]|[.measurements[]|select(.is_warp==false)|.gpu_ms] + [.measurements[]|select(.is_warp==true)|.gpu_ms]])
        | {family:$screen.family,rows:$screen.rows,result:.} end] as $results
| ($results|map(select(.family=="vocabulary" and .rows==1))[0].result) as $head
| ($profile[0].by_step[]|select(.generated_step==255)|.stages[]|select(.stage=="lm_head")|.median_share) as $head_share
| (1/(1-$head_share+$head_share/(1+$head.throughput_gain))-1) as $ceiling
| require($complete.is_complete and $complete.cases==8 and $complete.device_bytes_before==0
    and $complete.device_bytes_after==0 and ($complete.runtime_defaults_changed|not);"completion and cleanup")
| {passes_audit:true,source:$manifest.revision,executable_sha256:$manifest.executable_sha256,
    primary_pairs:40,gpu_boundaries:80,clocks:$clocks,results:$results,
    vocabulary_optimistic_e2e_gain:$ceiling,vocabulary_clears_e2e_feasibility:($ceiling>=0.03),
    all_one_row_down_cases_clear_complete_ffn_gate:all($results[]|select(.family=="down" and .rows==1);.complete_ffn.passes_micro_screen),
    runtime_integration_allowed:false,is_e2e_claim:false,changes_runtime_defaults:false}
