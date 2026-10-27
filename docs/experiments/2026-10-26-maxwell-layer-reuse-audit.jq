# Validate same-work pairs, temporal-order statistics and frozen provenance.
def require($ok;$why): if $ok then . else error($why) end;
def median: sort | .[length/2|floor];
def one($kind): map(select(.kind==$kind)) | require(length==1;"record multiplicity") | .[0];
. as $records | one("ffn_reuse_manifest") as $m
| $records | one("ffn_reuse_complete") as $done
| $records | map(select(.kind=="ffn_reuse_validation")) as $checks
| $records | map(select(.kind=="ffn_reuse_pair")) as $pairs
| $records | map(select(.kind=="ffn_reuse_warmup")) as $warmups
| $records | map(select(.kind=="ffn_reuse_case_complete")) as $cases
| [[0,17],[17,35],[35,0]] as $ids
| require(($records|length)==26 and ($pairs|length)==15
    and all([$checks,$warmups,$cases][];length==3 and map(.layer_ids)==$ids)
    and $done.is_complete and $done.cases==3 and $done.device_bytes_before==0 and $done.device_bytes_after==0;"coverage/cleanup")
| require($m.revision=="75ffdf78d37a3ff1fd691b1ab611ac4633818509"
    and $m.executable_sha256=="db70d00408f9444b753b8b1a81a1692d22d5178c09823e7aae812ce67066c554"
    and $m.worktree=="" and $m.build_profile=="release"
    and $m.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $m.fixture_sha256=="eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687"
    and $m.capture_report_sha256=="d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801"
    and $m.execution_configuration==$capture[0].execution_configuration
    and ($m.prepared_weights|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))
      ==($capture[0].prepared_weights|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))
    and $m.memory=={original_weights:3279519744,prepared_weights:1192935424,resident_kv:0,tracked_device_bytes:4472455168}
    and $m.rows==[1] and $m.trials==5 and $m.iterations==10
    and $m.layer_reuse_contract=={pairs:$ids,orders:["grouped","alternating"],one_stream:"decode",
      has_inter_layer_hidden_dependency:false,has_measured_dram_traffic:false,
      sensitivity_threshold_percent:3,runtime_integration_allowed:false};"frozen source/control")
| require(all($checks[];. as $check|
    .rows==1 and .additional_prepared_bytes==95813632 and .scratch_bytes==402304
    and .tracked_device_bytes==4568671104 and .runtime_integration_allowed==false
    and (.checks|length)==2 and (.preparation|length)==2
    and all(range(0;2);. as $index | $check.preparation[$index] as $p |
      $p.layout=="SeparateScalesValues" and $p.additional_prepared_bytes==47906816
      and ($p.keys|length)==2 and all($p.keys[];
        .layout=="SeparateScalesValues" and .source.dtype=="Q8_0"
        and .source.shape=={input_dim:2048,output_dim:11008}
        and .source.model_sha256==$m.prepared_weights.tensors[0].key.source.model_sha256)
      and ($p.keys|map(.source.tensor_name))==[
        "blk.\($check.layer_ids[$index]).ffn_gate.weight","blk.\($check.layer_ids[$index]).ffn_up.weight"])
    and all(.checks[];.variant=="split-f32x4-128" and .quantization==null
      and .guards_and_immutable_inputs and .intermediate_capture_matches_normal_kernel
      and all([.gate_vs_cpu,.up_vs_cpu,.swiglu_vs_cpu,.complete_down_vs_cpu_on_actual_swiglu,.isolated_down_vs_cpu_on_frozen_swiglu][];
        .max_abs_diff>=0 and .max_abs_diff<0.001 and .mean_abs_diff>=0 and .mean_abs_diff<=.max_abs_diff)));"actual independent layers/CPU/ownership")
| require(all($warmups[];(.warmups|map(.order))==["grouped","alternating"]
    and all(.warmups[].result;.minimum_gpu_ms==1000 and .gpu_ms>=1000
      and .iterations_per_batch==256 and .maximum_batches==256 and .batches<=256));"separate bounded warmups")
| require(all($pairs|group_by(.layer_ids)[];map(.trial)==[0,1,2,3,4])
    and ([$pairs[].layer_ids]|unique)==$ids
    and all($pairs[];. as $pair|
      .rows==1 and .tracked_device_bytes==4568671104 and .has_matching_output_bits
      and [.measurements[].stage]==["gate_up_swiglu","gate_up_swiglu","isolated_down","isolated_down","complete","complete"]
      and [.measurements[].order]==(if .trial%2==0 then ["grouped","alternating","grouped","alternating","grouped","alternating"]
        else ["alternating","grouped","alternating","grouped","alternating","grouped"] end)
      and all(.measurements[];.enqueue_counters as $c|
        .iterations_per_layer==10 and .layer_operations==20
        and .gate_up_calls==(if .stage=="isolated_down" then 0 else 20 end)
        and .down_calls==(if .stage=="gate_up_swiglu" then 0 else 20 end)
        and $c.launches==.gate_up_calls+.down_calls
        and all(["device_allocs","device_frees","h2d_copies","h2d_bytes","d2h_copies","d2h_bytes","stream_syncs","stream_waits","cublas_calls"][];$c[.]==0)
        and .gpu_interval_ms>0 and (.gpu_ms_per_layer-.gpu_interval_ms/20|fabs)<1e-12
        and .completed_unix_us>.started_unix_us and .has_measured_dram_traffic==false));"same actual work and order")
| require(all($cases[];. as $case|
    .has_intact_inputs_and_guards and .has_matching_output_bits
    and .tracked_bytes_before==4472455168 and .tracked_bytes_after==4472455168
    and .runtime_integration_allowed==false
    and (.results|map(.stage))==["gate_up_swiglu","isolated_down","complete"]
    and all(.results[];.stage as $stage | .result as $r |
      [$pairs[]|select(.layer_ids==$case.layer_ids)|
        [.measurements[]|select(.stage==$stage and .order=="grouped")|.gpu_ms_per_layer]+
        [.measurements[]|select(.stage==$stage and .order=="alternating")|.gpu_ms_per_layer]] as $raw |
      $r.pairs==$raw and $r.grouped_median_ms==($raw|map(.[0])|median)
      and $r.alternating_median_ms==($raw|map(.[1])|median)
      and ($r.alternating_latency_change-($r.alternating_median_ms/$r.grouped_median_ms-1)|fabs)<1e-12
      and $r.bootstrap_resamples==10000 and $r.bootstrap_seed==0
      and ($r.latency_change_ci95|length)==2 and $r.latency_change_ci95[0]<=$r.latency_change_ci95[1]
      and $r.latency_change_ci95[0]>=((($raw|map(.[1])|min)/($raw|map(.[0])|max))-1)-1e-12
      and $r.latency_change_ci95[1]<=((($raw|map(.[1])|max)/($raw|map(.[0])|min))-1)+1e-12
      and $r.has_material_order_sensitivity==(($r.alternating_latency_change|fabs)>=0.03
        and ($r.latency_change_ci95[0]>0 or $r.latency_change_ci95[1]<0))
      and $r.has_bounded_small_order_effect==($r.latency_change_ci95[0]> -0.03 and $r.latency_change_ci95[1]<0.03)
      and $r.passes_micro_screen==null));"raw pairs, normalized medians and order-only decisions")
| ([$pairs[].measurements[]]|sort_by(.started_unix_us)) as $intervals
| require(($intervals|length)==90 and all(range(1;90);
    $intervals[.].started_unix_us>=$intervals[.-1].completed_unix_us);"serial primary intervals")
| require(all([$pairs[]|.gpu_before,.gpu_after][];
    (.device|contains("GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, Tesla M40 24GB"))
    and (.device|contains(", P0,")) and (.device|contains("1113 MHz, 3004 MHz"))
    and .processes=="3614633, target/release/examples/q8_execution_bench, 4468 MiB");"isolated stable GPU")
| {passes_audit:true,source:$m.revision,executable_sha256:$m.executable_sha256,
    pairs:15,stage_comparisons:9,serial_primary_intervals:90,gpu_boundaries:30,
    has_all_small_order_effects:all($cases[].results[].result;.has_bounded_small_order_effect),
    has_measured_cache_traffic:false,runtime_integration_allowed:false,
    cases:$cases,tracked_device_bytes_after:0}
