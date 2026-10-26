# jq -s --slurpfile capture ...ffn-capture.jsonl -f THIS SCREEN
def require($ok;$why): if $ok then . else error($why) end;
def median: sort|.[length/2|floor];
def key: [.family,.layer_id,.rows];
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity")|.[0];
def resident: if .family=="down" then (if .rows==1 then 4544516544 else 4545118656 end)
  else (if .rows==1 then 4803683968 else 4805531776 end) end;
def summary_matches($s;$raw):
  $s.pairs==$raw and $s.native_median_ms==($raw|map(.[0])|median) and $s.candidate_median_ms==($raw|map(.[1])|median)
  and $s.throughput_gain==($s.native_median_ms/$s.candidate_median_ms-1)
  and $s.bootstrap_resamples==10000 and $s.bootstrap_seed==0 and ($s.bootstrap_gain_ci95|length)==2
  and $s.bootstrap_gain_ci95[0]<=$s.bootstrap_gain_ci95[1]
  and $s.bootstrap_gain_ci95[0]>=((($raw|map(.[0])|min)/($raw|map(.[1])|max))-1)
  and $s.bootstrap_gain_ci95[1]<=((($raw|map(.[0])|max)/($raw|map(.[1])|min))-1)
  and $s.passes_micro_screen==($s.throughput_gain>=0.05 and $s.bootstrap_gain_ci95[0]>0);
. as $records
| one("prepared_projection_manifest") as $m
| $records|one("prepared_projection_complete") as $done
| $records|map(select(.kind=="prepared_projection_pair")) as $pairs
| $records|map(select(.kind=="prepared_projection_validation")) as $checks
| $records|map(select(.kind=="prepared_projection_warmup")) as $warmups
| $records|map(select(.kind=="prepared_projection_screen")) as $screens
| [["down",0,1],["down",0,4],["down",17,1],["down",17,4],["down",35,1],["down",35,4],["vocabulary",null,1],["vocabulary",null,4]] as $shapes
| require($m.revision=="966ead2439b95d56cd59f8a20ff1ec1ba5151b98" and $m.worktree=="" and $m.build_profile=="release"
    and $m.executable_sha256=="ff037b7b54937e4d91242b12cd265d300c192e443c006667e70fb1295d91a926"
    and $m.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $m.fixture_sha256=="eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687"
    and $m.capture_report_sha256=="d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801"
    and $m.execution_configuration==$capture[0].execution_configuration
    and ($m.prepared_weights|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))
      ==($capture[0].prepared_weights|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))
    and ($m.prepared_weights.tensors|length)==37 and $m.original_weight_bytes==3279519744 and $m.tracked_device_bytes==4472455168
    and $m.rows==[1,4] and $m.trials==5 and $m.iterations==10 and $m.candidate_kernel=="WarpPacket"
    and ($m.is_weight_read_diagnostic|not) and ($m.runtime_defaults_changed|not) and ($m.is_e2e_claim|not)
    and $m.cuda_flags==["-O3","sm_52","compute_52","no fast math"];"clean frozen control/source")
| require(($records|length)==68 and ($pairs|length)==40 and ($checks|length)==8 and ($screens|length)==8
    and ([$checks[]|key]|sort)==$shapes and ([$screens[]|key]|sort)==$shapes
    and $done.is_complete and $done.cases==8 and $done.device_bytes_before==0 and $done.device_bytes_after==0;"coverage/cleanup")
| require(all($checks[];. as $check|.packet_preparation as $p|
    $p.layout=="WarpTaskPacket4x8V1" and $p.key.format_version==1 and $p.source_roundtrip_bits_equal
    and $p.source_bytes==$p.additional_prepared_bytes and $p.source_bytes==$p.retained_host_prepared_bytes
    and $p.source_bytes==$p.inverse_roundtrip_temporary_host_bytes and $p.scale_bytes+$p.payload_bytes==$p.source_bytes
    and $p.tracked_bytes_after==$p.tracked_bytes_before+$p.source_bytes and $p.tracked_bytes_after==($check|resident)
    and ([$m.prepared_weights.tensors[]|select(.key.source==$p.key.source)]|length)==1
    and ($p.original_source_sha256|test("^[a-f0-9]{64}$"))
    and (if .family=="down" then $p.source_bytes==23953408 and $p.extra_budget_bytes==86310912
      and $p.key.source.tensor_name==.source.name and $p.original_source_sha256==.source.source_sha256
      and .gate_up_preparation.additional_prepared_bytes==47906816 and .scratch_bytes==(if .rows==1 then 201152 else 803264 end)
      and .control_hidden_bits_equal and .control_complete_output_bits_equal and .control_down_bits_equal
      and all([.control_checks,.candidate_checks][];.variant=="split-f32x4-128" and .quantization==null
        and .guards_and_immutable_inputs and .intermediate_capture_matches_normal_kernel
        and all([.gate_vs_cpu,.up_vs_cpu,.swiglu_vs_cpu,.complete_down_vs_cpu_on_actual_swiglu,.isolated_down_vs_cpu_on_frozen_swiglu][];
          .max_abs_diff>=0 and .max_abs_diff<0.001 and .mean_abs_diff>=0 and .mean_abs_diff<=.max_abs_diff))
    else $p.source_bytes==330612736 and $p.extra_budget_bytes==536870912 and .control_bits_equal
      and .cpu_sample_max_abs_diff>=0 and .cpu_sample_max_abs_diff<0.001 and .scratch_bytes==(if .rows==1 then 616064 else 2463872 end) end));"lossless source/case budget/control bits/CPU bounds")
| require(($warmups|length)==10 and all($warmups[];
    if .family=="down" then .control.minimum_gpu_ms==1000 and .control.elapsed_gpu_ms>=1000 and .candidate.minimum_gpu_ms==100
      and .candidate.elapsed_gpu_ms>=100 and .control.maximum_batches==256 and .candidate.maximum_batches==256
    else .elapsed_gpu_ms>=(if .is_candidate then 100 else 1000 end) end);"bounded warmup")
| require(([$pairs[]|key]|unique|sort)==$shapes and all($pairs|group_by(key)[];map(.trial)==[0,1,2,3,4])
    and all($pairs[];. as $pair|.resident_bytes==resident
      and (.measurements|map(.is_candidate))==(if .trial%2==0 then [false,true] else [true,false] end)
      and all(.measurements[];. as $arm|
        (if $pair.family=="down" then [.timings[].stage]==(if $pair.trial%2==0 then ["isolated_down","gate_up_swiglu","complete"] else ["complete","gate_up_swiglu","isolated_down"] end) else true end)
        and all((if $pair.family=="down" then .timings[] else . end);.enqueue_counters as $c|
          .iterations==10 and .gpu_ms>0 and .completed_unix_us>.started_unix_us
          and all(["device_allocs","device_frees","h2d_copies","h2d_bytes","d2h_copies","d2h_bytes","cublas_calls","stream_syncs"][];$c[.]==0)
          and $c.launches==(if .stage=="complete" then 20 else 10 end)
          and $c.stream_waits==(if $pair.family=="down" and $pair.rows==4 then (if .stage=="complete" then 20 else 10 end) else 0 end)
          and .projection_operation==(if $arm.is_candidate then "gemm_f32xq8_warp_packet" else "gemm_f32xq8_separate_f32" end)
          and .projection_launches==(if .stage=="gate_up_swiglu" then 0 else 10 end)
          and (if $pair.family=="down" then .gate_up_operation=="ffn_q8_split_f32x4_128" and .gate_up_launches==(if .stage=="isolated_down" then 0 else 10 end) else true end))));"actual complete work/dispatch/no enqueue side effects")
| require(all($screens[];. as $screen|
    ([$pairs[]|select(key==($screen|key))]) as $shape|
    (if .family=="down" then all(["complete","isolated_down"][];. as $stage|
      [$shape[]|[.measurements[]|select(.is_candidate==false)|.timings[]|select(.stage==$stage)|.gpu_ms]
        +[.measurements[]|select(.is_candidate==true)|.timings[]|select(.stage==$stage)|.gpu_ms]] as $raw|
      summary_matches((if $stage=="complete" then $screen.complete_ffn else $screen.isolated_projection end);$raw))
    else [$shape[]|[.measurements[]|select(.is_candidate==false)|.gpu_ms]+[.measurements[]|select(.is_candidate==true)|.gpu_ms]] as $raw|
      summary_matches($screen.result;$raw) end)
    and (.runtime_integration_allowed|not));"raw pairs/medians/paired bounds/unchanged gates")
| [$pairs[]|.gpu_before,.gpu_after] as $gpu
| require(all($gpu[];(.device|contains("GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, Tesla M40 24GB")) and (.device|contains(", P0,"))
    and (.device|contains("1113 MHz, 3004 MHz")) and (.processes|test("^3074139, target/release/examples/q8_execution_bench, [0-9]+ MiB$")));"GPU isolation/clocks")
| ([$pairs[]|if .family=="down" then .measurements[].timings[] else .measurements[] end]|sort_by(.started_unix_us)) as $intervals
| require(($intervals|length)==200 and all(range(1;200);$intervals[.].started_unix_us>=$intervals[.-1].completed_unix_us);"serial primary intervals")
| ($screens|map(select(.family=="vocabulary" and .rows==1))|.[0].result) as $head
| 0.07980905884300443 as $head_fraction
| {passes_audit:true,source:$m.revision,executable_sha256:$m.executable_sha256,pairs:40,serial_primary_intervals:200,gpu_boundaries:80,
    tracked_device_bytes_after:0,screens:$screens,
    down_complete_one_row_qualifies:all($screens[]|select(.family=="down" and .rows==1);.complete_ffn.passes_micro_screen),
    vocabulary_projection_qualifies:$head.passes_micro_screen,qualified_middle_vocabulary_fraction:$head_fraction,
    vocabulary_only_optimistic_e2e_gain:(1/(1-$head_fraction*$head.throughput_gain/(1+$head.throughput_gain))-1),
    runtime_integration_allowed:false,interpretation:"complete FFN gates exclude failed down; vocabulary-only scope is hypothetical, not HTTP performance"}
