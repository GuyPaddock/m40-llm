# jq -n --slurpfile initial INITIAL --slurpfile fresh FRESH
# --slurpfile capture CAPTURE --slurpfile profile PROFILE -f THIS
def require($ok;$message): if $ok then . else error($message) end;
def close($a;$b): ($a-$b|fabs)<1e-12;
def median: sort | length as $n | if $n%2==0 then (.[($n/2)-1]+.[$n/2])/2 else .[$n/2|floor] end;
def key: [.case.stage,(.case.layer_id//.case.distribution),.case.rows];
def audit($records;$trials;$pid):
  ($records|map(select(.kind=="decode_reductions_manifest"))) as $manifests
  | require(($manifests|length)==1;"manifest multiplicity")
  | $manifests[0] as $m
  | require($m.revision=="61a48d9147adfaca4252f8d721de1bae7236467a" and $m.worktree==""
      and $m.executable_sha256=="1e655a9e5fdc2ca3e60092d8ad4602d647c187290717e0c7e524d9b4f9f1eb34"
      and $m.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
      and $m.fixture_sha256=="eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687"
      and $m.capture_report_sha256=="d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801"
      and $m.execution_configuration==$capture[0].execution_configuration
      and ($m.prepared_weights|del(.model_hash_elapsed_ms,.preparation_elapsed_ms))
        ==($capture[0].prepared_weights|del(.model_hash_elapsed_ms,.preparation_elapsed_ms))
      and ($m.prepared_weights.tensors|length)==37
      and $m.memory=={original_weights:3279519744,prepared_weights:1192935424,resident_kv:0,tracked_device_bytes:4472455168}
      and $m.build_profile=="release" and $m.cuda_flags==["-O3","sm_52","compute_52","no fast math"]
      and $m.trials==$trials and $m.iterations==10 and $m.rows==[1,4]
      and $m.profile_source.sha256=="3ff95bf79269368145bf3044bb4d351ec826f11444b87cadafbfe7868e29177e"
      and $m.profile_source.generated_step==255;"frozen release/model/input/control identity")
  | ($records|map(select(.kind=="decode_reductions_validation"))) as $checks
  | ($records|map(select(.kind=="decode_reductions_warmup"))) as $warmups
  | ($records|map(select(.kind=="decode_reductions_pair"))) as $pairs
  | ($records|map(select(.kind=="decode_reductions_screen"))) as $screens
  | ($records|map(select(.kind=="decode_reductions_scope"))) as $scopes
  | ($records|map(select(.kind=="decode_reductions_complete"))) as $complete
  | ([(["attn_norm","ffn_norm"][] as $stage | [0,17,35][] as $layer | [1,4][] | [$stage,$layer,.]),
      (["random","ascending","late-tie"][] as $distribution | [1,4][] | ["argmax",$distribution,.])]|sort) as $keys
  | require(($records|length)==18*($trials+3)+3 and ($checks|length)==18
      and ($warmups|length)==18 and ($screens|length)==18 and ($pairs|length)==18*$trials
      and ([$checks[]|key]|sort)==$keys and ([$screens[]|key]|sort)==$keys and ([$warmups[]|key]|sort)==$keys
      and all($pairs|group_by(key)[];map(.trial)==[range($trials)])
      and ($scopes|length)==1 and ($complete|length)==1 and $complete[0].is_complete
      and $complete[0].device_bytes_before==0 and $complete[0].device_bytes_after==0;"coverage/cleanup")
  | require(all($warmups[];.control.minimum_gpu_ms==1000 and .candidate.minimum_gpu_ms==100
      and all([.control,.candidate][];.gpu_ms>=.minimum_gpu_ms and .iterations_per_batch==4096
        and .maximum_batches==256 and .batches>0 and .batches<=256));"bounded untimed warmup")
  | require(all($checks[];. as $check|.geometry_and_input as $g|
      .checks.control_bits_equal and .checks.guards_and_immutable_read_spans
      and $g.rows==.case.rows and $g.threads==256
      and (.tracked_device_bytes==4472455168+$g.scratch_capacity_bytes)
      and (if .case.stage=="argmax" then
        $g.dim==151936 and $g.candidate_partial_payload_bytes==384 and $g.shared_bytes_per_block==2048
        and $g.candidate_grid==[48,1] and $g.control_grid==1
        and $g.scratch_capacity_bytes==(if .case.rows==1 then 608324 else 2431568 end)
        and .checks.cpu.actual_indices==.checks.cpu.expected_indices
        and (.checks.cpu.actual_indices|length)==.case.rows
        and all(.checks.cpu.actual_indices[];.>=0 and .<151936)
        and $g.input_sha256==($m.logit_inputs[]|select(.distribution==$check.case.distribution)|if $check.case.rows==1 then .rows1_sha256 else .rows4_sha256 end)
      else
        $g.dim==2048 and $g.candidate_partial_payload_bytes==0 and $g.shared_bytes_per_block==1024
        and $g.candidate_grid==[.case.rows] and $g.control_grid==.case.rows
        and $g.scratch_capacity_bytes==(if .case.rows==1 then 16512 else 65664 end)
        and .case.weight.dtype=="F32" and .case.weight.shape==[2048]
        and .case.weight.name==("blk."+(.case.layer_id|tostring)+"."+.case.stage+".weight")
        and (.case.weight.sha256|test("^[a-f0-9]{64}$"))
        and .case.eps>0 and .case.eps<0.00001
        and (.case.row_positions|length)==.case.rows
        and .checks.cpu.compared_values==.case.rows*2048
        and .checks.cpu.max_abs_diff<0.001 and .checks.cpu.mean_abs_diff>=0
      end));"numerics/guards/identity/shape/scratch")
  | require(all($pairs[];. as $pair|
      ($checks[]|select(key==($pair|key))) as $check|
      .case==$check.case and .tracked_device_bytes==$check.tracked_device_bytes
      and [.measurements[].is_candidate]==(if .trial%2==0 then [false,true] else [true,false] end)
      and all(.measurements[];. as $arm|.enqueue_counters as $c|
        .iterations==10 and .gpu_ms>0 and close(.gpu_ms_per_request;.gpu_ms/$pair.case.rows)
        and .host_enqueue_and_completion_ms>0 and .completed_unix_us>.started_unix_us
        and all(["device_allocs","device_frees","h2d_copies","h2d_bytes","d2h_copies","d2h_bytes","stream_syncs","stream_waits","cublas_calls"][];$c[.]==0)
        and .operations==(if $pair.case.stage=="argmax" then
          if .is_candidate then [{name:"argmax_f32_partition_partials",launches:10*$pair.case.rows},{name:"argmax_f32_partition_reduce",launches:10*$pair.case.rows}]
          else [{name:"argmax_f32",launches:10*$pair.case.rows}] end
        else [{name:(if .is_candidate then "rms_norm_f32_weighted_warp" else "rms_norm_f32_weighted" end),launches:10}] end)
        and $c.launches==([.operations[].launches]|add)));"alternation/dispatch/side effects")
  | require(all($screens[];. as $screen|
      [$pairs[]|select(key==($screen|key))|[.measurements[]|select(.is_candidate|not)|.gpu_ms]+[.measurements[]|select(.is_candidate)|.gpu_ms]] as $raw|
      .result.pairs==$raw and close(.result.control_median_ms;($raw|map(.[0])|median))
      and close(.result.candidate_median_ms;($raw|map(.[1])|median))
      and close(.result.throughput_gain;.result.control_median_ms/.result.candidate_median_ms-1)
      and .result.bootstrap_resamples==10000 and .result.bootstrap_seed==0
      and (.result.bootstrap_gain_ci95|length)==2 and .result.bootstrap_gain_ci95[0]<=.result.bootstrap_gain_ci95[1]
      and .result.bootstrap_gain_ci95[0]>=((($raw|map(.[0])|min)/($raw|map(.[1])|max))-1)
      and .result.bootstrap_gain_ci95[1]<=((($raw|map(.[0])|max)/($raw|map(.[1])|min))-1)
      and .result.passes_micro_screen==(.result.throughput_gain>=0.05 and .result.bootstrap_gain_ci95[0]>0)
      and (.runtime_integration_allowed|not));"independent raw medians/gates/bootstrap bounds")
  | ($profile[0].by_step[]|select(.generated_step==255)|.stages) as $profile_stages
  | $scopes[0].result as $scope
  | require(($scope.stages|map(.stage))==["attn_norm","ffn_norm","argmax"]
      and all($scope.stages[];. as $stage|
        ($screens|map(select(.case.stage==$stage.stage and .case.rows==1))) as $one|
        ($profile_stages[]|select(.stage==(if $stage.stage=="argmax" then "greedy_argmax" else $stage.stage end))|.median_share) as $fraction|
        .passes_all_stage_cases==all($one[];.result.passes_micro_screen)
        and close(.minimum_throughput_gain;($one|map(.result.throughput_gain)|min))
        and close(.profile_fraction;$fraction)
        and close(.eligible_saved_fraction;(if .passes_all_stage_cases then $fraction*(1-1/(1+.minimum_throughput_gain)) else 0 end)))
      and close($scope.optimistic_profile_weighted_gain;1/(1-([$scope.stages[].eligible_saved_fraction]|add))-1)
      and $scope.passes_profile_scope_gate==($scope.optimistic_profile_weighted_gain>=0.03)
      and ($scope.runtime_integration_allowed|not);"passing-stage-only profile scope")
  | [$pairs[]|.gpu_before,.gpu_after] as $gpu
  | require(all($gpu[];(.device|contains("GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, Tesla M40 24GB"))
      and (.device|contains(", P0,")) and (.device|contains("1113 MHz, 3004 MHz"))
      and (.processes|test("^"+$pid+", target/release/examples/q8_execution_bench, [0-9]+ MiB$")));"GPU clocks/isolation")
  | {trials:$trials,pairs:($pairs|length),primary_intervals:2*($pairs|length),gpu_boundary_samples:($gpu|length),
      screens:[$screens[]|{case,result}],profile_scope:$scope,
      max_norm_cpu_error:([$checks[]|select(.case.stage!="argmax")|.checks.cpu.max_abs_diff]|max),
      input_identities:[$checks[]|{case,geometry_and_input}],
      intervals:[$pairs[].measurements[]|{started_unix_us,completed_unix_us}]};

audit($initial;5;"2846382") as $a
| audit($fresh;10;"2850083") as $b
| require($a.input_identities==$b.input_identities;"initial/fresh input identity changed")
| ($a.intervals+$b.intervals|sort_by(.started_unix_us)) as $intervals
| require(($intervals|length)==540 and all(range(1;540);$intervals[.].started_unix_us>=$intervals[.-1].completed_unix_us);"overlapping primary intervals")
| {passes_audit:true,initial:($a|del(.intervals,.input_identities)),fresh:($b|del(.intervals,.input_identities)),
    primary_intervals:540,tracked_device_bytes_after:0,control_bits_equal:true,qualified_prepared_cache_unchanged:true,
    bootstrap_scope:"raw medians/gates and resampling bounds independently checked; benchmark records seeded 10000-resample intervals",
    runs_pooled:false,measured_http_gain:null,runtime_defaults_changed:false}
