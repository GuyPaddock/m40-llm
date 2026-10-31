# jq -n --slurpfile qkv ...qkv.jsonl --slurpfile output ...output.jsonl
# --slurpfile capture ...ffn-capture.jsonl --slurpfile previous ...initial-qkv.jsonl
# --slurpfile incomplete ...incomplete-output.jsonl --slurpfile profile ...packed-stage-audit.json -f THIS
def require($ok;$message): if $ok then . else error($message) end;
def median: sort|.[length/2|floor];
def close($a;$b): ($a-$b|fabs)<1e-12;
def feasibility($fraction;$gain): 1/(1-$fraction+$fraction/(1+$gain))-1;
def key: [.layer_id,.rows];
def audit_stage($records;$prefix;$control;$candidate;$pid;$fraction;$prepared;$columns;$shared):
  ($records|map(select(.kind==($prefix+"_manifest")))) as $manifests
  | ($records|map(select(.kind==($prefix+"_pair")))) as $pairs
  | ($records|map(select(.kind==($prefix+"_validation")))) as $checks
  | ($records|map(select(.kind==($prefix+"_screen")))) as $screens
  | ($records|map(select(.kind==($prefix+"_complete")))) as $complete
  | ($records|map(select(.kind==($prefix+"_warmup")))) as $warmups
  | [[0,1],[0,4],[17,1],[17,4],[35,1],[35,4]] as $shapes
  | require(($manifests|length)==1;"manifest multiplicity")
  | $manifests[0] as $manifest
  | require($manifest.revision=="5719ffcbfbcf57c91acad67c691120ffc95ecc01" and $manifest.worktree==""
      and $manifest.executable_sha256=="517829832eea508f4c71f10e442a817306ef3886e90b8657c7c36f48970f43eb"
      and $manifest.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
      and $manifest.fixture_sha256=="eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687"
      and $manifest.capture_report_sha256=="d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801"
      and $manifest.execution_configuration==$capture[0].execution_configuration
      and $manifest.build_profile=="release" and $manifest.rows==[1,4]
      and $manifest.cuda_flags==["-O3","sm_52","compute_52","no fast math"]
      and $manifest.trials==5 and $manifest.iterations==10 and $manifest.comparison=="split-planes"
      and $manifest.profile_source.sha256=="3ff95bf79269368145bf3044bb4d351ec826f11444b87cadafbfe7868e29177e"
      and $manifest.profile_source.generated_step==255 and $manifest.qkv_peer==null
      and $manifest.memory=={additional_prepared_bytes:0,original_weights:3279519744,prepared_weights:1192935424,resident_kv:0,tracked_device_bytes:4472455168}
      and ($manifest.prepared_weights.tensors|length)==37
      and ($manifest.prepared_weights.tensors|map(.key))==($previous[0].prepared_weights.tensors|map(.key));"source/configuration/prepared control")
  | require(($records|length)==50 and ($pairs|length)==30 and ($checks|length)==6 and ($screens|length)==6
      and ($warmups|length)==6 and ([$warmups[]|key]|sort)==$shapes
      and ([$screens[]|key]|sort)==$shapes and ([$checks[]|key]|sort)==$shapes
      and all($warmups[];.control.minimum_gpu_ms==1000 and .candidate.minimum_gpu_ms==100
        and all([.control,.candidate][];.gpu_ms>=.minimum_gpu_ms
          and .iterations_per_batch==256 and .maximum_batches==256 and .batches>0 and .batches<=256))
      and ($complete|length)==1 and $complete[0].is_complete
      and $complete[0].device_bytes_before==0 and $complete[0].device_bytes_after==0;"coverage/warmup/cleanup")
  | require(all($checks[];. as $check|
      .checks.control_bits_equal and .checks.guards_and_immutable_read_spans
      and (.sources|length)==(if $prefix=="qkv_split" then 3 else 1 end)
      and [.checks.outputs[].source]==.sources
      and all(.checks.outputs[];.cpu.max_abs_diff>=0 and .cpu.max_abs_diff<0.001
        and .cpu.compared_values==$check.rows*.source.shape[1])
      and .preparation.layout=="SeparateScalesValues"
      and .preparation.additional_prepared_bytes==$prepared
      and .preparation.retained_host_prepared_bytes==$prepared
      and .preparation.estimated_all_layers_extra_bytes==$prepared*36
      and .preparation.tracked_bytes_before==4472455168+.scratch_bytes
      and .preparation.tracked_bytes_after==.tracked_device_bytes
      and .preparation.tracked_bytes_after-.preparation.tracked_bytes_before==$prepared
      and ([.preparation.planes[].resident_bytes]|add)==$prepared
      and (.preparation.keys|map(.source.tensor_name))==(.sources|map(.name))
      and (.preparation.keys|map([.source.shape.input_dim,.source.shape.output_dim]))==(.sources|map(.shape))
      and (.preparation.keys|map(.source.byte_offset))==(.sources|map(.byte_offset))
      and all(.preparation.keys[];.layout=="SeparateScalesValues" and .source.dtype=="Q8_0"
        and .source.model_sha256==$manifest.prepared_weights.tensors[0].key.source.model_sha256));"numerical validation/typed candidate preparation")
  | require(all(($checks+$pairs)[];
      .scratch_bytes==(if $prefix=="qkv_split" then (if .rows==1 then 18688 else 73984 end) else (if .rows==1 then 16512 else 65664 end) end)
      and .tracked_device_bytes==4472455168+.scratch_bytes+$prepared);"fixed scratch/residency")
  | require(([$pairs[]|key]|unique|sort)==$shapes
      and all($pairs|group_by(key)[];map(.trial)==[0,1,2,3,4])
      and all($pairs[];. as $pair|
        .additional_prepared_bytes==$prepared
        and [.measurements[].is_candidate]==(if .trial%2==0 then [false,true] else [true,false] end)
        and all(.measurements[];. as $arm|.enqueue_counters as $c|
          .iterations==10 and .is_warp and .has_packed_loads
          and .operation==(if .is_candidate then $candidate else $control end)
          and .operation_launches==($pair.rows*10) and $c.launches==($pair.rows*10)
          and .gpu_ms>0 and close(.gpu_ms_per_request;.gpu_ms/$pair.rows)
          and .completed_unix_us>.started_unix_us
          and all(["device_allocs","device_frees","h2d_copies","h2d_bytes","d2h_copies","d2h_bytes","stream_syncs","stream_waits","cublas_calls"][];$c[.]==0)
          and .split_projection_dispatch==(if .is_candidate then {operation:$candidate,
            launches:($pair.rows*10),threads:256,columns:$columns,shared_bytes:$shared,
            payload_load_bytes:8,additional_prepared_bytes:$prepared} else null end)));
      "actual dispatch/alternating pairs/side effects")
  | require(all($screens[];. as $screen|
      [$pairs[]|select(key==($screen|key))|
        [.measurements[]|select(.is_candidate|not)|.gpu_ms]
        + [.measurements[]|select(.is_candidate)|.gpu_ms]] as $actual|
      .result.pairs==$actual and .result.control_median_ms==($actual|map(.[0])|median)
      and .result.candidate_median_ms==($actual|map(.[1])|median)
      and close(.result.throughput_gain;.result.control_median_ms/.result.candidate_median_ms-1)
      and .result.bootstrap_seed==0 and .result.bootstrap_resamples==10000
      and (.result.bootstrap_gain_ci95|length)==2 and all(.result.bootstrap_gain_ci95[];type=="number")
      and .result.bootstrap_gain_ci95[0]<=.result.bootstrap_gain_ci95[1]
      and .result.bootstrap_gain_ci95[0]>=((($actual|map(.[0])|min)/($actual|map(.[1])|max))-1)
      and .result.bootstrap_gain_ci95[1]<=((($actual|map(.[0])|max)/($actual|map(.[1])|min))-1)
      and .result.passes_micro_screen==(.result.throughput_gain>=0.05 and .result.bootstrap_gain_ci95[0]>0)
      and .profile_fraction==$fraction and close(.optimistic_profile_weighted_e2e_gain;feasibility($fraction;.result.throughput_gain))
      and .optimistic_combined_e2e_gain==null and (.runtime_integration_allowed|not));"raw medians/independent stage gates")
  | [$pairs[]|.gpu_before,.gpu_after] as $gpu
  | require(all($gpu[];(.device|contains("GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, Tesla M40 24GB"))
      and (.device|contains(", P0,")) and (.device|contains("1113 MHz, 3004 MHz"))
      and (.processes|test("^"+$pid+", target/release/examples/q8_execution_bench, [0-9]+ MiB$")));"GPU clocks/isolation")
  | {stage:$prefix,source:$manifest.revision,executable:$manifest.executable_sha256,
      pairs:30,validation_cases:6,gpu_boundary_samples:60,profile_fraction:$fraction,
      additional_prepared_bytes:$prepared,
      screens:($screens|map({layer_id,rows,result,optimistic_profile_weighted_e2e_gain})),
      max_cpu_error:([$checks[].checks.outputs[].cpu.max_abs_diff]|max),
      intervals:[$pairs[].measurements[]|{started_unix_us,completed_unix_us}]};

($profile[0].by_step[]|select(.generated_step==255)|.stages) as $shares
| ($shares[]|select(.stage=="qkv_project")|.median_share) as $fq
| ($shares[]|select(.stage=="out_project")|.median_share) as $fo
| audit_stage($qkv;"qkv_split";"qkv_f32xq8_0_gguf_packed_warp";"qkv_f32xq8_separate_packed_warp";"2743769";$fq;5570560;1;1024) as $q
| audit_stage($output;"output_split";"gemm_f32xq8_0_gguf_coltile4_packed_warp";"gemm_f32xq8_separate_coltile4_packed_warp";"2745566";$fo;4456448;4;4096) as $o
| ($q.intervals+$o.intervals|sort_by(.started_unix_us)) as $intervals
| require(($intervals|length)==120 and all(range(1;$intervals|length);
    $intervals[.].started_unix_us >= $intervals[.-1].completed_unix_us);"overlapping primary intervals")
| require(($incomplete|length)==2
    and [$incomplete[].kind]==["output_split_manifest","output_split_validation"]
    and $incomplete[0].revision=="34a242770f0851f206b36b44f8a4831b114275e7"
    and $incomplete[0].executable_sha256=="fd27330265709ae50bcb2c2ef58d5487f5fd8feb5b96c823371f7e77bcce8f79"
    and $incomplete[1].checks.control_bits_equal;"incomplete output is separate, not primary evidence")
| [$q.screens[] as $qs|$o.screens[]|select(key==($qs|key))|
    ($qs.result.passes_micro_screen and .result.passes_micro_screen) as $eligible|
    {layer_id,rows,independent_stages_pass:$eligible,
      optimistic_combined_e2e_gain:(if $eligible then
        1/(1-$fq-$fo+$fq/(1+$qs.result.throughput_gain)+$fo/(1+.result.throughput_gain))-1 else null end)}
    | .permits_runtime_quality=(.rows==1 and .independent_stages_pass and .optimistic_combined_e2e_gain>=0.03)] as $combined
| {passes_audit:true,primary_intervals:120,stages:[$q,$o]|map(del(.intervals)),
    control_bits_equal:true,qualified_prepared_cache_unchanged:true,tracked_device_bytes_after:0,
    scope:"fresh common-source screens only; initial QKV not pooled; incomplete output has no primary pairs",
    bootstrap_scope:"seeded benchmark intervals recorded; raw medians/gains and possible resample bounds independently checked",
    combined:$combined,permits_runtime_quality:all($combined[]|select(.rows==1);.permits_runtime_quality),
    measured_http_gain:null,defaults_changed:false}
