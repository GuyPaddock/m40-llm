# jq -n --slurpfile qkv ...qkv-packed.jsonl --slurpfile output ...output-packed.jsonl --slurpfile capture ...ffn-capture.jsonl --slurpfile previous ...byte-ffn-screen.jsonl --slurpfile profile ...projection-stage-audit.json -f THIS
def require($ok;$message): if $ok then . else error($message) end;
def median: sort | .[length/2|floor];
def close($a;$b): ($a-$b|fabs)<1e-12;
def feasibility($fraction;$gain): 1/(1-$fraction+$fraction/(1+$gain))-1;
def audit_stage($records;$prefix;$control;$candidate;$pid;$fraction):
  ($records|map(select(.kind==($prefix+"_manifest")))) as $manifests
  | ($records|map(select(.kind==($prefix+"_pair")))) as $pairs
  | ($records|map(select(.kind==($prefix+"_validation")))) as $checks
  | ($records|map(select(.kind==($prefix+"_screen")))) as $screens
  | ($records|map(select(.kind==($prefix+"_complete")))) as $complete
  | ($records|map(select(.kind==($prefix+"_warmup")))) as $warmups
  | require(($manifests|length)==1;"manifest multiplicity")
  | $manifests[0] as $manifest
  | require($manifest.revision=="8b7b20c4ae0e9840a9a67a139e66865a0e7a16a7" and $manifest.worktree==""
      and $manifest.executable_sha256=="1a82c50a6f74686fb151ac3582d2c4c4ae62337f2a8a10facdcb911c7fdcd251"
      and $manifest.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
      and $manifest.fixture_sha256=="eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687"
      and $manifest.capture_report_sha256=="d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801"
      and $manifest.execution_configuration==$capture[0].execution_configuration
      and $manifest.build_profile=="release" and $manifest.rows==[1,4]
      and $manifest.cuda_flags==["-O3","sm_52","compute_52","no fast math"]
      and $manifest.trials==5 and $manifest.iterations==10 and $manifest.comparison=="packed-loads"
      and $manifest.profile_source.sha256=="9445906c521ae61700935e0b934d8744d98eb0db247f7abeee667e2cc9e3066d"
      and $manifest.profile_source.generated_step==255 and $manifest.qkv_peer==null
      and $manifest.memory=={additional_prepared_bytes:0,original_weights:3279519744,prepared_weights:1192935424,resident_kv:0,tracked_device_bytes:4472455168}
      and ($manifest.prepared_weights.tensors|map(.key))==($previous[0].prepared_weights.tensors|map(.key));"source/configuration/preparation")
  | require(($pairs|length)==30 and ($checks|length)==6 and ($screens|length)==6
      and ($warmups|length)==6 and all($warmups[];.control.gpu_ms>=1000 and .candidate.gpu_ms>=100)
      and [$screens[]|[.layer_id,.rows]]==[[0,1],[0,4],[17,1],[17,4],[35,1],[35,4]]
      and [$checks[]|[.layer_id,.rows]]==[[0,1],[0,4],[17,1],[17,4],[35,1],[35,4]]
      and ($complete|length)==1 and $complete[0].is_complete
      and $complete[0].device_bytes_before==0 and $complete[0].device_bytes_after==0;"coverage/cleanup")
  | require(all($checks[];.checks.control_bits_equal and .checks.guards_and_immutable_read_spans
      and all(.checks.outputs[];.cpu.max_abs_diff<0.001));"numerical validation")
  | require(all(($checks+$pairs)[];
      .scratch_bytes==(if $prefix=="qkv_packed" then (if .rows==1 then 18688 else 73984 end) else (if .rows==1 then 16512 else 65664 end) end)
      and .tracked_device_bytes==4472455168+.scratch_bytes);"fixed scratch/residency")
  | require(all($pairs|group_by(.layer_id,.rows)[];map(.trial)==[0,1,2,3,4])
      and all($pairs[];. as $pair|
        [.measurements[].is_candidate]==(if .trial%2==0 then [false,true] else [true,false] end)
        and all(.measurements[];. as $arm|.enqueue_counters as $c|
          .iterations==10 and .is_warp and .has_packed_loads==.is_candidate
          and .operation==(if .is_candidate then $candidate else $control end)
          and .operation_launches==($pair.rows*10) and $c.launches==($pair.rows*10)
          and .gpu_ms>0 and close(.gpu_ms_per_request;.gpu_ms/$pair.rows)
          and .completed_unix_us>.started_unix_us
          and all(["device_allocs","device_frees","h2d_copies","h2d_bytes","d2h_copies","d2h_bytes","stream_syncs","stream_waits","cublas_calls"][];$c[.]==0)));"actual dispatch/alternating pairs/side effects")
  | require(all($screens[];. as $screen|
      [$pairs[]|select(.layer_id==$screen.layer_id and .rows==$screen.rows)|
        [.measurements[]|select(.is_candidate|not)|.gpu_ms]
        + [.measurements[]|select(.is_candidate)|.gpu_ms]] as $actual|
      .result.pairs==$actual and .result.control_median_ms==($actual|map(.[0])|median)
      and .result.candidate_median_ms==($actual|map(.[1])|median)
      and close(.result.throughput_gain;.result.control_median_ms/.result.candidate_median_ms-1)
      and .result.bootstrap_seed==0 and .result.bootstrap_resamples==10000
      and (.result.bootstrap_gain_ci95|length)==2
      and all(.result.bootstrap_gain_ci95[];type=="number")
      and .result.bootstrap_gain_ci95[0]>0 and .result.bootstrap_gain_ci95[0]<=.result.bootstrap_gain_ci95[1]
      and .result.passes_micro_screen and .result.throughput_gain>=0.05
      and .profile_fraction==$fraction and close(.optimistic_profile_weighted_e2e_gain;feasibility($fraction;.result.throughput_gain))
      and .optimistic_combined_e2e_gain==null
      and .runtime_integration_allowed==(.rows==1 and .optimistic_profile_weighted_e2e_gain>=0.03));"independent stage screens")
  | [$pairs[]|.gpu_before,.gpu_after] as $gpu
  | require(all($gpu[];(.device|contains(", P0,")) and (.device|contains("1113 MHz, 3004 MHz"))
      and (.processes|test("^"+$pid+", target/release/examples/q8_execution_bench, [0-9]+ MiB$")));"GPU clocks/isolation")
  | {stage:$prefix,source:$manifest.revision,executable:$manifest.executable_sha256,
      pairs:30,validation_cases:6,gpu_boundary_samples:60,profile_fraction:$fraction,
      screens:($screens|map({layer_id,rows,result,optimistic_profile_weighted_e2e_gain,runtime_integration_allowed})),
      max_cpu_error:([$checks[].checks.outputs[].cpu.max_abs_diff]|max),
      intervals:[$pairs[].measurements[]|{started_unix_us,completed_unix_us}]};

($profile[0].by_step[]|select(.generated_step==255)|.stages) as $shares
| ($shares[]|select(.stage=="qkv_project")|.median_share) as $fq
| ($shares[]|select(.stage=="out_project")|.median_share) as $fo
| audit_stage($qkv;"qkv_packed";"qkv_f32xq8_0_gguf_warp";"qkv_f32xq8_0_gguf_packed_warp";"1676964";$fq) as $q
| audit_stage($output;"output_packed";"gemm_f32xq8_0_gguf_coltile4_warp";"gemm_f32xq8_0_gguf_coltile4_packed_warp";"1678416";$fo) as $o
| ($q.intervals+$o.intervals|sort_by(.started_unix_us)) as $intervals
| require(all(range(1;$intervals|length);$intervals[.].started_unix_us >= $intervals[.-1].completed_unix_us);"overlapping stage/process intervals")
| [$q.screens[] as $qs|$o.screens[]|select(.layer_id==$qs.layer_id and .rows==$qs.rows)|
    (1/(1-$fq-$fo+$fq/(1+$qs.result.throughput_gain)+$fo/(1+.result.throughput_gain))-1) as $combined|
    {layer_id,rows,qkv_gain:$qs.result.throughput_gain,output_gain:.result.throughput_gain,
      optimistic_combined_e2e_gain:$combined,
      permits_runtime_quality:(.rows==1 and $qs.result.passes_micro_screen and .result.passes_micro_screen and $combined>=0.03)}] as $combined
| require(all($combined[]|select(.rows==1);.permits_runtime_quality);"combined feasibility")
| {is_complete:true,primary_intervals:($intervals|length),stages:[$q,$o]|map(del(.intervals)),
    control_bits_equal:true,prepared_cache_identities_unchanged:true,tracked_device_bytes_after:0,
    combined:$combined,runtime_quality_required:true,measured_http_gain:null,defaults_changed:false}
