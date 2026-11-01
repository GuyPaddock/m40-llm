# jq -s --slurpfile stages STAGES --slurpfile events EVENTS -f THIS OFFICIAL_AUDIT
def require($condition;$message): if $condition then . else error($message) end;
def median: sort|length as $n|if $n%2==0 then (.[($n/2)-1]+.[($n/2)])/2 else .[($n/2)|floor] end;
.[0] as $official
| require($official.passes_audit and $official.mode=="ollama";"completed official qualification")
| $official.comparison.candidate.median as $base
| $stages[0].by_step|map(select(.generated_step==255))|.[0] as $middle
| def cost($names): [$middle.stages[]|select(.stage as $name|$names|index($name))|.median_ms]|add;
  [
    {stage:"gate_up_swiglu",ms:cost(["mlp_gate_up_swiglu"])},
    {stage:"down",ms:cost(["mlp_down"])},
    {stage:"complete_ffn",ms:cost(["mlp_gate_up_swiglu","mlp_down"])},
    {stage:"qkv_and_output",ms:cost(["qkv_project","out_project"])},
    {stage:"vocabulary",ms:cost(["lm_head"])},
    {stage:"attention",ms:cost(["attention"])}
  ] as $groups
| {scope:"optimistic equivalent per-token budget using sampled instrumented middle stages; not a speedup forecast or unprofiled critical-path proof",
   qualified_http_tps:$base,official_http_tps:$official.comparison.control.median,
   sampled_middle_stage_sum_ms:$middle.median_stage_sum_ms,
   sampled_middle_token_wall_ms:$middle.median_token_wall_ms,
   targets:([53.06,$official.original_relative_target]|map(. as $target|
     (512000/$base-512000/$target) as $save_ms|
     {required_http_tps:$target,required_throughput_gain:($target/$base-1),
      required_request_time_saved_ms:$save_ms,required_request_latency_reduction:(1-$base/$target),
      equivalent_saved_ms_per_generated_token:($save_ms/512),
      sampled_stage_scope:($groups|map(. as $g|($save_ms/512/$g.ms) as $fraction|
        {stage:$g.stage,sampled_gpu_ms:$g.ms,required_latency_reduction_if_only_this_stage:$fraction,
         eliminating_entire_sampled_stage_insufficient:($fraction>=1),
         required_stage_throughput_gain:(if $fraction<1 then 1/(1-$fraction)-1 else null end)}))})),
   interval_medians:($events|sort_by(.session_id,.generated_step)|group_by(.session_id)|.[1:-1]|add
     |group_by(.generated_step)|map({step:.[0].generated_step,
       gpu_union_ms:(map(.gpu_stage_union_ms)|median),gpu_timeline_ms:(map(.gpu_timeline_span_ms)|median),
       wall_minus_gpu_union_ms:(map(.wall_minus_gpu_union_ms)|median)})),
   does_not_admit_new_kernel:true,hardware_ceiling_claim:false}
