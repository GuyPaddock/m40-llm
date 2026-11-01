def require($condition;$message): if $condition then . else error($message) end;
def record($kind): map(select(.kind==$kind)) | require(length==1;"record multiplicity: \($kind)") | .[0];
. as $rows
| record("manifest") as $manifest
| $rows | record("profile_comparison_summary") as $summary
| $rows | record("profile_token_comparison") as $parity
| $quality | map(select(.kind=="q8_runtime_browser_generation" and .label=="candidate"))[0] as $expected
| $rows | map(select(.kind=="trial" and (.phase=="profile-off" or .phase=="profile-on"))) as $primary
| $rows | map(select(.kind=="token_trace")) as $traces
| $rows
| require($manifest.worktree=="" and $manifest.revision=="05e52760640cfee057e81d14dfc02a4430a5e69e";"profile source")
| require($manifest.benchmark_executable_sha256=="9540358f1eb1eee6c4658125f27387694125f8cf670e377b54bb3ee1c85a0e2e"
    and $manifest.primary_runtime_config=={ffn:"split-f32x4-128",prefill:"dequant-sgemm"};"selected runtime")
| require(($primary|length)==10 and ([$primary[]|select(.phase=="profile-off")]|length)==5
    and ([$primary[]|select(.phase=="profile-on")]|length)==5;"off/on pairs")
| require(all($primary[]; .is_fixed_workload and .matches_completed_quality_output
    and .metrics.generated_tokens==512 and .metrics.prompt_tokens==65 and .metrics.stop_reason=="length"
    and .output==$expected.generation.output and .cache.reused_prompt_tokens==0
    and .cache.evaluated_prompt_tokens==65 and .q8_prepared_bytes==2917580800
    and .launch_deltas.ffn_q8_split_f32x4_128==18432 and .enqueue_counters.cublas_calls==252);"request contract")
| require(($traces|length)==2 and ($traces|map(.stage_profile_enabled)|sort)==[false,true]
    and all($traces[];.generated_token_ids==$expected.generation.generated_token_ids
      and .prompt_token_ids==$expected.prompt_token_ids and .matches_measured_output);"actual ID parity")
| require($parity.actual_ids_equal and $parity.actual_generated_tokens==512
    and $parity.matches_completed_quality and ($parity.is_optimization_claim|not);"parity summary")
| require(($events|length)==21 and ($events|map(.process_id)|unique)==[1271892]
    and ($events|group_by(.session_id)|length)==7
    and all(($events|group_by(.session_id))[]; map(.generated_step)==[0,255,511])
    and all($events[]; .absolute_position==64+.generated_step and (.stages|length)==400);"event coverage")
| require(all($events[]; . as $sample |
    all(["qkv_project","mlp_gate_up_swiglu","mlp_down","out_project","attention"][];
      . as $stage | [$sample.stages[]|select(.stage==$stage)|.layer_id]|sort==[range(0;36)])
    and ([$sample.stages[]|select(.stage=="lm_head")]|length)==1);"layer/global stage accounting")
| ([$primary[]|(.gpu_before,.gpu_after)|.gpus_csv|split(",")|map(gsub("^ +| +$";""))|{sm:.[6],memory:.[7]}]|unique) as $clocks
| require($clocks==[{sm:"1113",memory:"3004"}];"profile clocks")
| {passes_audit:true,profile_off:$summary.profile_off,profile_on:$summary.profile_on,
   throughput_change:$summary.throughput_change,actual_ids_equal:512,
   samples:21,primary_profile_sessions:5,gpu_boundary_samples:20,clocks:$clocks,
   stage_scope:"36 layer samples per core stage, lm_head once per token; event spans not unprofiled latency or measured bandwidth",
   is_optimization_claim:false}
