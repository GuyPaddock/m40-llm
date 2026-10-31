# Freeze the selected projection runtime, actual IDs and complete stage coverage.
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
| require($manifest.worktree=="" and $manifest.revision=="be149f69495856a2b7d5e352d2c0b8b21ed06555";"profile source")
| require($manifest.benchmark_executable_sha256=="4b7f15aa9361a29e42472e7c0194c92863b050ed1c3ab436f2f68514f4ff9cf7"
    and $manifest.primary_runtime_config=={ffn:"split-f32x4-128",prefill:"dequant-sgemm",attention_projection:"packed-warp-256"};"selected runtime")
| ($quality | record("q8_runtime_manifest")) as $quality_manifest
| ($rows | record("q8_runtime_quality_evidence")) as $evidence
| require($manifest.actual_q8_attention_projection_backend=="packed-warp-256"
    and ($manifest.environment|del(.M40LLM_STAGE_PROFILE_REPORT))==$quality_manifest.environment
    and $manifest.model_sha256==$quality_manifest.candidate.sha256
    and $manifest.context_tokens==1024 and $manifest.kv_mode=="off" and $manifest.build_profile=="release"
    and $evidence.passes_exact_quality_gate
    and $evidence.sha256=="8b499787179f484aeb32e37caf837208e0994da4995918e18e34ba922d3d949e"
    and ($manifest.control_q8_execution|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))==
        ($quality_manifest.prepared_candidate|del(.preparation_elapsed_ms,.model_hash_elapsed_ms));"completed quality and preparation")
| require(all($primary[];.q8_attention_projection_backend=="packed-warp-256" and .final_kv_bytes==37748736
    and .enqueue_counters==$expected.dispatch.totals
    and .launch_deltas==($expected.dispatch.operations|with_entries(select(.value.launches>0))|map_values(.launches)));"actual selected dispatch and memory")
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
| require(($events|length)==21 and ($events|map(.process_id)|unique)==[2372464]
    and ($events|group_by(.session_id)|length)==7
    and all(($events|group_by(.session_id))[]; map(.generated_step)==[0,255,511])
    and all($events[]; .absolute_position==64+.generated_step and (.stages|length)==400);"event coverage")
| require(all($events[]; . as $sample |
    all(["qkv_project","mlp_gate_up_swiglu","mlp_down","out_project","attention"][];
      . as $stage | [$sample.stages[]|select(.stage==$stage)|.layer_id]|sort==[range(0;36)])
    and ([$sample.stages[]|select(.stage=="lm_head")]|length)==1);"layer/global stage accounting")
| ([$primary[]|(.gpu_before,.gpu_after)|.gpus_csv|split(",")|map(gsub("^ +| +$";""))|{sm:.[6],memory:.[7]}]|unique) as $clocks
| require($clocks==[{sm:"1113",memory:"3004"}];"profile clocks")
| require(all($primary[];all([.gpu_before,.gpu_after][];
    (.gpus_csv|split(",")[3]|gsub("^ +| +$";""))=="P0"
    and (.compute_processes_csv|split("\n")|map(split(",")|map(gsub("^ +| +$";""))|.[1])|sort)==["2372464","2373330"]));"profile isolation and power state")
| def median: sort|.[length/2|floor];
  ([$primary[]|select(.phase=="profile-off")|.e2e_tps]|median) as $off
| ([$primary[]|select(.phase=="profile-on")|.e2e_tps]|median) as $on
| require($off==$summary.profile_off.median and $on==$summary.profile_on.median
    and (($summary.throughput_change-($on/$off-1))|fabs)<1e-12
    and ($summary.is_optimization_claim|not);"measured perturbation")
| {passes_audit:true,profile_off:$summary.profile_off,profile_on:$summary.profile_on,
   throughput_change:$summary.throughput_change,actual_ids_equal:512,
   samples:21,primary_profile_sessions:5,gpu_boundary_samples:20,clocks:$clocks,
   stage_scope:"36 layer samples per core stage, lm_head once per token; event spans not unprofiled latency or measured bandwidth",
   is_optimization_claim:false}
