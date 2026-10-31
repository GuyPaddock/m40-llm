# Audit recorded behavior and costs, not source text or a new speed claim.
def require($condition;$message): if $condition then . else error($message) end;
def record($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity: \($kind)")|.[0];
def median: sort|.[length/2|floor];
def interval_union:
  sort_by(.gpu_start_ms)
  | reduce .[] as $span ({end:0,total:0};
      .total += ([$span.gpu_end_ms-([.end,$span.gpu_start_ms]|max),0]|max)
      | .end=([.end,$span.gpu_end_ms]|max)) | .total;
. as $rows
| record("manifest") as $manifest
| $rows|record("profile_comparison_summary") as $summary
| $rows|record("profile_token_comparison") as $parity
| $quality|record("q8_runtime_manifest") as $quality_manifest
| $quality|map(select(.kind=="q8_runtime_browser_generation" and .label=="candidate"))[0] as $expected
| $rows|record("q8_runtime_quality_evidence") as $evidence
| $rows|map(select(.kind=="trial" and (.phase=="profile-off" or .phase=="profile-on"))) as $primary
| $rows|map(select(.kind=="token_trace")) as $traces
| $events|map(select(.kind=="packed_prefill_gpu_profile")) as $prefix
| $events|map(select(.kind==null)) as $decode
| $rows
| require($manifest.revision=="cf411b63c042c5b3b5146f8b3c56508c754d6f72" and $manifest.worktree==""
    and $manifest.benchmark_executable_sha256=="0998e7e9940cac90d99319d080f111659b47314f515bd7473aba0d376daa12aa"
    and $manifest.primary_runtime_config=={ffn:"split-f32x4-128",prefill:"dequant-sgemm",attention_projection:"packed-warp-256"}
    and $manifest.actual_q8_attention_projection_backend=="packed-warp-256"
    and $manifest.environment.M40LLM_STAGE_PROFILE=="0"
    and $manifest.environment.M40LLM_STAGE_PROFILE_PREFILL=="1"
    and ($manifest.environment|del(.M40LLM_STAGE_PROFILE_REPORT,.M40LLM_STAGE_PROFILE_PREFILL,.M40LLM_STAGE_PROFILE))==$quality_manifest.environment
    and $manifest.model_sha256==$quality_manifest.candidate.sha256
    and $manifest.context_tokens==1024 and $manifest.kv_mode=="off" and $manifest.build_profile=="release"
    and $evidence.passes_exact_quality_gate
    and $evidence.sha256=="8b499787179f484aeb32e37caf837208e0994da4995918e18e34ba922d3d949e"
    and ($manifest.control_q8_execution|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))==
        ($quality_manifest.prepared_candidate|del(.preparation_elapsed_ms,.model_hash_elapsed_ms));"clean source, selected runtime and qualified preparation")
| require(($primary|length)==10 and all($primary|group_by(.pair)[];length==2
    and (map(.phase)|sort)==["profile-off","profile-on"])
    and ([$primary[].pair]|unique)==[0,1,2,3,4]
    and all($primary[];.is_fixed_workload and .matches_completed_quality_output
      and .prefill_stage_profile_requested and .stage_profile_enabled==(.phase=="profile-on")
      and .metrics.generated_tokens==512 and .metrics.prompt_tokens==65 and .metrics.stop_reason=="length"
      and .output==$expected.generation.output and .cache.reused_prompt_tokens==0
      and .cache.evaluated_prompt_tokens==65 and .final_kv_bytes==37748736
      and .q8_prepared_bytes==2917580800 and .q8_attention_projection_backend=="packed-warp-256"
      and .enqueue_counters==$expected.dispatch.totals
      and .launch_deltas==($expected.dispatch.operations|with_entries(select(.value.launches>0))|map_values(.launches)));"full request work, counters and memory")
| ($primary|sort_by(.client_started_unix_ms)) as $ordered
| require(all(range(1;$ordered|length);$ordered[.-1].client_finished_unix_ms<=$ordered[.].client_started_unix_ms)
    and all($ordered|group_by(.pair)[];sort_by(.client_started_unix_ms)|
      map(.phase)==(if .[0].pair%2==0 then ["profile-off","profile-on"] else ["profile-on","profile-off"] end));"serial alternating request intervals")
| require(($traces|length)==2 and ($traces|map(.stage_profile_enabled)|sort)==[false,true]
    and all($traces[];.prefill_stage_profile_requested
      and .generated_token_ids==$expected.generation.generated_token_ids
      and .prompt_token_ids==$expected.prompt_token_ids and .matches_measured_output
      and .q8_prefill_temporary_weight_bytes==90177536)
    and $parity.actual_ids_equal and $parity.actual_generated_tokens==512
    and $parity.matches_completed_quality and ($parity.is_optimization_claim|not);"actual 512 token parity")
| require(($events|length)==28 and ($prefix|length)==7 and ($decode|length)==21
    and ($events|map(.process_id)|unique)==[3753060]
    and ($prefix|map(.session_id)|sort)==[1,2,3,4,5,6,7]
    and ($decode|map(.session_id)|unique)==[1,2,3,4,5,6,7]
    and all($decode|group_by(.session_id)[];map(.generated_step)==[0,255,511])
    and all($decode[];.absolute_position==64+.generated_step and (.stages|length)==400
      and .event_pool_reused and .event_pool_capacity==880)
    and all($prefix[];(.stages|length)==829 and .prefix_tokens==64 and (.event_pool_reused|not)
      and .event_pool_capacity==880 and (has("generated_step")|not)
      and .counters=={launches:14292,cublas_calls:252,stream_syncs:70,stream_waits:324,
        device_allocs:21,device_frees:20,h2d_copies:68,d2h_copies:64,h2d_bytes:524304,d2h_bytes:139264});"scope identity, pool reuse and unchanged prefix work")
| require(all($prefix[];. as $sample|
    all(["prefill_attn_norm","prefill_rope_kv","prefill_attention","prefill_attn_residual",
         "prefill_ffn_norm","prefill_swiglu","prefill_mlp_residual","qkv_bias"][];
      . as $stage|[$sample.stages[]|select(.stage==$stage)|.layer_id]|sort==[range(0;36)])
    and ([$sample.stages[]|select(.stage=="prefill_hidden_copy")|.layer_id]|sort)==[range(0;35)]
    and all(["prefill_embeddings","prefill_output_copy"][];. as $stage|
      [$sample.stages[]|select(.stage==$stage)]|length==1 and .[0].layer_id==null)
    and all(range(0;36);. as $layer|
      ([$sample.stages[]|select(.layer_id==$layer and .stage=="q8_prefill_dequantize")]|length)==7
      and ([$sample.stages[]|select(.layer_id==$layer and .stage=="q8_prefill_sgemm")]|length)==7));"all grouped stages and layer coverage")
| require(all($prefix[];all(.stages[];.counters as $c|
    .gpu_elapsed_ms>=0 and .gpu_end_ms>=.gpu_start_ms
    and ((.gpu_elapsed_ms-(.gpu_end_ms-.gpu_start_ms))|fabs)<0.001
    and (if .stage=="prefill_embeddings" then .stream=="decode"
      and $c=={launches:0,cublas_calls:0,stream_syncs:64,stream_waits:0,device_allocs:0,
        device_frees:0,h2d_copies:64,d2h_copies:64,h2d_bytes:524288,d2h_bytes:139264}
    else
      all(["device_allocs","device_frees","h2d_copies","d2h_copies","h2d_bytes","d2h_bytes","stream_syncs","stream_waits"][];$c[.]==0)
      and (if .stage=="q8_prefill_sgemm" then .stream=="prefill" and $c.launches==0 and $c.cublas_calls==1
        elif .stage=="q8_prefill_dequantize" or .stage=="prefill_attention" then .stream=="prefill" and $c.launches==1 and $c.cublas_calls==0
        elif .stage=="prefill_rope_kv" or .stage=="qkv_bias" then .stream=="decode" and $c.launches==192 and $c.cublas_calls==0
        elif .stage=="prefill_hidden_copy" or .stage=="prefill_output_copy" then .stream=="decode" and $c.launches==0 and $c.cublas_calls==0
        else .stream=="decode" and $c.launches==1 and $c.cublas_calls==0 end) end))
    and ((.gpu_stage_sum_ms-(.stages|map(.gpu_elapsed_ms)|add))|fabs)<0.001
    and ((.gpu_stage_union_ms-(.stages|interval_union))|fabs)<0.001
    and ((.wall_minus_gpu_union_ms-(.prefill_wall_ms-.gpu_stage_union_ms))|fabs)<0.001);"stage timing, streams and allocation-free grouped work")
| require(all($primary[];all([.gpu_before,.gpu_after][];
    (.gpus_csv|split(",")|map(gsub("^ +| +$";""))|[.[3],.[6],.[7]])==["P0","1113","3004"]
    and (.compute_processes_csv|split("\n")|map(split(",")|map(gsub("^ +| +$";""))|.[1])|sort)==["3753060","3754140"]));"GPU state and owned process isolation")
| ([$primary[]|select(.phase=="profile-off")|.e2e_tps]|median) as $off
| ([$primary[]|select(.phase=="profile-on")|.e2e_tps]|median) as $on
| require($off==$summary.profile_off.median and $on==$summary.profile_on.median
    and (($summary.throughput_change-($on/$off-1))|fabs)<1e-12
    and $off/$on-1<0.03 and $summary.profile_off.mad_over_median<0.05
    and $summary.profile_on.mad_over_median<0.05
    and $summary.prefill_stage_profile_requested and ($summary.is_optimization_claim|not);"instrumentation perturbation admission")
| ($prefix|map(select(.session_id>=2 and .session_id<=6))) as $primary_prefix
| ($primary_prefix|map([.stages[]|select(.stage|startswith("prefill_"))|.gpu_elapsed_ms]|add)|median) as $new_group_ms
| ($primary_prefix|map([.stages[]|select((.stage|startswith("prefill_")) or .stage=="qkv_bias")|.gpu_elapsed_ms]|add)|median) as $new_and_bias_ms
| ([$primary[]|select(.phase=="profile-off")|.client_elapsed_ms]|median) as $wall_ms
| {passes_audit:true,source:$manifest.revision,executable_sha256:$manifest.benchmark_executable_sha256,
    pairs:5,gpu_boundaries:20,primary_prefix_samples:5,all_prefix_samples:7,all_decode_samples:21,
    spans_per_prefix:829,event_pool_capacity:880,profile_off:$summary.profile_off,profile_on:$summary.profile_on,
    throughput_change:$summary.throughput_change,latency_increase:($off/$on-1),actual_generated_ids_equal:512,
    prefix_wall_ms:($primary_prefix|map(.prefill_wall_ms)|median),
    prefix_profiled_union_ms:($primary_prefix|map(.gpu_stage_union_ms)|median),
    prefix_unattributed_wall_ms:($primary_prefix|map(.wall_minus_gpu_union_ms)|median),
    prefix_stages:([$primary_prefix[]|.stages|group_by(.stage)|map({stage:.[0].stage,
      gpu_ms:map(.gpu_elapsed_ms)|add,cpu_ms:map(.cpu_call_elapsed_ms)|add})|.[]]
      |group_by(.stage)|map({stage:.[0].stage,median_gpu_ms:map(.gpu_ms)|median,median_cpu_ms:map(.cpu_ms)|median})),
    off_http_ms:$wall_ms,new_groups_ms:$new_group_ms,new_groups_plus_bias_ms:$new_and_bias_ms,
    optimistic_new_group_elimination_gain:($wall_ms/($wall_ms-$new_group_ms)-1),
    optimistic_new_group_and_bias_elimination_gain:($wall_ms/($wall_ms-$new_and_bias_ms)-1),
    elimination_scope:"optimistic screen only, not unprofiled exclusive costs or a causal bound",
    per_request_counts_unchanged:true,prepared_cache_unchanged:true,is_optimization_claim:false,changes_defaults:false}
