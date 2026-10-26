# jq -s --slurpfile quality QUALITY -f THIS RAW
def require($condition;$message): if $condition then . else error($message) end;
def median: sort|.[length/2|floor];
def identity: del(.preparation_elapsed_ms,.model_hash_elapsed_ms);
. as $rows
| map(select(.kind=="kv_stack_manifest"))|require(length==1;"manifest")|.[0] as $m
| $rows|map(select(.kind=="kv_stack_generation")) as $all
| $all|map(select(.phase=="measured")) as $measured
| $measured|map(select(.case=="dense"))|.[0] as $dense
| $quality|map(select(.kind=="q8_runtime_browser_generation" and .label=="candidate"))|.[0] as $qualified
| $rows
| require($m.revision=="0ea619bb4113359136eecb2169632fedcac3ae13" and $m.worktree==""
    and $m.build_profile=="release" and $m.context==1024 and $m.max_tokens==512
    and ($m.executable_sha256|split(" ")[0])=="ab22e0fe9a55f8b306e6000145702c9226ea4dd19e01075be7e29813492420af"
    and ($m.model_sha256|split(" ")[0])=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d";"source/build/workload")
| require(($m.prepared|identity)==($quality[0].prepared_candidate|identity)
    and $m.prepared.resident_bytes==2917580800 and ($m.prepared.tensors|length)==109;"fixed prepared weights")
| require(($all|length)==24 and ($measured|length)==20
    and all($all|group_by(.case)[];length==6 and ([.[].round]|sort)==[0,1,2,3,4,5])
    and all($measured|group_by(.round)[];length==4)
    and ([$all[]|select(.round==0)|.case])==["dense","compressed-default","compressed-window128-top8","compressed-window128-top4"]
    and all(range(1;6);. as $round|
      ([$all[]|select(.round==$round)|.case])==(if $round%2==0
        then ["dense","compressed-default","compressed-window128-top8","compressed-window128-top4"]
        else ["compressed-window128-top4","compressed-window128-top8","compressed-default","dense"] end));"warmups/alternating coverage")
| require(all($all[];.metrics.generated_tokens==512 and (.generated_ids|length)==512
    and .metrics.prompt_tokens==65 and .metrics.stop_reason=="length"
    and .prompt_ids==$qualified.prompt_token_ids and .temporary_prefill_weight_bytes==90177536
    and .operations.qkv_f32xq8_0_gguf_packed_warp==18432
    and .operations.gemm_f32xq8_0_gguf_coltile4_packed_warp==18432
    and .operations.ffn_q8_split_f32x4_128==18432 and .operations.gemm_f32xq8_separate_f32==18944
    and .operations.dequantize_q8_0_to_f32_colmajor==252 and .counters.cublas_calls==252);"actual fixed workload/stack dispatch")
| require($dense.generated_ids==$qualified.generation.generated_token_ids
    and all($all[];if .case=="dense" then .generated_ids==$dense.generated_ids and .final_kv_bytes==37748736
      and .operations.attention_head128_parallel==18432 and .prefill_mode=="packed-prefix-dequant-sgemm"
      else .operations.attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct==18432
      and .kv.exact_old_backing=="fp16-k-q4-v" and .kv.exact_old_attention=="fp16-k-q4-v-direct"
      and .kv.policy=="topk" and .kv.block_size==32 and .kv.representatives==0
      and .prefill_mode=="packed-prefix-block-select-exact-dequant-sgemm" end);"dense reference/actual attention backend")
| require(all($all[];if .case=="compressed-default" then .kv.recent_window==1024 and .kv.top_blocks==8
      and .old_tokens_final_forward==0 and .final_kv_bytes==61747488 and .generated_ids==$dense.generated_ids
    elif .case=="compressed-window128-top8" then .kv.recent_window==128 and .kv.top_blocks==8
      and .old_tokens_final_forward==448 and .final_kv_bytes==31707216 and .first_divergence_vs_dense==323
    elif .case=="compressed-window128-top4" then .kv.recent_window==128 and .kv.top_blocks==4
      and .old_tokens_final_forward==448 and .final_kv_bytes==31707216 and .first_divergence_vs_dense==196
    else .case=="dense" end);"explicit memory/selection cases")
| require(all($all|group_by(.case)[];([.[].generated_ids]|unique|length)==1);"per-case output stability")
| ([$measured[]|(.gpu_before,.gpu_after)|.clocks]|unique) as $clocks
| ([$all[]|(.gpu_before,.gpu_after)|.processes|split("\n")
    |map(split(",")|.[0:3]|map(gsub("^ +| +$";"")))|sort]|unique) as $processes
| require($clocks==["GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, P0, 1113, 3004"]
    and ($processes|length)==1 and ($processes[0]|length)==1
    and $processes[0][0][0]=="GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852"
    and ($processes[0][0][2]|endswith("qwen_decode_throughput-8127bb5a855b8b5c"));"stable clocks/single process")
| require(any($rows[];.kind=="kv_stack_complete" and .tracked_initial==0 and .tracked_after==0);"completion/cleanup")
| {passes_audit:true,scope:"resident generation, not HTTP/Ollama/exact-goal qualification",
    source:$m.revision,executable_sha256:($m.executable_sha256|split(" ")[0]),
    warmups:4,measured_requests:20,rounds_per_case:5,actual_tokens_per_request:512,
    prepared_tensor_identities_equal:109,source_plus_prepared_bytes:6197100544,
    observed_measured_clocks:$clocks,observed_processes:$processes,measured_gpu_boundaries:40,warmup_gpu_boundaries:8,
    warmup_clocks:([$all[]|select(.phase=="warmup")|(.gpu_before,.gpu_after)|.clocks]|unique),
    cases:($measured|group_by(.case)|map({case:.[0].case,requests:length,
      resident_tps:(map(.resident_tps)|median),min_tps:(map(.resident_tps)|min),max_tps:(map(.resident_tps)|max),
      prefill_ms:(map(.metrics.prefill_elapsed_ms)|median),decode_ms:(map(.metrics.decode_elapsed_ms)|median),
      total_ms:(map(.elapsed_ms)|median),kv_allocation_ms:(map(.kv_allocation_ms)|median),
      final_kv_bytes:.[0].final_kv_bytes,dense_equivalent_kv_bytes:.[0].dense_equivalent_kv_bytes,
      final_active_tokens_upper_bound_per_head:.[0].active_tokens_upper_bound_per_head_final_forward,
      first_divergence_vs_dense:.[0].first_divergence_vs_dense,
      recent_fp16_bytes:.[0].recent_fp16_bytes,old_k_fp16_bytes:.[0].old_k_fp16_bytes,
      old_v_q4_payload_bytes:.[0].old_v_q4_payload_bytes,old_v_q4_scale_bytes:.[0].old_v_q4_scale_bytes,
      summary_index_bytes:.[0].summary_index_bytes})),
    tracked_device_bytes_after:0,changes_runtime_defaults:false}
