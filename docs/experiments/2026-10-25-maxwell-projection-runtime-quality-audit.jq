# jq -s --slurpfile prior PRIOR --slurpfile capture CAPTURE -f THIS RAW
def require($condition;$message): if $condition then . else error($message) end;
def one($kind): map(select(.kind==$kind)) | require(length==1;"record multiplicity: \($kind)") | .[0];
def cache_identity: del(.preparation_elapsed_ms,.model_hash_elapsed_ms);
. as $rows
| one("q8_runtime_manifest") as $manifest
| $rows | one("q8_runtime_summary") as $summary
| $rows | one("q8_runtime_browser_comparison") as $comparison
| $rows | map(select(.kind=="q8_runtime_browser_generation")) as $generations
| $generations | map(select(.label=="control")) | require(length==1;"control generation") | .[0] as $a
| $generations | map(select(.label=="candidate")) | require(length==1;"candidate generation") | .[0] as $b
| $prior | map(select(.kind=="q8_runtime_browser_generation" and .label=="candidate"))
| require(length==1;"prior qualified runtime") | .[0] as $qualified
| $rows | map(select(.kind=="model_quality_task")) as $tasks
| $rows | map(select(.kind=="q8_runtime_prefix_logits")) as $logits
| $rows
| require($manifest.revision=="1349613b4844a6e097eeb4fdddafa8ead587531d" and $manifest.worktree==""
    and $manifest.executable_sha256=="6b01f34771bb491ba95f6e0b28ffe5acb49dcb63d0e93936b9948881f6743c71"
    and $manifest.build_profile=="release" and $manifest.schema_version==2
    and $manifest.prefill_execution_contract=="prefix-decode-completed-v1";"source/release/prefix contract")
| require($manifest.control_config=={attention_projection:"native",ffn:"split-f32x4-128",prefill:"dequant-sgemm"}
    and $manifest.candidate_config=={attention_projection:"warp-256",ffn:"split-f32x4-128",prefill:"dequant-sgemm"}
    and ($manifest.is_numerical_track|not) and $manifest.original_exact_goal_eligible;"current control/candidate/exact track")
| require($manifest.environment==$capture[0].execution_configuration.environment
    and $manifest.corpus_sha256=="2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38"
    and $manifest.reference.sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $manifest.candidate.sha256==$manifest.reference.sha256;"frozen inputs/environment")
| require($manifest.reference.active_layer_count==36 and $manifest.candidate.active_layer_count==36
    and $manifest.reference.head_dim==128 and $manifest.candidate.rope_layout==1
    and $manifest.reference.tokenizer_metadata_sha256==$manifest.candidate.tokenizer_metadata_sha256;"unchanged Qwen architecture/tokenizer")
| require(($manifest.browser_prompt_ids|length)==65 and $a.prompt_token_ids==$qualified.prompt_token_ids
    and $b.prompt_token_ids==$a.prompt_token_ids and $manifest.browser_prompt_ids==$a.prompt_token_ids
    and $a.generation.generated_token_ids==$qualified.generation.generated_token_ids
    and $a.generation.generated_token_ids==$b.generation.generated_token_ids
    and $a.generation.output==$b.generation.output;"actual canonical IDs/output")
| require(($generations|length)==2 and all($generations[];
    (.generation.generated_token_ids|length)==512 and .generation.metrics.generated_tokens==512
    and .generation.metrics.prompt_tokens==65 and .generation.metrics.stop_reason=="length"
    and .generation.prefill_mode=="packed-prefix-dequant-sgemm"
    and .generation.q8_prefill_temporary_weight_bytes==90177536
    and .dispatch.operations.dequantize_q8_0_to_f32_colmajor.launches==252
    and .dispatch.operations.ffn_q8_split_f32x4_128.launches==18432
    and .dispatch.operations.gemm_f32xq8_separate_f32.launches==18944
    and .dispatch.totals.cublas_calls==252);"complete generation/unchanged SGEMM/FFN/down/head")
| require($a.actual_attention_projection_backend=="native" and $b.actual_attention_projection_backend=="warp-256"
    and $a.dispatch.operations.qkv_f32xq8_0_gguf_decode.launches==18432
    and $a.dispatch.operations.gemm_f32xq8_0_gguf_f32_decode.launches==18432
    and $b.dispatch.operations.qkv_f32xq8_0_gguf_warp.launches==18432
    and $b.dispatch.operations.gemm_f32xq8_0_gguf_coltile4_warp.launches==18432
    and ($a.dispatch.operations|del(.qkv_f32xq8_0_gguf_decode,.gemm_f32xq8_0_gguf_f32_decode))
        ==($b.dispatch.operations|del(.qkv_f32xq8_0_gguf_warp,.gemm_f32xq8_0_gguf_coltile4_warp))
    and $a.dispatch.totals==$b.dispatch.totals;"actual projection dispatch and unchanged other work")
| require(($tasks|length)==12 and all($tasks[];.has_identical_generation
    and .reference.output==.candidate.output and .reference_score==.candidate_score)
    and ([$tasks[]|select(.reference_score.passed)]|length)==7
    and ([$tasks[]|select(.reference_score.passed and (.candidate_score.passed|not))]|length)==0;"frozen task outcomes")
| require(all($tasks[];. as $task|
    [$prior[]|select(.kind=="model_quality_task" and .id==$task.id)|.candidate.output]
    ==[$task.reference.output]);"previously qualified task outputs")
| require(($logits|map(.generated_step_zero_based))==[0,1,8,32,128,256,511]
    and all($logits[];.top10_overlap==10 and .max_abs_logit_diff==0 and .mean_abs_logit_diff==0
      and .actual_reference_greedy_token_id==.actual_candidate_greedy_token_id);"full-vocabulary prefix equality")
| require(($manifest.prepared_control|cache_identity)==($manifest.prepared_candidate|cache_identity)
    and ($manifest.prepared_control|cache_identity)==($prior[0].prepared_candidate|cache_identity)
    and $manifest.prepared_control.resident_bytes==2917580800
    and $manifest.prepared_control.total_weight_payload_bytes==6197100544
    and ($manifest.prepared_control.tensors|length)==109;"all existing prepared GU/down/head identities and accounting")
| require(all([$manifest.reference_memory,$manifest.candidate_memory][];
    .resident_kv_bytes==37748736 and .allocated_context_tokens==1024
    and .materialized_f32_bytes==0 and .activation_sparsity.is_enabled==false
    and .active_layer_count==36);"dense/full-model memory semantics")
| require($comparison.has_fixed_workload and $comparison.has_identical_generation
    and $summary.passes_exact_quality_gate and $summary.dense_valid_tasks==7
    and $summary.dense_valid_regressions==0 and $summary.tracked_device_bytes_initial==0
    and $summary.tracked_device_bytes_after==0;"completion/cleanup")
| {passes_quality_audit:true,source:$manifest.revision,executable:$manifest.executable_sha256,
    scope:"actual full-model quality against qualified split-FFN/SGEMM; not HTTP timing",
    actual_browser_ids_equal:512,task_outputs_equal:12,dense_valid_tasks:7,regressions:0,
    sampled_full_vocabulary_logit_max_diff:0,sampled_full_vocabulary_logit_mean_diff:0,
    top10_overlap:10,prepared_tensor_identities_equal:109,additional_device_weight_bytes:0,
    source_plus_prepared_bytes:6197100544,kv_bytes_per_model:37748736,temporary_prefill_bytes:90177536,
    candidate_qkv_warp_launches:18432,candidate_output_warp_launches:18432,
    unchanged_other_dispatch:true,unchanged_enqueue_counters:true,tracked_device_bytes_after:0,
    diagnostic_resident_generation_ms:[$a.generation.resident_generation_wall_ms,$b.generation.resident_generation_wall_ms],
    makes_http_or_default_claim:false}
