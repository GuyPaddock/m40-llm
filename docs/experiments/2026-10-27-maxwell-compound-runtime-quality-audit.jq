# jq -s --arg revision EXPECTED --slurpfile prior PRIOR -f THIS RAW
def require($condition;$message): if $condition then . else error($message) end;
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity: \($kind)")|.[0];
def identity: del(.preparation_elapsed_ms,.model_hash_elapsed_ms);
. as $rows
| one("q8_runtime_manifest") as $m
| $rows|one("q8_runtime_summary") as $s
| $rows|one("q8_runtime_browser_comparison") as $comparison
| $rows|map(select(.kind=="q8_runtime_browser_generation")) as $generations
| $generations|map(select(.label=="control"))|require(length==1;"control generation")|.[0] as $a
| $generations|map(select(.label=="candidate"))|require(length==1;"candidate generation")|.[0] as $b
| $prior|map(select(.kind=="q8_runtime_browser_generation" and .label=="candidate"))|.[0] as $qualified
| $rows|map(select(.kind=="model_quality_task")) as $tasks
| $rows|map(select(.kind=="q8_runtime_prefix_logits")) as $logits
| $rows
| require($m.revision==$revision and ($revision|test("^[0-9a-f]{40}$")) and $m.worktree==""
    and $m.executable_sha256=="9a659afbdfbf02249588b9669b42f14e18cc86266ef24227b4f05ad0de1738d3"
    and $m.build_profile=="release" and $m.schema_version==2
    and $m.prefill_execution_contract=="prefix-decode-completed-v1";"clean release/source")
| require($m.control_config==$prior[0].candidate_config
    and $m.candidate_config==($m.control_config|.ffn="joint-shared-scales-128")
    and ($m.is_numerical_track|not) and $m.original_exact_goal_eligible
    and $m.environment==$prior[0].environment
    and $m.corpus_sha256==$prior[0].corpus_sha256
    and $m.reference.sha256==$prior[0].candidate.sha256
    and $m.candidate.sha256==$m.reference.sha256
    and $m.selected_splits==["calibration","held_out"];"frozen exact model/config/input/environment")
| require(($generations|length)==2 and ($m.browser_prompt_ids|length)==65
    and $a.prompt_token_ids==$qualified.prompt_token_ids and $b.prompt_token_ids==$a.prompt_token_ids
    and $a.generation.generated_token_ids==$qualified.generation.generated_token_ids
    and $b.generation.generated_token_ids==$a.generation.generated_token_ids
    and $a.generation.output==$b.generation.output;"actual canonical512 IDs/output")
| require(all($generations[];(.generation.generated_token_ids|length)==512
    and .generation.metrics.generated_tokens==512 and .generation.metrics.prompt_tokens==65
    and .generation.metrics.stop_reason=="length"
    and .generation.prefill_mode=="packed-prefix-dequant-sgemm"
    and .generation.q8_prefill_temporary_weight_bytes==90177536
    and .dispatch.operations.dequantize_q8_0_to_f32_colmajor.launches==252
    and .dispatch.operations.gemm_f32xq8_separate_f32.launches==18944
    and .dispatch.operations.qkv_f32xq8_0_gguf_packed_warp.launches==18432
    and .dispatch.operations.gemm_f32xq8_0_gguf_coltile4_packed_warp.launches==18432
    and .dispatch.totals.cublas_calls==252);"unchanged full-model prefix/down/head/projections")
| require($a.dispatch.operations.ffn_q8_split_f32x4_128.launches==18432
    and $b.dispatch.operations.ffn_q8_joint_f32x4_shared_scales_128.launches==18432
    and ($a.dispatch.operations|del(.ffn_q8_split_f32x4_128))
        ==($b.dispatch.operations|del(.ffn_q8_joint_f32x4_shared_scales_128))
    and $a.dispatch.totals==$b.dispatch.totals;"only compound dispatch changed")
| require(($tasks|length)==12 and all($tasks[];.has_identical_generation
    and .reference.output==.candidate.output and .reference_score==.candidate_score)
    and all($tasks[];. as $t|[$prior[]|select(.kind=="model_quality_task" and .id==$t.id)|.candidate.output]==[$t.reference.output])
    and ([$tasks[].id]|sort)==([$prior[]|select(.kind=="model_quality_task")|.id]|sort)
    and ([$tasks[]|select(.reference_score.passed)]|length)==7;"frozen task outcomes")
| require(($logits|map(.generated_step_zero_based))==[0,1,8,32,128,256,511]
    and all($logits[];.top10_overlap==10 and .max_abs_logit_diff==0 and .mean_abs_logit_diff==0
    and .actual_reference_greedy_token_id==.actual_candidate_greedy_token_id);"full-vocabulary prefix equality")
| require(($m.prepared_control|identity)==($m.prepared_candidate|identity)
    and ($m.prepared_control|identity)==($prior[0].prepared_candidate|identity)
    and $m.prepared_control.resident_bytes==2917580800
    and ($m.prepared_control.tensors|length)==109;"all original prepared owners")
| require(($m.joint_control.is_enabled|not) and $m.joint_control.resident_bytes==0
    and $m.joint_candidate.is_enabled and $m.joint_candidate.pair_count==36
    and $m.joint_candidate.resident_bytes==1724645376
    and $m.joint_candidate.combined_prepared_bytes==4642226176
    and $m.joint_candidate.budget_bytes==6442450944
    and $m.joint_candidate.peak_temporary_host_pair_bytes==47906816
    and $m.joint_candidate.retained_host_pair_bytes==0
    and $b.joint_ffn_pairs==$m.joint_candidate and $a.joint_ffn_pairs==$m.joint_control;"joint preparation accounting")
| require(all($m.joint_candidate.pairs[]; . as $pair|all(["gate","up"][];. as $role|
    ([$m.prepared_candidate.tensors[]|select(.key.source==$pair[$role])]|length)==1));"joint source identities")
| require(all([$m.reference_memory,$m.candidate_memory][];.resident_kv_bytes==37748736
    and .allocated_context_tokens==1024 and .materialized_f32_bytes==0
    and (.activation_sparsity.is_enabled|not) and .active_layer_count==36)
    and all([$m.reference,$m.candidate][];.layers==36 and .active_layer_count==36 and .head_dim==128
        and .rope_layout==1 and .tokenizer_metadata_sha256==$prior[0].candidate.tokenizer_metadata_sha256)
    and $m.candidate_memory.prepared_weight_bytes==2917580800
    and $m.candidate_memory.joint_ffn_pairs==$m.joint_candidate
    and $m.candidate_memory.weight_payload_bytes_including_joint==7921745920;"dense memory/full model")
| require($comparison.has_fixed_workload and $comparison.has_identical_generation
    and $s.passes_exact_quality_gate and $s.dense_valid_tasks==7 and $s.dense_valid_regressions==0
    and $s.tracked_device_bytes_initial==0 and $s.tracked_device_bytes_after==0;"quality/cleanup completion")
| {passes_quality_audit:true,source:$m.revision,executable_sha256:$m.executable_sha256,
    actual512_ids_equal:true,all12_task_outputs_equal:true,dense_valid_tasks:7,regressions:0,
    sampled_full_vocabulary_max_abs_diff:0,sampled_full_vocabulary_mean_abs_diff:0,top10_overlap:10,
    original_prepared_identities:109,joint_pairs:36,joint_bytes:1724645376,
    combined_prepared_bytes:4642226176,candidate_total_weight_payload_bytes:7921745920,
    unchanged_other_dispatch:true,unchanged_enqueue_counters:true,tracked_device_bytes_after:0,
    cold_preparation:{control:$m.prepared_control.preparation_elapsed_ms,
        candidate_split:$m.prepared_candidate.preparation_elapsed_ms,joint:$m.joint_candidate},
    makes_http_claim:false}
