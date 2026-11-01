# jq -s --arg geometry 128 --slurpfile prior CORRECTED --slurpfile capture CAPTURE --rawfile log LOG -f THIS RAW
def require($condition;$message): if $condition then . else error($message) end;
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity: \($kind)")|.[0];
def cache_identity: del(.preparation_elapsed_ms,.model_hash_elapsed_ms);
def near($a;$b): (($a-$b)|fabs)<1e-8;
. as $rows
| one("q8_runtime_manifest") as $m
| $rows|one("q8_runtime_summary") as $summary
| $rows|one("q8_runtime_browser_comparison") as $comparison
| $rows|map(select(.kind=="q8_runtime_browser_generation")) as $generations
| $generations|map(select(.label=="control"))|require(length==1;"control")|.[0] as $a
| $generations|map(select(.label=="candidate"))|require(length==1;"candidate")|.[0] as $b
| $prior|map(select(.kind=="q8_runtime_browser_generation" and .label=="control"))
| require(length==1;"corrected reference")|.[0] as $qualified
| $rows|map(select(.kind=="model_quality_task")) as $tasks
| $rows|map(select(.kind=="q8_runtime_prefix_logits")) as $logits
| $rows|map(select(.kind=="q8_runtime_passage")) as $passages
| $rows|map(select(.kind=="q8_runtime_loss_summary")) as $loss
| $rows
| require(($geometry=="128" or $geometry=="256") and length==31
    and $m.schema_version==2 and $m.worktree=="" and $m.build_profile=="release"
    and $m.revision=="33839478c35e20a59bf0a8f7930db66eb8109dda"
    and $m.executable_sha256=="bf581bbabe5eb0f21d30f9387c0d3bf398e65381c2ea6bae336852310cdf6b32"
    and $m.prefill_execution_contract=="prefix-decode-completed-v1";"clean source/repaired execution contract")
| require($m.control_config=={attention_projection:"native",ffn:"native",prefill:"sequential"}
    and $m.candidate_config=={attention_projection:"native",ffn:("q81-"+$geometry),prefill:"sequential"}
    and $m.is_numerical_track and ($m.original_exact_goal_eligible|not)
    and ($log|contains("[q8-ffn] kernel=Q81 threads=Some("+$geometry+") scope=single-row activation_bytes=2304 numerical_track=true"));"explicit numerical candidate/scratch")
| require($m.environment==$capture[0].execution_configuration.environment
    and $m.corpus_sha256=="2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38"
    and $m.reference.sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $m.candidate.sha256==$m.reference.sha256 and $m.selected_splits==["calibration","held_out"]
    and $m.reference.active_layer_ids==[range(36)] and $m.candidate.active_layer_ids==[range(36)]
    and $m.reference.head_dim==128 and $m.candidate.rope_layout==1
    and $m.reference.tokenizer_metadata_sha256==$m.candidate.tokenizer_metadata_sha256;"frozen source/corpus/architecture")
| require(($m.prepared_control|cache_identity)==($m.prepared_candidate|cache_identity)
    and ($m.prepared_control|cache_identity)==($capture[0].prepared_weights|cache_identity)
    and ($m.prepared_control.tensors|length)==37
    and $m.prepared_control.resident_bytes==1192935424
    and $m.prepared_control.total_weight_payload_bytes==4472455168;"prepared identities/accounting")
| require(all([$m.reference_memory,$m.candidate_memory][];.resident_kv_bytes==37748736
    and .allocated_context_tokens==1024 and .uploaded_weight_payload_bytes==3279519744
    and .materialized_f32_bytes==0 and (.activation_sparsity.is_enabled|not)
    and (.has_in_process_compaction|not));"dense full-model semantics")
| $a.generation.generated_token_ids as $ids
| $b.generation.generated_token_ids as $candidate_ids
| [range(0;([$ids|length,$candidate_ids|length]|min))|select($ids[.]!=$candidate_ids[.])] as $different
| (if ($different|length)>0 then $different[0] elif ($ids|length)!=($candidate_ids|length) then ([$ids|length,$candidate_ids|length]|min) else null end) as $divergence
| require(($generations|length)==2 and $ids==$qualified.generation.generated_token_ids
    and $a.generation.output==$qualified.generation.output
    and $a.prompt_token_ids==$qualified.prompt_token_ids and $b.prompt_token_ids==$a.prompt_token_ids
    and $m.browser_prompt_ids==$a.prompt_token_ids and ($m.browser_prompt_ids|length)==65
    and all($generations[];(.generation.generated_token_ids|length)==512 and .generation.metrics.generated_tokens==512
      and .generation.metrics.prompt_tokens==65 and .generation.metrics.stop_reason=="length"
      and .generation.prefill_mode=="sequential" and .generation.q8_prefill_temporary_weight_bytes==null
      and .actual_attention_projection_backend=="native")
    and $divergence!=null and $comparison.first_generated_divergence_step_zero_based==$divergence
    and $comparison.has_fixed_workload and ($comparison.has_identical_generation|not);"actual corrected reference/continuations")
| ("ffn_q8_1_threads"+$geometry) as $op
| require($b.dispatch.operations[$op].launches==20736 and $b.dispatch.operations.quantize_q8_1.launches==20736
    and $a.dispatch.operations.mlp_gate_up_swiglu_f32xq8_0_gguf_decode.launches==20736
    and ($a.dispatch.operations|del(.mlp_gate_up_swiglu_f32xq8_0_gguf_decode))
      ==($b.dispatch.operations|del(.quantize_q8_1)|del(.[$op]))
    and all($generations[];.dispatch.operations.gemm_f32xq8_separate_f32.launches==21248
      and .dispatch.operations.qkv_f32xq8_0_gguf_decode.launches==20736
      and .dispatch.operations.gemm_f32xq8_0_gguf_f32_decode.launches==20736
      and .dispatch.totals.device_allocs==5 and .dispatch.totals.device_frees==5
      and .dispatch.totals.cublas_calls==0 and .dispatch.totals.stream_syncs==1155 and .dispatch.totals.stream_waits==0)
    and $b.dispatch.totals.launches==$a.dispatch.totals.launches+20736
    and ($a.dispatch.totals|del(.launches))==($b.dispatch.totals|del(.launches));"actual full FFN/quantization and unchanged other dispatch")
| require(($tasks|length)==12 and ([$tasks[]|[.split,.id]]|unique|length)==12
    and all($tasks[];. as $task|[$prior[]|select(.kind=="model_quality_task" and .id==$task.id and .split==$task.split)
      |{output:.reference.output,score:.reference_score}]==[{output:$task.reference.output,score:$task.reference_score}])
    and $summary.dense_valid_tasks==([$tasks[]|select(.reference_score.passed)]|length)
    and $summary.dense_valid_regressions==([$tasks[]|select(.reference_score.passed and (.candidate_score.passed|not))]|length);"frozen task reference/derived outcomes")
| require(($logits|length)==8 and ($logits|map(.generated_step_zero_based))==([0,1,8,32,128,256,511,$divergence]|unique|sort)
    and all($logits[];. as $probe|.prefix_length==65+.generated_step_zero_based
      and .expected_actual_reference_token_id==$ids[.generated_step_zero_based]
      and .actual_reference_greedy_token_id==.reference.top_indices[0]
      and .actual_candidate_greedy_token_id==.candidate.top_indices[0]
      and .max_abs_logit_diff>=.mean_abs_logit_diff and .mean_abs_logit_diff>=0
      and .top10_overlap==([.reference.top_indices[]|select(. as $id|$probe.candidate.top_indices|index($id))]|length));"matching reference-prefix logit diagnostics")
| require(($passages|length)==4 and ($loss|map(.split)|sort)==["calibration","held_out"]
    and all($passages[];. as $p|.comparison.input_tokens==(.token_ids|length)
      and (.comparison.tokens|length)==(.comparison.input_tokens-1)
      and ([.comparison.tokens[].position])==[range(1;.comparison.input_tokens)]
      and all(.comparison.tokens[];.target_id==$p.token_ids[.position])
      and all(["reference","candidate"][];. as $label|
        $p.comparison[$label] as $metric|$metric.scored_tokens==($p.comparison.tokens|length)
        and near($metric.nll_sum;[$p.comparison.tokens[]|.[$label].nll]|add)
        and near($metric.mean_nll;$metric.nll_sum/$metric.scored_tokens)
        and near($metric.perplexity;($metric.mean_nll|exp)) and ($metric.perplexity_overflow|not)))
    and all($loss[];. as $s|[$passages[]|select(.split==$s.split)] as $group|
      all(["reference","candidate"][];. as $label|$s[$label] as $metric|
        $metric.scored_tokens==([$group[].comparison[$label].scored_tokens]|add)
        and $metric.scored_tokens==(if $s.split=="calibration" then 102 else 101 end)
        and near($metric.nll_sum;[$group[].comparison[$label].nll_sum]|add)
        and near($metric.mean_nll;$metric.nll_sum/$metric.scored_tokens)
        and near($metric.perplexity;($metric.mean_nll|exp)) and ($metric.perplexity_overflow|not)));
    "teacher-forced target coverage and token-weighted loss/PPL")
| require($summary.is_numerical_track and ($summary.passes_exact_quality_gate|not)
    and ($summary.has_identical_browser_generation|not) and $summary.has_fixed_browser_workload
    and $summary.tracked_device_bytes_initial==0 and $summary.tracked_device_bytes_after==0;"numerical-only conclusion and cleanup")
| {passes_audit:true,geometry:($geometry|tonumber),source:$m.revision,executable:$m.executable_sha256,
    reference_matches_corrected_actual_ids:512,candidate_actual_tokens:512,first_divergence_step:$divergence,
    dense_valid_tasks:$summary.dense_valid_tasks,dense_valid_regressions:$summary.dense_valid_regressions,
    inconclusive_tasks:([$tasks[]|select(.reference_score.passed|not)]|length),
    changed_task_outputs:([$tasks[]|select(.reference.output!=.candidate.output)]|length),
    prepared_identities_equal:37,source_plus_prepared_bytes:4472455168,dense_kv_bytes_per_model:37748736,
    activation_scratch_bytes_from_runtime_log:2304,quantizer_launches:20736,ffn_launches:20736,
    sampled_full_vocabulary_max_diff:($logits|map(.max_abs_logit_diff)|max),
    sampled_full_vocabulary_mean_diff_max:($logits|map(.mean_abs_logit_diff)|max),
    sampled_top10_overlap_min:($logits|map(.top10_overlap)|min),
    divergence_probe:[$logits[]|select(.generated_step_zero_based==$divergence)],
    loss:$loss,tracked_device_bytes_after:0,exact_goal_eligible:false,
    recommendation:"numerical research only; not a speed claim or default change"}
