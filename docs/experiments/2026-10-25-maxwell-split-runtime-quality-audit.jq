def require($condition; $message):
  if $condition then . else error($message) end;
def record($kind):
  map(select(.kind == $kind)) | require(length == 1; "record multiplicity: \($kind)") | .[0];

. as $rows
| record("q8_runtime_manifest") as $manifest
| $rows | record("q8_runtime_summary") as $summary
| $rows | record("q8_runtime_browser_comparison") as $comparison
| $rows | map(select(.kind == "q8_runtime_browser_generation")) as $generations
| $generations | map(select(.label == "control")) | require(length == 1; "control generation") | .[0] as $a
| $generations | map(select(.label == "candidate")) | require(length == 1; "candidate generation") | .[0] as $b
| $qualified | map(select(.kind == "q8_runtime_browser_generation" and .label == "candidate"))
| require(length == 1; "qualified prior candidate") | .[0] as $prior
| $rows | map(select(.kind == "model_quality_task")) as $tasks
| $rows | map(select(.kind == "q8_runtime_prefix_logits")) as $logits
| $rows
| require($manifest.revision == "cf8eba0621903fb5945948f5125b9ad6c0427318"; "source revision")
| require($manifest.worktree == "?? docs/experiments/2026-10-25-maxwell-split-runtime-quality.jsonl"; "unexpected dirty source")
| require($manifest.executable_sha256 == "65c8fca18f62c0859fa61144d6f54a8c7a195eabaddc5e238c9e5fec63f248fd"; "release executable")
| require($manifest.prefill_execution_contract == "prefix-decode-completed-v1"; "prefix contract")
| require($manifest.control_config == {ffn:"packed-f32-128",prefill:"dequant-sgemm"}; "incremental control")
| require($manifest.candidate_config == {ffn:"split-f32x4-128",prefill:"dequant-sgemm"}; "candidate")
| require($manifest.environment == $capture[0].execution_configuration.environment; "frozen environment")
| require($manifest.corpus_sha256 == "2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38"; "frozen corpus")
| require($manifest.reference.sha256 == "4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $manifest.candidate.sha256 == $manifest.reference.sha256; "original model")
| require(($manifest.browser_prompt_ids | length) == 65
    and $a.prompt_token_ids == $prior.prompt_token_ids
    and $b.prompt_token_ids == $a.prompt_token_ids; "canonical prompt")
| require($a.generation.generated_token_ids == $prior.generation.generated_token_ids
    and $a.generation.generated_token_ids == $b.generation.generated_token_ids
    and $a.generation.output == $b.generation.output; "actual continuation equality")
| require(($generations | length) == 2 and all($generations[];
    (.generation.generated_token_ids | length) == 512
    and .generation.metrics.generated_tokens == 512
    and .generation.metrics.stop_reason == "length"
    and .generation.prefill_mode == "packed-prefix-dequant-sgemm"
    and .generation.q8_prefill_temporary_weight_bytes == 90177536
    and .dispatch.totals.cublas_calls == 252
    and .dispatch.operations.dequantize_q8_0_to_f32_colmajor.launches == 252); "complete generation/SGEMM dispatch")
| require($a.dispatch.operations.ffn_q8_f32_packed_warp_128.launches == 18432
    and $b.dispatch.operations.ffn_q8_split_f32x4_128.launches == 18432
    and $a.dispatch.totals == $b.dispatch.totals; "FFN coverage and unchanged enqueue work")
| require(($tasks | length) == 12 and all($tasks[]; .has_identical_generation and .reference.output == .candidate.output)
    and ([$tasks[] | select(.reference_score.passed)] | length) == 7
    and ([$tasks[] | select(.reference_score.passed and (.candidate_score.passed | not))] | length) == 0; "frozen task outcomes")
| require(($logits | map(.generated_step_zero_based)) == [0,1,8,32,128,256,511]
    and all($logits[]; .top10_overlap == 10 and .max_abs_logit_diff >= 0); "prefix diagnostics")
| require($manifest.prepared_control.resident_bytes == 1192935424
    and $manifest.prepared_candidate.resident_bytes == 2917580800
    and $manifest.prepared_candidate.total_weight_payload_bytes == 6197100544
    and ($manifest.prepared_candidate.tensors | length) == 109; "observed cache accounting")
| require(all([$manifest.reference_memory, $manifest.candidate_memory][];
    .resident_kv_bytes == 37748736 and .materialized_f32_bytes == 0
    and .activation_sparsity.is_enabled == false and .active_layer_count == 36); "unchanged model/KV semantics")
| require($comparison.has_identical_generation and $comparison.has_fixed_workload
    and $summary.passes_exact_quality_gate and $summary.dense_valid_tasks == 7
    and $summary.dense_valid_regressions == 0
    and $summary.tracked_device_bytes_initial == 0 and $summary.tracked_device_bytes_after == 0; "completion/cleanup")
| {
    passes_quality_audit:true, source:$manifest.revision,
    scope:"actual full-model quality against current packed128/SGEMM; not HTTP timing",
    actual_browser_ids_equal:512, task_outputs_equal:12, dense_valid_tasks:7, regressions:0,
    prefix_logit_maximum:($logits | map(.max_abs_logit_diff) | max),
    prefix_mean_diff_maximum:($logits | map(.mean_abs_logit_diff) | max),
    top10_overlap_minimum:($logits | map(.top10_overlap) | min),
    additional_gate_up_device_bytes:($manifest.prepared_candidate.resident_bytes - $manifest.prepared_control.resident_bytes),
    source_plus_prepared_candidate_bytes:$manifest.prepared_candidate.total_weight_payload_bytes,
    preparation_ms:[$manifest.prepared_control.preparation_elapsed_ms,$manifest.prepared_candidate.preparation_elapsed_ms],
    diagnostic_resident_generation_ms:[$a.generation.resident_generation_wall_ms,$b.generation.resident_generation_wall_ms],
    makes_http_or_default_claim:false
  }
