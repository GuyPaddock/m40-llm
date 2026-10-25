def require($condition; $message): if $condition then . else error($message) end;
def record($kind): map(select(.kind == $kind)) | require(length == 1; "record multiplicity: \($kind)") | .[0];
def median: sort | .[length / 2 | floor];
def stable_manifest:
  del(.prepared_control.model_hash_elapsed_ms,.prepared_control.preparation_elapsed_ms,
      .prepared_candidate.model_hash_elapsed_ms,.prepared_candidate.preparation_elapsed_ms);
def clock:
  .gpus_csv | split(",") | map(gsub("^ +| +$"; ""))
  | {pstate:.[3],sm_mhz:(.[6]|tonumber),memory_mhz:(.[7]|tonumber)};

require($mode == "incremental" or $mode == "ollama"; "audit mode")
| . as $rows
| record("manifest") as $manifest
| $rows | record("q8_runtime_quality_evidence") as $evidence
| $quality | record("q8_runtime_manifest") as $quality_manifest
| $quality | map(select(.kind == "q8_runtime_browser_generation" and .label == "candidate"))
| require(length == 1; "quality candidate") | .[0].generation as $expected
| $rows | map(select(.kind == "trial" and .phase == "screen")) as $primary
| $primary | map(select(.runtime == "m40-llm")) as $m40
| $primary | map(select(.runtime == "ollama")) as $ollama
| $rows | map(select(.kind == "token_trace" and .runtime == "m40-llm")) as $traces
| $rows | record(if $mode == "incremental" then "q8_execution_summary" else "summary" end) as $summary
| $summary.comparison as $comparison
| $primary | map(select(if $mode == "incremental" then .model_variant == "control" else .runtime == "ollama" end)) as $controls
| $primary | map(select(if $mode == "incremental" then .model_variant == "prepared" else .runtime == "m40-llm" end)) as $candidates
| $rows
| require($manifest.worktree == ""; "dirty benchmark source")
| require($manifest.revision == (if $mode == "incremental" then "69baad2719f2590775f9212654fadbf2d1d64815" else "f93bb89767cb8d47b926903975ece5211d06d6cb" end); "source revision")
| require($manifest.benchmark_executable_sha256 == (if $mode == "incremental" then "63d616833d1f732b55a19d19c5a2bb8da495aab9bf2d9be4e7980f86d7378560" else "9540358f1eb1eee6c4658125f27387694125f8cf670e377b54bb3ee1c85a0e2e" end); "release executable")
| require($manifest.candidate_runtime_config == $quality_manifest.candidate_config
    and $manifest.quality_control_runtime_config == $quality_manifest.control_config
    and $manifest.prefill_execution_contract == "prefix-decode-completed-v1"
    and $evidence.passes_exact_quality_gate
    and $evidence.sha256 == "f10fe1f6adbba9476ebc8ec26c8e135534230607c624ffa4757c69417e3c2515"
    and ($evidence.manifest|stable_manifest) == ($quality_manifest|stable_manifest)
    and all(["prepared_control","prepared_candidate"][]; . as $part |
      all(["model_hash_elapsed_ms","preparation_elapsed_ms"][]; . as $field |
        (($evidence.manifest[$part][$field] - $quality_manifest[$part][$field])|fabs) < 1e-9)); "completed quality binding")
| require(($primary | length) == 10 and ($controls|length) == 5 and ($candidates|length) == 5
    and $comparison.pairs == 5; "primary pair count")
| require(all($primary[]; .is_fixed_workload and .cache.reused_prompt_tokens == 0
    and .cache.evaluated_prompt_tokens == 65); "workload/cache validation")
| require(all($m40[]; .metrics.generated_tokens == 512 and .metrics.prompt_tokens == 65
    and .metrics.stop_reason == "length" and .output == $expected.output
    and .matches_completed_quality_output and .invalid_reason == null
    and .final_kv_bytes == 37748736
    and .resident_q8_activation_scratch_bytes_after_request == 0
    and .resident_prefill_weight_scratch_bytes_after_request == 0
    and .enqueue_counters.cublas_calls == 252
    and .launch_deltas.dequantize_q8_0_to_f32_colmajor == 252
    and .q8_prefill_backend == "dequant-sgemm"
    and (if $mode == "incremental" and .model_variant == "control" then
      .q8_prepared_bytes == 1192935424 and .q8_ffn.kernel == "packed-f32"
      and .launch_deltas.ffn_q8_f32_packed_warp_128 == 18432
    else .q8_prepared_bytes == 2917580800 and .q8_ffn.kernel == "split-f32x4-128"
      and .launch_deltas.ffn_q8_split_f32x4_128 == 18432 end)); "M40 dispatch/storage/output")
| require(($traces | length) == (if $mode == "incremental" then 2 else 1 end)
    and all($traces[]; .generated_token_ids == $expected.generated_token_ids
      and .prompt_token_ids == $quality_manifest.browser_prompt_ids
      and .output == $expected.output and .matches_measured_output
      and .q8_prefill_temporary_weight_bytes == 90177536); "actual replay IDs")
| require(all($ollama[]; .generated_tokens == 512 and .prompt_tokens == 65
    and .stop_reason == "length" and .cache_error == null and .workload_error == null
    and .runner_log_end > .runner_log_offset); "Ollama fixed work and runner log")
| ($ollama|sort_by(.runner_log_offset)) as $runner_order
| require(all(range(1;$runner_order|length);
    $runner_order[.-1].runner_log_end <= $runner_order[.].runner_log_offset); "overlapping runner log regions")
| require(($controls|map(.e2e_tps)|median) == $comparison.control.median
    and ($candidates|map(.e2e_tps)|median) == $comparison.candidate.median
    and (($comparison.candidate.median / $comparison.control.median - 1 - $comparison.gain)|fabs) < 1e-12
    and $comparison.gain > $comparison.promotion_threshold
    and $comparison.bootstrap_gain_ci95[0] > 0
    and $comparison.passes_promotion_gate
    and ($comparison.needs_variation_extension|not)
    and ($comparison.needs_fresh_confirmation|not); "paired timing decision")
| ([$primary[] | (.gpu_before,.gpu_after) | clock] | unique) as $clocks
| require($clocks == [{pstate:"P0",sm_mhz:1113,memory_mhz:3004}]; "clock variation")
| if $mode == "incremental" then
    ($m40 | sort_by(.client_started_unix_ms)) as $ordered
    | require(all(range(1;$ordered|length); $ordered[.-1].client_finished_unix_ms <= $ordered[.].client_started_unix_ms); "overlapping M40 intervals")
    | $rows | record("q8_promotion_decision")
    | require(.qualifies_as_benchmark_candidate and .actual_generated_ids_equal
      and .passes_completed_runtime_quality_gate and (.changes_runtime_defaults|not); "incremental promotion")
  else
    $rows | record("selected_runtime_ollama_qualification")
    | require(.actual_ids_match_completed_quality and .passes_completed_exact_quality
      and .actual_generated_tokens == 512 and .required_absolute_http_tps == 53.06
      and ((.required_relative_http_tps - 1.30 * $comparison.control.median)|fabs) < 1e-12
      and (.reaches_original_target|not) and (.changes_runtime_defaults|not); "original target status")
  end
| {
    passes_audit:true,comparison_kind:$mode,source:$manifest.revision,
    executable_sha256:$manifest.benchmark_executable_sha256,
    primary_pairs:5,primary_requests:10,gpu_boundary_samples:20,observed_clocks:$clocks,
    completed_quality_and_actual_ids_bound:true,all_prompt_cache_resets_verified:true,
    comparison:$comparison,
    phase_medians:($primary | group_by(if $mode == "incremental" then .model_variant else .runtime end)
      | map({label:(if $mode == "incremental" then .[0].model_variant else .[0].runtime end),requests:length,
        http_ms:(map(.client_elapsed_ms)|median),
        prefill_ms:(map(if .runtime == "m40-llm" then .metrics.prefill_elapsed_ms else .prefill_elapsed_ms end)|median),
        decode_ms:(map(if .runtime == "m40-llm" then .metrics.decode_elapsed_ms else .decode_elapsed_ms end)|median)})),
    memory_scope:"requested prepared payload plus separate native dense KV; two M40 models only for incremental pairs, plus idle Ollama; no concurrent inference",
    prepared_weight_bytes:($m40|map(.q8_prepared_bytes)|unique),dense_kv_bytes:37748736,
    original_goal_achieved:false,changes_runtime_defaults:false
  }
