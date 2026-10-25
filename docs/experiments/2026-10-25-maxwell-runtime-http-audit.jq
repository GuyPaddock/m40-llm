def median: sort | if length % 2 == 1 then .[length/2|floor] else (.[length/2-1]+.[length/2])/2 end;
def require($condition; $message): if $condition then . else error($message) end;
def phase_medians:
  group_by(.model_variant) | map({model_variant:.[0].model_variant, requests:length,
    http_ms:map(.client_elapsed_ms)|median,
    prefill_ms:map(.metrics.prefill_elapsed_ms)|median,
    decode_ms:map(.metrics.decode_elapsed_ms)|median,
    runtime_total_ms:map(.metrics.total_elapsed_ms)|median,
    allocations:map({allocs:.enqueue_counters.device_allocs,frees:.enqueue_counters.device_frees})|unique,
    counters:map(.enqueue_counters)|unique,
    resident_activation_scratch:map(.resident_q8_activation_scratch_bytes_after_request)|unique,
    resident_prefill_scratch:map(.resident_prefill_weight_scratch_bytes_after_request)|unique});
. as $rows |
($rows | map(select(.kind=="manifest"))[0]) as $manifest |
($rows | map(select(.kind=="q8_execution_summary"))[-1]) as $summary |
($rows | map(select(.kind=="trial" and .phase!="q8-warmup"))) as $all |
($all | map(select(if $summary.phase=="confirmation20" then .phase=="confirmation" else .phase=="screen" end))) as $primary |
($rows | map(select(.kind=="token_trace" and .model_variant=="control"))[0]) as $control |
($rows | map(select(.kind=="token_trace" and .model_variant=="prepared"))[0]) as $candidate |
($reference | map(select(.kind=="token_trace" and .model_variant=="prepared"))[0]) as $qualified |
($all | sort_by(.client_started_unix_ms)) as $ordered |
require(($primary|length)==2*$summary.comparison.pairs; "wrong primary pair count") |
require(all($all[]; .is_fixed_workload and .invalid_reason==null and .metrics.generated_tokens==512 and .metrics.prompt_tokens==65 and .metrics.stop_reason=="length" and .cache.reused_prompt_tokens==0 and .cache.evaluated_prompt_tokens==65); "invalid workload") |
require(all(range(1;$ordered|length); $ordered[.-1].client_finished_unix_ms <= $ordered[.].client_started_unix_ms); "overlapping client intervals") |
require($control.generated_token_ids==$qualified.generated_token_ids; "control differs from qualified trace") |
require(all($all[]; .q8_prepared_bytes==1192935424 and .final_kv_bytes==37748736); "memory control changed") |
{source_revision:$manifest.revision, benchmark_executable_sha256:$manifest.benchmark_executable_sha256,
 candidate_runtime_config:$manifest.candidate_runtime_config,
 primary_phase:$summary.phase,primary_pairs:$summary.comparison.pairs,primary_requests:($primary|length),
 all_screen_and_confirmation_requests:($all|length), client_intervals_are_serial:true,
 control_actual_ids_match_qualified_trace:true,
 actual_generated_ids_equal:($control.generated_token_ids==$candidate.generated_token_ids),
 comparison:$summary.comparison,
 promotion_decision:($rows|map(select(.kind=="q8_promotion_decision"))[0]),
 token_comparison:($rows|map(select(.kind=="q8_token_comparison"))[0]),
 phase_medians:($primary|phase_medians),
 gpu_boundary_samples:($primary|length)*2,
 observed_gpu_clocks:([$primary[]|(.gpu_before,.gpu_after)|.gpus_csv|split(",")|map(gsub("^ +| +$";""))|{sm_mhz:.[6],memory_mhz:.[7]}]|unique),
 prepared_weight_bytes:($primary|map(.q8_prepared_bytes)|unique),
 final_kv_bytes:($primary|map(.final_kv_bytes)|unique),
 actual_prefill_trace:($candidate|{prefill_mode,prefill_chunk_size,q8_prefill_temporary_weight_bytes})}
