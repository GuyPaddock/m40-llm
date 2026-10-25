def median:
  sort | if length % 2 == 1 then .[length/2|floor]
  else (.[length/2-1]+.[length/2])/2 end;
def require($condition; $message): if $condition then . else error($message) end;
. as $rows |
($rows|map(select(.kind=="manifest"))[0]) as $manifest |
($rows|map(select(.kind=="summary"))[-1]) as $summary |
($rows|map(select(.kind=="selected_runtime_ollama_qualification"))[0]) as $qualification |
($summary.comparison) as $comparison |
(if $summary.phase=="confirmation20" then "confirmation" else "screen" end) as $phase |
($rows|map(select(.kind=="trial" and .phase==$phase))) as $primary |
($primary|map(select(.runtime=="m40-llm"))) as $m40 |
($primary|map(select(.runtime=="ollama"))) as $ollama |
($rows|map(select(.kind=="token_trace" and .runtime=="m40-llm"))[0]) as $trace |
($quality|map(select(.kind=="q8_runtime_browser_generation" and .label=="candidate"))[0]) as $expected |
require(($m40|length)==$comparison.pairs and ($ollama|length)==$comparison.pairs; "wrong pair count") |
require($manifest.comparison_kind=="selected-runtime-vs-ollama" and
  $manifest.primary_runtime_config==($quality|map(select(.kind=="q8_runtime_manifest"))[0].candidate_config) and
  $manifest.kv_mode=="off" and $manifest.context_tokens==1024; "wrong runtime/configuration") |
require(all($m40[]; .is_fixed_workload and .matches_completed_quality_output and
  .metrics.generated_tokens==512 and .metrics.prompt_tokens==65 and .metrics.stop_reason=="length" and
  .cache.reused_prompt_tokens==0 and .cache.evaluated_prompt_tokens==65 and
  .q8_prepared_bytes==1192935424 and .final_kv_bytes==37748736); "M40 workload/quality mismatch") |
require(all($ollama[]; .is_fixed_workload and .generated_tokens==512 and .prompt_tokens==65 and
  .stop_reason=="length" and .cache.reused_prompt_tokens==0 and .cache.evaluated_prompt_tokens==65 and
  .prompt_eval_cached_count==0 and .runner_log_end>.runner_log_offset); "Ollama workload/cache mismatch") |
require(($m40|map(.output_sha256)|unique|length)==1 and
  ($ollama|map(.output_sha256)|unique|length)==1; "output changed within engine") |
require($trace.generated_token_ids==$expected.generation.generated_token_ids and
  $trace.output==$expected.generation.output and $trace.matches_measured_output; "actual IDs/output differ from quality") |
require($qualification.passes_completed_exact_quality and $qualification.actual_ids_match_completed_quality and
  $qualification.actual_generated_tokens==512; "missing completed qualification") |
require($qualification.reaches_original_target==($comparison.candidate.median>=53.06 and
  $comparison.candidate.median>=1.30*$comparison.control.median); "inconsistent original target decision") |
{source_revision:$manifest.revision,benchmark_executable_sha256:$manifest.benchmark_executable_sha256,
 primary_runtime_config:$manifest.primary_runtime_config,primary_phase:$phase,pairs:$comparison.pairs,
 comparison:$comparison,qualification:$qualification,actual_ids_match_completed_quality:true,
 all_primary_cache_resets_verified:true,outputs_stable_within_each_engine:true,
 scheduling_scope:"serial awaited HTTP requests; absolute client intervals recorded only for M40, Ollama has per-request runner-log byte ranges",
 observed_gpu_clocks:([$primary[]|(.gpu_before,.gpu_after)|.gpus_csv|split(",")|
   map(gsub("^ +| +$";""))|{sm_mhz:.[6],memory_mhz:.[7]}]|unique),
 phase_medians:($primary|group_by(.runtime)|map({runtime:.[0].runtime,requests:length,
   http_ms:map(.client_elapsed_ms)|median,
   prompt_ms:map(if .runtime=="m40-llm" then .metrics.prefill_elapsed_ms else .prefill_elapsed_ms end)|median,
   decode_ms:map(if .runtime=="m40-llm" then .metrics.decode_elapsed_ms else .decode_elapsed_ms end)|median})),
 m40_counters:($m40|map(.enqueue_counters)|unique),
 prepared_weight_bytes:($m40|map(.q8_prepared_bytes)|unique),dense_kv_bytes:($m40|map(.final_kv_bytes)|unique),
 actual_prefill:($trace|{prefill_mode,prefill_chunk_size,q8_prefill_temporary_weight_bytes})}
