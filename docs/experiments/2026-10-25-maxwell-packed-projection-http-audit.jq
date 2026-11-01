# jq -s --arg mode incremental|ollama --arg revision EXPECTED_GIT_HEAD --slurpfile quality QUALITY -f THIS RAW
def require($condition;$message): if $condition then . else error($message) end;
def one($kind): map(select(.kind==$kind)) | require(length==1;"record multiplicity: \($kind)") | .[0];
def median: sort | length as $n
    | if $n%2==0 then (.[($n/2)-1]+.[($n/2)])/2 else .[($n/2)|floor] end;
def measure_variation: . as $values|median as $middle
    | ([$values[]|(. - $middle)|fabs]|median)/$middle;
def cache_identity: del(.preparation_elapsed_ms,.model_hash_elapsed_ms);
def stable_manifest: del(.prepared_control.model_hash_elapsed_ms,.prepared_control.preparation_elapsed_ms,
    .prepared_candidate.model_hash_elapsed_ms,.prepared_candidate.preparation_elapsed_ms);
def clock: .gpus_csv|split(",")|map(gsub("^ +| +$";""))
    | {pstate:.[3],sm_mhz:(.[6]|tonumber),memory_mhz:(.[7]|tonumber)};
def processes: .compute_processes_csv|split("\n")|map(split(",")|map(gsub("^ +| +$";""))
    | {uuid:.[0],pid:.[1],name:.[2]})|sort_by(.pid);
require($mode=="incremental" or $mode=="ollama";"audit mode")
| . as $rows
| one("manifest") as $manifest
| $rows|one("q8_runtime_quality_evidence") as $evidence
| $quality|one("q8_runtime_manifest") as $quality_manifest
| $quality|map(select(.kind=="q8_runtime_browser_generation" and .label=="candidate"))
| require(length==1;"quality candidate")|.[0] as $expected
| $quality|map(select(.kind=="q8_runtime_browser_generation" and .label=="control"))|.[0] as $control_quality
| $rows|map(select(.kind=="trial" and .phase=="screen")) as $screen
| $rows|map(select(.kind=="trial" and .phase=="confirmation")) as $confirmation
| ($screen+$confirmation) as $all_measured
| (if ($confirmation|length)>0 then $confirmation else $screen end) as $primary
| $all_measured|map(select(.runtime=="m40-llm")) as $m40
| $all_measured|map(select(.runtime=="ollama")) as $ollama
| $rows|map(select(.kind=="token_trace" and .runtime=="m40-llm")) as $traces
| $rows|map(select(.kind==(if $mode=="incremental" then "q8_execution_summary" else "summary" end)))|last as $summary
| $summary.comparison as $comparison
| $comparison.pairs as $pair_count
| $primary|map(select(if $mode=="incremental" then .model_variant=="control" else .runtime=="ollama" end)) as $controls
| $primary|map(select(if $mode=="incremental" then .model_variant=="prepared" else .runtime=="m40-llm" end)) as $candidates
| $rows
| require($manifest.worktree=="" and $manifest.revision==$revision
    and ($revision|test("^[0-9a-f]{40}$"))
    and $manifest.benchmark_executable_sha256=="4b7f15aa9361a29e42472e7c0194c92863b050ed1c3ab436f2f68514f4ff9cf7"
    and $manifest.build_profile=="release" and $manifest.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $manifest.ollama_version.version=="0.40.11"
    and $manifest.ollama_executable_sha256=="b0a4becfaf872e3ad5d2e9051244e0dfe55409f03d6e2cd9bd9c34400e201b4c"
    and $manifest.environment==$quality_manifest.environment
    and $manifest.ollama_environment.LLAMA_ARG_CACHE_RAM=="0"
    and $manifest.ollama_environment.OLLAMA_NUM_PARALLEL=="1"
    and $manifest.kv_mode=="off" and $manifest.context_tokens==1024 and $manifest.max_tokens==512;"clean source/binaries/controlled workload")
| require($manifest.candidate_runtime_config==$quality_manifest.candidate_config
    and $manifest.quality_control_runtime_config==$quality_manifest.control_config
    and $manifest.primary_runtime_config==(if $mode=="incremental" then $quality_manifest.control_config else $quality_manifest.candidate_config end)
    and $manifest.prefill_execution_contract=="prefix-decode-completed-v1"
    and $evidence.passes_exact_quality_gate
    and $evidence.sha256=="8b499787179f484aeb32e37caf837208e0994da4995918e18e34ba922d3d949e"
    and ($evidence.manifest|stable_manifest)==($quality_manifest|stable_manifest)
    and all(["prepared_control","prepared_candidate"][];. as $part|
      all(["model_hash_elapsed_ms","preparation_elapsed_ms"][];. as $field|
        (($evidence.manifest[$part][$field]-$quality_manifest[$part][$field])|fabs)<1e-9));"completed quality/configuration binding")
| require(($manifest.control_q8_execution|cache_identity)==($quality_manifest.prepared_control|cache_identity);"primary prepared cache identity")
| require(($pair_count==5 and ($confirmation|length)==0 and $summary.phase=="screen5")
    or ($pair_count==20 and ($confirmation|length)==40 and ($screen|length)==10
      and $summary.phase=="confirmation20"
      and any($rows[];.kind=="q8_execution_summary" and .phase=="screen5"
        and .comparison.needs_fresh_confirmation));"prespecified screen/fresh-confirmation sequence")
| require(($primary|length)==2*$pair_count and ($controls|length)==$pair_count and ($candidates|length)==$pair_count
    and all($primary|group_by(.pair)[];length==2)
    and ([$primary[].pair]|unique)==[range(0;$pair_count)];"complete paired coverage")
| require(all($all_measured[];.is_fixed_workload and .cache.reused_prompt_tokens==0
    and .cache.evaluated_prompt_tokens==65);"fixed work/cache reset")
| require(all($m40[];. as $trial|
    (if $mode=="incremental" and .model_variant=="control" then $control_quality else $expected end) as $generation|
    .metrics.generated_tokens==512 and .metrics.prompt_tokens==65 and .metrics.stop_reason=="length"
    and .output==$expected.generation.output and .matches_completed_quality_output and .invalid_reason==null
    and .final_kv_bytes==37748736 and .q8_prepared_bytes==2917580800
    and .q8_execution_families==["gate-up","down","vocabulary"]
    and .q8_ffn=={kernel:"split-f32x4-128",threads:128} and .q8_prefill_backend=="dequant-sgemm"
    and .resident_q8_activation_scratch_bytes_after_request==0 and .resident_prefill_weight_scratch_bytes_after_request==0
    and .enqueue_counters==$generation.dispatch.totals
    and .launch_deltas==($generation.dispatch.operations|with_entries(select(.value.launches>0))|map_values(.launches)));
    "actual M40 dispatch/output/storage")
| require(($traces|length)==(if $mode=="incremental" then 2 else 1 end)
    and all($traces[];.generated_token_ids==$expected.generation.generated_token_ids
      and .prompt_token_ids==$quality_manifest.browser_prompt_ids and .output==$expected.generation.output
      and .matches_measured_output and .q8_prefill_temporary_weight_bytes==90177536);"actual replay IDs and scratch")
| require(all($ollama[];.generated_tokens==512 and .prompt_tokens==65 and .stop_reason=="length"
    and .cache_error==null and .workload_error==null and .runner_log_end>.runner_log_offset);"Ollama work and log evidence")
| ($ollama|sort_by(.runner_log_offset)) as $runner_order
| require(all(range(1;$runner_order|length);$runner_order[.-1].runner_log_end<=$runner_order[.].runner_log_offset);"overlapping runner log regions")
| require(($controls|map(.e2e_tps)|median)==$comparison.control.median
    and ($candidates|map(.e2e_tps)|median)==$comparison.candidate.median
    and (($comparison.candidate.median/$comparison.control.median-1-$comparison.gain)|fabs)<1e-12
    and $comparison.promotion_threshold==0.03 and $comparison.gain>$comparison.promotion_threshold
    and $comparison.bootstrap_gain_ci95[0]>0 and $comparison.passes_promotion_gate
    and ($comparison.needs_variation_extension|not) and ($comparison.needs_fresh_confirmation|not);"original paired performance gate")
| require(((($controls|map(.e2e_tps)|measure_variation)-$comparison.control.mad_over_median)|fabs)<1e-12
    and ((($candidates|map(.e2e_tps)|measure_variation)-$comparison.candidate.mad_over_median)|fabs)<1e-12
    and $comparison.promotion_threshold==([0.03,2*$comparison.control.mad_over_median,2*$comparison.candidate.mad_over_median]|max)
    and $comparison.bootstrap_resamples==10000 and $comparison.bootstrap_seed==0;"actual variation and prespecified threshold")
| ([$all_measured[]|(.gpu_before,.gpu_after)|clock]|unique) as $clocks
| require($clocks==[{pstate:"P0",sm_mhz:1113,memory_mhz:3004}];"clock stability")
| ($primary[0].gpu_before|processes) as $processes
| require(($processes|length)==2 and ([$processes[].name]|sort)==[
    "/tmp/m40-llm-tools/ollama-0.40.11/lib/ollama/llama-server","target/release/examples/compare_ollama"]
    and all($all_measured[];(.gpu_before|processes)==$processes and (.gpu_after|processes)==$processes);"process isolation")
| ($m40|sort_by(.client_started_unix_ms)) as $ordered
| require(all(range(1;$ordered|length);$ordered[.-1].client_finished_unix_ms<=$ordered[.].client_started_unix_ms);"overlapping M40 client intervals")
| if $mode=="incremental" then
    ($rows|one("q8_execution_setup")) as $setup
    | require(($setup.control_execution|cache_identity)==($setup.candidate_execution|cache_identity)
        and ($setup.control_execution|cache_identity)==($quality_manifest.prepared_control|cache_identity)
        and $setup.tracked_device_payload_before_candidate_bytes==6197100544
        and $setup.tracked_device_payload_after_candidate_bytes==12394201088;"unchanged complete preparation and setup memory")
    | $rows|one("q8_promotion_decision")
    | require(.qualifies_as_benchmark_candidate and .actual_generated_ids_equal
        and .passes_completed_runtime_quality_gate and (.changes_runtime_defaults|not);"incremental qualification")
  else
    $rows|one("selected_runtime_ollama_qualification")
    | require(.actual_ids_match_completed_quality and .passes_completed_exact_quality
        and .actual_generated_tokens==512 and .required_absolute_http_tps==53.06
        and ((.required_relative_http_tps-1.30*$comparison.control.median)|fabs)<1e-12
        and .reaches_original_target==($comparison.candidate.median>=53.06 and $comparison.candidate.median>=1.30*$comparison.control.median)
        and (.changes_runtime_defaults|not);"original objective accounting")
  end
| {passes_audit:true,comparison_kind:$mode,source:$manifest.revision,
    executable_sha256:$manifest.benchmark_executable_sha256,
    primary_pairs:$pair_count,primary_requests:($primary|length),screen_requests:($screen|length),
    fresh_confirmation_requests:($confirmation|length),gpu_boundary_samples:(2*($all_measured|length)),observed_clocks:$clocks,
    observed_compute_processes:$processes,completed_quality_and_actual_ids_bound:true,
    prompt_cache_resets_verified:true,all_existing_prepared_tensor_identities_unchanged:109,comparison:$comparison,
    phase_medians:($primary|group_by(if $mode=="incremental" then .model_variant else .runtime end)
      |map({label:(if $mode=="incremental" then .[0].model_variant else .[0].runtime end),requests:length,
        http_ms:(map(.client_elapsed_ms)|median),
        prefill_ms:(map(if .runtime=="m40-llm" then .metrics.prefill_elapsed_ms else .prefill_elapsed_ms end)|median),
        decode_ms:(map(if .runtime=="m40-llm" then .metrics.decode_elapsed_ms else .decode_elapsed_ms end)|median)})),
    prepared_bytes:2917580800,source_plus_prepared_bytes_per_m40_model:6197100544,dense_kv_bytes_per_model:37748736,
    original_absolute_target_http_tps:53.06,
    original_relative_target_http_tps:(if $mode=="ollama" then 1.30*$comparison.control.median else null end),
    reaches_both_original_targets:(if $mode=="ollama" then $comparison.candidate.median>=53.06 and $comparison.candidate.median>=1.30*$comparison.control.median else false end),
    changes_runtime_defaults:false}
