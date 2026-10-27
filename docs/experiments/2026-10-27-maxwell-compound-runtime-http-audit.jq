# jq -s --arg mode incremental|ollama --arg revision EXPECTED --slurpfile quality QUALITY -f THIS RAW
def require($condition;$message): if $condition then . else error($message) end;
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity: \($kind)")|.[0];
def median: sort|length as $n|if $n%2==0 then (.[($n/2)-1]+.[($n/2)])/2 else .[($n/2)|floor] end;
def variation: . as $v|median as $m|([$v[]|(. - $m)|fabs]|median)/$m;
def identity: del(.model_hash_elapsed_ms,.preparation_elapsed_ms,.packing_elapsed_ms,.upload_elapsed_ms);
def stable_quality: reduce ["prepared_control","prepared_candidate","joint_control","joint_candidate"][] as $part (. ; .[$part]|=identity);
def clock: .gpus_csv|split(",")|map(gsub("^ +| +$";""))|{pstate:.[3],sm_mhz:(.[6]|tonumber),memory_mhz:(.[7]|tonumber)};
def processes: .compute_processes_csv|split("\n")|map(split(",")|map(gsub("^ +| +$";""))|{uuid:.[0],pid:.[1],name:.[2]})|sort_by(.pid);
require($mode=="incremental" or $mode=="ollama";"audit mode")
| . as $rows|one("manifest") as $m
| $quality|one("q8_runtime_manifest") as $q
| $quality|map(select(.kind=="q8_runtime_browser_generation" and .label=="control"))|.[0] as $a
| $quality|map(select(.kind=="q8_runtime_browser_generation" and .label=="candidate"))|.[0] as $b
| $rows|one("q8_runtime_quality_evidence") as $evidence
| $rows|map(select(.kind=="trial" and .phase=="screen")) as $screen
| $rows|map(select(.kind=="trial" and .phase=="confirmation")) as $confirmation
| ($screen+$confirmation) as $all
| (if ($confirmation|length)>0 then $confirmation else $screen end) as $primary
| $rows|map(select(.kind==(if $mode=="incremental" then "q8_execution_summary" else "summary" end)))|last as $s
| $s.comparison as $c
| $primary|map(select(if $mode=="incremental" then .model_variant=="control" else .runtime=="ollama" end)) as $controls
| $primary|map(select(if $mode=="incremental" then .model_variant=="prepared" else .runtime=="m40-llm" end)) as $candidates
| $all|map(select(.runtime=="m40-llm")) as $m40
| $all|map(select(.runtime=="ollama")) as $ollama
| $rows|map(select(.kind=="token_trace" and .runtime=="m40-llm")) as $traces
| $rows
| require($m.worktree=="" and $m.revision==$revision and $m.build_profile=="release"
    and $m.benchmark_executable_sha256=="eb6dc69f94a2e67ecd62bc07a02a12e47c26067236ac6f7df40e2c4f78b5b86f"
    and $m.environment==$q.environment and $m.model_sha256==$q.candidate.sha256
    and $m.ollama_version.version=="0.40.11"
    and $m.ollama_executable_sha256=="b0a4becfaf872e3ad5d2e9051244e0dfe55409f03d6e2cd9bd9c34400e201b4c"
    and $m.ollama_environment.LLAMA_ARG_CACHE_RAM=="0" and $m.ollama_environment.OLLAMA_NUM_PARALLEL=="1"
    and $m.kv_mode=="off" and $m.context_tokens==1024 and $m.max_tokens==512;"clean release/control/isolation")
| require($m.candidate_runtime_config==$q.candidate_config and $m.quality_control_runtime_config==$q.control_config
    and $m.primary_runtime_config==(if $mode=="incremental" then $q.control_config else $q.candidate_config end)
    and $m.prefill_execution_contract=="prefix-decode-completed-v1"
    and $evidence.passes_exact_quality_gate and ($evidence.manifest|stable_quality)==($q|stable_quality)
    and all(["prepared_control","prepared_candidate","joint_control","joint_candidate"][];. as $part|
        all(["model_hash_elapsed_ms","preparation_elapsed_ms","packing_elapsed_ms","upload_elapsed_ms"][];. as $field|
            ((($evidence.manifest[$part][$field]//0)-($q[$part][$field]//0))|fabs)<1e-9));"quality configuration binding")
| require(($m.control_q8_execution|identity)==($q.prepared_control|identity);"original prepared identities")
| require(($c.pairs==5 and ($confirmation|length)==0 and $s.phase=="screen5")
    or ($c.pairs==20 and ($confirmation|length)==40 and ($screen|length)==10 and $s.phase=="confirmation20"
        and any($rows[];.phase=="screen5" and .comparison.needs_fresh_confirmation));"prespecified fresh confirmation")
| require(($primary|length)==2*$c.pairs and ($controls|length)==$c.pairs and ($candidates|length)==$c.pairs
    and all($primary|group_by(.pair)[];length==2) and ([$primary[].pair]|unique)==[range(0;$c.pairs)];"complete pairs")
| require(all($all[];.is_fixed_workload and .cache.reused_prompt_tokens==0 and .cache.evaluated_prompt_tokens==65);"actual work/cache reset")
| require(all($m40[]; (if .model_variant=="control" and $mode=="incremental" then $a else $b end) as $g|
    .metrics.generated_tokens==512 and .metrics.prompt_tokens==65 and .metrics.stop_reason=="length"
    and .output==$b.generation.output and .matches_completed_quality_output and .invalid_reason==null
    and .q8_prepared_bytes==2917580800 and .final_kv_bytes==37748736
    and .q8_ffn.kernel==$g.config.ffn and .q8_ffn.threads==128 and .q8_prefill_backend=="dequant-sgemm"
    and (.joint_ffn_pairs|identity)==($g.joint_ffn_pairs|identity)
    and .weight_payload_bytes_including_joint==(6197100544+$g.joint_ffn_pairs.resident_bytes)
    and .resident_q8_activation_scratch_bytes_after_request==0 and .resident_prefill_weight_scratch_bytes_after_request==0
    and .enqueue_counters==$g.dispatch.totals
    and .launch_deltas==($g.dispatch.operations|with_entries(select(.value.launches>0))|map_values(.launches)));"actual dispatch/memory/output")
| require(($traces|length)==(if $mode=="incremental" then 2 else 1 end)
    and all($traces[];.generated_token_ids==$b.generation.generated_token_ids and .prompt_token_ids==$q.browser_prompt_ids
    and .matches_measured_output and .q8_prefill_temporary_weight_bytes==90177536);"actual512 traces")
| require(all($ollama[];.generated_tokens==512 and .prompt_tokens==65 and .stop_reason=="length"
    and .cache_error==null and .workload_error==null and .runner_log_end>.runner_log_offset);"official cache/work evidence")
| ($ollama|sort_by(.runner_log_offset)) as $runner_order
| require(all(range(1;$runner_order|length);$runner_order[.-1].runner_log_end<=$runner_order[.].runner_log_offset);"distinct runner log regions")
| require(($controls|map(.e2e_tps)|median)==$c.control.median and ($candidates|map(.e2e_tps)|median)==$c.candidate.median
    and (($c.candidate.median/$c.control.median-1-$c.gain)|fabs)<1e-12
    and ((($controls|map(.e2e_tps)|variation)-$c.control.mad_over_median)|fabs)<1e-12
    and ((($candidates|map(.e2e_tps)|variation)-$c.candidate.mad_over_median)|fabs)<1e-12
    and $c.promotion_threshold==([0.03,2*$c.control.mad_over_median,2*$c.candidate.mad_over_median]|max)
    and $c.passes_promotion_gate==($c.gain>=$c.promotion_threshold and $c.bootstrap_gain_ci95[0]>0
        and ([$c.control.mad_over_median,$c.candidate.mad_over_median]|max)<=0.05)
    and $c.bootstrap_resamples==10000 and $c.bootstrap_seed==0;"original performance/variation gate")
| ([$all[]|(.gpu_before,.gpu_after)|clock]|unique) as $clocks
| require($clocks==[{pstate:"P0",sm_mhz:1113,memory_mhz:3004}];"stable clocks")
| ($primary[0].gpu_before|processes) as $processes
| require(($processes|length)==2 and ([$processes[].name]|sort)==[
    "/tmp/m40-llm-tools/ollama-0.40.11/lib/ollama/llama-server","target/release/examples/compare_ollama"]
    and all($all[];(.gpu_before|processes)==$processes and (.gpu_after|processes)==$processes);"isolated processes")
| ($m40|sort_by(.client_started_unix_ms)) as $ordered
| require(all(range(1;$ordered|length);$ordered[.-1].client_finished_unix_ms<=$ordered[.].client_started_unix_ms);"serial timing")
| if $mode=="incremental" then
    ($rows|one("q8_execution_setup")) as $setup
    | require(($setup.control_execution|identity)==($setup.candidate_execution|identity)
        and $setup.tracked_device_payload_before_candidate_bytes==6197100544
        and $setup.tracked_device_payload_after_candidate_bytes==14118846464;"combined resident preparation")
    | $rows|one("q8_promotion_decision")
    | require(.qualifies_as_benchmark_candidate and .actual_generated_ids_equal
        and .passes_completed_runtime_quality_gate and (.changes_runtime_defaults|not);"runtime qualification")
  else
    $rows|one("selected_runtime_ollama_qualification")
    | require(.actual_ids_match_completed_quality and .passes_completed_exact_quality
        and .actual_generated_tokens==512 and .required_absolute_http_tps==53.06
        and ((.required_relative_http_tps-1.30*$c.control.median)|fabs)<1e-12
        and .reaches_original_target==($c.candidate.median>=53.06 and $c.candidate.median>=1.30*$c.control.median)
        and (.changes_runtime_defaults|not);"unchanged original targets")
  end
| {passes_audit:true,mode:$mode,source:$m.revision,primary_pairs:$c.pairs,
    screen_requests:($screen|length),fresh_confirmation_requests:($confirmation|length),
    gpu_boundaries:(2*($all|length)),clocks:$clocks,compute_processes:$processes,
    comparison:$c,actual512_ids_bound:true,original_prepared_identities:109,
    extra_candidate_joint_bytes:1724645376,candidate_weight_payload_bytes:7921745920,
    phase_medians:($primary|group_by(if $mode=="incremental" then .model_variant else .runtime end)
        |map({label:(if $mode=="incremental" then .[0].model_variant else .[0].runtime end),
            http_ms:(map(.client_elapsed_ms)|median),
            prefill_ms:(map(if .runtime=="m40-llm" then .metrics.prefill_elapsed_ms else .prefill_elapsed_ms end)|median),
            decode_ms:(map(if .runtime=="m40-llm" then .metrics.decode_elapsed_ms else .decode_elapsed_ms end)|median)})),
    original_absolute_target:53.06,original_relative_target:(if $mode=="ollama" then 1.30*$c.control.median else null end),
    changes_runtime_defaults:false}
