# jq -s --slurpfile quality QUALITY --slurpfile events EVENTS -f THIS HTTP
def require($condition;$message): if $condition then . else error($message) end;
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity: \($kind)")|.[0];
def median: sort|length as $n|if $n%2==0 then (.[($n/2)-1]+.[($n/2)])/2 else .[($n/2)|floor] end;
def identity: del(.preparation_elapsed_ms,.model_hash_elapsed_ms,.packing_elapsed_ms,.upload_elapsed_ms);
def clock: .gpus_csv|split(",")|map(gsub("^ +| +$";""))|{pstate:.[3],sm_mhz:(.[6]|tonumber),memory_mhz:(.[7]|tonumber)};
def processes: .compute_processes_csv|split("\n")|map(split(",")|map(gsub("^ +| +$";""))|{uuid:.[0],pid:.[1],name:.[2]})|sort_by(.pid);
. as $rows|one("manifest") as $m
| $rows|one("profile_comparison_summary") as $s
| $rows|one("profile_token_comparison") as $parity
| $quality|one("q8_runtime_manifest") as $q
| $quality|map(select(.kind=="q8_runtime_browser_generation" and .label=="candidate"))|.[0] as $expected
| $rows|one("q8_runtime_quality_evidence") as $evidence
| $rows|map(select(.kind=="trial" and (.phase=="profile-on" or .phase=="profile-off"))) as $primary
| $rows|map(select(.kind=="token_trace" and .runtime=="m40-llm")) as $traces
| $rows
| require($m.revision=="e83d4613b72c2d38deb947f6b9962ef420b2d69e" and $m.worktree==""
    and $m.benchmark_executable_sha256=="eb6dc69f94a2e67ecd62bc07a02a12e47c26067236ac6f7df40e2c4f78b5b86f"
    and $m.primary_runtime_config==$q.candidate_config and $m.primary_runtime_config.ffn=="joint-shared-scales-128"
    and $m.candidate_runtime_config==$q.candidate_config and $m.quality_control_runtime_config==$q.control_config
    and $m.build_profile=="release" and $m.context_tokens==1024 and $m.kv_mode=="off"
    and $m.max_tokens==512 and $m.prompt_token_ids==$q.browser_prompt_ids
    and $m.model_sha256==$q.candidate.sha256;"clean qualified release/source/model")
| require(($m.environment|del(.M40LLM_STAGE_PROFILE_REPORT))==$q.environment
    and $m.actual_q8_attention_projection_backend=="packed-warp-256"
    and ($m.control_q8_execution|identity)==($q.prepared_candidate|identity)
    and $m.prefill_execution_contract=="prefix-decode-completed-v1"
    and $evidence.passes_exact_quality_gate
    and $evidence.sha256=="adf09f0077f7ec24e7ca61f1a846f8b4e97db4b631fa82c86a6af0201b5e4f24"
    and $m.ollama_version.version=="0.40.11"
    and $m.ollama_executable_sha256=="b0a4becfaf872e3ad5d2e9051244e0dfe55409f03d6e2cd9bd9c34400e201b4c"
    and $m.ollama_environment.LLAMA_ARG_CACHE_RAM=="0" and $m.ollama_environment.OLLAMA_NUM_PARALLEL=="1";"frozen configuration/quality/oracle")
| require(($primary|length)==10 and ([$primary[]|select(.phase=="profile-off")]|length)==5
    and ([$primary[]|select(.phase=="profile-on")]|length)==5
    and ([$primary[].pair]|unique)==[0,1,2,3,4]
    and all($primary|group_by(.pair)[];length==2 and (map(.phase)|sort)==["profile-off","profile-on"])
    and $s.pairs==5 and $s.selected_runtime_config==$q.candidate_config;"prescribed paired coverage")
| require(all($primary[];.is_fixed_workload and .invalid_reason==null and .matches_completed_quality_output
    and .stage_profile_enabled==(.phase=="profile-on") and (.prefill_stage_profile_requested|not)
    and .metrics.generated_tokens==512 and .metrics.prompt_tokens==65 and .metrics.stop_reason=="length"
    and .output==$expected.generation.output and .cache.reused_prompt_tokens==0 and .cache.evaluated_prompt_tokens==65
    and .q8_ffn=={kernel:"joint-shared-scales-128",threads:128}
    and .q8_prefill_backend=="dequant-sgemm" and .q8_attention_projection_backend=="packed-warp-256"
    and .q8_prepared_bytes==2917580800 and .weight_payload_bytes_including_joint==7921745920
    and (.joint_ffn_pairs|identity)==($q.joint_candidate|identity)
    and .final_kv_bytes==37748736
    and .enqueue_counters==$expected.dispatch.totals
    and .launch_deltas==($expected.dispatch.operations|with_entries(select(.value.launches>0))|map_values(.launches)));
    "actual dispatch/output/storage/cache")
| require(($traces|length)==2 and ([$traces[].stage_profile_enabled]|sort)==[false,true]
    and all($traces[];.generated_token_ids==$expected.generation.generated_token_ids
        and .prompt_token_ids==$q.browser_prompt_ids and .matches_measured_output)
    and $parity.actual_ids_equal and $parity.actual_generated_tokens==512
    and $parity.matches_completed_quality and ($parity.is_optimization_claim|not);"actual512 replay equality")
| require(($events|length)==21 and ($events|map(.process_id)|unique)==[611360]
    and ($events|group_by(.session_id)|length)==7
    and all($events|group_by(.session_id)[];map(.generated_step)==[0,255,511]
        and map(.event_pool_reused)==[false,true,true])
    and all($events[];.absolute_position==64+.generated_step and (.stages|length)==400
        and .event_pool_capacity==592 and .gpu_stage_union_ms>0 and .token_wall_ms>0);"event/pool coverage")
| require(all($events[];. as $sample|
    all(["qkv_project","mlp_gate_up_swiglu","mlp_down","out_project","attention","attn_norm","ffn_norm",
        "attn_residual","mlp_residual","rope_q","kv_append_rope_k"][];. as $stage|
        ([$sample.stages[]|select(.stage==$stage)|.layer_id]|sort)==[range(0;36)])
    and all(["embedding","output_norm","lm_head","greedy_argmax"][];. as $stage|
        ([$sample.stages[]|select(.stage==$stage)]|length)==1)
    and all(.stages[];.stream=="decode" and .gpu_elapsed_ms>=0
        and .gpu_end_ms>=.gpu_start_ms));"full layer/global stage coverage")
| ([$primary[]|(.gpu_before,.gpu_after)|clock]|unique) as $clocks
| require($clocks==[{pstate:"P0",sm_mhz:1113,memory_mhz:3004}];"stable clocks")
| ($primary[0].gpu_before|processes) as $processes
| require($processes==[
    {uuid:"GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852",pid:"611360",name:"target/release/examples/compare_ollama"},
    {uuid:"GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852",pid:"612193",name:"/tmp/m40-llm-tools/ollama-0.40.11/lib/ollama/llama-server"}]
    and all($primary[];(.gpu_before|processes)==$processes and (.gpu_after|processes)==$processes);"process isolation")
| ($primary|sort_by(.client_started_unix_ms)) as $ordered
| require(all(range(1;$ordered|length);$ordered[.-1].client_finished_unix_ms<=$ordered[.].client_started_unix_ms);"serial intervals")
| ([$primary[]|select(.phase=="profile-off")|.e2e_tps]|median) as $off
| ([$primary[]|select(.phase=="profile-on")|.e2e_tps]|median) as $on
| ($off/$on-1) as $latency_change
| require($off==$s.profile_off.median and $on==$s.profile_on.median
    and (($on/$off-1-$s.throughput_change)|fabs)<1e-12 and ($latency_change|fabs)<=0.03
    and ($s.is_optimization_claim|not) and ($s.prefill_stage_profile_requested|not);"perturbation attribution gate")
| {passes_audit:true,source:$m.revision,executable:$m.benchmark_executable_sha256,
    actual_ids_equal:512,profile_off:$s.profile_off,profile_on:$s.profile_on,
    throughput_change:$s.throughput_change,latency_change:$latency_change,attribution_limit:0.03,
    original_prepared_identities:109,joint_pairs:36,joint_bytes:1724645376,
    event_rows:21,primary_sessions:5,primary_samples:15,spans_per_sample:400,
    event_pool_capacity:592,gpu_boundaries:20,clocks:$clocks,processes:$processes,
    unchanged_enqueue_counters:true,is_optimization_claim:false}
