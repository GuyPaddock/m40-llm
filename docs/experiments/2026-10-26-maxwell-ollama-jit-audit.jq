def require($ok;$why): if $ok then . else error($why) end;
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity")|.[0];
. as $e | $e.identity as $i
| $e.trace|one("cuda_trace_start") as $start
| $e.trace|one("cuda_trace_environment") as $env
| $e.trace|one("cuda_trace_complete") as $done
| $e.trace|map(select(.kind=="cuda_module")) as $modules
| $e.trace|map(select(.kind=="cuda_launch")) as $launches
| $launches|map(select(.domain=="driver")) as $driver
| "_Z13mul_mat_vec_qIL9ggml_type8ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj" as $mmvq
| "_Z13quantize_q8_1PKfPvlllllj5uint3" as $quantizer
| $e|require($i.source_revision=="3b9eb51806f277b9deeef2aab0c1f6b3f3dc7df0"
    and $i.ollama_sha256==$e.original.binaries.ollama_sha256
    and $i.runner_sha256==$e.original.binaries.llama_server_sha256
    and $i.backend_sha256==$e.original.binaries.cuda_backend_sha256
    and $i.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $i.observer_sha256=="cc155e8fdda002984c2354eac831a0d97be2d699b48244289072efe32a3bda26"
    and $i.runner_pid==3470246 and $i.server_pid==3469761 and $i.runner_port==42173
    and $i.is_timing_measurement==false and $i.changes_runtime_defaults==false;
    "pinned actual binaries/source and observation scope")
| require($start.pid==$i.runner_pid and $start.is_timing_measurement==false and $start.captures_modules==true
    and $start.record_limit==100000 and $start.module_record_limit==64 and $start.module_byte_limit==268435456
    and $env=={kind:"cuda_trace_environment",cuda_force_ptx_jit:null,cuda_disable_ptx_jit:null,cuda_cache_disable:"1"}
    and $done=={kind:"cuda_trace_complete",launch_records:12660,dropped_launches:0,failed_launches:0,
      unknown_symbols:0,unsubscribe_status:0,module_records:16,module_bytes:13956848}
    and ($modules|length)==16 and ($modules|map(.cubin_bytes)|add)==13956848
    and ($modules|map(.image_index))==[range(0;16)]
    and all($modules[];.cubin_bytes>0 and .cubin_bytes<=67108864)
    and ($e.modules|map(del(.sha256,.retained_path)))==$modules
    and all($e.modules[];(.sha256|test("^[0-9a-f]{64}$")) and (.retained_path|startswith("target/research-tools/ollama-jit-inspection/")))
    and $e.modules[1].sha256=="558de8bac091ba17555c23e60465daf8f99f2255b10a22869449985bd2441e6e"
    and $e.modules[2].sha256=="2f84eb83bc0ad4f951d53768044965aeb138f4cf5125ed09e9cf3de7b60d8687";
    "complete actual-runner trace and bounded module images")
| require(($launches|length)==12660 and ($driver|length)==6330
    and all($launches[];.launch_status==0 and (.symbol|length)>0)
    and all($launches|group_by([.context_id,.correlation_id])[];
      length==2 and (map(.domain)|sort)==["driver","runtime"]
      and (map({symbol,grid,block,shared_bytes,launch_status})|unique|length)==1)
    and ([$driver[]|select(.symbol==$mmvq)]|length)==771
    and ([$driver[]|select(.symbol==$quantizer)]|length)==1269
    and ([$driver[]|select(.symbol|startswith("_Z13mul_mat_vec_qIL9ggml_type8ELi2ELb0ELb0ELb0"))]|length)==498
    and all($driver[]|select(.symbol==$mmvq);.block==[32,4,1] and .shared_bytes==0)
    and all($driver[]|select(.symbol==$quantizer);.block==[256,1,1]);
    "actual quantizer/MMVQ launch coverage without domain double counting")
| require($e.request.model=="qwen2.5:3b-instruct-q8_0" and $e.request.raw==true and $e.request.stream==false
    and $e.request.prompt==$e.frozen.prompt and $e.request.options=={num_ctx:1024,num_predict:4,num_gpu:99,
      temperature:0,top_k:1,top_p:1,min_p:0,repeat_penalty:1,presence_penalty:0,frequency_penalty:0,seed:0}
    and $e.response.model==$e.request.model and $e.response.done==true and $e.response.done_reason=="length"
    and $e.response.prompt_eval_count==65 and $e.response.prompt_eval_cached_count==0 and $e.response.eval_count==4
    and $e.tokenizer.tokens==$e.frozen.prompt_token_ids and ($e.tokenizer.tokens|length)==65;
    "canonical actual IDs, bounded completed response and no prompt reuse")
| require($i.loaded_mmvq_words_sha256==$i.offline_mmvq_words_sha256
    and $i.loaded_mmvq_words_sha256=="9c27a7b45565707b1f4094d1d3c022bce84c238c54a9e3faf2b5f9830742f5cb"
    and $i.quantizer_words_sha256=="972c76d2c6c57d489d1d3bf9f71131795b12b7af22632efb3794bb8a8b3430f1"
    and $i.mmvq_word_count==480 and $i.mmvq_xmad_sites==102 and $i.mmvq_dp4a_sites==0;
    "actual selected code matches prior offline assumption")
| {passes_audit:true,source_revision:$i.source_revision,runner_pid:$i.runner_pid,launch_records:12660,
    driver_runtime_pairs:6330,module_images:16,module_bytes:13956848,prompt_ids:65,generated_tokens:4,
    response:$e.response.response,has_loaded_offline_mmvq_word_equality:true,
    mmvq_registers:48,mmvq_static_shared_bytes:384,mmvq_stack_bytes:0,mmvq_local_bytes:0,
    mmvq_static_barrier_sites:1,mmvq_static_shuffle_sites:5,mmvq_static_xmad_sites:102,
    quantizer_registers:23,quantizer_static_shared_bytes:0,
    is_timing_measurement:false,changes_runtime_defaults:false,should_promote_new_runtime_path:false,
    scope:"actual selected official JIT code and whole-process launch coverage, not stage latency or E2E throughput"}
