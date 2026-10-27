# Input clocks are normalized with exact 64-bit integer subtraction first.
def require($ok;$why): if $ok then . else error($why) end;
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity")|.[0];
def key: [.context_id,.correlation_id]|map(tostring)|join(":");
def median: sort|.[length/2|floor];
def mmvq: startswith("_Z13mul_mat_vec_qIL9ggml_type8ELi1ELb0ELb0ELb0");
def quantizer: .=="_Z13quantize_q8_1PKfPvlllllj5uint3";
def role($index): if $index==252 then "vocabulary" elif $index%7<3 then "qkv"
  elif $index%7==3 then "out" elif $index%7<6 then "gate_up" else "down" end;
def decode_frame:
  . as $frame|map(select(.symbol|mmvq)) as $dots
  | require(($dots|length)==253 and [$dots[].grid]==[
      range(0;36)|[2048,256,256,2048,11008,11008,2048][]|[.,1,1]]+[[151936,1,1]];
      "decode projection shape/order")
  | (map({key:(.correlation_id|tostring),value:.})|from_entries) as $by_id
  | require(all($dots[];$by_id[(.correlation_id-1|tostring)].symbol|quantizer);
      "projection quantizer dependency")
  | ([range(0;253)|. as $i|$dots[$i] as $d|
      {key:($d.correlation_id|tostring),value:(role($i)+"_projection")},
      {key:($d.correlation_id-1|tostring),value:(role($i)+"_quantization")} ]|from_entries) as $roles
  | map(.+{stage:($roles[(.correlation_id|tostring)] //
      (if (.symbol|startswith("_Z21unary_gated_op_kernel")) then "swiglu"
       elif (.symbol|startswith("_Z13mul_mat_vec_fI6__halffLi1E")) or (.symbol|startswith("_Z12soft_max_f32"))
       then "attention" else "other" end))}) as $tagged
  | {kernels:length,busy_ms:(map(.duration_ns)|add)/1e6,
      span_ms:((map(.relative_end_ns)|max)-(map(.relative_start_ns)|min))/1e6,
      first_correlation_id:.[0].correlation_id,last_correlation_id:.[-1].correlation_id,
      by_stage:($tagged|group_by(.stage)|map({key:.[0].stage,value:{launches:length,ms:(map(.duration_ns)|add)/1e6}})|from_entries)}
  | .ffn_ms=([.by_stage|to_entries[]|select(.key|test("^(gate_up|down)_|^swiglu$"))|.value.ms]|add)
  | .quantization_ms=([.by_stage|to_entries[]|select(.key|endswith("_quantization"))|.value.ms]|add);
. as $e|$e.trace as $records
| $records|one("cuda_trace_start") as $start
| $records|one("cuda_trace_complete") as $done
| $records|one("cuda_activity_complete") as $activity_done
| $records|one("cuda_trace_environment") as $env
| $records|map(select(.kind=="cuda_launch" and .domain=="driver")) as $driver
| $records|map(select(.kind=="cuda_launch" and .domain=="runtime")) as $runtime
| $records|map(select(.kind=="cuda_kernel_activity"))|sort_by(.correlation_id) as $gpu
| require(($records|length)==55462 and ($driver|length)==18486 and ($runtime|length)==18486
    and ($gpu|length)==18486 and $start.pid==3636932 and $start.captures_kernel_activity
    and ($start.captures_modules|not) and ($start.is_timing_measurement|not)
    and $done.launch_records==36972 and $done.dropped_launches==0 and $done.failed_launches==0
    and $done.unknown_symbols==0 and $done.unsubscribe_status==0 and $done.module_records==0
    and $activity_done.kernel_records==18486 and $activity_done.dropped_records==0
    and $activity_done.invalid_timestamps==0 and $activity_done.buffer_bytes==1048576
    and $activity_done.maximum_buffers==8 and $activity_done.record_limit==100000;
    "complete bounded actual runner")
| require($env=={kind:"cuda_trace_environment",cuda_force_ptx_jit:null,cuda_disable_ptx_jit:null,cuda_cache_disable:null};
    "unmodified driver compiler/cache environment")
| require($e.identity.source_revision=="be82fa4c3cbad4cf97b59dfff1b0570be8aa0adb"
    and $e.identity.observer_sha256=="c5539c951fe5d0309f31139bae00efb1882e81e9abe02ab1cefb057c3ebd3d78"
    and $e.identity.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $e.identity.ollama_sha256==$e.pin.ollama_sha256
    and $e.identity.runner_sha256==$e.pin.runner_sha256
    and $e.identity.backend_sha256==$e.pin.backend_sha256
    and $e.identity.runner_pid==3636932 and $e.identity.server_pid==3636581
    and $e.identity.runner_port==44745 and $e.identity.has_authoritative_throughput==false;
    "actual binary/source provenance")
| ($driver|map({key:key,value:.})|from_entries) as $launch_index
| ($runtime|map({key:key,value:.})|from_entries) as $runtime_index
| require(($launch_index|length)==18486 and ($runtime_index|length)==18486
    and ($gpu|map(key)|unique|length)==18486 and all($gpu[];. as $k|
      $launch_index[($k|key)] as $d|$runtime_index[($k|key)] as $r|
      .device_id==0 and .context_id==1 and .stream_id==15 and .has_valid_timestamps
      and (.start_ns|type)=="string" and (.end_ns|type)=="string"
      and .duration_ns>0 and .duration_ns==.relative_end_ns-.relative_start_ns
      and $d.launch_status==0 and $r.launch_status==0
      and all([$d,$r][];.symbol==$k.symbol and .grid==$k.grid and .block==$k.block
        and .shared_bytes==$k.dynamic_shared_bytes))
    and all(range(1;18486);$gpu[.].relative_start_ns>=$gpu[.-1].relative_end_ns);
    "one-to-one activity/callback correlation and serial GPU execution")
| require($e.request.model=="qwen2.5:3b-instruct-q8_0" and $e.request.raw and ($e.request.stream|not)
    and $e.request.prompt==$e.frozen.prompt and $e.request.options=={num_ctx:1024,num_predict:16,num_gpu:99,
      temperature:0,top_k:1,top_p:1,min_p:0,repeat_penalty:1,presence_penalty:0,frequency_penalty:0,seed:0}
    and $e.tokenizer.tokens==$e.frozen.prompt_token_ids and ($e.tokenizer.tokens|length)==65
    and $e.response.done and $e.response.done_reason=="length" and $e.response.eval_count==16
    and $e.response.prompt_eval_count==65 and $e.response.prompt_eval_cached_count==0;
    "canonical prompt IDs, natural length stop and fresh KV")
| [$gpu[]|select((.symbol|mmvq) and .grid==[151936,1,1])|.correlation_id] as $ends
| require(($ends|length)==18 and $ends[0:3]==[1050,2101,3295] and $ends[-1]==19030;
    "warmup/prefill/decode output boundaries")
| [range(2;18)|. as $i|
    [$gpu[]|select(.correlation_id>$ends[$i-1] and .correlation_id<=$ends[$i])]] as $frames
| require(($frames[0]|length)==1193 and all($frames[1:][];length==1013);
    "actual prefill and fifteen one-token decode frames")
| [$frames[1:][]|decode_frame] as $decoded
| {passes_audit:true,source_revision:$e.identity.source_revision,driver_runtime_pairs:18486,
    gpu_activities:18486,generated_tokens:16,prompt_tokens:65,response:$e.response.response,
    startup_kernel_count:2098,prefill_kernel_count:1193,decode_forwards:15,
    prefill:{busy_ms:($frames[0]|map(.duration_ns)|add)/1e6,
      span_ms:(($frames[0]|map(.relative_end_ns)|max)-($frames[0]|map(.relative_start_ns)|min))/1e6},
    decode_frames:$decoded,
    median_decode:{busy_ms:($decoded|map(.busy_ms)|median),span_ms:($decoded|map(.span_ms)|median),
      ffn_ms:($decoded|map(.ffn_ms)|median),quantization_ms:($decoded|map(.quantization_ms)|median),
      by_stage:([$decoded[].by_stage|to_entries[]]|group_by(.key)|map({key:.[0].key,
        value:{launches:.[0].value.launches,median_ms:(map(.value.ms)|median)}})|from_entries)},
    has_authoritative_throughput:false,changes_runtime_defaults:false,
    scope:"one instrumented cold request, short-context GPU spans; not steady512 throughput or matched-clock engine qualification"}
