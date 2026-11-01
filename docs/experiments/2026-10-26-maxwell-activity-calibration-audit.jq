def require($ok;$why): if $ok then . else error($why) end;
def median: sort|.[length/2|floor];
def variation: . as $v|median as $m|map((.-$m)|fabs)|median/$m;
def phase_metrics:
  if .engine=="m40" then .data.generations[4].metrics|
    {decode_ms:.decode_elapsed_ms,prefill_ms:.prefill_elapsed_ms,total_ms:.total_elapsed_ms}
  else .data.generations[4].timings|
    {decode_ms:.predicted_ms,prefill_ms:.prompt_ms,total_ms:(.predicted_ms+.prompt_ms)} end;
# All 5^5 paired bootstrap draws, not a random or pooled confirmation run.
def bootstrap_ci:
  . as $pairs|[1,5,25,125,625] as $powers|
  [range(0;3125)|. as $draw|
    [range(0;5)|$pairs[((($draw/$powers[.])|floor)%5)]]|
    (map(.[1])|median)/(map(.[0])|median)-1]|sort|[.[78],.[3046]];
def summarize_phase:
  . as $pairs|map(.[0]) as $off|map(.[1]) as $on|
  {pairs:$pairs,off_median_ms:($off|median),on_median_ms:($on|median),
    on_latency_change:(($on|median)/($off|median)-1),paired_ci95:($pairs|bootstrap_ci),
    off_variation:($off|variation),on_variation:($on|variation),
    bootstrap:"all 3125 ordered paired resamples; percentile indices 78/3046"};
. as $e|.cases as $cases
| [$e.frozen[]|select(.kind=="q8_runtime_browser_generation" and .label=="candidate")][0] as $reference
| $reference.prompt_token_ids as $prompt|$reference.generation.generated_token_ids[0:16] as $ids
| require(($cases|length)==20 and ($cases|map([.engine,.pair,.mode])|unique|length)==20
    and all(["m40","native"][];. as $engine|
      all(range(0;5);. as $pair|[$cases[]|select(.engine==$engine and .pair==$pair)|.mode]|sort==["off","on"]));
    "five complete independent pairs per engine")
| require($e.original.revision=="1e12a727eda5e752d11d4beeb22dd341b6227708"
    and $e.resumed.revision=="49beaebbefa928d6187da1367b3f60671a99816f"
    and $e.original.engines==null and $e.resumed.engines=="native"
    and all([$e.original,$e.resumed][];.pairs==5 and .warmups==4 and .generated_tokens==16
      and (.observer|startswith("cdebace8dfd0a502f9d8395a75f19e1ba404b84622a85cee525dce6c1018f2f2 "))
      and (.probe|startswith("725b35fc534e7ae62bf17871765a9545c845b401aca1a6af3761aa36af59cb2a "))
      and (.runner|startswith("234b05b2138264f8fb263c3205e85f4c290e8afe5067e280a4f6f90cdac5696b ")));
    "committed actual binaries and native-only resume")
| require(all($cases[]|select(.engine=="m40");. as $case|
    .data.manifest.pid==.pid and (.data.manifest.revision|startswith("1e12a72"))
    and .data.manifest.worktree=="" and .data.manifest.build_profile=="release"
    and .data.manifest.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and .data.manifest.prepared.resident_bytes==2917580800
    and (.data.manifest.prepared.tensors|length)==109
    and .data.manifest.prompt_ids==$prompt and .data.manifest.max_tokens==16
    and .data.manifest.warmups==4
    and (.data.generations|map(.round))==[0,1,2,3,4]
    and (.data.generations|map(.phase))==["warmup","warmup","warmup","warmup","measured"]
    and .data.complete=={kind:"activity_complete",tracked_initial:0,tracked_after:0}
    and all(.data.generations[];.prompt_ids==$prompt and .generated_ids==$ids
      and .metrics.generated_tokens==16 and .metrics.prompt_tokens==65 and .metrics.stop_reason=="length"
      and .prefill_mode=="packed-prefix-dequant-sgemm"
      and .operations.ffn_q8_split_f32x4_128==576
      and .operations.qkv_f32xq8_0_gguf_packed_warp==576
      and .operations.gemm_f32xq8_0_gguf_coltile4_packed_warp==576
      and .operations.dequantize_q8_0_to_f32_colmajor==252
      and .operations.gemm_f32xq8_separate_f32==592
      and .operations.kvcache_reset==1 and .counters.launches==21253 and .counters.cublas_calls==252
      and all([.gpu_before,.gpu_after][];
        (.processes|split("\n")|length)==1 and (.processes|split(",")[1]|gsub("^ +| +$";"")|tonumber)==$case.pid))
    and all([.data.generations[4]|.gpu_before,.gpu_after][];
      .clocks=="GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, P0, 1113, 3004"));
    "real M40 IDs, dispatch, preparation, GPU ownership and cleanup")
| require(all($cases[]|select(.engine=="native");
    .data.request.prompt==$prompt and .data.request.cache_prompt==false
    and .data.request.return_tokens and .data.request.n_predict==16
    and .data.request.temperature==0 and .data.request.top_k==1
    and (.data.generations|length)==5
    and all(.data.generations[];.tokens==$ids and .timings.cache_n==0 and .timings.prompt_n==65
      and .timings.predicted_n==16 and .stop_type=="limit"
      and .generation_settings.ignore_eos==false and .timings.predicted_ms>0 and .timings.prompt_ms>0));
    "native actual IDs, fresh KV and natural bounded stopping")
| require(all($cases[];if .mode=="off" then .trace==null else
    .trace.start.pid==.pid and .trace.start.captures_kernel_activity
    and .trace.start.captures_launch_callbacks==false
    and .trace.callbacks=={kind:"cuda_trace_complete",launch_records:0,dropped_launches:0,failed_launches:0,
      unknown_symbols:0,unsubscribe_status:0,module_records:0,module_bytes:0}
    and .trace.activity.kind=="cuda_activity_complete"
    and .trace.activity.invalid_timestamps==0 and .trace.activity.record_limit==100000
    and .trace.activity.buffer_bytes==1048576 and .trace.activity.maximum_buffers==8
    and .trace.actual_records==.trace.activity.kernel_records
    and (.trace.sha256|test("^[0-9a-f]{64}$"))
    and if .engine=="m40" then .trace.actual_records==100000 and .trace.activity.dropped_records==7520
      else .trace.actual_records==84038 and .trace.activity.dropped_records==0 end end);
    "honest truncated M40 and complete native capture accounting")
| ["m40","native"]|map(. as $engine|
    [$cases[]|select(.engine==$engine)] as $rows|
    [range(0;5)|. as $pair|[$rows[]|select(.pair==$pair)]|
      map({mode,metrics:phase_metrics})|sort_by(.mode)|map(.metrics)] as $pairs|
    (["decode_ms","prefill_ms","total_ms"]|map(. as $phase|
      {key:$phase,value:([$pairs[]|map(.[$phase])]|summarize_phase)})|from_entries) as $stats|
    {engine:$engine,statistics:$stats,
      has_complete_captures:($engine=="native"),
      has_small_decode_overhead:($stats.decode_ms.on_latency_change<=0.03),
      has_stable_decode:($stats.decode_ms.off_variation<=0.05 and $stats.decode_ms.on_variation<=0.05),
      passes_calibration:($engine=="native" and $stats.decode_ms.on_latency_change<=0.03
        and $stats.decode_ms.off_variation<=0.05 and $stats.decode_ms.on_variation<=0.05),
      interpretation:if $engine=="m40" then "capped collector; timing cannot calibrate complete capture"
        else "complete capture but decode overhead exceeds 3%" end}) as $summary
| {passes_integrity_audit:true,cases:20,generations:100,measured_generations:20,
    pairs_per_engine:5,summary:$summary,passes_joint_calibration:all($summary[];.passes_calibration),
    has_quantitative_engine_stage_comparison:false,has_new_throughput_qualification:false,
    changes_runtime_defaults:false,scope:"resident/native 16-token observer cost, not HTTP or actual512 qualification"}
